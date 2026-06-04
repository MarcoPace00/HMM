"""
HMM regime-switching trading pipeline — HMM_2: volatility targeting.

Identical to HMM_1, except the long/flat position is scaled by a causal
volatility-targeting overlay (`vol_target_leverage`) so trailing realized vol
tracks `Config.target_vol`. This de-risks in turbulent regimes to cut
drawdowns, and is applied consistently on both the VAL window (so model
selection sees the traded series) and the TEST window.

For each asset:
  1. Download daily simple returns.
  2. Split 60% in-sample / 40% OOS.
  3. Grid-search (TRAIN_WINDOW, VAL_WINDOW, TEST_WINDOW, THRESHOLD) on the
     in-sample part, scoring each combo by the Sharpe of the stitched in-sample
     TEST segments produced by a walk-forward that, at every step, picks the
     best of 4 HMMs on the VAL window and trades it on the TEST window.
  4. Re-run the walk-forward with the chosen combo on the OOS part.
  5. Plot the 5 stitched equity curves + drawdowns; legend carries OOS stats.

Models (per step the best on VAL-Sharpe is traded on TEST):
  - CategoricalHMM, 2 states   (obs: DOWN/-1, FLAT/0, UP/+1 via THRESHOLD)
  - CategoricalHMM, 3 states
  - GMMHMM, 2 states x 2 mixtures   (obs: raw returns)

Signal (long/flat, no look-ahead):
  next_state = filtered_posterior(x_{1:t-1}) @ transmat
  score      = next_state @ per_state_expected_direction
  signal_t   = 1 if score > 0 else 0      # applied to return_t
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")  # hmmlearn convergence chatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hmm")

# hmmlearn emits "Degenerate mixture covariance" via *logging* (not warnings),
# so filterwarnings above misses it. It fires when a GMM mixture collapses onto
# (near-)identical returns -> zero variance. Hard-flooring the variance enough
# to prevent it (covars_weight) would impose a ~1% std floor that distorts the
# model, and fit_model already guards degenerate fits (try/except + best-LL),
# while walk_forward's val-Sharpe selection rarely picks a collapsed model.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
ModelKind = Literal["cat", "gmm"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: ModelKind
    n_states: int
    n_mix: int = 1  # only used by GMMHMM


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("cat_2s", "cat", 2),
    ModelSpec("cat_3s", "cat", 3),
    ModelSpec("gmm_2s2m", "gmm", 2, n_mix=2),
)


@dataclass(frozen=True)
class HParams:
    train_window: int
    val_window: int
    test_window: int
    threshold: float


@dataclass
class Config:
    tickers: tuple[str, ...] = ("SPY", "QQQ", "TQQQ", "UPRO", "BTC-USD")
    start: str = "2016-01-01"
    end: str | None = None
    split_frac: float = 0.60

    train_grid: tuple[int, ...] = (21, 252, 504)
    val_grid: tuple[int, ...] = (21, 42)
    test_grid: tuple[int, ...] = (1, 7, 21)
    threshold_grid: tuple[float, ...] = (0.0, 0.005, 0.01)

    hmm_iter: int = 50
    seeds: tuple[int, ...] = (0,)          # add seeds for more restarts (slower)
    min_steps: int = 2                     # require >=2 walk-forward steps to score

    # volatility targeting: scale the long/flat position so trailing realized
    # vol tracks target_vol. De-risk-only by default (cap leverage at 1.0).
    target_vol: float = 0.15               # annualized vol target
    vol_lookback: int = 20                 # trailing window for realized vol
    max_leverage: float = 1.0              # cap; 1.0 = never lever beyond fully long

    def grid(self) -> list[HParams]:
        return [
            HParams(tr, va, te, th)
            for tr, va, te, th in product(
                self.train_grid, self.val_grid, self.test_grid, self.threshold_grid
            )
        ]


def ann_factor(ticker: str) -> int:
    return 365 if "BTC" in ticker.upper() else 252


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def download_returns(cfg: Config) -> dict[str, pd.Series]:
    """One simple-return Series per ticker, on each ticker's own calendar."""
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for tk in cfg.tickers:
        px = yf.download(tk, start=cfg.start, end=cfg.end, auto_adjust=True,
                         progress=False)["Close"]
        px = px.squeeze("columns") if isinstance(px, pd.DataFrame) else px
        ret = px.pct_change().dropna()
        ret.name = tk
        if len(ret) < 600:
            log.warning("%s has only %d returns", tk, len(ret))
        out[tk] = ret
        log.info("loaded %-8s %d returns  %s -> %s", tk, len(ret),
                 ret.index[0].date(), ret.index[-1].date())
    return out


# --------------------------------------------------------------------------- #
# Observations & models                                                       #
# --------------------------------------------------------------------------- #
DIR_VALUE = np.array([-1.0, 0.0, 1.0])  # DOWN, FLAT, UP  -> codes 0,1,2


def discretize(ret: np.ndarray, threshold: float) -> np.ndarray:
    """0=DOWN (< -thr), 1=FLAT, 2=UP (> thr)."""
    obs = np.ones(len(ret), dtype=int)
    obs[ret < -threshold] = 0
    obs[ret > threshold] = 2
    return obs


def build_model(spec: ModelSpec, cfg: Config, seed: int):
    from hmmlearn import hmm

    if spec.kind == "cat":
        return hmm.CategoricalHMM(
            n_components=spec.n_states, n_features=3,
            n_iter=cfg.hmm_iter, random_state=seed, tol=1e-3,
        )
    return hmm.GMMHMM(
        n_components=spec.n_states, n_mix=spec.n_mix, covariance_type="diag",
        n_iter=cfg.hmm_iter, random_state=seed, tol=1e-3,
    )


def fit_model(spec: ModelSpec, X: np.ndarray, cfg: Config):
    """Fit with restarts; return best-likelihood model or None on failure."""
    best, best_ll = None, -np.inf
    for seed in cfg.seeds:
        model = build_model(spec, cfg, seed)
        try:
            model.fit(X)
            ll = model.score(X)
        except Exception:  # noqa: BLE001 - degenerate covariances etc.
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = model, ll
    return best


def state_direction(model, spec: ModelSpec) -> np.ndarray:
    """Per-state expected directional score (sign is what matters)."""
    if spec.kind == "cat":
        return model.emissionprob_ @ DIR_VALUE            # (n_states,)
    # GMMHMM: mixture-weighted mean per state (n_features == 1)
    return (model.weights_ * model.means_[:, :, 0]).sum(axis=1)


# --------------------------------------------------------------------------- #
# Signals & backtest                                                          #
# --------------------------------------------------------------------------- #
def segment_signals(model, spec: ModelSpec, obs2d: np.ndarray,
                    seg_lo: int, seg_hi: int, train_window: int) -> np.ndarray:
    """
    Long/flat signal for each day t in [seg_lo, seg_hi).
    Uses only obs strictly before t  ->  no look-ahead.
    """
    s_dir = state_direction(model, spec)
    A = model.transmat_
    sig = np.zeros(seg_hi - seg_lo)
    for i, t in enumerate(range(seg_lo, seg_hi)):
        hist = obs2d[max(0, t - train_window):t]
        if len(hist) == 0:
            continue
        filt = model.predict_proba(hist)[-1]      # filtered P(z_{t-1}|x_{1:t-1})
        score = (filt @ A) @ s_dir
        sig[i] = 1.0 if score > 0 else 0.0
    return sig


def safe_sharpe(r: np.ndarray, ann: int) -> float:
    r = np.asarray(r, float)
    if r.size < 2:
        return -np.inf
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(ann))


def cagr(equity: np.ndarray, ann: int) -> float:
    n = len(equity)
    if n < 2 or equity[-1] <= 0:
        return float("nan")
    return float(equity[-1] ** (ann / n) - 1.0)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def vol_target_leverage(ret: np.ndarray, cfg: Config, ann: int) -> np.ndarray:
    """
    Causal per-day leverage that scales the position toward cfg.target_vol.
    leverage_t uses only returns strictly before t (trailing realized vol
    shifted by one day), so it adds no look-ahead. Capped at cfg.max_leverage;
    0 until vol_lookback history exists (i.e. flat during warm-up).
    """
    s = pd.Series(ret)
    daily_target = cfg.target_vol / np.sqrt(ann)
    realized = s.rolling(cfg.vol_lookback).std(ddof=1).shift(1)  # window ends at t-1
    lev = (daily_target / realized).clip(upper=cfg.max_leverage)
    return lev.fillna(0.0).to_numpy()


# --------------------------------------------------------------------------- #
# Walk-forward                                                                #
# --------------------------------------------------------------------------- #
def walk_forward(ret: np.ndarray, dates: pd.DatetimeIndex, hp: HParams,
                 cfg: Config, ann: int, ts_start: int, ts_end: int,
                 ) -> tuple[pd.Series, Counter]:
    """
    Tile non-overlapping TEST segments over [ts_start, ts_end).
    Each step: fit all models on TRAIN, pick best VAL-Sharpe, trade TEST.
    History for the earliest steps may reach before ts_start (allowed OOS).
    Returns (stitched daily strategy returns over TEST, count of how many
    steps each model spec was the selected one).
    """
    cat_obs = discretize(ret, hp.threshold).reshape(-1, 1)
    ret_obs = ret.reshape(-1, 1)
    lev = vol_target_leverage(ret, cfg, ann)   # causal vol-targeting overlay
    pieces: list[pd.Series] = []
    picks: Counter = Counter()

    ts = ts_start
    while ts + hp.test_window <= ts_end:
        train_lo = ts - hp.val_window - hp.train_window
        val_lo = ts - hp.val_window
        if train_lo < 0:                       # not enough history yet
            ts += hp.test_window
            continue

        best_model, best_spec, best_val = None, None, -np.inf
        for spec in MODELS:
            obs2d = cat_obs if spec.kind == "cat" else ret_obs
            model = fit_model(spec, obs2d[train_lo:val_lo], cfg)
            if model is None:
                continue
            val_sig = segment_signals(model, spec, obs2d, val_lo, ts, hp.train_window)
            val_sh = safe_sharpe(val_sig * lev[val_lo:ts] * ret[val_lo:ts], ann)
            if val_sh > best_val:
                best_model, best_spec, best_val = model, spec, val_sh

        if best_model is not None:
            picks[best_spec.name] += 1
            obs2d = cat_obs if best_spec.kind == "cat" else ret_obs
            test_sig = segment_signals(best_model, best_spec, obs2d,
                                       ts, ts + hp.test_window, hp.train_window)
            test_ret = (test_sig * lev[ts:ts + hp.test_window]
                        * ret[ts:ts + hp.test_window])
            pieces.append(pd.Series(test_ret, index=dates[ts:ts + hp.test_window]))
        ts += hp.test_window

    if not pieces:
        return pd.Series(dtype=float), picks
    return pd.concat(pieces), picks


# --------------------------------------------------------------------------- #
# Per-asset driver                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class AssetResult:
    ticker: str
    best_hp: HParams
    split_date: pd.Timestamp
    equity: pd.Series          # stitched IS + OOS, normalised to 1
    drawdown: pd.Series
    oos_sharpe: float
    oos_cagr: float
    oos_maxdd: float


def fmt_picks(picks: Counter) -> str:
    """Model-selection tally in MODELS order, e.g. 'cat_2s:5 cat_3s:3 gmm_2s2m:2'."""
    total = sum(picks.values())
    if total == 0:
        return "none"
    return "  ".join(f"{s.name}:{picks[s.name]}" for s in MODELS)


def run_asset(ticker: str, ret_s: pd.Series, cfg: Config) -> AssetResult | None:
    ret = ret_s.to_numpy(float)
    dates = ret_s.index
    n = len(ret)
    split = int(n * cfg.split_frac)
    ann = ann_factor(ticker)

    # ---- grid-search on in-sample 60% --------------------------------------
    best_hp, best_score, best_is_ret, best_is_picks = None, -np.inf, None, None
    for hp in cfg.grid():
        if hp.train_window + hp.val_window + hp.test_window > split:
            continue
        is_start = hp.train_window + hp.val_window
        is_ret, is_picks = walk_forward(ret, dates, hp, cfg, ann, is_start, split)
        n_steps = len(is_ret) // hp.test_window
        if n_steps < cfg.min_steps:
            continue
        score = safe_sharpe(is_ret.to_numpy(), ann)
        if score > best_score:
            best_hp, best_score, best_is_ret, best_is_picks = hp, score, is_ret, is_picks

    if best_hp is None:
        log.warning("%s: no valid hyper-parameter combo", ticker)
        return None
    log.info("%-8s best IS sharpe=%.2f  %s", ticker, best_score, best_hp)
    log.info("%-8s IS  model picks: %s", ticker, fmt_picks(best_is_picks))

    # ---- OOS walk-forward with chosen combo --------------------------------
    oos_ret, oos_picks = walk_forward(ret, dates, best_hp, cfg, ann, split, n)
    if oos_ret.empty:
        log.warning("%s: empty OOS", ticker)
        return None
    log.info("%-8s OOS model picks: %s", ticker, fmt_picks(oos_picks))

    # ---- stitch IS (winning combo) + OOS for plotting ----------------------
    full_ret = pd.concat([best_is_ret, oos_ret])
    full_ret = full_ret[~full_ret.index.duplicated(keep="last")].sort_index()
    equity = (1.0 + full_ret).cumprod()
    equity /= equity.iloc[0]
    dd = equity / equity.cummax() - 1.0

    oos_eq = (1.0 + oos_ret).cumprod().to_numpy()
    return AssetResult(
        ticker=ticker,
        best_hp=best_hp,
        split_date=dates[split],
        equity=equity,
        drawdown=dd,
        oos_sharpe=safe_sharpe(oos_ret.to_numpy(), ann),
        oos_cagr=cagr(oos_eq, ann),
        oos_maxdd=max_drawdown(oos_eq),
    )


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def plot_results(results: list[AssetResult], out_html: str,
                 benchmark: pd.Series | None = None,
                 benchmark_name: str = "SPY B&H") -> go.Figure:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=("Equity (stitched: IS tuned | OOS)",
                        "Drawdown"),
    )

    # Align every curve to a common start so the comparison is fair. Assets
    # have different tuned warm-ups (and BTC trades a denser 365-day calendar),
    # so without this BTC's equity would begin years before the others. Clip
    # each curve to the latest per-asset start and renormalise to 1 there;
    # drawdown is recomputed over that visible window for consistency.
    common_start = max(r.equity.index[0] for r in results)
    log.info("aligning equity curves to common start %s", common_start.date())

    # Buy-and-hold benchmark over the same displayed window (always invested,
    # no tuning), normalised to 1 at common_start so it lines up with the rest.
    if benchmark is not None:
        bench = benchmark.loc[common_start:]
        if not bench.empty:
            bench_eq = (1.0 + bench).cumprod()
            bench_eq /= bench_eq.iloc[0]
            bench_dd = bench_eq / bench_eq.cummax() - 1.0
            bench_np = bench_eq.to_numpy()
            label = (f"{benchmark_name} | Sh {safe_sharpe(bench.to_numpy(), 252):.2f} | "
                     f"CAGR {cagr(bench_np, 252):.1%} | DD {max_drawdown(bench_np):.1%}")
            fig.add_trace(
                go.Scatter(x=bench_eq.index, y=bench_eq.values, name=label,
                           legendgroup="benchmark",
                           line=dict(color="#555555", width=2.0, dash="dash")),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=bench_dd.index, y=bench_dd.values, name=label,
                           legendgroup="benchmark", showlegend=False,
                           line=dict(color="#555555", width=1.2, dash="dash")),
                row=2, col=1)

    for i, r in enumerate(results):
        c = palette[i % len(palette)]
        eq = r.equity.loc[common_start:]
        if eq.empty:
            continue
        eq = eq / eq.iloc[0]
        dd = eq / eq.cummax() - 1.0
        label = (f"{r.ticker} | Sh {r.oos_sharpe:.2f} | "
                 f"CAGR {r.oos_cagr:.1%} | DD {r.oos_maxdd:.1%}")
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq.values, name=label,
                       legendgroup=r.ticker, line=dict(color=c, width=1.6)),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd.values, name=label,
                       legendgroup=r.ticker, showlegend=False,
                       line=dict(color=c, width=1.0)),
            row=2, col=1)

    # 60% split reference (equity ETFs share a calendar; BTC differs slightly)
    ref = next((r for r in results if r.ticker == "SPY"), results[0])
    for row in (1, 2):
        fig.add_vline(x=ref.split_date, line=dict(color="black", dash="dash",
                                                  width=1), row=row, col=1)
    fig.add_annotation(x=ref.split_date, y=1, yref="paper", showarrow=False,
                       text="  OOS →", xanchor="left", font=dict(size=11))

    fig.update_yaxes(row=1, col=1, title_text="growth of 1")
    fig.update_yaxes(row=2, col=1, title_text="drawdown", tickformat=".0%")
    fig.update_layout(
        title=f"HMM regime strategy — OOS metrics in legend "
              f"(split @ {ref.split_date.date()}, SPY ref)",
        template="plotly_white", height=820, hovermode="x unified",
        legend=dict(font=dict(size=11)),
    )
    fig.write_html(out_html)
    log.info("wrote %s", out_html)
    return fig


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(cfg: Config | None = None,
         returns: dict[str, pd.Series] | None = None,
         out_html: str = "hmm_equity_voltarget.html",
         n_jobs: int | None = None) -> go.Figure:
    """
    Run every ticker through `run_asset` and plot the results.

    Tickers are independent and CPU-bound, so each is processed in its own
    process. `n_jobs` caps the worker count (default: min(#tickers, #cpus));
    pass n_jobs=1 to force the serial path for debugging.
    """
    cfg = cfg or Config()
    returns = returns or download_returns(cfg)
    tickers = [tk for tk in cfg.tickers if tk in returns]

    # Pin BLAS to one thread per worker. Spawned children inherit os.environ
    # and import numpy fresh, so they pick up the limit; this prevents the
    # processes from oversubscribing cores via nested BLAS threads (which can
    # be slower than serial). HMM matrices are tiny, so we lose nothing.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    results: list[AssetResult] = []
    if n_jobs == 1 or len(tickers) <= 1:
        for tk in tickers:
            res = run_asset(tk, returns[tk], cfg)
            if res is not None:
                results.append(res)
    else:
        max_workers = min(len(tickers), n_jobs or os.cpu_count() or 1)
        log.info("running %d tickers across %d processes", len(tickers), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_asset, tk, returns[tk], cfg): tk for tk in tickers}
            for fut in as_completed(futs):
                tk = futs[fut]
                try:
                    res = fut.result()
                except Exception:  # noqa: BLE001 - surface worker failures, keep going
                    log.exception("%s: worker failed", tk)
                    continue
                if res is not None:
                    results.append(res)
        # restore the configured ticker order (palette/legend stability)
        results.sort(key=lambda r: cfg.tickers.index(r.ticker))

    if not results:
        raise RuntimeError("no asset produced results")
    return plot_results(results, out_html, benchmark=returns.get("SPY"))


if __name__ == "__main__":
    main()