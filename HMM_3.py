"""
HMM regime-switching trading pipeline — HMM_3: feature-based observations.

Vol-targeting overlay + equal-weight model ensemble, with the HMM fit on a
causal trailing FEATURE matrix (momentum + volatility, see `build_features`)
rather than raw daily returns / discretised up-down-flat codes. Daily returns
are ~all noise, so states fit on them flicker; states fit on trailing
momentum/vol should track the persistent regime structure the strategy
actually cares about.

Universe: SPY / QQQ / TQQQ / UPRO, each benchmarked against its own buy-and-
hold. Two figures are produced:
  * the stitched equity + drawdown comparison (plot_results), and
  * a per-asset regime map (plot_regimes): the OOS buy-and-hold equity line
    with the detected regime (bull / bear / flat) shaded behind it.

For each asset:
  1. Download daily simple returns; derive the trailing feature matrix.
  2. Split 60% in-sample / 40% OOS.
  3. Grid-search (TRAIN_WINDOW, TEST_WINDOW) on the in-sample part, scoring
     each combo by the Sharpe of the stitched in-sample TEST segments produced
     by a walk-forward that, at every step, fits all HMMs on TRAIN (right up to
     TEST) and trades their equal-weight ensemble (vol-targeted) on TEST.
  4. Re-run the walk-forward with the chosen combo on the OOS part, recording
     the per-day regime label on the TEST segments.
  5. Plot the equity/drawdown comparison and the per-asset regime map.

Models (each step ALL are fit and averaged into an equal-weight ensemble):
  - GaussianHMM, 2 states          (GMMHMM, n_mix=1, over the feature matrix)
  - GaussianHMM, 3 states
  - GMMHMM, 2 states x 2 mixtures

Observation features (causal, trailing; one row per day):
  col 0 = momentum = EWMA-mean daily return (span cfg.mom_window)
  col 1 = vol      = trailing std of daily return over cfg.vol_feat_window

Signal (long/flat, no look-ahead), per model m at day t:
  next_state_m = filtered_posterior_m(feat_{1:t-1}) @ transmat_m
  score_m      = (next_state_m @ per_state_mean_momentum_m) / max|.|
  ens_score    = mean_m score_m
  signal_t     = 1 if ens_score > 0 else 0   # applied to (vol-targeted) return_t

Regime (per TEST day): from ens_score with a symmetric dead-band
cfg.regime_band:  bull if ens_score > +band, bear if < -band, else flat.
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")  # hmmlearn convergence chatter
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hmm")

# hmmlearn emits "Degenerate mixture covariance" via *logging* (not warnings),
# so filterwarnings above misses it. It fires when a GMM mixture collapses onto
# (near-)identical feature rows -> zero variance. fit_model already guards
# degenerate fits (try/except + best-LL), and the equal-weight ensemble dilutes
# any single collapsed member, so the message is benign here.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_states: int
    n_mix: int = 1  # GMM mixtures per state (n_mix=1 => plain Gaussian)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gauss_2s", 2, n_mix=1),
    ModelSpec("gauss_3s", 3, n_mix=1),
    ModelSpec("gmm_2s2m", 2, n_mix=2),
)


@dataclass(frozen=True)
class HParams:
    train_window: int
    test_window: int


@dataclass
class Config:
    tickers: tuple[str, ...] = ("SPY", "QQQ", "TQQQ", "UPRO")
    start: str = "2016-01-01"
    end: str | None = None
    split_frac: float = 0.60

    train_grid: tuple[int, ...] = (21, 252, 504)
    test_grid: tuple[int, ...] = (7, 21)        # test_window=1 dropped (slow/noisy)

    # observation features (causal, trailing): the HMM is fit on these instead
    # of raw daily returns, so its hidden states track persistent momentum/vol
    # regimes rather than daily noise.
    mom_window: int = 5                    # EWMA span for the momentum feature
    vol_feat_window: int = 20              # trailing-vol feature lookback

    hmm_iter: int = 50
    seeds: tuple[int, ...] = (0,)          # add seeds for more restarts (slower)
    min_steps: int = 2                     # require >=2 walk-forward steps to score

    # volatility targeting: scale the long/flat position so trailing realized
    # vol tracks target_vol. De-risk-only by default (cap leverage at 1.0).
    target_vol: float = 0.15               # annualized vol target
    vol_lookback: int = 20                 # trailing window for realized vol
    max_leverage: float = 1.0              # cap; 1.0 = never lever beyond fully long

    regime_band: float = 0.15              # dead-band on normalised score -> flat

    def grid(self) -> list[HParams]:
        return [
            HParams(tr, te)
            for tr, te in product(self.train_grid, self.test_grid)
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
# Observations (features) & models                                            #
# --------------------------------------------------------------------------- #
MOM_COL = 0   # column of build_features() holding trailing momentum


def build_features(ret: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Causal trailing observation features, one row per day:
      col 0 = momentum: EWMA-mean daily return (span cfg.mom_window) -- the
              exponential weighting tracks recent moves faster than a flat
              window of the same length, reducing regime lag.
      col 1 = vol:      trailing std of daily return over cfg.vol_feat_window
    Each row uses returns up to and including that day (it is the emission AT t;
    the day-t signal only conditions on rows strictly before t, so no look-
    ahead). Warm-up rows lacking history are filled with 0 (neutral).
    """
    s = pd.Series(ret)
    mom = s.ewm(span=cfg.mom_window, adjust=False).mean()  # causal EWMA
    vol = s.rolling(cfg.vol_feat_window).std(ddof=1)
    feat = np.column_stack([mom.to_numpy(), vol.to_numpy()])
    return np.nan_to_num(feat, nan=0.0)


def build_model(spec: ModelSpec, cfg: Config, seed: int):
    from hmmlearn import hmm

    # All models are Gaussian(-mixture) HMMs over the continuous feature matrix
    # (n_mix=1 is a plain Gaussian per state).
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
    """
    Per-state directional score = mixture-weighted mean of the MOMENTUM
    feature in each state. A state whose typical trailing momentum is positive
    is a bull regime (only the sign matters downstream).
    """
    return (model.weights_ * model.means_[:, :, MOM_COL]).sum(axis=1)


# --------------------------------------------------------------------------- #
# Signals & backtest                                                          #
# --------------------------------------------------------------------------- #
def segment_scores(model, spec: ModelSpec, obs2d: np.ndarray,
                   seg_lo: int, seg_hi: int, train_window: int) -> np.ndarray:
    """
    Per-day normalised directional score (~[-1, 1]) for ONE model over
    [seg_lo, seg_hi). Uses only obs strictly before t -> no look-ahead.
    Dividing by max|s_dir| puts each model's scores on a comparable scale so
    they can be averaged into an ensemble.
    """
    s_dir = state_direction(model, spec)
    A = model.transmat_
    scale = float(np.max(np.abs(s_dir))) + 1e-12
    out = np.zeros(seg_hi - seg_lo)
    for i, t in enumerate(range(seg_lo, seg_hi)):
        hist = obs2d[max(0, t - train_window):t]
        if len(hist) == 0:
            continue
        filt = model.predict_proba(hist)[-1]      # filtered P(z_{t-1}|x_{1:t-1})
        out[i] = ((filt @ A) @ s_dir) / scale
    return out


def ensemble_signals(members, feat: np.ndarray,
                     seg_lo: int, seg_hi: int, train_window: int,
                     regime_band: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Equal-weight ensemble of the fitted models. Averages each member's
    normalised per-day score, then:
      * sig: 1.0 long / 0.0 flat        (avg score > 0)
      * reg: {+1 bull, 0 flat, -1 bear} (avg score vs +/-regime_band)
    All members share the same feature matrix `feat`.
    """
    scores = [
        segment_scores(model, spec, feat, seg_lo, seg_hi, train_window)
        for model, spec in members
    ]
    avg = np.mean(scores, axis=0)
    sig = (avg > 0).astype(float)
    reg = np.where(avg > regime_band, 1.0,
                   np.where(avg < -regime_band, -1.0, 0.0))
    return sig, reg


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
                 ) -> tuple[pd.Series, Counter, pd.Series]:
    """
    Tile non-overlapping TEST segments over [ts_start, ts_end).
    Each step: fit all models on TRAIN and trade their EQUAL-WEIGHT ENSEMBLE
    (averaged normalised scores) on TEST (vol-targeted). History for the
    earliest steps may reach before ts_start.
    Returns (stitched daily strategy returns over TEST, per-spec ensemble-
    membership counts, stitched per-day regime labels over TEST in {+1,0,-1}).
    """
    feat = build_features(ret, cfg)            # causal observation features
    lev = vol_target_leverage(ret, cfg, ann)   # causal vol-targeting overlay
    pieces: list[pd.Series] = []
    regs: list[pd.Series] = []
    picks: Counter = Counter()      # per-spec ensemble-membership (fit success)

    ts = ts_start
    while ts + hp.test_window <= ts_end:
        train_lo = ts - hp.train_window
        if train_lo < 0:                       # not enough history yet
            ts += hp.test_window
            continue

        # Fit every model on TRAIN (right up to TEST -- no selection step, so no
        # held-out VAL gap is needed); ensemble all that converge.
        members = []
        for spec in MODELS:
            model = fit_model(spec, feat[train_lo:ts], cfg)
            if model is not None:
                members.append((model, spec))
                picks[spec.name] += 1

        if members:
            test_sig, test_reg = ensemble_signals(
                members, feat, ts, ts + hp.test_window,
                hp.train_window, cfg.regime_band)
            seg_idx = dates[ts:ts + hp.test_window]
            test_ret = (test_sig * lev[ts:ts + hp.test_window]
                        * ret[ts:ts + hp.test_window])
            pieces.append(pd.Series(test_ret, index=seg_idx))
            regs.append(pd.Series(test_reg, index=seg_idx))
        ts += hp.test_window

    if not pieces:
        return pd.Series(dtype=float), picks, pd.Series(dtype=float)
    return pd.concat(pieces), picks, pd.concat(regs)


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
    oos_regimes: pd.Series       # per-day regime over OOS TEST: {+1, 0, -1}
    oos_bench_equity: pd.Series  # asset buy&hold over OOS TEST, normalised to 1


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
        if hp.train_window + hp.test_window > split:
            continue
        is_start = hp.train_window
        is_ret, is_picks, _ = walk_forward(ret, dates, hp, cfg, ann, is_start, split)
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
    log.info("%-8s IS  ensemble fits: %s", ticker, fmt_picks(best_is_picks))

    # ---- OOS walk-forward with chosen combo --------------------------------
    oos_ret, oos_picks, oos_reg = walk_forward(ret, dates, best_hp, cfg, ann, split, n)
    if oos_ret.empty:
        log.warning("%s: empty OOS", ticker)
        return None
    log.info("%-8s OOS ensemble fits: %s", ticker, fmt_picks(oos_picks))

    # ---- stitch IS (winning combo) + OOS for plotting ----------------------
    full_ret = pd.concat([best_is_ret, oos_ret])
    full_ret = full_ret[~full_ret.index.duplicated(keep="last")].sort_index()
    equity = (1.0 + full_ret).cumprod()
    equity /= equity.iloc[0]
    dd = equity / equity.cummax() - 1.0

    oos_eq = (1.0 + oos_ret).cumprod().to_numpy()

    # asset's own buy-and-hold over the OOS TEST dates (the per-asset benchmark
    # shown in the regime map), normalised to 1 at the first OOS test day.
    bench_ret = ret_s.reindex(oos_ret.index)
    bench_eq = (1.0 + bench_ret).cumprod()
    bench_eq /= bench_eq.iloc[0]

    return AssetResult(
        ticker=ticker,
        best_hp=best_hp,
        split_date=dates[split],
        equity=equity,
        drawdown=dd,
        oos_sharpe=safe_sharpe(oos_ret.to_numpy(), ann),
        oos_cagr=cagr(oos_eq, ann),
        oos_maxdd=max_drawdown(oos_eq),
        oos_regimes=oos_reg,
        oos_bench_equity=bench_eq,
    )


# --------------------------------------------------------------------------- #
# Plot — equity / drawdown comparison                                         #
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
    # have different tuned warm-ups, so clip each curve to the latest per-asset
    # start and renormalise to 1 there; drawdown is recomputed over that window.
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

    # 60% split reference (the three equity ETFs share a calendar)
    ref = next((r for r in results if r.ticker == "SPY"), results[0])
    for row in (1, 2):
        fig.add_vline(x=ref.split_date, line=dict(color="black", dash="dash",
                                                  width=1), row=row, col=1)
    fig.add_annotation(x=ref.split_date, y=1, yref="paper", showarrow=False,
                       text="  OOS →", xanchor="left", font=dict(size=11))

    fig.update_yaxes(row=1, col=1, title_text="growth of 1")
    fig.update_yaxes(row=2, col=1, title_text="drawdown", tickformat=".0%")
    fig.update_layout(
        title=f"HMM regime strategy (vol-targeted) — OOS metrics in legend "
              f"(split @ {ref.split_date.date()}, SPY ref)",
        template="plotly_white", height=820, hovermode="x unified",
        legend=dict(font=dict(size=11)),
    )
    fig.write_html(out_html)
    log.info("wrote %s", out_html)
    return fig


# --------------------------------------------------------------------------- #
# Plot — per-asset regime map                                                 #
# --------------------------------------------------------------------------- #
def _regime_spans(reg: pd.Series):
    """Yield (x0, x1, label) for maximal contiguous runs of equal regime."""
    vals = reg.to_numpy()
    idx = reg.index
    if len(vals) == 0:
        return
    start = 0
    for k in range(1, len(vals)):
        if vals[k] != vals[start]:
            yield idx[start], idx[k], int(vals[start])   # tile up to next run
            start = k
    step = (idx[-1] - idx[-2]) if len(idx) > 1 else pd.Timedelta(days=1)
    yield idx[start], idx[-1] + step, int(vals[start])


def plot_regimes(results: list[AssetResult], out_html: str) -> go.Figure:
    """
    One subplot per asset: the OOS buy-and-hold equity line with the detected
    regime (bull / bear / flat) shaded behind it over the TEST phase.
    """
    REG_COLOR = {1: "#2ca02c", 0: "#9e9e9e", -1: "#d62728"}   # bull / flat / bear
    REG_NAME = {1: "bull", 0: "flat", -1: "bear"}
    n = len(results)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[f"{r.ticker} — buy&hold equity + detected regime (OOS)"
                        for r in results],
    )

    for i, r in enumerate(results, start=1):
        reg, eq = r.oos_regimes, r.oos_bench_equity
        # Trace MUST be added before the vrects: add_vrect(row=, col=) defaults
        # to exclude_empty_subplots=True and silently drops shapes on a subplot
        # that has no traces yet. We also pass exclude_empty_subplots=False.
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq.values, name=r.ticker,
                       line=dict(color="black", width=1.4), showlegend=False),
            row=i, col=1)
        if reg is not None and len(reg) > 0:
            for x0, x1, lab in _regime_spans(reg):
                fig.add_vrect(x0=x0, x1=x1, fillcolor=REG_COLOR[lab],
                              opacity=0.20, line_width=0, layer="below",
                              row=i, col=1, exclude_empty_subplots=False)
        fig.update_yaxes(title_text="growth of 1", row=i, col=1)

    # one shared regime legend (dummy markers)
    for lab in (1, 0, -1):
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=12, color=REG_COLOR[lab]),
                       name=REG_NAME[lab], legendgroup="regime"),
            row=1, col=1)

    fig.update_layout(
        title="HMM detected regimes over the OOS test phase "
              "(shaded) vs each asset's buy & hold",
        template="plotly_white", height=300 * n, hovermode="x unified",
        legend=dict(title="regime", font=dict(size=11)),
    )
    fig.write_html(out_html)
    log.info("wrote %s", out_html)
    return fig


# --------------------------------------------------------------------------- #
# Regime-quality diagnostic                                                   #
# --------------------------------------------------------------------------- #
def regime_diagnostics(results: list[AssetResult]) -> None:
    """
    Log whether the detected regimes actually carry signal. For each asset, group
    the asset's OWN daily returns over the OOS test phase by the regime label the
    model assigned that day (causal -> a genuine forward-looking test) and report
    the annualised mean return, hit-rate and day count per regime. A useful
    detector shows bull > flat > bear in mean return; a bull-minus-bear spread
    near zero means the regimes are noise, not signal.
    """
    NAME = {1: "bull", 0: "flat", -1: "bear"}
    for r in results:
        ann = ann_factor(r.ticker)
        ret = r.oos_bench_equity.pct_change()      # asset's return on each OOS day
        df = pd.DataFrame({"reg": r.oos_regimes, "ret": ret}).dropna()
        means, parts = {}, []
        for lab in (1, 0, -1):
            sub = df.loc[df["reg"] == lab, "ret"]
            if sub.empty:
                means[lab] = float("nan")
                parts.append(f"{NAME[lab]}: n/a")
                continue
            means[lab] = sub.mean() * ann
            parts.append(f"{NAME[lab]}: {means[lab]:+6.1%} ret "
                         f"{(sub > 0).mean():4.0%} hit {len(sub):4d}d")
        spread = means[1] - means[-1]
        if np.isfinite(spread):
            tag = "GOOD bull>bear" if spread > 0 else "BAD bull<=bear"
            spread_txt = f"{spread:+.1%}/yr"
        else:
            tag, spread_txt = "n/a", "n/a"
        log.info("%-8s regime quality (OOS) | %s | bull-bear %s [%s]",
                 r.ticker, "  ".join(parts), spread_txt, tag)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(cfg: Config | None = None,
         returns: dict[str, pd.Series] | None = None,
         out_html: str = "hmm3_equity.html",
         regimes_html: str = "hmm3_regimes.html",
         n_jobs: int | None = None) -> go.Figure:
    """
    Run every ticker through `run_asset`, then write two figures: the
    equity/drawdown comparison and the per-asset regime map.

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
    fig = plot_results(results, out_html, benchmark=returns.get("SPY"))
    plot_regimes(results, regimes_html)
    regime_diagnostics(results)
    return fig


if __name__ == "__main__":
    main()
