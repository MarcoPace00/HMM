# HMM Regime-Switching Trading Experiments

A research sandbox for testing whether **Hidden Markov Models (HMMs)** can detect
tradeable market regimes (bull / bear / flat) in liquid ETFs and turn them into a
long/flat timing strategy. The repo is an iteration log: each `HMM_*.py` script is
a successive attempt to fix the weaknesses of the previous one.

> ⚠️ **Research only — not investment advice.** Backtests use free `yfinance`
> data, model **no transaction costs or slippage**, and are single-path. Treat
> every number as illustrative.

---

## TL;DR — what we learned

- **Volatility targeting is the only robust win.** Scaling position size so
  trailing realized vol tracks a target cut max drawdowns **3–5×** on the
  volatile/leveraged names (TQQQ, UPRO, BTC) and improved Sharpe almost
  everywhere. This is *risk control*, and it works.
- **The directional signal is weak.** Predicting next-day ETF direction from an
  HMM is close to a coin flip. Every variant confirmed this.
- **Discrete models beat the GMM** at the selection game — `CategoricalHMM`
  states were chosen ~2× more often than `GMMHMM`.
- **Cleverer model *selection* didn't help** (Calmar-based selection was a wash);
  **how you size risk matters far more than which model you pick.**
- **Nothing beat SPY buy-&-hold on raw return** — but the vol-targeted variants
  matched its risk-adjusted return with a fraction of the drawdown.

Bottom line: as a *market-timing alpha* engine this doesn't work; as a
*risk-control overlay* it does.

---

## Requirements

```bash
pip install numpy pandas plotly hmmlearn yfinance
```

Python 3.11+ (developed on 3.14). Each script downloads its own data and writes a
self-contained interactive Plotly `.html` you can open in a browser.

## Running

```bash
python HMM_1.py      # baseline
python HMM_2.py      # + volatility targeting
python HMM_3.py      # feature-based ensemble + regime map
```

Each `main()` accepts overrides, e.g. serial mode for clean logs:

```python
from HMM_3 import main, Config
main(Config(tickers=("SPY", "QQQ")), n_jobs=1)
```

---

## The scripts

### `HMM_1.py` — Baseline regime-switching strategy

The original pipeline. For each ticker:

1. Download daily returns; split **60% in-sample / 40% out-of-sample (OOS)**.
2. **Grid-search** hyper-parameters (train / validation / test window lengths,
   discretization threshold) on the in-sample part.
3. At every walk-forward step, fit **3 HMMs**, pick the best on the **validation
   window by Sharpe**, and trade it long/flat on the test window.
4. Re-run with the chosen hyper-parameters on the OOS slice.
5. Plot stitched equity + drawdown, with OOS Sharpe / CAGR / MaxDD in the legend
   and a SPY buy-&-hold benchmark.

**Models:** `CategoricalHMM` (2 & 3 states, on DOWN/FLAT/UP-discretized returns)
and `GMMHMM` (2 states × 2 mixtures, on raw returns).
**Signal:** `next_state = filtered_posterior @ transition_matrix`, then
`long if next_state · per_state_direction > 0 else flat` — strictly causal (no
look-ahead).
**Output:** `hmm_equity.html`. **Universe:** SPY, QQQ, TQQQ, UPRO, BTC-USD.

Per-ticker work runs in parallel across processes (with BLAS pinned to 1 thread
per worker to avoid core oversubscription).

### `HMM_2.py` — Volatility targeting

`HMM_1` plus a causal **volatility-targeting overlay** (`vol_target_leverage`):
the long/flat position is scaled by `target_vol / trailing_realized_vol` so the
strategy de-risks in turbulent regimes. De-risk-only by default
(`max_leverage = 1.0`, i.e. it never adds leverage on top of already-leveraged
ETFs). Applied consistently to both the validation and test windows.

**Output:** `hmm_equity_voltarget.html`. This was the most effective single change
in the whole project (see results).

### `HMM_3.py` — Feature-based observations + model ensemble + regime map

The current head of the line. Three ideas stacked on top of `HMM_2`:

- **Feature-based observations.** Instead of raw daily returns (~all noise), every
  HMM is fit on a causal **feature matrix**: `[EWMA momentum (span 5), 20-day
  trailing volatility]`. The intent is for hidden states to track *persistent*
  momentum/vol regimes rather than daily noise. All models are now
  Gaussian / Gaussian-mixture HMMs (`gauss_2s`, `gauss_3s`, `gmm_2s2m`); the
  categorical models and the discretization threshold are gone.
- **Equal-weight ensemble.** Rather than *selecting* one model per step, all
  models are fit and their normalized directional scores are **averaged** — this
  removes the step-to-step model-switching whipsaw. (With no selection step, the
  validation window was dropped; models train right up to the test window.)
- **Per-state direction** is read from each state's mean **momentum** feature
  (positive trailing momentum ⇒ bull state).

**Outputs:**
- `hmm3_equity.html` — equity + drawdown comparison.
- `hmm3_regimes.html` — a per-asset **regime map**: each asset's OOS buy-&-hold
  equity line with the detected bull / bear / flat regime shaded behind it.

**Regime-quality diagnostic.** At the end of a run, `regime_diagnostics()` logs,
per asset, the annualized mean return / hit-rate / day-count of the asset's own
returns grouped by the regime the model assigned that day. A working detector
shows **bull > flat > bear** in mean return and a clearly positive *bull-minus-bear
spread*; a spread near zero means the regimes are noise. This is the objective
test of whether the regime detection is real.

**Universe:** SPY, QQQ, TQQQ, UPRO.

---

## Results

### Headline: baseline vs volatility targeting (OOS, 2016 split)

Out-of-sample metrics per asset. **B&H** = SPY buy-and-hold benchmark
(`Sh ≈ 0.90 / CAGR 15.7% / MaxDD −33.7%`).

| Asset | Metric | `HMM_1` baseline | `HMM_2` vol-targeted |
|-------|--------|:----------------:|:--------------------:|
| **SPY**  | Sharpe | 0.47 | **0.90** |
|          | MaxDD  | −14.0% | **−7.1%** |
| **QQQ**  | Sharpe | **1.03** | 1.01 |
|          | MaxDD  | −25.6% | **−12.8%** |
| **TQQQ** | Sharpe | 0.46 | **1.12** |
|          | MaxDD  | −55.6% | **−11.9%** |
| **UPRO** | Sharpe | 0.08 | **0.85** |
|          | MaxDD  | −54.4% | **−12.2%** |
| **BTC**  | Sharpe | 0.47 | **0.48** |
|          | MaxDD  | −49.7% | **−17.5%** |

Volatility targeting **reduced drawdown on every asset** — dramatically on the
leveraged/volatile names (TQQQ −55.6%→−11.9%, UPRO −54.4%→−12.2%) — while
*improving* Sharpe on four of five. It gives up some CAGR (de-risking caps upside)
but the risk-adjusted profile is far better.

### Other experiments

- **Calmar-based model selection** (an intermediate `HMM_3` variant): scoring the
  per-step model choice by Calmar (return/MaxDD) instead of Sharpe **did not
  reliably reduce drawdowns** — it even made some worse (BTC went to a negative
  Sharpe). Lesson: drawdowns come from *position sizing*, not *model selection*.
  This variant was retired.
- **Model-selection tally:** `CategoricalHMM` states were chosen ~2× more often
  than the `GMMHMM` — discrete observations suited this directional task better,
  which is partly why `HMM_3` moved away from raw-return GMMs toward engineered
  features.
- **Feature-based observations + ensemble (`HMM_3`):** intended to fix flickering,
  lagged regimes. Use the **regime-quality diagnostic** in the logs to judge it —
  if `bull-bear` spreads are consistently positive the regimes carry signal; if
  they hover near zero, they don't, and no amount of window-tuning will help.

### Honest conclusion

Across every variant, **no version beat SPY buy-&-hold on raw return**, and the
directional timing signal stayed weak. The durable, reproducible result is that
**the HMM + vol-targeting overlay is a good drawdown-reduction tool, not an alpha
source.** Daily-frequency regime *timing* of liquid ETFs appears to be at or
beyond the edge of what's extractable from price alone.

---

## Repo layout

```
HMM_1.py                    baseline: select-best-model HMM, long/flat, parallel per ticker
HMM_2.py                    HMM_1 + volatility-targeting overlay   (the key improvement)
HMM_3.py                    feature-based (momentum+vol) Gaussian-HMM ensemble + regime map
README.md                   this file
hmm_equity.html             HMM_1 report (equity + drawdown)
hmm_equity_voltarget.html   HMM_2 report (vol-targeted)
hmm3_equity.html            HMM_3 equity + drawdown
hmm3_regimes.html           HMM_3 per-asset regime map
```

### Viewing the reports

The `.html` reports are self-contained interactive Plotly charts. GitHub shows
their source rather than rendering them, so open them one of these ways:

- **Download** the file and open it in a browser, or
- **htmlpreview** (no setup — just replace `USER` / `REPO` below; branch is `main`):
  - [HMM_1 — equity & drawdown](https://htmlpreview.github.io/?https://github.com/USER/REPO/blob/main/hmm_equity.html)
  - [HMM_2 — vol-targeted](https://htmlpreview.github.io/?https://github.com/USER/REPO/blob/main/hmm_equity_voltarget.html)
  - [HMM_3 — equity & drawdown](https://htmlpreview.github.io/?https://github.com/USER/REPO/blob/main/hmm3_equity.html)
  - [HMM_3 — regime map](https://htmlpreview.github.io/?https://github.com/USER/REPO/blob/main/hmm3_regimes.html)

> htmlpreview requires the repo to be **public** and pulls the whole ~5–6 MB
> file through a proxy, so it can be slow. For snappier viewing, enable **GitHub
> Pages** and the files serve directly at
> `https://USER.github.io/REPO/hmm3_equity.html`.

## Design notes (shared across scripts)

- **No look-ahead:** the signal for day *t* conditions only on observations
  strictly before *t*; features are trailing; vol-target leverage is shifted one
  day. The 60/40 split is honored — hyper-parameters are tuned only on the
  in-sample half.
- **Parallelism:** tickers are independent and CPU-bound, so each runs in its own
  process; BLAS threads are pinned to 1 per worker to prevent core
  oversubscription.
- **`hmmlearn` "degenerate covariance" noise** is suppressed at the logger level
  (it fires when a Gaussian mixture collapses onto near-identical rows; the
  ensemble dilutes any single collapsed member and `fit_model` already guards it).

## Known limitations / next ideas (untried)

- **No transaction costs** — the biggest missing piece. High-turnover settings
  (e.g. short test windows) look better than they would trade. Adding a per-flip
  cost is the highest-value next step.
- **No regime persistence** — even with smoother features, a sticky transition
  prior, a Viterbi state path, or a minimum holding period would reduce
  flickering/lag (persistence and responsiveness trade off directly).
- **Single seed, single path** — more random restarts and a walk-forward over
  multiple eras would make the metrics more trustworthy.
- **A 200-day moving-average filter** would be a sane trivial baseline to check
  whether the HMM machinery earns its complexity.
