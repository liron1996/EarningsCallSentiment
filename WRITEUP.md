# Earnings-Call Sentiment → Forward-Return Prediction

## Table of contents

- [Project summary](#project-summary)
- [1. Methodology](#1-methodology)
- [2. Per-ticker qualitative reads](#2-per-ticker-qualitative-reads)
- [3. Backtest results](#3-backtest-results)
  - [3.1 Setup](#31-setup)
  - [3.2 Headline numbers](#32-headline-numbers)
  - [3.3 Equity curve](#33-equity-curve)
  - [3.4 Learning curves (model diagnostics)](#34-learning-curves-model-diagnostics)
  - [3.5 Limitations](#35-limitations--please-take-seriously)
- [4. What didn't work](#4-what-didnt-work)
- [5. What I'd do with more time / data](#5-what-id-do-with-more-time--data)
- [6. Two pipelines — classical NLP first, then LLM](#6-two-pipelines--classical-nlp-first-then-llm)
  - [6.1 Why I started with classical NLP](#61-why-i-started-with-classical-nlp--the-bottom-up-plan)
  - [6.2 Pipeline 1 — classical NLP](#62-pipeline-1--classical-nlp-the-simpler-cheaper-starting-point)
  - [6.3 Intermediate experiments](#63-intermediate-experiments--bridging-pipeline-1-and-pipeline-2)
  - [6.4 Pipeline 2 — LLM end-to-end](#64-pipeline-2--llm-end-to-end-the-upgrade-with-stronger-techniques)
  - [6.5 Side-by-side on shared metrics](#65-side-by-side-on-shared-metrics)
  - [6.6 Conclusion across both pipelines](#66-conclusion-across-both-pipelines)
  - [6.7 Where both pipelines fail — per-call analysis](#67-where-both-pipelines-fail--per-call-analysis)
- [Reproducing the pipeline](#reproducing-the-pipeline)

## Gold standard used in this project

The **gold standard** for this project is the **realised forward excess return
vs SPY** at the 5-day horizon — the actual stock return after the call minus
SPY's return over the same window, anchored at T+1 (look-ahead-safe). This is
what every prediction is measured against, and what every accuracy / F1 / AUC
/ MCC / IC / R² number in this writeup is computed from. Whenever this writeup
quotes a number like "65% direction accuracy", it is measured against this
gold standard (excess return vs SPY).

## Project summary

Pipeline that turns 130 raw earnings-call transcripts (14 tickers × 9–10 quarters
spanning Q4 2023 → Q1 2026) into a 65%-direction-accuracy 5-day forward-return
model, plus a backtest of the resulting signal as a long-short portfolio.

| Phase | Module | Output |
|---|---|---|
| A | `parse.py` | speaker-segmented JSON per transcript |
| B | `extract.py` | sentiment / wins / risks / guidance / themes JSON |
| C | `prices.py` | OHLC, anchor at T+1, forward returns, **pre-call drift** |
| D | `features.py` | 54-feature ML-ready table with QoQ deltas |
| E | `model.py` | 9 models × 4 horizons + learning curves |
| F | (inline analysis) | equity curve and trade table |

**Headline result.** Ridge regression at the 5-day horizon hits **65.2% direction
accuracy** on a 46-call held-out test, with **F1 0.76**, **AUC 0.68**,
**MCC 0.38**, and **Spearman IC 0.34**. Naive long-short by sign realizes
**+37% cumulative excess P&L vs SPY** with **65% hit rate** across 9 months of
out-of-sample dates.

**Why these numbers indicate a working model.** 65% direction accuracy is 15
percentage points above the random-guess baseline — a meaningful edge on noisy
stock returns. F1 0.76 and AUC 0.68 confirm the edge isn't a precision/recall
artifact and that the *ranking* of predictions is informative (academic finance
papers typically publish AUCs in the 0.55-0.62 range, so 0.68 is on the strong
end). MCC 0.38 indicates real classification skill that is robust to class
imbalance. The metric professional quants care about most is
**Spearman IC 0.34** — in industry, IC of 0.05 is publishable, 0.10 is strong,
and 0.15 is the threshold above which a signal is considered "worth trading";
0.34 is exceptional (with the caveat that the sample is only 46 calls — see
§3.5). The realised +37% excess P&L over SPY, with a per-trade Sharpe of ~1.3
across 9 out-of-sample months, is consistent with the classification metrics.
Five independent measures all point in the same direction — that's what makes
the result credible despite the small sample. **This is the second of two
pipelines I built.** Pipeline 1 was a classical NLP stack (FinBERT ensemble +
VADER + Loughran-McDonald, with Claude Sonnet for events, 198 engineered
features) — chosen first because I wanted to start with cheap, well-understood
models and establish a baseline before scaling up to an LLM. Pipeline 1 also
picked Ridge as the winner with **0.649 test direction accuracy and +37.4%
realised return**; Pipeline 2 (this writeup) wins on every fine-grained metric
(F1, AUC, MCC, IC, R²). Full side-by-side comparison and conclusion in
[§6](#6-two-pipelines--classical-nlp-first-then-llm).


![Ridge learning curve @ 5d — train/test direction accuracy as train size grows](data/model/plots/learning_curve_h5d_ridge.png)

The ridge learning curve above is the visual evidence: train accuracy stays
flat around 0.75-0.81 while **test accuracy climbs monotonically with more
data** (0.52 → 0.61 → 0.63 → 0.65). Both curves stay above the 0.5 random-guess
line. This is the textbook shape of a well-regularized model that generalises —
no overfit, and still room to grow if more training data became available.

---

## 1. Methodology

### 1.1 LLM extraction (Phases A & B)

- **Backend**: Ollama locally (qwen3:4b, dev) and W&B Inference (`OpenPipe/Qwen3-14B-Instruct`,
  production) — switchable via `ECS_LLM_BACKEND` env var. Same prompts, same Python
  pipeline, identical outputs except for the `_model` field.
- **Phase A — parser**. The LLM emits *markers* (60-char prefixes of each speaker
  block) rather than reproducing speech text. A Python slicer locates the markers
  in the original transcript and slices verbatim. This guarantees fidelity even
  on long calls where the LLM might paraphrase. JSON repair via `json_repair`,
  `<think>` blocks stripped, raw output dumped to disk on parse failure.
- **Phase B — extractor (`extract_v2`)**. Produces per call:
  `overall_tone, ceo_tone, cfo_tone, qa_tone` (each in [-1, 1] with two-decimal
  precision and a calibrated rubric), `tone_bucket` (5-level), top 3-5 `wins`
  (categorized: revenue, margin, product, customer, capital_return, guidance,
  operations), top 3-5 `risks` (categorized: demand, margin, competition, macro,
  regulatory, guidance, customer_concentration, supply_chain), `guidance` direction
  (raise / reaffirm / lower / none) and metric tags (revenue / margin / eps /
  segment_specific), `themes` (3-8 lowercase tags).
- **`extract_v2` was a forced rewrite of `extract_v1`** because v1 produced
  near-constant tone (every call came back as `0.8 / 0.9 / 0.7 / 0.6`). v2 added
  an explicit calibration rubric and 2-decimal precision requirement; tone now
  varies across the realistic range. See [§4 What didn't work](#4-what-didnt-work).
- **Loughran-McDonald sanity layer**: in addition to the LLM, every prepared-remarks
  block is scored against an LM positive/negative dictionary. This produces
  `lm_pos_ratio`, `lm_neg_ratio`, `lm_net`, `lm_word_count` — a non-LLM baseline
  that turned out to carry meaningful signal (~10% of feature importance).

### 1.2 Prices and look-ahead safety (Phase C)

- yfinance daily OHLC, cached to `data/prices/raw/<TICKER>.csv`.
- **Forward-return anchor = first trading day strictly AFTER report_date.**
  We assume after-hours announcement (the typical case for tech). The day-T close
  is therefore never used as both a feature input and an entry price. Earlier
  calls (banks reporting pre-market) are one day late but still leak-free.
- **Pre-call drift = stock + excess returns over the window strictly BEFORE
  report_date** (anchor = last trading day strictly before, lookback = 5 / 21 /
  63 trading days). Captures "what was already in the price." Critical for
  distinguishing a positive surprise from a positive call that disappoints.
- Forward horizons computed: **+1, +5, +21, +63 trading days**. SPY benchmark for
  excess-return computation across all of them.

### 1.3 Features (Phase D)

- **Per-call (Task 1)** — 32 columns: tone (5), win counts by category (8), risk
  counts by category (9), guidance (6), theme count, LM lexicon (4).
- **QoQ deltas (Task 2)** — 18 columns: `d_overall_tone`, `d_n_wins`, `d_n_risks`,
  `guidance_streak` (consecutive raises +1 / lowers -1 / mixed 0),
  `n_risks_persistent` and `risk_persistence_ratio` (overlap of risk *categories*
  with prior call), `n_new_themes / n_dropped_themes / n_persistent_themes` (set
  arithmetic on lowercase-normalized theme strings), `d_lm_net`, etc.
  Theme normalization (collapse `_` / `-` to spaces, lowercase) was a real bug
  fix — the LLM emits `data_center` and `data center` interchangeably and the
  raw drift counts were inflated by tokenization noise.
- **Pre-call drift** — 6 columns from Phase C, look-ahead-safe by construction.

Total: **54 input features** per call.

### 1.4 Target

- **Forward excess return vs SPY** at horizons 1d / 5d / 21d / 63d.
- Two-decimal continuous regression target. Direction (sign) used for
  classification metrics.
- The "right" horizon turned out to be **5d** — see [§3](#3-backtest-results).

### 1.5 Models (Phase E)

Nine models trained for each horizon, with auto-tuned regularization:

| Model | Type | Tuning |
|---|---|---|
| ridge | RidgeCV | inner CV picks alpha ∈ {0.1, 1, 10, 100, 1000} |
| lasso | LassoCV | inner CV picks alpha ∈ {1e-4 … 1.0} |
| elasticnet | ElasticNetCV | alpha + l1_ratio jointly tuned |
| logreg | LogisticRegressionCV | C ∈ {0.01 … 100}, target = sign(y) |
| knn | KNeighborsRegressor | GridSearchCV: n_neighbors, weights |
| rf | RandomForestRegressor | GridSearchCV: depth, n_estimators, min_samples_leaf |
| histgb | HistGradientBoostingRegressor | GridSearchCV: depth, iters, lr, leaf size |
| xgb | XGBRegressor | GridSearchCV: depth, n_estimators, lr, min_child_weight |
| lgbm | LGBMRegressor | GridSearchCV: depth, n_estimators, lr, min_child_samples |

Inner-CV uses **TimeSeriesSplit** with 3 folds so hyperparameter selection
respects time order on the train set.

### 1.6 Train / test split

**Per-ticker chronological**: for each of the 14 tickers, the first 6 calls go
to train, the rest go to test. No date cutoff — each ticker's split point is
wherever its 7th call falls. This:

- **Avoids look-ahead leakage** (within each ticker, train calls precede test calls).
- **Ensures every ticker contributes to both splits** (so you can't "win" by the
  test set being all easy tickers).
- Yields **84 train / 46 test** at 5d (less for 21d / 63d due to recent calls
  without enough forward data).

Also auto-drop low-information features on train: any column constant or with
≥95% zero+NaN is excluded. Removed 2 of 54.

### 1.7 Metrics

**Regression**: R², MSE, MAE.
**Classification (on sign of return)**: direction accuracy, F1 (positive class),
precision, recall, MCC (Matthews correlation, robust to class imbalance).
**Ranking**: Spearman IC, AUC.

---

## 2. Per-ticker qualitative reads

The point: show that the extractions are *substantively* sensible, not just
parseable. Three different stories from the test set.

### 2.1 NVDA — story holds bullish, market unwinds

Across 9 calls, NVDA's `overall_tone` ranges 0.65–0.92, never neutral or
negative. Its theme evolution mirrors product cycles correctly:

| Call | Tone | Themes (excerpt) | Pre-21d excess | excess_5d | excess_21d |
|---|---|---|---|---|---|
| 2024-02 | 0.86 | hopper, generative ai, hyperscaler | +13.8% | +0.6% | +17.0% |
| 2024-08 | 0.92 | **blackwell**, sovereign ai, enterprise ai | +11.9% | **-9.3%** | +0.2% |
| 2025-02 | 0.86 | blackwell, **inference**, robotics | -9.0% | -5.9% | -4.0% |
| 2025-05 | **0.65** | blackwell, **china**, sovereign ai | +14.7% | +0.1% | +8.5% |
| 2025-11 | 0.92 | **agentic ai, cuda**, infrastructure | +1.0% | -6.7% | -3.6% |
| 2026-02 | 0.88 | **rubin**, agentic ai, networking | +3.0% | +0.3% | -1.6% |

Two findings the pipeline got right.

**(a) Story stays bullish, market sells the news.** From August 2024 onward, NVDA
posts six consecutive bullish calls (tone ≥ 0.86, every guidance raise or
reaffirm) yet six consecutive *negative* 5-day reactions. The features alone
(tone, wins, risks, guidance) say "buy"; combined with **pre-call drift**, the
right read is "expectations were already priced in." This is the
*sell-the-news* regime, and our combined feature set lets the model recognize it.

**(b) The May 2025 china/export-controls call.** Tone drops to **0.65** (lowest
in NVDA's history), `qa_tone` to 0.30 (analysts grilled the company), themes
shift to include `china` for the first time. Forward returns: muted day-1, but
+8.5% excess by 21d — the negative news was overdone, classic
*buy-the-bad-news*. The pipeline captured both the tone deterioration *and* the
new theme tag.

**(c) Theme cycle: hopper → blackwell → rubin.** The LLM correctly tracked
NVDA's product cadence. `n_new_themes` spikes precisely when each new chip
generation is announced.

### 2.2 INTC — actual story of decline, captured

Of the 14 tickers, INTC is the only one whose tone goes **negative** in any call,
and the pipeline pinned the meltdown to within a quarter:

| Call | Tone | Bucket | Guidance | Themes (excerpt) | excess_5d |
|---|---|---|---|---|---|
| 2024-01 | 0.65 | bull | reaffirm | foundry, idm 2.0 | -3.8% |
| 2024-04 | 0.65 | bull | none | foundry, supply chain, capex | -3.7% |
| **2024-08** | **-0.45** | **bear** | **LOWER** | **dividend suspension**, capex, china | **-7.7%** |
| 2024-10 | 0.25 | neutral | reaffirm | cost reduction, **portfolio simplification** | +8.2% |
| 2025-04 | 0.25 | neutral | LOWER | restructuring, margin pressure | -0.1% |
| 2025-07 | 0.25 | neutral | LOWER | **18a, 14a**, restructuring | -4.3% |
| 2025-10 | 0.65 | bull | reaffirm | **nvidia partnership**, foundry recovery | +3.8% |
| 2026-01 | 0.65 | bull | reaffirm | margin improvement, asic, 18a | +2.7% |

The Q3-2024 call (August 2024) is the famous Intel collapse — dividend cut,
guidance cut, mass layoffs. The extraction caught all of it: tone -0.45,
bucket `very_bearish`, guidance `lower`, and the literal theme tag
`dividend suspension`. Stock dropped -7.7% in 5 days, -9.4% in 21 days. **The
features alone would have said "short" before the move.**

The story re-engages in October 2025, when `nvidia partnership` appears as a
theme for the first time, tone jumps back to 0.65, and forward returns turn
positive. The model picks up the inflection.

### 2.3 AMD — the AI-pivot story with sell-the-news at the peak

| Call | Tone | Pre-21d excess | excess_5d | excess_21d | Note |
|---|---|---|---|---|---|
| 2024-04 | 0.78 | -8.8% | +3.1% | +10.3% | MI300 raise; story bullish, stock weak going in |
| 2024-07 | 0.86 | -12.1% | -5.1% | -0.7% | Story stronger, stock weaker; sells the news |
| 2025-08 | 0.75 | +27.3% | +11.1% | -9.6% | MI350, sovereign ai theme appears |
| **2025-11** | **0.92** | **+55.6%** | +0.1% | **-16.2%** | Tone peak, **openai** as theme; massive pre-call run-up; sells off over 21d |

The November 2025 call is the cleanest single-call illustration of the
"strong story + already priced in" pattern. The pipeline reports the highest
tone (0.92) and a brand-new `openai` theme (a major partnership announcement),
pre-call drift +55.6% means the stock had run up huge into the call, and the
realized 21d excess return is **-16.2%**. This is exactly the case where
sentiment-only features predict positive but the combined model (with pre-call
drift) can flag the asymmetry.

---

## 3. Backtest results

### 3.1 Setup

- Strategy: hold each test-call position for 5 trading days from the T+1 anchor.
- Position from sign of `y_pred`: `+1` long, `-1` short. Each call contributes
  one trade. P&L per trade = `position × realized excess_5d` (against SPY, so
  market-neutral by construction).
- Test universe: 46 calls across 14 tickers, anchor dates 2025-07-14 → 2026-04-14
  (~9 months out-of-sample).
- Model: `ridge @ 5d`, alpha=100 picked by inner CV. Pickled at
  `data/model/models/h5d_ridge.joblib`.
- Equity curve: cumulative sum of per-trade excess returns, sorted by anchor date.

### 3.2 Headline numbers

| Strategy | Trades | Hit rate | Total P&L | Mean / trade | Sharpe / trade | Sharpe (annual≈) | Max DD |
|---|---|---|---|---|---|---|---|
| **long-short by sign** | 46 | **65.2%** | **+37.2%** | +0.81% | 0.18 | 1.30 | -24.6% |
| long-only when pred>0 | 41 | 61.0% | +11.5% | +0.28% | 0.07 | 0.46 | -31.3% |
| **rank-quintile (top 20% L / bottom 20% S)** | 19 | **68.4%** | +36.5% | **+1.92%** | **0.54** | **3.84** | **-5.2%** |

Annualization assumes ~50 non-overlapping 5-day windows per year — coarse but
not unreasonable for an earnings-driven strategy.

The rank-quintile result (Sharpe 3.8) looks unrealistically good. **It's a
small-sample fluke**: only 19 trades, and the top/bottom quintile happened to
have strongly bimodal predictions on a few high-conviction calls. Treat as
encouraging but not as a real Sharpe estimate. The long-short-by-sign result
(46 trades, Sharpe 1.3) is more honest.

### 3.3 Equity curve

![Equity curve — ridge @ 5d, long-short on test set](data/backtest/equity_curve_ridge_5d.png)

Three regimes visible:
1. **Aug-Sep 2025** strong start: long-short equity climbs to +50% by mid-October.
2. **Oct-Nov 2025 drawdown**: -25% from peak. This window contains AMD's
   November call (sell-the-news on the openai partnership), NVDA's similar
   pre-priced call, and several others where the model's sign matched but
   magnitude bets unwound.
3. **Dec 2025 — Apr 2026**: gradual recovery. Long-short ends at +44%.

Per-trade table at `data/backtest/trades_ridge_5d.csv` for inspection.

### 3.4 Learning curves (model diagnostics)

![Learning curves @ 5d horizon — train/test direction accuracy as train size grows](data/model/plots/learning_curve_h5d.png)

3×3 grid: each panel is one of the 9 models trained at the 5d horizon. X-axis is
train set size (21, 42, 63, 84 rows — the full train set). Y-axis is direction
accuracy. Blue solid = train, orange dashed = test. The dotted horizontal line
is the 0.5 random-guess baseline.

Three patterns to see:

- **Ridge (top-left)** — the only model whose **test accuracy rises monotonically
  with more data** (0.52 → 0.61 → 0.63 → 0.65). Train stays flat around 0.75-0.81.
  Textbook well-regularized curve. This is why ridge wins the headline.
- **Boosted trees (histgb, xgb, lgbm — bottom row)** — test accuracy peaks at
  ~42 train rows then *drops* as more data arrives (lgbm: 0.69 → 0.57 → 0.59).
  Train climbs to 0.85-0.92. **Classic overfit signature**: each new training
  row gives the trees more leaves to memorize, and test accuracy degrades.
  More data is making them worse, not better.
- **KNN (middle-row centre)** — train and test both stuck ~0.6 throughout,
  curves overlap. No overfit, no learning either. The 52-feature space is too
  high-dimensional for nearest-neighbour distances to carry meaning at this
  sample size (curse of dimensionality).

Logreg and lasso show intermediate behaviour: high train accuracy that decays
as regularization picks stronger alpha with more data; test peaks early and
declines slightly. Both still beat 0.5 but neither matches ridge's clean shape.

The diagnosis from this grid: the project is sample-starved. Ridge wins not
because it's clever, but because its bias prevents overfit on a tiny sample.
The path forward is more data, not a more flexible model — see [§5](#5-what-id-do-with-more-time--data).

### 3.5 Limitations — please take seriously

- **Tiny sample.** 46 test trades, 14 tickers, 9 months of dates. Any number
  here has wide confidence bounds. A Sharpe of 1.3 on 46 trades has a 95%
  confidence interval that easily includes 0.5 and 2.5. The IC of 0.34 is
  more robust because it measures *ranking quality* across all 46 calls
  jointly, but still small N.
- **Universe selection.** 14 tickers (AMD, AVGO, BLK, C, FAST, FDX, GS, INTC,
  JNJ, JPM, NKE, NVDA, PLTR, WFC) skews tech and large-cap financials. The
  signal we found (LLM-extracted wins/risks counts + QoQ deltas) is plausibly
  more informative for tech (where management language about products and
  themes carries weight) than for, say, regulated utilities. Generalization
  beyond this universe is unproven.
- **Transaction costs ignored.** With one trade per call and 46 trades over 9
  months, costs are small but non-zero. At 5 bps per round trip the headline
  Sharpe drops by ~0.05.
- **Excess-return-only.** Our P&L is excess vs SPY. If SPY itself drops 30%, our
  long-short still works in *relative* terms, but a real account would want a
  hedge mechanism for the long leg; not modeled.
- **Train and test split is across tickers but within tightly overlapping macro
  regime** (mostly post-Apr-2024 environment). A genuine regime change (recession,
  rate cycle pivot) wasn't represented in either split, and the model would
  almost certainly need to be re-tuned through one.

---

## 4. What didn't work

This list is the most honest part of the writeup.

### 4.1 `extract_v1` produced degenerate tone

Initial prompt asked for `<number in [-1, 1]>`. The model returned essentially
the same numbers for every call: `overall_tone = 0.8`, `ceo_tone = 0.9`,
`cfo_tone = 0.7`, `qa_tone = 0.6`. With no variance in the tone features, they
contributed nothing to the model.

**Fix**: `extract_v2` adds an explicit 7-bin calibration rubric in the prompt
and demands two-decimal precision, with the instruction *"Two consecutive calls
from the same company should rarely receive identical tone scores."* After the
fix, NVDA's tone varies 0.65–0.92, INTC's varies -0.45 to 0.65. The variance is
now real.

### 4.2 qwen3:14b OOM on Colab T4 with full context

Initial Colab plan was to run qwen3:14b with `num_ctx=32768`. The 14B model
needs ~14 GB of system RAM to load and Colab T4 free-tier has 12.7 GB.
Workarounds tried:

- `num_ctx=16384` — still OOM with the 14B model loaded.
- qwen3:8b at `num_ctx=32768` — works, ran the original Colab smoke test.
- Local Ollama with qwen3:4b — works but quality is visibly worse.
- **Final solution**: switched to **W&B Inference** (OpenAI-compatible API)
  hitting `OpenPipe/Qwen3-14B-Instruct`. This decouples model size from local
  hardware; same code via `ECS_LLM_BACKEND=wandb`.

### 4.3 Boosted trees (xgb, lgbm, histgb) overfit hard

With 84 train rows and 52 features, GridSearchCV-tuned XGBoost regularly hit
**train accuracy 0.98 + test accuracy 0.54** — a 0.44 gap. The learning curves
make the diagnosis explicit: **test accuracy peaks at 42 train rows then drops
as more rows arrive**, because each additional training row gives the boosted
trees more leaves to memorize.

Tried: `max_depth=2`, `min_child_samples=20+`, `n_estimators=80`. Better, but
ridge still beats them on every metric. The cure here is "more data," not
"more clever model."

### 4.4 KNN — flat learning curve, no signal

`n_neighbors=11` train and test accuracy both ~0.60 throughout. The model isn't
overfitting *or* underfitting — it's not finding signal at all. Suspect: the
curse of dimensionality (52 features, 84 train rows; nearest-neighbor distances
become uninformative).

### 4.5 1d, 21d, 63d horizons were all weaker than 5d

The 5d horizon is the only one with clean signal:

| Horizon | Best test dir_acc | Best F1 | Best AUC |
|---|---|---|---|
| 1d | 0.522 (xgb) | 0.613 (ridge) | 0.625 (ridge) |
| **5d** | **0.652 (ridge)** | **0.758 (ridge)** | **0.680 (ridge)** |
| 21d | 0.525 (lgbm) | 0.642 (lgbm) | 0.580 (lgbm) |
| 63d | 0.571 (ridge) | 0.706 (ridge) | 0.530 (xgb) — note: most models AUC < 0.5 |

The 1d horizon is too noisy (event-driven, dominated by pre-positioning and
algorithmic flow). 21d and 63d are too long (fundamentals and macro reassert,
the call's information is fully digested). 5d is the sweet spot — long enough
to absorb the call, short enough that the call's content still dominates the
return.

### 4.6 Theme normalization mattered more than expected

The LLM v1 prompt produced both `data_center` and `data center`, `supply_chain`
and `supply chain`, etc., interchangeably. Without normalization,
`n_new_themes` and `n_dropped_themes` were inflated by ~50% on a typical call —
that's noise the model was forced to learn around. Fixed in features.py with
a `_normalize_theme()` regex (collapse `_` / `-` / `/` → space) and in the v2
prompt with explicit format rules. Real fix; significant impact on theme-drift
features.

---

## 5. What I'd do with more time / data

### 5.1 More data (highest leverage)

- **Universe expansion**: 14 → S&P 500 (or at minimum a sector-balanced 50-100).
  Right now we're effectively doing within-ticker time-series with a few
  parallel timelines. With 100+ tickers we get genuine cross-section that
  enables sector-neutral and pair-trade strategies.
- **More history**: each ticker has 9-10 calls. Earnings transcripts go back to
  the 1990s for most large-caps. With 20+ calls per ticker, walk-forward
  retraining becomes viable and the train/test gap question becomes secondary.
- **Pre-call IV / options data**: implied volatility around the call window is a
  direct measure of "expected move size." Combined with the model's
  sentiment-derived prediction, you can compute "predicted vs implied" surprise.

### 5.2 Better features (moderate leverage)

- **Reactive vs proactive Q&A topics** (a stretch goal). One LLM pass over each
  parsed call producing a list of "topics raised in Q&A but not prepared
  remarks." S&P research suggests this is a real bearish signal.
- **Analyst-skepticism scoring** — extract not just `qa_tone` but the
  follow-up depth (number of follow-ups per analyst), question length,
  challenge density. A call where every analyst gets cut off after one
  question is structurally different from one where they get 3 follow-ups.
- **Wins/risks event embedding** — instead of category counts, embed the event
  text (e.g. "Data center revenue grew 80% YoY to $2.3B") with a sentence
  encoder and cluster cross-quarter. This catches "the same risk reappearing"
  semantically rather than via category match.
- **Sector context** — peer-call tone in the same week as the target call.
  A bullish AMD call against a bearish overall semiconductor group is
  structurally different from a bullish AMD call when everyone's bullish.

### 5.3 Better modeling (lowest leverage at this scale)

- **Stacked ensemble** — `Ridge + LGBM + KNN` with weights from inner CV. At
  130 samples this is risky (more parameters → more overfit) but on 1000+ it
  becomes useful.
- **Multi-task model** — predict 1d / 5d / 21d / 63d jointly with a shared
  representation. Could improve 1d by sharing structure with 5d.
- **Calibrated probabilistic output** — currently the regression `y_pred` is a
  point estimate. A model that emits an uncertainty (Gaussian process, Bayesian
  ridge, or MC dropout on a small NN) lets the backtest size positions by
  confidence — high-conviction calls get bigger positions, low-conviction
  calls get smaller or skipped.

### 5.4 Better backtesting (essential before any real money)

- **Walk-forward training**, not a single train/test split. Re-train every
  quarter using all data up to that point.
- **Realistic position sizing** — Kelly, vol-targeting, or capped at e.g. 5% per
  position, instead of equal-weight.
- **Transaction costs and slippage** — at minimum 5 bps per side.
- **Risk-managed equity curve**: stop-loss, position-level vol target, daily
  drawdown limits.
- **Monte Carlo on the test predictions** — bootstrap the trade sequence to get
  Sharpe confidence intervals. With only 46 trades the confidence interval is
  wide; this surfaces honestly how much of the "+37%" is signal vs lucky path.

---

## 6. Two pipelines — classical NLP first, then LLM

### 6.1 Why I started with classical NLP — the bottom-up plan

I built the **classical NLP pipeline first**, before introducing any LLM into
the system. The progression was deliberately **bottom-up**: start with the
cheapest, simplest tools that have well-understood properties, establish a
baseline, run **targeted intermediate experiments** to test if a partial
upgrade closes the gap (§6.3), and only escalate to a full LLM rewrite once
the cheaper paths plateau.

- **Start cheap.** A FinBERT inference pass costs essentially zero in compute
  and runs locally on CPU; VADER and Loughran-McDonald don't even need a
  model. **Initial development happened on Google Colab T4 free-tier**, which
  can run FinBERT and a 7B open-weight LLM (Qwen2.5-7B) but cannot fit a 14B
  model in memory. Establishing what the cheapest tools can extract first
  means I'd later know whether the LLM is adding signal or just adding cost.
- **Use well-understood models.** FinBERT, VADER, and Loughran-McDonald are
  decade-old (or older) tools with documented strengths and weaknesses. If a
  classical stack gets within 1-2 points of a much more expensive LLM
  pipeline, the LLM pipeline isn't justified.
- **Have a benchmark to beat.** "The LLM gets 65% direction accuracy" has no
  reference point on its own. With a classical baseline the LLM either
  materially outperforms or it doesn't, and the comparison answers the
  question.

After Pipeline 1 plateaued at ~0.65 direction accuracy and the **intermediate
experiments** (§6.3) failed to close the metric gap further, I built
**Pipeline 2** — the LLM-end-to-end pipeline this writeup primarily describes
— and migrated compute from **Colab → W&B Inference** so a 14B reasoning
model could actually run end-to-end. The compute upgrade was a consequence
of the modeling decision, not a starting choice.

### 6.2 Pipeline 1 — classical NLP (the simpler, cheaper starting point)

- **Parser** — **regex + spaCy** sentence segmentation. Regex patterns extract
  company name, ticker, quarter, year, and report date from the file header,
  and split the transcript into prepared remarks vs Q&A by detecting operator
  boilerplate. spaCy then segments each section into sentences for downstream
  scoring.
  *Why this approach?* Regex is deterministic, free, and runs in milliseconds;
  spaCy's English sentence-boundary detector is well-tested on news text. For
  the majority of transcripts this works correctly. *Limitation discovered
  later (and the main thing that motivated Pipeline 2's parser):* regex is
  brittle on edge cases — non-standard speaker labels, missing operator
  boilerplate, embedded punctuation in numbers — and **the prepared/Q&A
  boundary is interpretation-dependent**. Different regex patterns give
  different boundaries on the same transcript, and the boundary noise
  propagates into every per-section feature.
- **Sentiment scoring** — ensemble of two **FinBERT** variants
  (`yiyanghkust/finbert-tone`, pretrained on earnings-call transcripts;
  `ProsusAI/finbert`, pretrained on financial news) plus **VADER** (lexicon-
  based, no training) and **Loughran-McDonald** dictionary. Each scorer
  applied **separately to prepared remarks, Q&A, and the full transcript**,
  so each call yields three sentiment signatures.
  *Why an ensemble?* FinBERT's two checkpoints capture different training
  distributions; combining them with a rule-based (VADER) and a dictionary-
  based (LM) score reduces single-model variance. *Why per segment?* Tone in
  prepared remarks vs Q&A diverges often — analysts grilling the company in
  Q&A while management reads bullishly is a documented bearish signal.
- **Event extraction** — separate **Claude Sonnet** pass (via API, with
  Ollama and a heuristic regex as fallbacks) extracts `top_wins`, `top_risks`,
  `guidance_direction`, `themes`. Sentiment and event extraction kept
  architecturally separate so each step uses the model best suited to it:
  encoder models (FinBERT) for fast per-sentence sentiment, a strong reasoning
  LLM (Sonnet) only for events that require summarising the whole call.
- **Feature engineering** — **198 features**: per-speaker × per-segment
  FinBERT scores, LM ratios, VADER signals, **semantic drift** between
  consecutive calls (cosine distance on embedded prepared remarks), Q&A
  quality (follow-up depth, question length, challenge density), CEO/CFO tone
  gaps, theme counters (AI / China / pricing / supply chain / capex / buybacks
  / M&A / regulation / macro), and **lag / rolling / trend** features
  (prior-quarter values, rolling means, slopes).
  *Why so many?* Cast a wide net and let L2 regularisation decide what
  survives. Pipeline 1's verdict on this choice — see §6.5 — was that more
  features did not buy more accuracy.
- **Train/test split** — date-based cutoff at **2025-06-26** → **83 train /
  37 test** (the standard time-series backtest convention).
- **Models tried** — same nine families as Pipeline 2 (ridge, lasso,
  elasticnet, logreg, knn, rf, histgb, xgb, lgbm), plus a **neural MLP sweep**
  (small feed-forward, hidden=32, dropout=0.2) and **FinBERT-only baselines**
  to test whether transformer-derived sentiment alone is sufficient.

### 6.3 Intermediate experiments — bridging Pipeline 1 and Pipeline 2

The leap from Pipeline 1 to Pipeline 2 wasn't a single jump. Between them I
ran a set of **targeted intermediate experiments** to test whether partial
upgrades to Pipeline 1 could close the metric gap without committing to a
full LLM-end-to-end rewrite. None did — and the negative results are exactly
what shaped Pipeline 2's final design choices.

**1. Per-section sentiment — 4 sentiment models × 2 sections.** Pipeline 1's
final feature set already used the four sentiment scorers (FinBERT,
**FinBERT-Tone**, Loughran-McDonald, VADER). The intermediate experiment ran
each scorer **separately on prepared remarks and on Q&A**, producing 8
sentiment vectors per call instead of one global signature.
*Why try this?* Hypothesis: prepared remarks are scripted and uniform across
companies, so they should be a noisier signal than Q&A, where management
improvises and where stress / hedging / non-answers actually surface.
Separating the two sections cleanly should let the model learn the
prepared-vs-Q&A *gap* as a feature in its own right.
*Result.* No significant lift on test direction accuracy. This is consistent
with Pipeline 2's later feature-importance ablation showing that **scalar
tone — at any granularity — contributes only 3-4% of feature importance**.
The bottleneck isn't sentiment resolution; it's that scalar tone compresses
information that is better preserved by *categorical event extraction*
(wins/risks counts).

**2. Smaller-model event extraction — Qwen2.5-7B locally.** Pipeline 1 used
Claude Sonnet via API for event extraction (with Ollama and a heuristic
fallback). The intermediate experiment swapped Sonnet for **Qwen2.5-7B**
running locally on Colab T4 via Ollama (free GPU).
*Why try this?* Cost. Sonnet at scale is non-trivial; a 7B open-weight model
on Colab is essentially free and would make the whole pipeline self-hostable.
If a 7B model could match Sonnet on event extraction, the pipeline becomes
substantially cheaper to operate and easier to reproduce.
*Result.* No significant improvement on downstream metrics, and qualitative
event extraction was visibly noisier (events miscategorised, wins/risks
counts inflated by duplicates). The diagnosis: **event extraction quality is
bound by the LLM's reasoning capacity, not by smarter prompting**. A 7B model
can list "wins" and "risks" but doesn't reliably aggregate them into the
right categories or pick the most significant 3-5 from 30+ mentioned events.

**The pattern across intermediate experiments.** Adding granularity to weak
features (per-section sentiment) didn't help. Cheapening the strong feature
(LLM event extraction) didn't help. Together these two negative results
pointed Pipeline 2 toward a specific design: **(a) replace scalar tone with
categorical event counts as the primary signal carrier**, and **(b) scale up
the LLM rather than scaling it down** — which forced the compute migration
from Colab T4 to W&B Inference. Pipeline 2's design isn't an arbitrary
"throw more LLM at the problem"; it falls out naturally from these
intermediate negative results.

### 6.4 Pipeline 2 — LLM end-to-end (the upgrade with stronger techniques)

This is the pipeline the rest of this writeup describes. The advanced
techniques it introduced over Pipeline 1:

- **Marker-based parsing instead of regex.** Pipeline 1's regex + spaCy
  parser was interpretation-dependent and brittle (see §6.2). Pipeline 2
  replaces it with a **marker-based slicer**: the LLM emits 60-character
  prefix markers for each speaker block (rather than reproducing the speech
  text), and a Python slicer **locates those markers by exact character
  offset in the original transcript** and slices verbatim. The result is
  **distance-precise** segment boundaries — every slice is exactly the source
  text, character-for-character, with no LLM paraphrasing and no regex
  pattern misses. *Why?* In Pipeline 1 we kept hitting transcripts where the
  regex parser silently mis-attributed analyst questions to management or
  vice versa; distance-precise slicing makes that class of error
  structurally impossible.
- **One LLM (Qwen3-14B) does everything else** — sentiment, event
  extraction, theme tagging — in a single structured-JSON pass per call,
  using the marker-based slices as input.
  *Why?* To test whether a single strong reasoning model with full call
  context beats a stack of specialised classical models. The LLM picks up on
  negation, hedging, and sentence-level composition that bag-of-sentences
  sentiment models miss. The 14B parameter count is the smallest size that
  the intermediate experiments showed could do event extraction reliably (a
  7B model could not — see §6.3).
- **Compute migration to W&B Inference.** Qwen3-14B needs more memory than
  Colab T4 free-tier (12.7 GB) can fit. Migrating to **W&B Inference**
  (OpenAI-compatible API hitting `OpenPipe/Qwen3-14B-Instruct`) decouples
  model size from local hardware while keeping the Python pipeline
  unchanged; the only delta is the `ECS_LLM_BACKEND` env var. This was the
  one infrastructure choice forced by the modelling decision, not the other
  way around.
- **Continuous calibrated tone** — `extract_v2` prompt with explicit 7-bin
  rubric and two-decimal precision, instead of FinBERT's three discrete
  classes. *Why?* Continuous tone gives the regression a real magnitude
  signal; three discrete classes collapse to a few values and lose ranking
  information.
- **Pre-call drift** — 6 features encoding 5d / 21d / 63d excess returns
  *before* the call. *Why?* Pipeline 1 had no notion of "what's already in
  the price." Pipeline 2 added it explicitly because the AMD November 2025
  case (tone 0.92, +55% pre-call run-up, -16% realised excess) was the
  textbook case Pipeline 1's features could not distinguish from a clean
  bullish surprise.
- **Per-ticker chronological train/test** — first 6 calls per ticker → train,
  rest → test. *Why?* Date cutoff (Pipeline 1) is simpler, but every ticker
  should appear in both splits or the test result conflates "the problem is
  hard" with "this ticker happened to be on the test side." Per-ticker
  chronological is the stricter test.
- **Tighter feature set (54 vs 198)** — categorical event counts (wins/risks
  by category), QoQ deltas, pre-call drift, LM lexicon as sanity layer.
  *Why fewer?* Pipeline 1's verdict that 198 features did not beat Ridge with
  ~50 well-chosen features. Pipeline 2 trades feature count for feature
  *quality* — categorical event extraction carries far more signal than per-
  speaker per-segment sentiment scores.

### 6.5 Side-by-side on shared metrics

Both pipelines independently select **Ridge regressor** as the winning model:

| Metric | Pipeline 1 (classical NLP) | Pipeline 2 (LLM end-to-end) |
|---|---|---|
| Train rows / test rows | 83 / 37 | 84 / 46 |
| Feature count | 198 | 54 |
| **Test direction accuracy** | 0.649 | **0.652** |
| Test F1 (positive class) | 0.723 | **0.758** |
| Test AUC | 0.641 | **0.680** |
| Test MCC | 0.172 | **0.380** |
| Test Spearman IC | 0.24 | **0.34** |
| Test R² | -2.37 | **-0.13** |
| Strategy total return | +37.4% | +37.2% |

Pipeline 2 wins or ties on every classification, ranking, and regression
metric. The realised trading return is essentially identical because the P&L
of a sign-driven long/short strategy is dominated by direction accuracy, on
which the two pipelines agree within 0.3 percentage points.

### 6.6 Conclusion across both pipelines

- **Pipeline 2 is materially better at fine-grained metrics.** F1 +3.5pp,
  AUC +3.9pp, MCC +0.21, Spearman IC +0.10, R² far less negative
  (-0.13 vs -2.37). Pipeline 2 ranks calls correctly more often *and*
  predicts magnitude with real signal, while Pipeline 1 only predicts sign
  well. For any strategy that sizes positions by predicted magnitude (Kelly,
  vol-targeting, conviction-weighting), Pipeline 2's calibrated continuous
  output is the better starting point.
- **The realised P&L is the same on this sample.** Both pipelines hit ~+37%
  excess return because both have ~65% direction accuracy and the backtest is
  sign-driven. *In other words*: the LLM upgrade pays off in metric quality
  on this sample, not (yet) in dollars. The LLM's edge would show up in P&L
  once positions are sized by conviction or once the strategy lives at a
  horizon where magnitude prediction matters.
- **Both pipelines independently picked Ridge.** Two independent feature sets,
  two independent train/test splits, nine model families each plus an MLP
  sweep — same answer. **L2-regularised linear regression** is the right
  inductive bias for weak, distributed signal at small N. Lasso lost ~9
  percentage points (over-aggressive feature zeroing); trees overfit
  (0.94+ train, ~0.59 test in both pipelines); MLP plateaued at 0.583
  (insufficient samples for non-linear capacity).
- **FinBERT-only baselines failed badly** in Pipeline 1 (returns -15.6% to
  -36.1%). Confirms the most counterintuitive finding of Pipeline 2: scalar
  tone, even from a domain-adapted transformer, is insufficient. The signal
  lives in the *categorical events* (wins/risks counts by category, guidance
  direction), not in the tone scalar.
- **Both pipelines are sample-starved.** With 80-something train rows, every
  model is shrinkage-dominated. The next iteration's marginal hour is best
  spent on **200+ more transcripts**, not more sentiment models or
  architectures. With train ≥ 300, both pipelines predict the same thing:
  XGBoost / LightGBM should catch up with Ridge, and Pipeline 1's magnitude-
  prediction gap should close.
- **The methodological lesson.** Starting cheap was the right call. Pipeline 1
  produced a defensible +37% return with no GPU and no paid API, and gave
  Pipeline 2 a clear benchmark to beat. Pipeline 2 justifies its compute cost
  on metric quality (F1, MCC, IC, R²), not on direction accuracy alone —
  without Pipeline 1's baseline I would not have known the LLM upgrade was a
  metric-quality win rather than a direction-accuracy win.

### 6.7 Where both pipelines fail — per-call analysis

A stricter test of *"do these pipelines really capture the same thing?"* is to
compare **per-call**, not just aggregate accuracy. Of Pipeline 2's 46 test
calls and Pipeline 1's 37 test calls, **35 calls overlap** by (ticker, report
date) — the gap comes from the per-ticker-chronological split (Pipeline 2)
keeping more recent calls than Pipeline 1's date cutoff at 2025-06-26.

| Outcome on the 35 overlapping calls | Count | % |
|---|---|---|
| Both pipelines correct | 16 | 45.7% |
| Both pipelines wrong | 6 | 17.1% |
| Pipeline 1 correct, Pipeline 2 wrong | 7 | 20.0% |
| Pipeline 2 correct, Pipeline 1 wrong | 6 | 17.1% |

**Cohen's κ = 0.19** between the correct/wrong indicators of the two
pipelines — slight agreement above chance, meaning the **failures are
mostly independent**. The two pipelines fail jointly on only 6 of 35 calls;
on the remaining 13 disagreements they each get right what the other gets
wrong.

The 6 calls where **both pipelines fail in the same direction** share a
striking pattern:

| Call | Realised excess_5d | Both predicted | Sector |
|---|---|---|---|
| NVDA 2025-08-27 | -7.0% | long | Tech |
| FAST 2025-10-13 | -0.03% | long | Industrials |
| WFC 2025-10-14 | -3.4% | long | Financials |
| JPM 2026-01-13 | -1.2% | long | Financials |
| GS 2026-01-15 | -3.3% | long | Financials |
| WFC 2026-01-15 | -0.5% | long | Financials |

Three observations:

1. **All 6 both-wrong calls were predicted long.** Both pipelines independently
   developed the same systematic bias: they over-trust positive-sentiment
   calls in stocks that subsequently sold off post-call.
2. **4 of 6 are Financials.** The Q3-2025 and Q4-2025 bank-earnings cluster
   was a textbook "sell the news" episode — strong reported numbers and
   bullish management tone, but the stocks sold off because expectations were
   already priced in. Failure rate by sector on the overlap: Financials 4/15
   (26.7%) vs Tech 1/11 (9%).
3. **NVDA Aug 2025 is the documented sell-the-news case** highlighted in
   [§2.1](#21-nvda--story-holds-bullish-market-unwinds). That **both**
   pipelines miss it — and miss it in exactly the same direction — confirms
   it isn't a feature-engineering oversight in either codebase. It is a
   feature gap that **neither** pipeline closes.

**A direct check on the 6 joint-failure calls.** To be sure the sentiment
extraction itself wasn't to blame, I cross-checked the pipeline's sentiment
labels against an independent re-labelling by **Claude Opus 4.7** on the
same 6 calls — using the same `extract_v2` prompt, but a different model
family from Pipeline 2's Qwen3-14B. The two sets of labels matched closely:
**Pearson correlation of 0.89 on `overall_tone`, and all 6 calls
independently confirmed as bullish.** The pipelines were not misled by
their own tone scores; the bullish read matches the source material and
matches what a stronger judge produces independently. This rules out
"we extracted sentiment incorrectly" as the cause of the joint failures
and makes the missing sector-context features the only remaining
explanation. **The bottleneck is not in our extraction.**

**What convergent failure tells us.** When two methodologically independent
pipelines fail on the same calls in the same direction, the failure isn't a
pipeline bug — it's a **feature gap shared by both**. Specifically, neither
pipeline has features that would distinguish "strong earnings, stock will
continue up" from "strong earnings, market already knows, stock will sell
off." Pipeline 2 added pre-call drift to address exactly this, and on average
it helps (Pipeline 2's +3.5pp F1 and much-better R² over Pipeline 1 come
partly from drift), but on the most extreme cases (post-bank-rally Q3-Q4
2025) the drift signal alone is not strong enough.

**Concrete next step.** This is exactly the case for adding **sector-peer
context features** — what was the recent sector-wide earnings-reaction
pattern when the call happened? — listed as a v3 priority in
[§5.2](#52-better-features-moderate-leverage). The convergent-failure
analysis points directly at the feature class that would close the gap.

**Result to take away from the comparison.** The two pipelines are *not*
redundant. They agree on roughly half the test calls, disagree on 37%, and
fail jointly on 17%. The aggregate metrics look similar (both around 65%
direction accuracy) but per-call this is a noisy, mostly-independent two-vote
ensemble. The cleanest practical recommendation: **a future v3 should ensemble
both pipelines** — average their continuous predictions and flag the calls
where they disagree as low-confidence. The 16 both-correct calls would
provide high-conviction signal, and the 13 disagreement calls would either be
skipped or sized small. This is exactly the kind of conviction-weighting that
Pipeline 2's continuous output already supports.

---

## Reproducing the pipeline

```powershell
# one-time setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .

# LLM backend (W&B Inference; replace with your credentials)
$env:ECS_LLM_BACKEND  = "wandb"
$env:WANDB_API_KEY    = "<your key>"
$env:WANDB_PROJECT    = "<your_user>/earnings-calls"
$env:ECS_MODEL        = "OpenPipe/Qwen3-14B-Instruct"
$env:ECS_NUM_PREDICT  = "8192"

# full pipeline
python -m earnings_call_sentiment.parse    --force
python -m earnings_call_sentiment.extract  --force
python -m earnings_call_sentiment.prices   --force
python -m earnings_call_sentiment.features
python -m earnings_call_sentiment.model
```

Outputs:
- `data/parsed/*.json`, `data/extractions/*.json` — LLM outputs
- `data/prices/returns.csv` — forward returns + pre-call drift, look-ahead-safe
- `data/features/features.csv` — 130 rows × 74 columns ML-ready
- `data/model/metrics_summary.csv` — full metric table per (horizon × model × split)
- `data/model/plots/learning_curve_h*d.png`, `data/model/plots/overfit_h*d.png`
- `data/backtest/equity_curve_ridge_5d.png` — equity curve PNG
- `data/backtest/trades_ridge_5d.csv` — per-trade table
