# Machine Learning in the Chinese Factor Zoo — clean-room reproduction

Our implementation of Leippold, Wang & Zhou (2022). The project has two modes:

1. **Synthetic demo** — runs without proprietary data and proves the whole pipeline works end-to-end.
2. **Real-data replication** — same pipeline, but using WIND/CSMAR/NBS-compatible exports.

The synthetic mode is an **extension / validation exercise**, not a claim that synthetic numbers reproduce the paper.

## Fastest way to understand the project

You only need to open two scripts first:

```text
scripts/generate_synthetic_data.py   # creates paper-shaped synthetic inputs
scripts/run_replication.py           # runs the whole experiment
```

Everything in `src/` is the reusable implementation behind those two scripts.

## One-command synthetic run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest

python scripts/run_replication.py --data synthetic --quick --regenerate
```

`--quick` keeps the paper-style data construction and time split, but uses a much smaller compute budget and three representative models (`OLS-3`, `RF`, `NN4`). It is for demonstration/debugging, not final empirical replication.

The included example run is already saved in:

```text
SYNTHETIC_RESULTS.md
outputs/synthetic/RESULTS.md
outputs/synthetic/tables/
outputs/synthetic/figures/
```

## What happens inside the pipeline

```text
raw stock characteristics + macro data + returns
                    ↓
cross-sectional rank to [-1, 1]
                    ↓
90 stock characteristics
+ 90 × 11 stock×macro interactions
+ 80 industry dummies
                    ↓
1,160 predictors
                    ↓
train 2000–2008
validation 2009–2011
out-of-sample test from 2012 onward
annual refits
                    ↓
R²_oos + subgroup analysis
                    ↓
decile long-only / long-short portfolios
                    ↓
turnover-based transaction costs
                    ↓
paper-reference comparison
```

## Synthetic data

The generator creates a monthly panel with the same **structure** needed by the paper pipeline:

- 90 stock characteristics;
- 11 macro variables;
- 80 industries;
- market capitalization;
- SOE flag;
- shareholder-size proxy;
- persistent/nonlinear predictive signal;
- noisy next-month stock returns.

It does **not** tune the synthetic data to force the paper's published R² or Sharpe ratios. The point is to test whether the code can discover a known signal and complete the full workflow without WIND/CSMAR.

Generate only the data:

```bash
python scripts/generate_synthetic_data.py --n-stocks 120
```

Generated files go to `data/synthetic/raw/`. Parquet is preferred; CSV is used automatically if a Parquet engine is unavailable.

## Real-data replication

Place equivalent exports in `data/raw/`:

```text
data/raw/stock_characteristics.parquet
data/raw/macro.parquet
data/raw/monthly_returns.parquet
```

Expected stock-characteristic columns:

```text
stock_id, date,
<90 names from factor_zoo.features.catalog.STOCK_CHARACTERISTICS>,
industry, market_cap, soe,
avg_mcap_per_shareholder
```

Macro columns:

```text
date, dp, de, bm, svar, ep, ntis, tms, infl, mtr, m2gr, itgr
```

Returns:

```text
stock_id, date, return, risk_free
```

Then run:

```bash
python scripts/run_replication.py --data real --models all
```

The paper's exact sample uses proprietary WIND + CSMAR data plus NBS macro data. Therefore a true empirical claim of “similar results” requires original-equivalent exports.

## Models

Full mode supports:

```text
OLS + Huber
OLS-3 + Huber
PLS
LASSO + Huber
Elastic Net + Huber
GBRT + paper-style fixed Huber threshold
Random Forest
VASA
NN1, NN2, NN3, NN4, NN5
```

Paper-style settings are in `configs/models.yaml`. `configs/synthetic.yaml` only controls the synthetic experiment and the small quick-run model subset.

## Main paper-specific choices implemented

- 90 stock characteristics + 11 macro variables + 80 industry dummies;
- all 990 characteristic×macro interactions;
- exactly 1,160 predictors when all 80 industries are observed;
- cross-sectional characteristic ranking into `[-1, 1]`;
- non-demeaned out-of-sample R²;
- time-ordered train/validation/test split;
- annual expanding-window refits;
- Huber threshold `M = 1.35`;
- value-weighted D10 and D10−D1 portfolios;
- transaction-cost scenarios 0/20/40/60/80 bps;
- paper values stored separately as reference checkpoints, never as generated results.

## Project structure

```text
configs/                     model / experiment / synthetic settings
src/factor_zoo/data/         loaders, panel construction, synthetic generator
src/factor_zoo/features/     90-characteristic catalog and preprocessing
src/factor_zoo/models/       linear, trees, VASA, neural networks
src/factor_zoo/training/     temporal splits, tuning, experiment runner
src/factor_zoo/evaluation/   R², subgroups, paper comparison
src/factor_zoo/portfolio/    portfolio sorts, turnover, costs, statistics
scripts/run_replication.py   simple end-to-end entry point
scripts/01...08              individual research stages
outputs/                     generated results
```

## Included synthetic results

The bundled quick run used **120 synthetic stocks, 246 months, 90 characteristics, 11 macro variables and 80 industries**. It produced actual out-of-sample predictions and portfolio results. See `SYNTHETIC_RESULTS.md` for the values and the explicit warning that they are synthetic.

## Tests

```bash
pytest -q
```

Current suite: **17 tests**. It checks timing, custom R², feature construction, Huber loss, VASA logic, point-in-time merging, portfolios, transaction costs, paper checkpoints and the synthetic-data generator.

## What should be shown in the seminar

A clean presentation can separate three things:

1. **Method reproduction:** our own implementation of the article's pipeline.
2. **Synthetic validation/extension:** the code runs end-to-end and recovers some predictive structure without proprietary data.
3. **Empirical replication:** only claimed after running the same code on original-equivalent WIND/CSMAR/NBS data and comparing the resulting tables with the article.

See `IMPLEMENTATION_NOTES.md` and `REPLICATION_CHECKLIST.md` before submission.
