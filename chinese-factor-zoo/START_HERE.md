# Start here

If you only want to understand or demonstrate the project, ignore most folders at first.

## 1. Run the tests

```bash
pip install -e .[dev]
pytest -q
```

## 2. Run the self-contained synthetic experiment

```bash
python scripts/run_replication.py --data synthetic --quick --regenerate
```

This performs, in order:

```text
create synthetic data
→ build 1,160 predictors
→ paper-style temporal split
→ train OLS-3 / RF / NN4
→ calculate OOS R²
→ build long-only and long-short portfolios
→ apply transaction costs
→ compare scale with paper reference values
```

## 3. Look at the result

Open:

```text
SYNTHETIC_RESULTS.md
outputs/synthetic/RESULTS.md
```

The bundled example is a real run of this repository, but on synthetic data. It is not presented as a reproduction of WIND/CSMAR numbers.

## 4. For the actual empirical replication

Put the required exports in `data/raw/` and run:

```bash
python scripts/run_replication.py --data real --models all
```

## Files worth reading first

```text
scripts/run_replication.py              high-level experiment
src/factor_zoo/data/synthetic.py        synthetic DGP
src/factor_zoo/data/panel.py            1,160-feature construction
src/factor_zoo/training/runner.py       expanding train/validation/test loop
src/factor_zoo/evaluation/metrics.py    paper R²
src/factor_zoo/portfolio/sorts.py       decile portfolios
```

Everything else supports those pieces.
