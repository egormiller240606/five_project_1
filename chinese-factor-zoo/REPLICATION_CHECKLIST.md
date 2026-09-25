# Final reproduction checklist

Use this before the presentation/repository submission.

- [ ] Export WIND monthly/daily A-share market data, 2000-01 through 2020-06.
- [ ] Export CSMAR quarterly statements with **actual disclosure dates** and one-year government bond yield.
- [ ] Export/build the 11 macro predictors from CSMAR/NBS.
- [ ] Map vendor column codes to canonical names; do not use fiscal period-end as information date.
- [ ] Build/verify all 90 characteristics; inspect missingness by characteristic and year.
- [ ] Confirm exactly 80 industry dummies and 1,160 model predictors.
- [ ] Run `pytest` and the point-in-time/leakage tests.
- [ ] Run all 13 models without `--max-candidates`.
- [ ] Run `03_evaluate_models.py` and `06_compare_with_paper.py`; explain every material Table-1 discrepancy.
- [ ] Run portfolios and compare Table 6 / Table 10 directionally.
- [ ] Run top-70% market-cap robustness, SOE/non-SOE results, model averaging, and price-limit robustness if flags are available.
- [ ] Put the paper figure/table beside the reproduced version on slides with identical units/axes.
- [ ] Report data-source differences and unresolved replication gaps explicitly.
