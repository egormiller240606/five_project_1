# Implementation notes — exact paper choices vs transparent clean-room choices

## Encoded directly from Leippold, Wang & Zhou (2022) / Internet Appendix

- Huber transition: `M = 1.35` with `H(e)=e²` for `|e|<=M`, otherwise `2M|e|-M²`.
- OLS-3 predictors: `bm`, `mve`, `mom1m`.
- Monthly timing: train 2000–2008, validation 2009–2011, next 12 months test; refit annually; expanding train / rolling 3-year validation.
- 90 stock characteristics, 11 macro variables, 80 industry dummies; `90 + 90*11 + 80 = 1,160` predictors.
- Cross-sectional rank transform of continuous stock characteristics into `[-1,1]` each period.
- Models: OLS, OLS-3, PLS, LASSO, Elastic Net, GBRT, RF, VASA, NN1–NN5.
- NN widths: `[32]`, `[32,16]`, `[32,16,8]`, `[32,16,8,4]`, `[32,16,8,4,2]`.
- NN: ReLU, Adam, batch normalization, dropout, early stopping; 100 epochs, patience 5, ensemble of 10.
- Hyperparameter ranges from Table A.2 are encoded in `configs/models.yaml`.
- Value-weighted D10 long-only and D10−D1 long-short portfolios, rebalanced monthly.
- Transaction-cost scenarios: 0/20/40/60/80 bps.
- Published Tables 1, 2, 6 and selected Table 10 checkpoints are transcribed in `evaluation/paper_targets.py` for automatic comparison, never as generated results.

## Improvements over the first draft

1. **GBRT no longer uses sklearn's quantile-parameterized Huber loss.** `PaperHuberGBRT` uses the paper's fixed `M=1.35` transition and clipped Huber pseudo-gradients.
2. **VASA no longer samples predictors uniformly.** Sampling probabilities are proportional to univariate in-sample R², then OLS submodels are averaged.
3. **Transaction costs are turnover-based.** The code builds constituent target weights, drifts previous weights by realized returns, computes half-L1 turnover, and applies bps × turnover rather than subtracting a flat monthly bps number.
4. **Point-in-time accounting merge helper.** `asof_merge_financials` only makes a statement available after its disclosure date, explicitly guarding against look-ahead bias.
5. **Paper result regression checks.** `scripts/06_compare_with_paper.py` measures deviations from Table 1 after a real-data run.
6. **Robustness/extension code.** Top-70%-market-cap filtering, price-limit buy exclusion hook, and equal-weight forecast model averaging are included.
7. **Dropout is tunable.** The appendix states dropout is used but does not give the exact probability; the code tunes `{0, .1, .2, .5}` on the validation sample instead of silently hard-coding one value.
8. **17 unit tests** cover timing, feature count, Huber loss, VASA probabilities, custom GBRT, portfolio exposures, turnover costs, point-in-time merging, macro construction, and published-table checkpoints.


## Synthetic validation mode

The repository now includes a structured synthetic-data generator and a single end-to-end runner. The generator reproduces the **shape of the information set** (90 characteristics, 11 macro variables, 80 industries, firm size/SOE metadata) and creates next-month returns from a documented nonlinear signal plus noise. It is intentionally not calibrated to the paper's published R²/Sharpe values. Synthetic outputs are labelled separately from empirical replication outputs.

## Remaining empirical limitation (cannot be solved honestly in code alone)

The exact raw WIND/CSMAR/NBS dataset is proprietary and is not contained in the paper. Therefore the repository must not claim that its generated numbers match the paper until the original-equivalent exports are supplied and `06_compare_with_paper.py` has been run. The authors also state that their constructed data can be requested from them.

Vendor field codes are deliberately not invented. `raw_characteristics.py` and `macro.py` use canonical field names; map your export schema to those names in the ingestion layer.

## Clean-room choices where the paper is underspecified

- Continuous ranges in Table A.2 are represented by finite grids inside the stated ranges.
- The exact dropout rate is not reported; it is validation-tuned here.
- VASA is implemented from the described variable-subsample aggregation principle with R²-based sampling probabilities. Any unpublished implementation details of the authors are necessarily unavailable.
- Transaction-cost turnover uses the standard half-L1 rebalancing convention and drifted prior weights. This is a stronger implementation than a flat monthly deduction; if an instructor requires a different turnover convention, it is isolated in `portfolio/costs.py`.
