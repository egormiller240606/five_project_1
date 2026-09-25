# Synthetic demonstration results

> **Important:** these are real outputs of this repository on generated data. They are **not** a replication of the paper’s proprietary WIND/CSMAR sample.

## Dataset

- 120 synthetic stocks, 246 months (2000-01-31 to 2020-06-30)
- 90 stock characteristics, 11 macro variables, 80 industries
- approximate predictable signal variance share in the DGP: 4.13%

## Out-of-sample R² (%)

| model   |   all |   soe |   non_soe |   bottom_30_mcap |   top_70_mcap |   bottom_30_shareholder |   top_70_shareholder |
|:--------|------:|------:|----------:|-----------------:|--------------:|------------------------:|---------------------:|
| nn4     | 0.028 | 0.036 |     0.021 |            0.074 |         0.008 |                   0.116 |               -0.012 |
| ols3_h  | 1.513 | 0.854 |     2.133 |            2.164 |         1.221 |                   2.031 |                1.278 |
| rf      | 0.585 | 0.260 |     0.892 |            0.987 |         0.406 |                   0.964 |                0.414 |

## Portfolio results, 0 bps

| model   | strategy   |   avg_monthly_pct |   sharpe_ann |
|:--------|:-----------|------------------:|-------------:|
| nn4     | long_only  |             0.227 |        0.265 |
| nn4     | long_short |             0.475 |        0.603 |
| ols3_h  | long_only  |             1.154 |        1.492 |
| ols3_h  | long_short |             2.272 |        2.978 |
| rf      | long_only  |             0.776 |        0.861 |
| rf      | long_short |             1.607 |        1.887 |

## Paper reference vs synthetic run (full sample)

This comparison is shown only as a scale/reference check. The synthetic generator is not calibrated to reproduce the paper’s exact values.

| model   |   paper_pct |   synthetic_pct |   abs_diff_pp |
|:--------|------------:|----------------:|--------------:|
| nn4     |       2.490 |           0.028 |         2.462 |
| ols3_h  |       0.770 |           1.513 |         0.743 |
| rf      |       2.440 |           0.585 |         1.855 |

## What this proves

The project can run end-to-end without proprietary data: data generation → 1,160-feature panel → temporal training → out-of-sample predictions → R² → decile portfolios → transaction costs → paper-reference comparison. Numerical equality with the article still requires original-equivalent data.
