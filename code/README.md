# Code for the 1.5-year monthly return-based study

Each model folder has four regime notebooks (`2008-2009`, `2013-2014`, `2018-2019`, `2019-2020`). Those notebooks estimate **P** from lookback returns and simulate paths. The scripts below load them headless for LSM and stock-path reports.

| Folder | Model |
|--------|--------|
| `gbm/` | Geometric Brownian Motion |
| `md_gbm/` | MD-GBM: 1-lag Markov direction + folded-normal sizes matched to sample \|R\| |
| `garch/` | GARCH(1,1), Duan LRNVR for Q |
| `merton/` | Merton jumps; Pan \(\mu_J^*\) for Q |
| `garch_merton/` | GARCH block + Merton jumps |

## Scripts that actually run the study

| Script | Role |
|--------|------|
| `run_v4_1p5y_10k_monthly.py` | Assemble LSM + stock + MD-GBM spec |
| `run_v4_1p5y_10k_monthly_empirical_study.py` | LSM config and cache |
| `run_v4_1p5y_10k_monthly_empirical_study_groups.py` | Write the 5-model LSM report |
| `run_v4_1p5y_10k_monthly_stock_study.py` | P-measure stock-path report |
| `run_v3_5y_monthly_empirical_study.py` | Shared LSM engine |
| `run_optimal_stopping_study.py` | Headless notebook loader + LSM loop |
| `american_lsm.py` | Longstaff–Schwartz |
| `option_filters.py` | Call filters (ATM band, DTE, liquidity) |
| `garch_duan_lrnvr.py` | GARCH P/Q |
| `garch_merton_pq.py` | GARCH–Merton P/Q |
| `pq_risk_premium.py` | Merton Pan \(\mu_J^*\) (and leftover unused Bates helpers) |
| `data_fetch.py` / `data_prepare.py` / `options_fetch.py` | Build `data/` |

`results/` next to these folders is the **cache** for this study, not the published PDFs. Published reports are in `../results/1p5y_monthly_return_based/`.
