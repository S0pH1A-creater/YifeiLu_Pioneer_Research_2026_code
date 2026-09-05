# Return-based 1.5-year monthly study

This folder holds two V4 return-based studies (same MD-GBM meanfix model):

1. **1.5-year monthly** — daily equity, 18-month lookback, monthly rolling, SPY/AAPL/MSFT/AMZN
2. **5-day hourly** — 1-minute RTH data, 5-day lookback, hourly rolling, SPY/AAPL/MSFT

## Layout

```
code/                          # model notebooks + scripts
  md_gbm/                      # MD-GBM (1-lag Markov Directional GBM, meanfix)
  results/empirical_study_1p5y_monthly_10000/
  results/empirical_study_5d_hourly_return_based/
data/
  equity/                      # daily adj close
  equity/short_interval/       # 1-minute RTH (5-day study)
  options/processed/
docs/figures/                  # stock-price data-collection plots
results/1p5y_monthly_return_based/
results/5d_hourly_return_based/
```

## Re-run

From the repo root, with Anaconda Python (the repo `.venv` may hang on numpy):

```bash
export MPLCONFIGDIR="$(pwd)/.mplconfig"
/opt/anaconda3/bin/python code/scripts/run_v4_1p5y_10k_monthly.py
```

Stored LSM cells in `code/results/empirical_study_1p5y_monthly_10000/` are reused unless you force a recompute.
