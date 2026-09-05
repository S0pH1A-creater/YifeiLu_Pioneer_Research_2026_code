# V4 5-day hourly return-based study (10,000 paths)

1-minute RTH data · **5-day** lookback · **hourly** rolling · return-based models only.

MD-GBM here is the same meanfix model as in the 1.5-year monthly study (`code/md_gbm/`).

| | |
|---|---|
| Models | MD-GBM, GBM, GARCH, Merton, GARCH–Merton |
| Tickers | SPY, AAPL, MSFT |
| Windows | 12 Friday-before-expiry weeks (Oct 2022 – Sep 2023) |
| Lookback | 5 RTH days (1950 one-minute bars) |
| Rolling | hourly |
| Paths | 10,000 · Δt = 5 minutes |

- Report: `01_optimal_stopping/`
- Cache: `code/results/empirical_study_5d_hourly_return_based/`
- Minute data: `data/equity/short_interval/`, `data/options/processed/short_interval/`
- Runner: `code/scripts/run_v4_5d_hourly_return_based_study.py`
