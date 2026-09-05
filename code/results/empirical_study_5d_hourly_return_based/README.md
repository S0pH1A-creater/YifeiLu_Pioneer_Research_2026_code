# V4 5-day hourly return-based empirical study (10,000 paths)

1-minute RTH data · **5-day** lookback · **hourly** rolling · return-based models only.

| | |
|---|---|
| Models | MD-GBM, GBM, GARCH, Merton, GARCH–Merton |
| Tickers | SPY, AAPL, MSFT |
| Windows | 12 Friday-before-expiry weeks (Oct 2022 – Sep 2023) |
| Lookback | 5 RTH days (1950 one-minute bars) |
| Rolling | hourly |
| Paths | 10,000 · Δt = 5 minutes |
| P→Q | GBM μ→r_f; GARCH Duan LRNVR; Merton Pan μ_J*; GARCH–Merton Duan+Pan; MD-GBM additive Q |

Canonical cache: `V4-Models_result/results/empirical_study_5d_hourly_return_based/`

- PDF: `V4_5d_hourly_empirical_study_return_based.pdf`
- Notebook: `Notebooks/V4_5d_hourly_empirical_study_return_based.ipynb`

Short report: `Results_In_Short/V4/V4 5-day hourly return-based empirical study 10000 paths/`
