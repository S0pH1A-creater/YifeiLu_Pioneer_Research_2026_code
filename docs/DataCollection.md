# Data

Tickers in this study: **SPY, AAPL, MSFT, AMZN**.

```
data/
  equity/
    prices_clean.csv           # adj close, 2003-12-01 → 2020-12-31
    log_returns_all.csv
    log_returns_by_regime.csv
    summary_stats.csv
  rates/
    risk_free_dgs3mo.csv       # FRED DGS3MO
  options/processed/
    {SPY,AAPL,MSFT,AMZN}_calls_panel.csv
    {SPY,AAPL,MSFT,AMZN}_options_panel.csv
    calls_panel_{all,crisis,normal,late,covid}.csv
    options_panel_{all,crisis,normal,late,covid}.csv
    options_summary.csv
```

## Equity

- SPY: State Street NAV
- AAPL / MSFT / AMZN: Yahoo adj-close mirror
- Log return \(R_t=\ln(S_t/S_{t-1})\)
- Common sample on disk starts 2003-12-01 (SPY NAV)

## Listed American calls

- Dates: 2008-01-01 → 2020-12-31
- Near ATM: \|K/S − 1\| ≤ 10%
- DTE 7–60
- Used for LSM comparison and for Merton / GARCH–Merton Q jump-size premia

## Risk-free

FRED `DGS3MO`, joined as decimal `r`.

## Rebuild

```bash
/opt/anaconda3/bin/python code/scripts/data_fetch.py
/opt/anaconda3/bin/python code/scripts/data_prepare.py
/opt/anaconda3/bin/python code/scripts/options_fetch.py
```
