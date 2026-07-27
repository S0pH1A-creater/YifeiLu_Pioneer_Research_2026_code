# Research Step 1 — Status (Data Acquisition & Preparation)

**Status: complete** (equity primary + SPY options primary; JPM/XOM equity & equity options optional)

## Scope

Build the data foundation for Merton/Kou jump-diffusion research:

1. Equity prices and log returns across three volatility regimes  
2. American option panels (SPY primary) with \(S_t\), \(K\), maturity, type, premium, trade date, \(r\)  
3. Risk-free rate series and exploratory figures  

## Regimes

| Regime | Equity window | Options window |
|--------|---------------|----------------|
| Crisis | 2007-01-01 → 2009-12-31 | 2008-01-01 → 2009-12-31 (open options history starts 2008) |
| Normal | 2013-01-01 → 2014-12-31 | same |
| High vol | 2017-01-01 → 2018-12-31 | same |

## Tickers

| Role | Ticker | Equity prices | Options |
|------|--------|---------------|---------|
| Primary | SPY | yes (SSGA NAV) | yes (~112k filtered quotes) |
| Secondary | AAPL | yes (Twelve Data) | pending (drop parquet in `data/options/raw/`) |
| Secondary | JPM | pending (`TWELVEDATA_API_KEY` or Yahoo) | pending |
| Secondary | XOM | pending | pending |

## Scripts

| Script | Role |
|--------|------|
| [`data_fetch.py`](data_fetch.py) | Download/clean equity prices → `prices_clean.csv` |
| [`data_prepare.py`](data_prepare.py) | Log returns, regimes, stats, figures |
| [`options_fetch.py`](options_fetch.py) | SPY options download/filter + join \(S_t\), \(r\) |

```bash
cd research
../.venv/bin/python data_fetch.py
../.venv/bin/python data_prepare.py
../.venv/bin/python options_fetch.py
```

## Deliverables

### Equity
- `data/prices_clean.csv` — 3,023 days (2007-01-03 → 2018-12-31), SPY + AAPL  
- `data/log_returns_all.csv`, `log_returns_by_regime.csv`, `summary_stats.csv`  
- `figures/01–04_*.png` — price/regime, return distributions, vol comparison, rolling vol  

### Rates
- `data/risk_free_dgs3mo.csv` — FRED 3M Treasury  

### Options (American)
- `data/options/raw/SPY_options.parquet` — full source dump (~600 MB)  
- `data/options/processed/SPY_options_panel.csv` — **primary** research panel  
- `options_panel_{all,crisis,normal,high_vol}.csv`, `options_summary.csv`  

Panel fields: `underlying`, `trading_date`, `S_t`, `K`, `expiration`, `option_type`, `option_price`, `r`, plus `T_years`, `dte`, `moneyness`, `regime`, `style=American`.

### SPY options summary (after quality filters)

| Regime | Quotes | Dates | Calls | Puts |
|--------|--------|-------|-------|------|
| Crisis | 1,667 | 59 | 798 | 869 |
| Normal | 33,073 | 101 | 15,046 | 18,027 |
| High vol | 76,847 | 101 | 35,074 | 41,773 |

Filters: ATM ±10%, DTE 7–60, volume ≥ 1, valid bid/ask, premium ≥ max(0.05, 0.5×intrinsic), every 5th trading date.

## SPY equity sanity (from `summary_stats.csv`)

| Regime | n | Ann. vol | Excess kurtosis |
|--------|---|----------|-----------------|
| Crisis | 755 | 29.9% | 6.13 |
| Normal | 505 | 11.2% | 1.33 |
| High vol | 502 | 13.0% | 6.34 |

## Sources

- Equity SPY: State Street NAV history  
- Equity AAPL: Twelve Data (demo); JPM/XOM via env key or Yahoo when available  
- Options SPY: [lambdaclass data-v1](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1) (MIT)  
- Risk-free: FRED `DGS3MO`  

Details: [`data/DATA_README.md`](data/DATA_README.md)

## Next (Step 2)

Fit **GBM baseline** on prepared log returns (SPY primary), by regime; optionally use option panels later for pricing checks against Merton/Kou.
