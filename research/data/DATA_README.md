# Research data layout (Step 1)

Step status: see [`../STEP1.md`](../STEP1.md).

## Equity prices
| File | Description |
|------|-------------|
| `prices_clean.csv` | Daily closes: SPY (primary), AAPL, (+ JPM/XOM when available) |
| `log_returns_*.csv` | Log returns and regime splits |
| `summary_stats.csv` | Moments by ticker × regime |
| `risk_free_dgs3mo.csv` | FRED 3-Month Treasury yield (%) |

## Options (American-style equity / ETF options)

```
data/options/
  raw/                         # full source dumps (large; gitignored ideally)
    SPY_options.parquet        # ~600 MB open release (2008–2025)
    AAPL_options.parquet       # optional drop-in
    JPM_options.parquet
    XOM_options.parquet
  processed/                   # research panels (filtered, joined)
    SPY_options_panel.csv      # PRIMARY
    AAPL_options_panel.csv
    JPM_options_panel.csv
    XOM_options_panel.csv
    options_panel_all.csv
    options_panel_crisis.csv
    options_panel_normal.csv
    options_panel_high_vol.csv
    options_summary.csv
```

### Processed panel columns
| Column | Meaning |
|--------|---------|
| `underlying` | Ticker (SPY primary) |
| `trading_date` | Quote / trade date |
| `S_t` | Underlying price (from `prices_clean.csv`) |
| `K` | Strike |
| `expiration` | Maturity date |
| `T_years` | Time to maturity in years |
| `dte` | Calendar days to expiration |
| `option_type` | `call` or `put` |
| `option_price` | Premium (mid/mark) |
| `r` | Risk-free rate (decimal, from DGS3MO) |
| `moneyness` | K / S_t |
| `regime` | crisis / normal / high_vol |
| `style` | American |

### Filters applied
- Dates: 2008-01-01 → 2018-12-31 (open options history starts 2008)
- Near ATM: \|K/S − 1\| ≤ 10%
- DTE: 7–60 days
- Volume ≥ 1
- Every 5th trading date (size control)

### Sources
- **SPY options:** [lambdaclass release data-v1](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1) (MIT; philippdubach/options-data)
- **Risk-free rate:** FRED `DGS3MO`
- **Underlying prices:** existing Step 1 equity pipeline
- **AAPL / JPM / XOM options:** place matching parquet/CSV in `raw/` and re-run `options_fetch.py` (Yahoo does not provide deep historical chains)

### Run
```bash
cd research
../.venv/bin/python options_fetch.py
```
