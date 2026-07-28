# Research data layout (Step 1)

Aligned with [`../DataCollection.md`](../../DataCollection.md) and [`../ResearchProposal-v2.md`](../../ResearchProposal-v2.md).

**Focus:** SPY (primary) · AAPL + MSFT (secondary) · JPM + XOM (auxiliary equity only).

```
data/
  equity/                         # §1 Stock market data
    prices_clean.csv              # Date + adj close: SPY, AAPL, MSFT, JPM, XOM
    log_returns_all.csv           # Daily log returns (full sample)
    log_returns_by_regime.csv     # Long format: ticker × regime × date
    summary_stats.csv             # Moments by ticker × regime
  rates/                          # §3 Risk-free rate
    risk_free_dgs3mo.csv          # FRED DGS3MO (%), raw CSV
  options/                        # §2 American options
    raw/                          # Large source dumps (gitignored)
      SPY_options.parquet         # Primary (~600 MB)
      AAPL_options.parquet        # Cobweb ToS EOD → parquet
      MSFT_options.parquet        # Cobweb ToS EOD → parquet
    processed/
      SPY_options_panel.csv       # Primary research panel (calls+puts)
      SPY_calls_panel.csv         # Calls only (optimal stopping)
      AAPL_options_panel.csv      # Secondary
      AAPL_calls_panel.csv
      MSFT_options_panel.csv      # Secondary
      MSFT_calls_panel.csv
      options_panel_{all,crisis,normal,high_vol}.csv   # core = SPY+AAPL+MSFT
      calls_panel_{all,crisis,normal,high_vol}.csv
      options_summary.csv
```

## 1. Equity (adj close → log returns)

| Ticker | Role | Status | Notes |
|--------|------|--------|-------|
| SPY | Primary | ✓ | SSGA NAV (common sample start Dec 2003) |
| AAPL | Secondary | ✓ | GitHub YF mirror adj close |
| MSFT | Secondary | ✓ | GitHub YF mirror adj close |
| JPM | Auxiliary | ✓ | Retained; not in options focus |
| XOM | Auxiliary | ✓ | Retained; not in options focus |

- **Requested window:** post-2000 → end of high-vol regime (2000-01-01 → 2018-12-31)
- **Common sample on disk:** 2003-12-01 → 2018-12-31 (SPY NAV history starts Dec 2003)
- **Transform:** \(r_t = \ln(S_t / S_{t-1})\) for every ticker
- **Regimes:** crisis 2007–2009 · normal 2013–2014 · high_vol 2017–2018

## 2. American options

| Ticker | Role | Status | Notes |
|--------|------|--------|-------|
| SPY | Primary | ✓ | Open release; calls + puts; call-only panels also written |
| AAPL | Secondary | ✓ | Cobweb Scripts ToS EOD (Mega) → `AAPL_options.parquet` |
| MSFT | Secondary | ✓ | Cobweb Scripts ToS EOD (Mega) → `MSFT_options.parquet` |
| JPM / XOM | Auxiliary | equity only | No Cobweb dump; CDN offline — optional later via Alpha Vantage |

### Processed panel columns
| Column | Meaning |
|--------|---------|
| `underlying` | Ticker (SPY primary; AAPL/MSFT secondary) |
| `trading_date` | Quote date |
| `S_t` | Underlying price (broker last when available, else adj close) |
| `K` | Strike |
| `expiration` | Maturity |
| `T_years` / `dte` | Time to maturity |
| `option_type` | `call` or `put` |
| `option_price` | Premium (mid/mark) |
| `r` | Risk-free (decimal, DGS3MO) |
| `moneyness` | K / S_t |
| `regime` | crisis / normal / high_vol |
| `style` | American |

### Filters
- Dates: 2008-01-01 → 2018-12-31 (open options history starts 2008)
- Near ATM: \|K/S − 1\| ≤ 10%
- DTE: 7–60 days
- Volume ≥ 1; valid bid/ask; premium ≥ max(0.05, 0.5×intrinsic)
- Every 5th trading date (size control)

## 3. Risk-free rate

- FRED `DGS3MO` (3-Month Treasury), percent in CSV; joined as decimal `r` in option panels

## Sources

- Equity SPY: State Street NAV history
- Equity AAPL/MSFT/JPM/XOM: [dieperdev/yfinance-stock-data](https://github.com/dieperdev/yfinance-stock-data) (Yahoo adj close mirror; Unlicense)
- Options SPY: [lambdaclass data-v1](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1) (MIT; philippdubach/options-data)
- Options AAPL / MSFT: [Cobweb Scripts ToS EOD](https://cobwebscripts.com/data/toseodoptiondata.html) (free; converted via `cobweb_to_parquet.py`)
- Risk-free: FRED `DGS3MO`

## Run

```bash
cd research
../.venv/bin/python data_fetch.py
../.venv/bin/python data_prepare.py
../.venv/bin/python options_fetch.py
# optional: convert a Cobweb ZIP already in data/options/raw/_staging/
../.venv/bin/python cobweb_to_parquet.py MSFT
```
