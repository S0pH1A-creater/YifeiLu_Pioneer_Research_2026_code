# Research data layout (Step 1)

Aligned with [`../../docs/DataCollection.md`](../../docs/DataCollection.md) and [`../../docs/ResearchProposal-v2.md`](../../docs/ResearchProposal-v2.md).

**Focus:** SPY (primary) · AAPL + MSFT (secondary) · JPM + XOM (auxiliary equity only).

```
data/
  equity/                         # §1 Stock market data
    prices_clean.csv              # Date + adj close: SPY, AAPL, MSFT, JPM, XOM
    log_returns_all.csv           # Daily log returns (full sample)
    log_returns_by_regime.csv     # Long format: ticker × regime × date
    summary_stats.csv             # Moments by ticker × regime
    short_interval/               # 2022-09-30 → 2023-09-29 (does not overwrite 2008–2020)
      prices_daily.csv
      prices_1min_rth/{SPY,AAPL,MSFT}.csv   # USED by 7d_1min_* and 1d_1min_* notebooks
      log_returns_1min_rth/
      log_returns_daily.csv
      metadata.json
    intraday/                     # FirstRate-derived windows (see README there)
      7d_1min/                    # dated 7-session slices (legacy cut; notebooks now use short_interval)
      60d_2min/ 60d_5min/         # stored, not used yet
      60d_15min/ 60d_30min/
      1h/                         # 1-hour bars (free sample is ~1y, not 2y)
      source_1min/                # full FirstRate 1-min sample
      metadata.json
  rates/                          # §3 Risk-free rate
    risk_free_dgs3mo.csv          # FRED DGS3MO (%), 2008–2020
    risk_free_dgs3mo_short_interval.csv  # FRED DGS3MO, 2022-03 → 2023-09
  options/                        # §2 American options
    raw/                          # Large source dumps (gitignored)
      SPY_options.parquet         # Primary (~600 MB)
      AAPL_options.parquet        # Cobweb ToS EOD → parquet
      MSFT_options.parquet        # Cobweb ToS EOD → parquet
    processed/
      SPY_options_panel.csv       # Primary research panel (calls+puts), 2008–2020
      SPY_calls_panel.csv         # Calls only (optimal stopping)
      AAPL_options_panel.csv      # Secondary
      AAPL_calls_panel.csv
      MSFT_options_panel.csv      # Secondary
      MSFT_calls_panel.csv
      options_panel_{all,crisis,normal,late,covid}.csv   # core = SPY+AAPL+MSFT
      calls_panel_{all,crisis,normal,late,covid}.csv
      options_summary.csv
      short_interval/             # 2022-09-30 → 2023-09-29 listed quotes (same filters)
```

## 1. Equity (adj close → log returns)

| Ticker | Role | Status | Notes |
|--------|------|--------|-------|
| SPY | Primary | ✓ | SSGA NAV (common sample start Dec 2003) |
| AAPL | Secondary | ✓ | GitHub YF mirror adj close |
| MSFT | Secondary | ✓ | GitHub YF mirror adj close |
| JPM | Auxiliary | ✓ | Retained; not in options focus |
| XOM | Auxiliary | ✓ | Retained; not in options focus |

- **Requested window:** post-2000 → end of late regime sample (2000-01-01 → 2020-12-31)
- **Common sample on disk:** 2003-12-01 → 2020-12-31 (SPY NAV history starts Dec 2003)
- **Transform:** \(r_t = \ln(S_t / S_{t-1})\) for every ticker
- **Regimes:** crisis 2008–2009 · normal 2013–2014 · late 2018–2019 · covid 2019–2020
- **Intraday (free):** `equity/short_interval/prices_1min_rth/` — SPY/AAPL/MSFT, 2022-09-30 → 2023-09-29. Notebooks `7d_1min_*.ipynb` and `1d_1min_*.ipynb` slice this series (7 sessions or 1 session), 1-hour/1-day lookback, hourly/minutely rolling. Source is FirstRate’s free 1-minute sample. Dated cuts under `equity/intraday/7d_1min/` are kept as a legacy slice. See `equity/intraday/README.md`.

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
| `regime` | crisis / normal / late / covid |
| `style` | American |

### Filters
- Dates: 2008-01-01 → 2020-12-31 (open options history starts 2008)
- Near ATM: \|K/S − 1\| ≤ 10%
- DTE: 7–60 days
- Volume ≥ 1; valid bid/ask; premium ≥ max(0.05, 0.5×intrinsic)
- Every 5th trading date (size control)
- **Short interval (2022-09-30 → 2023-09-29):** same ATM/DTE/volume filters, **no** every-5th-day subsample, written to `options/processed/short_interval/` (does not overwrite the 2008–2020 panels). Used by `7d_1min_*.ipynb` and `1d_1min_*.ipynb`.

## 3. Risk-free rate

- FRED `DGS3MO` (3-Month Treasury), percent in CSV; joined as decimal `r` in option panels

## Sources

- Equity SPY: State Street NAV history
- Equity AAPL/MSFT/JPM/XOM: [dieperdev/yfinance-stock-data](https://github.com/dieperdev/yfinance-stock-data) (Yahoo adj close mirror; Unlicense)
- Intraday SPY/AAPL/MSFT: Yahoo 7d/60d/2y when available; otherwise [FirstRate free 1-minute sample](https://firstratedata.com/free-intraday-data) (2022-09-30 → 2023-09-29), resampled into `equity/intraday/`
- Options SPY: [lambdaclass data-v1](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1) (MIT; philippdubach/options-data)
- Options AAPL / MSFT: [Cobweb Scripts ToS EOD](https://cobwebscripts.com/data/toseodoptiondata.html) (free; converted via `scripts/cobweb_to_parquet.py`)
- Risk-free: FRED `DGS3MO`

## Run

```bash
cd research
../.venv/bin/python scripts/data_fetch.py
../.venv/bin/python scripts/data_prepare.py
../.venv/bin/python scripts/options_fetch.py
# optional: convert a Cobweb ZIP already in data/options/raw/_staging/
../.venv/bin/python scripts/cobweb_to_parquet.py MSFT
```
