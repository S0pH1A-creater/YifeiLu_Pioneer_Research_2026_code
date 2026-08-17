# Intraday equity bars

- Source: FirstRate free 1-minute sample (Yahoo 7d/60d/2y blocked). 7d/60d/1h derived from that 1-minute file; 1h is ~1 year, not 2 years.
- Tickers: SPY (primary), AAPL, MSFT (same names as the daily notebooks)
- **Used now:** `equity/short_interval/prices_1min_rth/` — full-year 1-minute RTH bars used by `7d_1min_*.ipynb` and `1d_1min_*.ipynb`. Dated folders under `7d_1min/<YYYY-MM-DD_to_YYYY-MM-DD>/` are a legacy 7-session cut (current evaluation window: **2023-03-09 → 2023-03-15**). The earlier last-8-sessions cut is archived as `7d_1min/2023-09-21_to_2023-09-29/`.
- **Stored, not used yet:** `60d_2min`, `60d_5min`, `60d_15min`, `60d_30min`, `1h` (and `source_1min` when FirstRate is the source)
- See `metadata.json` for bar counts and timestamps.
