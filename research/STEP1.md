# Research Step 1 — Status (Data Acquisition & Preparation)

**Status: complete for equity + rates + core options (SPY / AAPL / MSFT)**  
Auxiliary JPM / XOM kept in equity; options not required.

Checklist from [`../docs/DataCollection.md`](../docs/DataCollection.md) / [`../docs/ResearchProposal-v2.md`](../docs/ResearchProposal-v2.md):

| # | Item | Status |
|---|------|--------|
| 1 | Stock adj closes: SPY, AAPL, MSFT (+ JPM, XOM) | ✓ |
| 1b | Log returns + regime stats (all tickers) | ✓ |
| 2 | American options: SPY (calls+puts; call panels) | ✓ |
| 2b | American options: AAPL, MSFT | ✓ |
| 2c | American options: JPM, XOM | skipped (auxiliary equity only) |
| 3 | U.S. Treasury risk-free (DGS3MO) | ✓ |

## Regimes

| Regime | Equity window | Options window |
|--------|---------------|----------------|
| Crisis | 2007-01-01 → 2009-12-31 | 2008-01-01 → 2009-12-31 |
| Normal | 2013-01-01 → 2014-12-31 | same |
| High vol | 2017-01-01 → 2018-12-31 | same |

## Tickers

| Role | Ticker | Equity + returns | Options |
|------|--------|------------------|---------|
| Primary | SPY | ✓ (SSGA NAV; common sample from 2003-12) | ✓ |
| Secondary | AAPL | ✓ | ✓ (Cobweb ToS EOD) |
| Secondary | MSFT | ✓ | ✓ (Cobweb ToS EOD) |
| Auxiliary | JPM | ✓ | — |
| Auxiliary | XOM | ✓ | — |

## Scripts

| Script | Role |
|--------|------|
| [`scripts/data_fetch.py`](scripts/data_fetch.py) | Download/clean equity → `data/equity/prices_clean.csv` |
| [`scripts/data_prepare.py`](scripts/data_prepare.py) | Log returns, regimes, stats, figures |
| [`scripts/options_fetch.py`](scripts/options_fetch.py) | Options + risk-free; join \(S_t\), \(r\) |
| [`scripts/cobweb_to_parquet.py`](scripts/cobweb_to_parquet.py) | Cobweb ZIP → regime parquet (`AAPL` / `MSFT`) |

```bash
cd research
../.venv/bin/python scripts/data_fetch.py
../.venv/bin/python scripts/data_prepare.py
../.venv/bin/python scripts/options_fetch.py
```

## Deliverables (organized)

```
scripts/       data_fetch, data_prepare, options_fetch, cobweb_to_parquet
data/equity/   prices_clean.csv, log_returns_*.csv, summary_stats.csv
data/rates/    risk_free_dgs3mo.csv
data/options/  raw/ + processed/ (SPY, AAPL, MSFT panels + combined core)
figures/       01–05 exploratory plots
notebooks/     Step 2 interactive model playgrounds
```

Details: [`data/DATA_README.md`](data/DATA_README.md)

## Next (Step 2)

Interactive Monte Carlo playgrounds (no market data) — one notebook per model with parameter sliders. See [`STEP2.md`](STEP2.md).
