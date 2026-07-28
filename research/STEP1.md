# Research Step 1 — Status (Data Acquisition & Preparation)

**Status: complete for equity + rates + core options (SPY / AAPL / MSFT)**  
Auxiliary JPM / XOM kept in equity; options not required.

Checklist from [`../DataCollection.md`](../DataCollection.md) / [`../ResearchProposal-v2.md`](../ResearchProposal-v2.md):

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
| [`data_fetch.py`](data_fetch.py) | Download/clean equity → `data/equity/prices_clean.csv` |
| [`data_prepare.py`](data_prepare.py) | Log returns, regimes, stats, figures |
| [`options_fetch.py`](options_fetch.py) | Options + risk-free; join \(S_t\), \(r\) |
| [`cobweb_to_parquet.py`](cobweb_to_parquet.py) | Cobweb ZIP → regime parquet (`AAPL` / `MSFT`) |

```bash
cd research
../.venv/bin/python data_fetch.py
../.venv/bin/python data_prepare.py
../.venv/bin/python options_fetch.py
```

## Deliverables (organized)

```
data/equity/   prices_clean.csv, log_returns_*.csv, summary_stats.csv
data/rates/    risk_free_dgs3mo.csv
data/options/  raw/ + processed/ (SPY, AAPL, MSFT panels + combined core)
figures/       01–05 exploratory plots
```

Details: [`data/DATA_README.md`](data/DATA_README.md)

## Next (Step 2)

Fit **GBM baseline** on prepared log returns (SPY primary), by regime; use American call panels for pricing / optimal stopping against Merton / Heston-Merton / GARCH-Merton. Secondary checks on AAPL / MSFT.
