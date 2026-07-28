# Research Step 1 — Status (Data Acquisition & Preparation)

**Status: complete for equity + rates + SPY options**  
Secondary equity options (AAPL/JPM/XOM) pending — open CDN offline.

Checklist from [`../DataCollection.md`](../DataCollection.md) / [`../ResearchProposal-v2.md`](../ResearchProposal-v2.md):

| # | Item | Status |
|---|------|--------|
| 1 | Stock adj closes: SPY, AAPL, JPM, XOM | ✓ |
| 1b | Log returns + regime stats (all tickers) | ✓ |
| 2 | American options: SPY (calls+puts; call panels) | ✓ |
| 2b | American options: AAPL, JPM, XOM | pending (CDN 404) |
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
| Primary | SPY | ✓ (SSGA NAV; common sample from 2003-12) | ✓ (~112k filtered quotes) |
| Secondary | AAPL | ✓ | pending |
| Secondary | JPM | ✓ | pending |
| Secondary | XOM | ✓ | pending |

## Scripts

| Script | Role |
|--------|------|
| [`data_fetch.py`](data_fetch.py) | Download/clean equity → `data/equity/prices_clean.csv` |
| [`data_prepare.py`](data_prepare.py) | Log returns, regimes, stats, figures |
| [`options_fetch.py`](options_fetch.py) | Options + risk-free; join \(S_t\), \(r\) |

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
data/options/  raw/ + processed/ (panels, call panels, summary)
figures/       01–05 exploratory plots
```

### Equity sanity (SPY from `summary_stats.csv`)

| Regime | n | Ann. vol | Excess kurtosis |
|--------|---|----------|-----------------|
| Crisis | 756 | 29.8% | 6.14 |
| Normal | 505 | 11.2% | 1.33 |
| High vol | 502 | 13.0% | 6.34 |

### SPY options (after filters)

| Regime | Quotes | Calls | Puts |
|--------|--------|-------|------|
| Crisis | 1,667 | 798 | 869 |
| Normal | 33,073 | 15,046 | 18,027 |
| High vol | 76,847 | 35,074 | 41,773 |

Details: [`data/DATA_README.md`](data/DATA_README.md)

## Next (Step 2)

Fit **GBM baseline** on prepared log returns (SPY primary), by regime; use American call panels for pricing / optimal stopping against Merton / Heston-Merton / GARCH-Merton.
