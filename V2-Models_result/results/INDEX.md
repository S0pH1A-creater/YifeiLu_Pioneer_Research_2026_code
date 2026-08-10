# V2 Optimal stopping — index (6-month lookback)

Underlyings: **SPY**, **AAPL**, **MSFT**. Same LSM harness, contracts/seeds per ticker (n=24, seed=42), n_paths=2000.

## Per-ticker indexes

- [SPY](SPY/INDEX.md)
- [AAPL](AAPL/INDEX.md)
- [MSFT](MSFT/INDEX.md)

## Comparison notebooks

| Notebook | Rolling |
|----------|---------|
| [compare_rolling_none.ipynb](../compare_rolling_none.ipynb) | none |
| [compare_rolling_monthly.ipynb](../compare_rolling_monthly.ipynb) | monthly |
| [compare_rolling_daily.ipynb](../compare_rolling_daily.ipynb) | daily |

Each compare notebook has three identical sections (SPY → AAPL → MSFT): RMSE table + 16 study blocks.

See `../V2_CHANGES.md` and `V1_vs_V2_RMSE.md`.

## Quick SPY RMSE (daily)

| Regime | Model | daily RMSE |
|--------|-------|-----------:|
| 2008-2009 | GBM | 0.6678 |
| 2008-2009 | Merton | 0.6560 |
| 2008-2009 | Heston–Merton | 0.6731 |
| 2008-2009 | GARCH–Merton | 0.6990 |
| 2013-2014 | GBM | 0.4021 |
| 2013-2014 | Merton | 0.4155 |
| 2013-2014 | Heston–Merton | 0.5309 |
| 2013-2014 | GARCH–Merton | 0.5655 |
| 2018-2019 | GBM | 1.5356 |
| 2018-2019 | Merton | 1.4499 |
| 2018-2019 | Heston–Merton | 2.3372 |
| 2018-2019 | GARCH–Merton | 2.0580 |
| 2019-2020 | GBM | 1.7976 |
| 2019-2020 | Merton | 1.8588 |
| 2019-2020 | Heston–Merton | 1.0636 |
| 2019-2020 | GARCH–Merton | 2.6584 |

