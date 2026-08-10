# Optimal stopping — index (6-month lookback)

16 studies × 3 rolling modes (`none` / `monthly` / `daily`). Same SPY American-call sample (n=24, seed=42), n_paths=2000. Lower RMSE better.

| Regime | Model | none | monthly | daily | best rolling | file |
|--------|-------|-----:|--------:|------:|--------------|------|
| 2008-2009 | GBM | 0.6200 | 0.8139 | 0.6678 | `none` | [2008-2009_gbm.md](2008-2009_gbm.md) |
| 2008-2009 | Merton | 0.6200 | 0.8325 | 0.6597 | `none` | [2008-2009_merton.md](2008-2009_merton.md) |
| 2008-2009 | Heston–Merton | 0.9793 | 1.1936 | 1.4352 | `none` | [2008-2009_heston_merton.md](2008-2009_heston_merton.md) |
| 2008-2009 | GARCH–Merton | 0.6538 | 0.6302 | 0.6990 | `monthly` | [2008-2009_garch_merton.md](2008-2009_garch_merton.md) |
| 2013-2014 | GBM | 0.7724 | 0.4572 | 0.4021 | `daily` | [2013-2014_gbm.md](2013-2014_gbm.md) |
| 2013-2014 | Merton | 0.7831 | 0.5367 | 0.4398 | `daily` | [2013-2014_merton.md](2013-2014_merton.md) |
| 2013-2014 | Heston–Merton | 3.8582 | 1.0534 | 0.7332 | `daily` | [2013-2014_heston_merton.md](2013-2014_heston_merton.md) |
| 2013-2014 | GARCH–Merton | 1.0072 | 0.6103 | 0.5655 | `daily` | [2013-2014_garch_merton.md](2013-2014_garch_merton.md) |
| 2018-2019 | GBM | 0.9143 | 1.6168 | 1.5356 | `none` | [2018-2019_gbm.md](2018-2019_gbm.md) |
| 2018-2019 | Merton | 0.8815 | 1.6459 | 1.5070 | `none` | [2018-2019_merton.md](2018-2019_merton.md) |
| 2018-2019 | Heston–Merton | 0.9086 | 1.9075 | 1.7732 | `none` | [2018-2019_heston_merton.md](2018-2019_heston_merton.md) |
| 2018-2019 | GARCH–Merton | 0.7625 | 3.0739 | 2.0580 | `none` | [2018-2019_garch_merton.md](2018-2019_garch_merton.md) |
| 2019-2020 | GBM | 2.0618 | 1.7747 | 1.7976 | `monthly` | [2019-2020_gbm.md](2019-2020_gbm.md) |
| 2019-2020 | Merton | 2.1481 | 1.9236 | 1.8986 | `daily` | [2019-2020_merton.md](2019-2020_merton.md) |
| 2019-2020 | Heston–Merton | 2.5245 | 1.9876 | 2.5020 | `monthly` | [2019-2020_heston_merton.md](2019-2020_heston_merton.md) |
| 2019-2020 | GARCH–Merton | 7.2019 | 3.2224 | 2.6584 | `daily` | [2019-2020_garch_merton.md](2019-2020_garch_merton.md) |

## Best model by regime (min RMSE across rolling modes)

- **2008-2009:** GBM (`none`, RMSE=0.6200)
- **2013-2014:** GBM (`daily`, RMSE=0.4021)
- **2018-2019:** GARCH–Merton (`none`, RMSE=0.7625)
- **2019-2020:** GBM (`monthly`, RMSE=1.7747)

Figures live under `figures/<study>/` (`{none,monthly,daily}_{panel,path}.png`).

## Comparison notebooks (one rolling mode each)

Copied graphs + RMSE only (no re-run). Each notebook has all **16** studies for that rolling mode:

| Notebook | Rolling mode | Contents |
|----------|--------------|----------|
| [compare_rolling_none.ipynb](compare_rolling_none.ipynb) | `none` | 16× (RMSE + 3-panel graph), labeled by model & regime |
| [compare_rolling_monthly.ipynb](compare_rolling_monthly.ipynb) | `monthly` | same layout |
| [compare_rolling_daily.ipynb](compare_rolling_daily.ipynb) | `daily` | same layout |

