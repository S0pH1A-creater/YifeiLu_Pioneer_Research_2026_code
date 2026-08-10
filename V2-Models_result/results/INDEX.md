# V2 Optimal stopping — index (6-month lookback)

After Merton residual-σ + Heston continuous-θ/v0 + κ/ξ AR(1) fixes. Same contracts/seeds/RMSE harness as V1.

| Regime | Model | none | monthly | daily | best rolling | file |
|--------|-------|-----:|--------:|------:|--------------|------|
| 2008-2009 | GBM | 0.6200 | 0.8139 | 0.6678 | `none` | [2008-2009_gbm.md](2008-2009_gbm.md) |
| 2008-2009 | Merton | 0.6200 | 0.8320 | 0.6560 | `none` | [2008-2009_merton.md](2008-2009_merton.md) |
| 2008-2009 | Heston–Merton | 0.5686 | 0.8097 | 0.6731 | `none` | [2008-2009_heston_merton.md](2008-2009_heston_merton.md) |
| 2008-2009 | GARCH–Merton | 0.6538 | 0.6302 | 0.6990 | `monthly` | [2008-2009_garch_merton.md](2008-2009_garch_merton.md) |
| 2013-2014 | GBM | 0.7724 | 0.4572 | 0.4021 | `daily` | [2013-2014_gbm.md](2013-2014_gbm.md) |
| 2013-2014 | Merton | 0.7667 | 0.4786 | 0.4155 | `daily` | [2013-2014_merton.md](2013-2014_merton.md) |
| 2013-2014 | Heston–Merton | 2.5391 | 0.6535 | 0.5309 | `daily` | [2013-2014_heston_merton.md](2013-2014_heston_merton.md) |
| 2013-2014 | GARCH–Merton | 1.0072 | 0.6103 | 0.5655 | `daily` | [2013-2014_garch_merton.md](2013-2014_garch_merton.md) |
| 2018-2019 | GBM | 0.9143 | 1.6168 | 1.5356 | `none` | [2018-2019_gbm.md](2018-2019_gbm.md) |
| 2018-2019 | Merton | 0.8772 | 1.5545 | 1.4499 | `none` | [2018-2019_merton.md](2018-2019_merton.md) |
| 2018-2019 | Heston–Merton | 0.8801 | 1.3155 | 2.3372 | `none` | [2018-2019_heston_merton.md](2018-2019_heston_merton.md) |
| 2018-2019 | GARCH–Merton | 0.7625 | 3.0739 | 2.0580 | `none` | [2018-2019_garch_merton.md](2018-2019_garch_merton.md) |
| 2019-2020 | GBM | 2.0618 | 1.7747 | 1.7976 | `monthly` | [2019-2020_gbm.md](2019-2020_gbm.md) |
| 2019-2020 | Merton | 2.1733 | 1.8482 | 1.8588 | `monthly` | [2019-2020_merton.md](2019-2020_merton.md) |
| 2019-2020 | Heston–Merton | 1.4705 | 1.0868 | 1.0636 | `daily` | [2019-2020_heston_merton.md](2019-2020_heston_merton.md) |
| 2019-2020 | GARCH–Merton | 7.2019 | 3.2224 | 2.6584 | `daily` | [2019-2020_garch_merton.md](2019-2020_garch_merton.md) |

## Best model by regime (min RMSE across rolling)

- **2008-2009:** Heston–Merton (`none`, RMSE=0.5686)
- **2013-2014:** GBM (`daily`, RMSE=0.4021)
- **2018-2019:** GARCH–Merton (`none`, RMSE=0.7625)
- **2019-2020:** Heston–Merton (`daily`, RMSE=1.0636)

## Comparison notebooks

| Notebook | Rolling |
|----------|---------|
| [compare_rolling_none.ipynb](../compare_rolling_none.ipynb) | none |
| [compare_rolling_monthly.ipynb](../compare_rolling_monthly.ipynb) | monthly |
| [compare_rolling_daily.ipynb](../compare_rolling_daily.ipynb) | daily |

See `../V2_CHANGES.md` and `V1_vs_V2_RMSE.md`.

