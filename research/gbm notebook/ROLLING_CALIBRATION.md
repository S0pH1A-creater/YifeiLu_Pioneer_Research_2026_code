# GBM notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** GBM parameters \((\hat\mu, \hat\sigma)\) using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Lookback choices

3 months · 6 months · 1 year · 2 years · 5 years (calendar lookback; uses available history if the series is shorter).

## Files

- `2008-2009_gbm.ipynb`
- `2013-2014_gbm.ipynb`
- `2018-2019_gbm.ipynb`
