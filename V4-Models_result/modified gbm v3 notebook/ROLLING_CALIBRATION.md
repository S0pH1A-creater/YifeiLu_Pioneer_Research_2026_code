# Modified GBM v3 notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Model

1. **Direction** — order-3 Markov chain on the last three signs. Eight states: UUU, UUD, UDU, UDD, DUU, DUD, DDU, DDD. The next bar is U with \(P(U\mid s)\) and D with \(1-P(U\mid s)\). The state then slides one step.
2. **Magnitude** — positive size from \(N(\mu_U,\sigma_U^2)\) or \(N(\mu_D,\sigma_D^2)\), taken absolute. Same as original Modified GBM.
3. **Price** — \(S_{t+1}=S_t e^{R_t}\) with \(R_t=+m_t\) (up) or \(R_t=-m_t\) (down).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** v3 parameters using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Files

- `2008-2009_modified_gbm_v3.ipynb`
- `2013-2014_modified_gbm_v3.ipynb`
- `2018-2019_modified_gbm_v3.ipynb`
- `2019-2020_modified_gbm_v3.ipynb`
- `7d_1min_modified_gbm_v3.ipynb`
- `1d_1min_modified_gbm_v3.ipynb`
