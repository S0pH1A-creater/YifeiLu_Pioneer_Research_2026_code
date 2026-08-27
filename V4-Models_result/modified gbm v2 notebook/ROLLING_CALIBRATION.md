# Modified GBM v2 notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Model

1. **Direction** — two-state Markov chain \(P(U\mid U),\ P(D\mid U),\ P(U\mid D),\ P(D\mid D)\).
2. **Calm / wild** — two-state Markov chain on size. A lookback bar is wild if \(|R|\) exceeds the window median of \(|R|\).
3. **Magnitude** — \(\mathrm{size}=\exp(N(\mu_{d,s},\sigma_{d,s}^2))\) (Way B, always positive). Four pairs: up/down × calm/wild.
4. **Price** — \(S_{t+1}=S_t e^{R_t}\) with \(R_t=+\mathrm{size}\) (up) or \(-\mathrm{size}\) (down).
5. **Q** — size-only: \(R^Q=\lambda R^P\) with \(\lambda\) chosen so \(E[e^{R^Q}]=e^{r_f\Delta t}\). Signs stay as drawn under \(P\).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** v2 parameters using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Files

- `2008-2009_modified_gbm_v2.ipynb`
- `2013-2014_modified_gbm_v2.ipynb`
- `2018-2019_modified_gbm_v2.ipynb`
- `2019-2020_modified_gbm_v2.ipynb`
- `7d_1min_modified_gbm_v2.ipynb`
- `1d_1min_modified_gbm_v2.ipynb`
