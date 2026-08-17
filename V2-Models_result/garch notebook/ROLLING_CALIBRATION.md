# GARCH notebooks — true rolling-window calibration

GARCH-only is **GARCH–Merton without the jump block**: same GARCH(1,1) MLE, rolling schedule, Monte Carlo, and LSM workflow; no \(\lambda,\mu_J,\sigma_J,\kappa\).

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period when rolling is `daily` or `monthly`.

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** GARCH parameters  
   \((\hat\mu, \hat\omega, \hat\alpha, \hat\beta, \hat\sigma_0)\)  
   using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window

From log returns \(r_t=\ln(S_t/S_{t-1})\):

| Step | What |
|------|------|
| 1. GARCH(1,1) MLE | Fit \(r_t=\mu_{\mathrm{day}}+\varepsilon_t\), \(\varepsilon_t=\sigma_t Z_t\), \(\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2\) by maximizing the Gaussian likelihood of returns given the implied \(\sigma_t\) path (\(\alpha+\beta<1\)). |
| 2. Drift | \(\hat\mu=\hat\mu_{\mathrm{day}}\times N_{\mathrm{days}}\) |
| 3. \(\sigma_0\) | last conditional \(\sigma\) from the fit (else \(\sqrt{\omega/(1-\alpha-\beta)}\)) |

No jump-day filter and no jump parameters.

§4 graphs show five series: \(\mu,\omega,\alpha,\beta,\sigma_0\).

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

**Important distinction**

- **Calibrated parameters** \((\mu,\omega,\alpha,\beta)\): fixed under `none`; refreshed under rolling.
- **Conditional volatility \(\sigma_t\)**: still evolves every step inside the Monte Carlo via the GARCH recursion, even when the calibrated law is fixed.

## Simulation step

\[
\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2,\quad
\varepsilon_t=\sigma_t Z_t,\quad
r_t=\mu\Delta t+\varepsilon_t,\quad
S\leftarrow S\exp(r_t)
\]

## Files

- `2008-2009_garch.ipynb`
- `2013-2014_garch.ipynb`
- `2018-2019_garch.ipynb`
- `2019-2020_garch.ipynb`
- `7d_1min_garch.ipynb`
- `1d_1min_garch.ipynb`

## Duplicate graphs / reopen

Same rules as the GBM / Merton notebooks: show figures once via PNG bytes + `plt.close(fig)`; after reopen use **Restart Kernel** then **Run All**.
