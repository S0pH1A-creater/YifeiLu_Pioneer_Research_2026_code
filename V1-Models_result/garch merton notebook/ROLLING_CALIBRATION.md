# GARCH–Merton notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period when rolling is `daily` or `monthly`.

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** GARCH–Merton parameters  
   \((\hat\mu, \hat\omega, \hat\alpha, \hat\beta, \hat\sigma_0, \hat\lambda, \hat\mu_J, \hat\sigma_J, \hat\kappa)\)  
   using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window

From daily log returns \(r_t=\ln(S_t/S_{t-1})\):

| Step | What |
|------|------|
| 1. GARCH(1,1) MLE | Fit \(r_t=\mu_{\mathrm{day}}+\varepsilon_t\), \(\varepsilon_t=\sigma_t Z_t\), \(\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2\) by maximizing the Gaussian likelihood of returns given the implied \(\sigma_t\) path (\(\alpha+\beta<1\)). |
| 2. Drift | \(\hat\mu=\hat\mu_{\mathrm{day}}\times 252\) |
| 3. \(\sigma_0\) | last conditional \(\sigma\) from the fit (else \(\sqrt{\omega/(1-\alpha-\beta)}\)) |
| 4. Jump days | \(\lvert\varepsilon_t/\sigma_t\rvert > 3\) (standardized residual) |
| 5. Jump params | \(\hat\lambda=n_{\mathrm{jumps}}/Y\); \(\hat\mu_J,\hat\sigma_J\) from jump-day returns; \(\kappa=e^{\hat\mu_J+\hat\sigma_J^2/2}-1\) |

§4 graphs show six series: \(\mu,\omega,\alpha,\beta,\lambda,\kappa\). \(\sigma_0,\mu_J,\sigma_J\) are estimated in the same windows and used in simulation.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

**Important distinction**

- **Calibrated parameters** \((\mu,\omega,\alpha,\beta,\lambda,\mu_J,\sigma_J,\kappa)\): fixed under `none`; refreshed under rolling.
- **Conditional volatility \(\sigma_t\)**: still evolves every day inside the Monte Carlo via the GARCH recursion, even when the calibrated law is fixed.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 5 years (calendar lookback; uses available history if the series is shorter). Minimum window length for a GARCH fit: 60 trading days.

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, rolling parameter charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; GARCH–Merton MC paths and expected path vs historical prices (one pair per ticker).

## Simulation step

\[
\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2,\quad
\varepsilon_t=\sigma_t Z_t,\quad
r_t=(\mu-\lambda\kappa)\Delta t+\varepsilon_t+\sum_{i=1}^{N_{\Delta t}}J_i,\quad
S\leftarrow S\exp(r_t)
\]

## Files

- `2008-2009_garch_merton.ipynb`
- `2013-2014_garch_merton.ipynb`
- `2018-2019_garch_merton.ipynb`
- `2019-2020_garch_merton.ipynb`

## Duplicate graphs / reopen

Same rules as the GBM / Merton notebooks: show figures once via PNG bytes + `plt.close(fig)`; after reopen use **Restart Kernel** then **Run All**.
