# Heston–Merton notebooks — true rolling-window calibration (Method A)

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period when rolling is `daily` or `monthly`.

**Estimation method in this folder:** **Method A only** — historical moment / realized-variance proxies + Merton jump threshold. **No** Monte Carlo / path-averaged likelihood in calibration (that lives later in a separate advanced folder).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** Heston–Merton parameters  
   \((\hat\mu, \hat\kappa, \hat\theta, \hat\xi, \hat\rho, \hat v_0, \hat\lambda, \hat\mu_J, \hat\sigma_J, \hat\kappa_J)\)  
   using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window (Method A — no random draws)

From daily log returns \(r_t=\ln(S_t/S_{t-1})\), \(RV_t=r_t^2\), \(\Delta t=1/252\):

| Quantity | Estimator |
|----------|-----------|
| Jump days | \(\lvert r_t\rvert > 3\cdot\hat\sigma_{\mathrm{day}}\) |
| \(\hat\mu\) | mean of **non-jump** returns \(\times 252\) |
| \(\hat\theta\) | \(\overline{RV}\times 252\) |
| \(\hat v_0\) | recent \(RV\) (last ≤21 days) \(\times 252\) |
| \(\hat\kappa\) | \(-\ln(\rho_1)/\Delta t\) from lag-1 autocorr of \(RV\) (clipped) |
| \(\hat\xi\) | moment scale of \(\Delta v\) residuals after mean-reversion drift (clipped) |
| \(\hat\rho\) | \(\mathrm{Corr}(r_t,\Delta v_t)\) on annualized variance proxy |
| \(\hat\lambda\) | \(n_{\mathrm{jumps}} / Y\) |
| \(\hat\mu_J,\hat\sigma_J\) | mean / std of jump-day returns |
| \(\kappa_J\) | \(e^{\hat\mu_J + \hat\sigma_J^2/2}-1\) (derived) |

Minimum window length: **60** trading days (else skip update).

§4 graphs show six series: \(\mu,\theta,\kappa,\xi,\rho,\lambda\). \(v_0,\mu_J,\sigma_J,\kappa_J\) are estimated in the same windows and used in simulation.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

**Important distinction**

- **Calibrated parameters** \((\mu,\kappa,\theta,\xi,\rho,\lambda,\mu_J,\sigma_J,\kappa_J)\): fixed under `none`; refreshed under rolling.
- **Variance state \(v_t\)**: still evolves every step inside the Monte Carlo via the Heston SDE, even when the calibrated law is fixed. \(\hat v_0\) initializes each path only.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 5 years (calendar lookback; uses available history if the series is shorter).

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, rolling parameter charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; Heston–Merton MC paths and expected path vs historical prices (one pair per ticker).

## Simulation step

\[
v_{t+\Delta t}=\max\big(0,\ v_t+\kappa(\theta-v_t)\Delta t+\xi\sqrt{v_t}\sqrt{\Delta t}\,Z_v\big)
\]

\[
S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\lambda\kappa_J-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S+\sum_{i=1}^{N_{\Delta t}}J_i\Big)
\]

with \(\mathrm{Corr}(Z_S,Z_v)=\rho\), \(N_{\Delta t}\sim\mathrm{Poisson}(\lambda\Delta t)\), \(J_i\sim N(\mu_J,\sigma_J^2)\).

## Files

- `2008-2009_heston_merton.ipynb`
- `2013-2014_heston_merton.ipynb`
- `2018-2019_heston_merton.ipynb`
- `2019-2020_heston_merton.ipynb`

## Duplicate graphs / reopen

Same rules as the GBM / Merton / GARCH–Merton notebooks: show figures once via PNG bytes + `plt.close(fig)`; after reopen use **Restart Kernel** then **Run All**.
