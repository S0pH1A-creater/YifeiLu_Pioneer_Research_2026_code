# Heston notebooks — true rolling-window calibration (Method A)

Heston-only is **Heston–Merton without the jump block**: same Method A realized-variance moments, rolling schedule, Monte Carlo, and LSM workflow; no \(\lambda,\mu_J,\sigma_J,\kappa_J\).

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period when rolling is `daily` or `monthly`.

**Estimation method in this folder:** **Method A only** — historical moment / realized-variance proxies on **all** returns in the window. **No** jump-day filter. **No** Monte Carlo / path-averaged likelihood in calibration (that lives in the Heston–Merton advanced folder).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** Heston parameters  
   \((\hat\mu, \hat\kappa, \hat\theta, \hat\xi, \hat\rho, \hat v_0)\)  
   using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window (Method A — no random draws)

From log returns \(r_t=\ln(S_t/S_{t-1})\), \(RV_t=r_t^2\), \(\Delta t=1/N_{\mathrm{days}}\):

| Quantity | Estimator |
|----------|-----------|
| \(\hat\mu\) | mean of **all** returns \(\times N_{\mathrm{days}}\) |
| \(\hat\theta\) | \(\overline{RV}\times N_{\mathrm{days}}\) |
| \(\hat v_0\) | recent \(RV\) (last ≤21 observations) \(\times N_{\mathrm{days}}\) |
| \(\hat\kappa\) | from lag-1 autocorr / AR(1) of \(RV\) (clipped) |
| \(\hat\xi\) | moment scale of \(\Delta v\) residuals after mean-reversion drift (clipped) |
| \(\hat\rho\) | \(\mathrm{Corr}(r_t,\Delta v_t)\) on annualized variance proxy |

Minimum window length: **60** observations (else skip update).

§4 graphs show six series: \(\mu,\theta,\kappa,\xi,\rho,v_0\).

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

**Important distinction**

- **Calibrated parameters** \((\mu,\kappa,\theta,\xi,\rho)\): fixed under `none`; refreshed under rolling.
- **Variance state \(v_t\)**: still evolves every step inside the Monte Carlo via the Heston SDE, even when the calibrated law is fixed. \(\hat v_0\) initializes each path only.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 5 years (2-year files) or 1 hour / 1 day (1-minute files); keep a **Reestimate** control.

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, rolling parameter charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; Heston MC paths and median vs historical prices (one pair per ticker).
- **§6 Optimal stopping:** LSM American calls with the same Heston simulator (risk-neutral drift \(\mu\to r\)).

## Simulation step

\[
v_{t+\Delta t}=\max\big(0,\ v_t+\kappa(\theta-v_t)\Delta t+\xi\sqrt{v_t}\sqrt{\Delta t}\,Z_v\big)
\]

\[
S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S\Big)
\]

with \(\mathrm{Corr}(Z_S,Z_v)=\rho\). No Poisson jumps and no compensator \(\lambda\kappa_J\).

## Files

- `2008-2009_heston.ipynb`
- `2013-2014_heston.ipynb`
- `2018-2019_heston.ipynb`
- `2019-2020_heston.ipynb`
- `7d_1min_heston.ipynb`
- `1d_1min_heston.ipynb`

## Duplicate graphs / reopen

Same rules as the other model notebooks: show figures once via PNG bytes + `plt.close(fig)`; after reopen use **Restart Kernel** then **Run All**.
