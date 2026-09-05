# GARCH notebooks — Duan (1995) GARCH-in-mean and LRNVR

GARCH-only is **GARCH–Merton without the jump block**. V4 estimates Duan’s GARCH-in-mean under \(P\) and prices with the Locally Risk-Neutral Valuation Relationship. No jump parameters \(\lambda_J,\mu_J,\sigma_J,\kappa\).

The extra parameter is Duan’s **unit risk premium** \(\lambda\) (not a jump intensity). It is estimated from lookback log returns and observed FRED DGS3MO.

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period when rolling is `daily` or `monthly`.

## Required behavior

1. At each update point, re-estimate **all** GARCH-in-mean parameters
   \((\hat\lambda, \hat\omega, \hat\alpha, \hat\beta, \hat\sigma_0)\)
   using **only** returns (and aligned \(r_{f,t}\)) inside the current lookback window.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window

From log returns \(r_t=\ln(S_t/S_{t-1})\) and one-period \(r_{f,t}=R_t^{\mathrm{DGS3MO}}/N_{\mathrm{days}}\):

| Step | What |
|------|------|
| 1. Duan GARCH-in-mean MLE | \(r_t=r_{f,t}+\lambda\sigma_t-\tfrac12\sigma_t^2+\sigma_t\varepsilon_t\), \(\sigma_t^2=\omega+\alpha\sigma_{t-1}^2\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2\) |
| 2. \(\sigma_0\) | last conditional \(\sigma\) from the fit (else \(\sqrt{\omega/(1-\alpha-\beta)}\)) |

§4 graphs show five series: \(\lambda,\omega,\alpha,\beta,\sigma_0\). After Reestimate, print **P parameters** and **Q dynamics** as two tables.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

**Important distinction**

- **Calibrated parameters** \((\lambda,\omega,\alpha,\beta)\): fixed under `none`; refreshed under rolling.
- **Conditional volatility \(\sigma_t\)**: still evolves every step via the GARCH recursion.

## Simulation

**§5 (\(P\)):**

\[
\ln\frac{S_{t+\Delta t}}{S_t}=r_{f,t}+\lambda\sigma_t-\tfrac12\sigma_t^2+\sigma_t\varepsilon_t,\quad
\sigma_{t+\Delta t}^2=\omega+\alpha\sigma_t^2\varepsilon_t^2+\beta\sigma_t^2
\]

**§6 LSM (\(Q\), Duan LRNVR):**

\[
\ln\frac{S_{t+\Delta t}}{S_t}=r_{f,t}-\tfrac12\sigma_t^2+\sigma_t\xi_t,\quad
\sigma_{t+\Delta t}^2=\omega+\alpha\sigma_t^2(\xi_t-\lambda)^2+\beta\sigma_t^2
\]

\(\omega,\alpha,\beta,\lambda,\sigma_0\) are the same numbers under \(Q\). Do **not** replace the mean by \(\mu\to r\) without the \(-\tfrac12\sigma^2\) term and the \((\xi-\lambda)\) variance shock.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 3 years · 5 years (calendar lookback). Headless V4 GARCH study: **1.5 years**, rolling **monthly**. Minimum window: 60 trading days.

## Files

- `2008-2009_garch.ipynb`
- `2013-2014_garch.ipynb`
- `2018-2019_garch.ipynb`
- `2019-2020_garch.ipynb`
- `7d_1min_garch.ipynb`
- `1d_1min_garch.ipynb`

Shared code: `code/scripts/garch_duan_lrnvr.py`.
