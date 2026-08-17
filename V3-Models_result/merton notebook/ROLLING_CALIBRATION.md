# Merton notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** Merton parameters \((\hat\mu, \hat\sigma, \hat\lambda, \hat\mu_J, \hat\sigma_J, \hat\kappa)\) using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window

From daily log returns \(r_t=\ln(S_t/S_{t-1})\):

| Quantity | Estimator |
|----------|-----------|
| Jump days | \(\lvert r_t\rvert > 3\cdot\hat\sigma_{\mathrm{day}}\) |
| \(\hat\mu,\hat\sigma\) | mean / std of **non-jump** returns, annualized (\(\times 252\), \(\times\sqrt{252}\)) |
| \(\hat\lambda\) | \(n_{\mathrm{jumps}} / Y\) |
| \(\hat\mu_J,\hat\sigma_J\) | mean / std of jump-day returns |
| \(\kappa\) | \(e^{\hat\mu_J + \hat\sigma_J^2/2}-1\) |

§4 graphs show five series: \(\mu,\sigma,\lambda,\mu_J,\kappa\). \(\sigma_J\) is estimated in the same windows and used in simulation.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Lookback choices

3 months · 6 months · 1 year · 2 years · 3 years · 5 years (calendar lookback; uses available history if the series is shorter). Default on 1-year regime notebooks: **3 years**, rolling **monthly**.

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, rolling parameter charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; Merton MC paths and expected path vs historical prices (one pair per ticker).

## Simulation step

\[
S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\lambda\kappa-\tfrac12\sigma^2\big)\Delta t+\sigma\sqrt{\Delta t}\,Z+\sum_{i=1}^{N_{\Delta t}}J_i\Big)
\]

## Files

- `2008-2009_merton.ipynb`
- `2013-2014_merton.ipynb`
- `2018-2019_merton.ipynb`
- `2019-2020_merton.ipynb`

## Duplicate graphs / reopen

Same rules as the GBM notebooks: show figures once via PNG bytes + `plt.close(fig)`; after reopen use **Restart Kernel** then **Run All**.
