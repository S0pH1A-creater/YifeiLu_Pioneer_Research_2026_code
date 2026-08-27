# Heston–Merton advanced notebooks — Method B (light MC likelihood)

**Separate from** [`../heston merton notebook/`](../heston%20merton%20notebook/) (Method A moments). Same period files and UI structure; different calibration engine.

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period when rolling is `daily` or `monthly`.

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate  
   \((\hat\mu, \hat\kappa, \hat\theta, \hat\xi, \hat\rho, \hat v_0, \hat\lambda, \hat\mu_J, \hat\sigma_J, \hat\kappa_J)\)  
   from **only** the current lookback window.
2. Use those parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Estimation in each window (Method B)

1. **Method A warm start** — moments + jump threshold (same formulas as the basic folder) for  
   \((\mu,\kappa,\theta,\xi,\rho,v_0,\lambda,\mu_J,\sigma_J,\kappa_J)\).
2. **Hold the jump block** \((\lambda,\mu_J,\sigma_J,\kappa_J)\) and drift \(\mu\) from Method A.
3. **Light MC likelihood** for the Heston block \((\kappa,\theta,\xi,\rho,v_0)\):
   - Draw \(M\) latent variance paths on the **daily historical grid** (no sub-steps).
   - On non-jump days, score the Gaussian return density given current \(v\); evolve \(v\) with shocks correlated to the return residual via \(\rho\).
   - Objective: minimize average negative log-likelihood over the \(M\) paths.
4. Optimize with a **capped** pure-NumPy coordinate/random search (tens of iterations), fixed calibration seed.

### Compute budget (keep the machine calm)

| Knob | Default | Notes |
|------|---------|--------|
| Latent paths \(M\) | **32** | estimation only; not research MC |
| Optimizer `maxiter` | **25** | capped NumPy search from Method A |
| Time grid | daily lookback days | no intra-day steps |
| Jump / drift | from Method A | not free in the MC stage |

Research Monte Carlo in §5 still uses `N_STEPS=500` and ~1000 paths — that is **separate** from estimation MC.

Minimum window length: **60** trading days.

§4 graphs show six series: \(\mu,\theta,\kappa,\xi,\rho,\lambda\). \(v_0,\mu_J,\sigma_J,\kappa_J\) are estimated in the same windows and used in simulation.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

Default UI rolling mode: **monthly** (daily Method B is available but slower).

**Important distinction**

- **Calibrated parameters**: fixed under `none`; refreshed under rolling.
- **Variance state \(v_t\)**: still evolves every step inside the §5 Monte Carlo via the Heston SDE.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 3 years · 5 years (calendar lookback; uses available history if the series is shorter). Default on 1-year regime notebooks: **3 years**, rolling **monthly**.

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, rolling parameter charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; Heston–Merton MC paths and expected path vs historical prices (one pair per ticker).

## Simulation step (§5)

The stock increment uses the current \(v_t\); variance is updated afterward.

\[
S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\lambda\kappa_J-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S+\sum_{i=1}^{N_{\Delta t}}J_i\Big)
\]

\[
v_{t+\Delta t}=\max\big(0,\ v_t+\kappa(\theta-v_t)\Delta t+\xi\sqrt{v_t}\sqrt{\Delta t}\,Z_v\big)
\]

## Files

- `2008-2009_heston_merton_advanced.ipynb`
- `2013-2014_heston_merton_advanced.ipynb`
- `2018-2019_heston_merton_advanced.ipynb`
- `2019-2020_heston_merton_advanced.ipynb`

## Duplicate graphs / reopen

Same rules as the other model notebooks: show figures once via PNG bytes + `plt.close(fig)`; after reopen use **Restart Kernel** then **Run All**.
