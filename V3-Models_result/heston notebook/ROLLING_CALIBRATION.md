# Heston notebooks — option-implied rolling calibration (V3)

Heston-only is **Heston without jumps**: the SDE, Euler Monte Carlo, and LSM workflow are the same as V2. **Parameter estimation is not Method A.**

**Estimation in this folder:** **option-implied nonlinear least squares.** Fit \((\kappa,\theta,\xi,\rho,v_0)\) so Heston European call prices (Fourier / characteristic function, Albrecher “little trap”) match listed market calls. **No** AR(1) / realized-variance moment matching. **No** Monte Carlo inside calibration.

Physical drift \(\mu\) is still the lookback mean of stock returns (option prices identify the risk-neutral law, not \(P\)-measure drift). LSM continues to replace \(\mu\to r\).

## Required behavior

1. At each update point (daily / monthly / minutely / hourly / or once if “none”), collect listed **call** quotes in the lookback ending at that update.
2. Calibrate \((\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0)\) by minimizing the (weighted) gap between market prices and Heston model prices.
3. Use those parameters for the **subsequent** Monte Carlo segment until the next update.
4. If the option snapshot is unchanged (typical for 1-minute files with daily option panels), reuse the previous Heston block and only refresh \(\mu\).

## Estimation in each window

| Quantity | Estimator |
|----------|-----------|
| \(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) | \(\min\sum_i w_i(C^{\mathrm{Heston}}(K_i,T_i)-C_i^{\mathrm{mkt}})^2\) |
| Quotes | calls, no-arbitrage \(C\ge\max(0,S-K)\), moneyness \(\lvert S/K-1\rvert\le 10\%\), DTE \(7\)–\(60\), liquid bid–ask; subsampled to \(\leq 24\) contracts |
| \(\hat\mu\) | mean of lookback log returns \(\times N_{\mathrm{days}}\) |

Pricer: Heston (1993) \(P_1,P_2\) inversion of the log-spot characteristic function.

## Modes

Same rolling schedule as V2 (`daily` / `monthly` / `none` on 1-year regime files; `minutely` / `hourly` on 1-minute files). Default on 1-year notebooks: **3 years** lookback, **monthly** rolling.

- **Calibrated parameters** \((\kappa,\theta,\xi,\rho,v_0)\): fixed under `none`; refreshed under rolling.
- **Variance state \(v_t\)**: still evolves every step inside the Monte Carlo via the Heston SDE.

## Simulation step

The stock increment over \([t,t+\Delta t]\) uses the **current** variance \(v_t\). Variance is updated afterward. Do not use \(v_{t+\Delta t}\) in the return.

\[
S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S\Big)
\]

\[
v_{t+\Delta t}=\max\big(0,\ v_t+\kappa(\theta-v_t)\Delta t+\xi\sqrt{v_t}\sqrt{\Delta t}\,Z_v\big)
\]

with \(\mathrm{Corr}(Z_S,Z_v)=\rho\). No Poisson jumps.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 3 years · 5 years. Default on 1-year regime notebooks: **3 years**, rolling **monthly**.

## Files

- `2008-2009_heston.ipynb`
- `2013-2014_heston.ipynb`
- `2018-2019_heston.ipynb`
- `2019-2020_heston.ipynb`
- `7d_1min_heston.ipynb`
- `1d_1min_heston.ipynb`

Shared pricer / optimizer: `V3-Models_result/scripts/heston_option_calibration.py`.
