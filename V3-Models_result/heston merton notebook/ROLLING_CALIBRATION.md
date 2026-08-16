# Heston–Merton notebooks — option-implied rolling calibration (V3)

The **Heston–Merton SDEs, Euler Monte Carlo, and LSM workflow are unchanged from V2.** **The Heston block is no longer Method A.**

**Estimation in this folder:** **option-implied Bates calibration** of \((\kappa,\theta,\xi,\rho,v_0)\) from listed call prices, with the Merton jump block \((\lambda,\mu_J,\sigma_J,\kappa_J)\) taken from a 3σ return threshold and **held fixed** while the Heston parameters are fitted. Model prices during calibration use the Bates characteristic function (Heston CF × Merton jump CF).

Physical drift \(\mu\) is the lookback mean of stock returns. LSM continues to replace \(\mu\to r\).

## Required behavior

1. At each update point, collect listed **call** quotes in the lookback ending there, and jump statistics from the same lookback of returns.
2. Calibrate \((\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0)\) by minimizing the gap between market prices and Bates model prices.
3. Use the full parameter vector for the **subsequent** Monte Carlo segment until the next update.
4. If the option snapshot is unchanged, reuse the previous Heston block and only refresh \(\mu\) and the jump block from returns.

## Estimation in each window

| Quantity | Estimator |
|----------|-----------|
| \(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) | \(\min\sum_i w_i(C^{\mathrm{Bates}}(K_i,T_i)-C_i^{\mathrm{mkt}})^2\) with jumps held from the return window |
| Quotes | calls, moneyness \(0.8\)–\(1.2\), DTE \(5\)–\(365\), subsampled to \(\leq 24\) contracts |
| \(\hat\mu\) | mean of lookback log returns \(\times N_{\mathrm{days}}\) |
| Jump days | \(\lvert r_t\rvert > 3\cdot\hat\sigma\) |
| \(\hat\lambda,\hat\mu_J,\hat\sigma_J\) | intensity / mean / std of jump-bar returns |
| \(\kappa_J\) | \(e^{\hat\mu_J+\hat\sigma_J^2/2}-1\) |

§4 graphs still show \(\mu,\theta,\kappa,\xi,\rho,\lambda\). \(v_0,\mu_J,\sigma_J,\kappa_J\) are stored and used in simulation.

## Modes

Same rolling schedule as V2. Variance \(v_t\) still evolves inside each path via the Heston SDE even when the calibrated law is held fixed.

## Simulation step

The stock increment over \([t,t+\Delta t]\) uses the **current** variance \(v_t\). Variance is updated afterward. Do not use \(v_{t+\Delta t}\) in the return.

\[
S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\lambda\kappa_J-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S+\sum_{i=1}^{N_{\Delta t}}J_i\Big)
\]

\[
v_{t+\Delta t}=\max\big(0,\ v_t+\kappa(\theta-v_t)\Delta t+\xi\sqrt{v_t}\sqrt{\Delta t}\,Z_v\big)
\]

with \(\mathrm{Corr}(Z_S,Z_v)=\rho\), \(N_{\Delta t}\sim\mathrm{Poisson}(\lambda\Delta t)\), \(J_i\sim N(\mu_J,\sigma_J^2)\).

## Files

- `2008-2009_heston_merton.ipynb`
- `2013-2014_heston_merton.ipynb`
- `2018-2019_heston_merton.ipynb`
- `2019-2020_heston_merton.ipynb`
- `7d_1min_heston_merton.ipynb`
- `1d_1min_heston_merton.ipynb`

Shared pricer / optimizer: `V3-Models_result/scripts/heston_option_calibration.py`.
