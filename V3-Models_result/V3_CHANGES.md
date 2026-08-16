# V3 methodological change — option-implied Heston

Copied from `V2-Models_result/`, then **estimation-only** edits in Heston and Heston–Merton.  
GBM / Merton / GARCH / GARCH–Merton, the Heston SDEs, Euler Monte Carlo, LSM, contracts, \(S_0\), horizons, RN \(\mu\to r\), seeds, and RMSE plumbing are unchanged. V1 and V2 are not modified.

## What changed

**Problem (V1/V2):** \((\kappa,\theta,\xi,\rho,v_0)\) came from rolling **Method A** historical moments (AR(1) on realized variance, RV-based \(\theta\), return-based \(\rho\), residual \(\xi\)). That is a \(P\)-measure proxy, not the standard option-implied (risk-neutral) Heston calibration used in quantitative finance.

**V3 method:** at each rolling update, fit

\[
(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0)
=\arg\min\sum_i w_i\bigl(C^{\mathrm{model}}(K_i,T_i)-C_i^{\mathrm{mkt}}\bigr)^2
\]

on listed call quotes in the lookback window. Model prices are European Heston (or Bates, for Heston–Merton) values from the characteristic-function inversion (Albrecher little-trap). Implementation: `scripts/heston_option_calibration.py`.

| Model | Option-implied | Still from returns |
|-------|----------------|--------------------|
| Heston | \(\kappa,\theta,\xi,\rho,v_0\) | \(\mu\) (P-measure drift for §5) |
| Heston–Merton | \(\kappa,\theta,\xi,\rho,v_0\) via Bates CF | \(\mu\) and \((\lambda,\mu_J,\sigma_J,\kappa_J)\) (3σ threshold) |

## Unchanged (comparison identity)

| Item | Status |
|------|--------|
| Heston / Heston–Merton SDEs and Euler step | Euler now uses \(v_t\) for the stock return, then updates \(v_{t+\Delta t}\) |
| Contract sample, \(S_0\), `dte`, \(\Delta t\), RN drift, path seeds, `n_paths`, LSM | same |
| Lookback / rolling UI and §4–§6 workflow | same |
| GBM / Merton / GARCH / GARCH–Merton | untouched |
| V1 / V2 trees | untouched |

## Outputs

Copied V2 `results/` are **Method A numbers** until the V3 study scripts are re-run from `V3-Models_result/scripts/`.
