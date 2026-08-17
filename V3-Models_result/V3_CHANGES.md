# V3 methodological change — option-implied Heston

Copied from `V2-Models_result/`, then **estimation-only** edits in Heston and Heston–Merton, plus a shared sampling / RMSE change that applies to every model. V1 and V2 are not modified.

## Sampling and RMSE (all models)

Random trading-date sampling (`n=24`, seed 42, moneyness×DTE strata) is removed.

| Window | Grid | Typical count | Rule |
|--------|------|---------------|------|
| 1-year regime | every Monday | ~50 | If Monday is closed, use the next session that week. Count follows actual trading weeks / option-panel coverage. |
| 7-day | every 15 RTH minutes | ~130 | 09:30–15:45; 5 sessions × 26. |
| 1-day | every 5 RTH minutes | ~78 | 09:30–15:55. |

At each grid time: nearest-ATM listed call, DTE 7–60, listed expiry. Quotes are **as-of** that timestamp (same session, else the prior session). No look-ahead. Same sampler in every model (`scripts/american_lsm.py`).

**Percentage RMSE** is the study metric everywhere evaluation is reported:

\[
\mathrm{RMSE}\% = 100\times\sqrt{\mathrm{mean}\bigl((C^{\mathrm{model}}-C^{\mathrm{mkt}})/C^{\mathrm{mkt}}\bigr)^2}
\]

Stock-path RMSE uses the same formula vs \(S_t\). The Heston Fourier NLS objective is still dollar SSE; it also stores `rmse_pct` as a diagnostic.

Existing `results/` tables are dollar RMSE until the V3 study scripts are re-run.

## 1-year evaluation windows

The four daily-regime notebooks are **one year** each (filenames still `2008-2009_*` etc.). 1-day / 7-day notebooks are unchanged.

| File stem | Evaluation window |
|-----------|-------------------|
| `2008-2009` | 2008-08-01 → 2009-07-31 |
| `2013-2014` | 2014-01-01 → 2014-12-31 |
| `2018-2019` | 2018-10-01 → 2019-09-30 |
| `2019-2020` | 2019-09-01 → 2020-08-31 |

Default on these notebooks: lookback **3 years**, rolling **monthly**. Headless study scripts use the same lookback.

**Data coverage:** equity `prices_clean.csv` and processed call panels now run through December 2020, so September 2019 – August 2020 is fully covered.

## What changed

**Problem (V1/V2):** \((\kappa,\theta,\xi,\rho,v_0)\) came from rolling **Method A** historical moments (AR(1) on realized variance, RV-based \(\theta\), return-based \(\rho\), residual \(\xi\)). That is a \(P\)-measure proxy, not the standard option-implied (risk-neutral) Heston calibration used in quantitative finance.

**V3 method:** at each rolling update, fit

\[
(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0)
=\arg\min\sum_i w_i\bigl(C^{\mathrm{model}}(K_i,T_i)-C_i^{\mathrm{mkt}}\bigr)^2
\]

on listed call quotes in the lookback window after the shared V3 estimation filters (no-arbitrage \(C\ge\max(0,S-K)\), DTE \(7\)–\(60\), \(\lvert S/K-1\rvert\le 10\%\), liquid bid–ask). Model prices are European Heston (or Bates, for Heston–Merton) values from the characteristic-function inversion (Albrecher little-trap). Implementation: `scripts/heston_option_calibration.py` and `scripts/option_filters.py`.

| Model | Option-implied | Still from returns |
|-------|----------------|--------------------|
| Heston | \(\kappa,\theta,\xi,\rho,v_0\) | \(\mu\) (P-measure drift for §5) |
| Heston–Merton | \(\kappa,\theta,\xi,\rho,v_0\) via Bates CF | \(\mu\) and \((\lambda,\mu_J,\sigma_J,\kappa_J)\) (3σ threshold) |

## Unchanged (comparison identity)

| Item | Status |
|------|--------|
| Heston / Heston–Merton SDEs and Euler step | Euler now uses \(v_t\) for the stock return, then updates \(v_{t+\Delta t}\) |
| Contract sample | systematic Mondays / 15-min / 5-min; percentage RMSE |
| Lookback / rolling UI and §4–§6 workflow | same |
| GBM / Merton / GARCH / GARCH–Merton SDEs | untouched (shared sampler + %RMSE only) |
| V1 / V2 trees | untouched |

## Outputs

Copied V2 `results/` are **Method A numbers** until the V3 study scripts are re-run from `V3-Models_result/scripts/`.
