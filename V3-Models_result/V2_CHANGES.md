# V2 methodological fixes

Copied from `V1-Models_result/`, then estimation-only edits in Merton and Heston–Merton (Method A).  
GBM / GARCH–Merton simulators, contracts, \(S_0\), horizons, RN \(\mu\to r\), seeds, LSM, and RMSE are unchanged.

## Fix 1 — Merton diffusion volatility (residual variance)

**Problem (V1):** \(\hat\sigma\) used **non-jump days only**, so jump second moments were removed from diffusion without being fully reallocated.

**V2 method:** same 3σ threshold for \((\lambda,\mu_J,\sigma_J,\kappa)\), then

\[
\sigma^2_{\mathrm{total}} = \widehat{\mathrm{Var}}(r_t)\cdot 252,\quad
\sigma^2_{\mathrm{diff}} = \max\bigl(\sigma^2_{\mathrm{total}} - \lambda(\mu_J^2+\sigma_J^2),\,\varepsilon\bigr),\quad
\hat\sigma=\sqrt{\sigma^2_{\mathrm{diff}}}.
\]

Files: `merton notebook/*_merton.ipynb` → `estimate_merton_params`.

## Fix 2 — Heston–Merton: no jump double-count in \(\theta,v_0\)

**Problem (V1):** \(\theta,v_0\) from **all** \(r_t^2\) while jumps were also simulated.

**V2 method:** continuous variance from non-jump days only:

\[
\theta = \overline{r_t^2}_{\mathrm{non\text{-}jump}}\cdot 252,\quad
v_0 = \overline{r_t^2}_{\mathrm{recent,\,non\text{-}jump}}\cdot 252.
\]

Jump block \((\lambda,\mu_J,\sigma_J,\kappa_J)\) unchanged in role.  
Files: `heston merton notebook/*_heston_merton.ipynb` → `estimate_heston_merton_params`.

## Fix 3 — Heston κ, ξ without artificial defaults/caps

**Problem (V1):** κ=2 when autocorr weak; ξ clipped to [0.05, 3]; variance lag used contemporaneous v_t.

**V2 method:**
1. AR(1) OLS on continuous daily RV: y_t = α + β y_{t-1} + e_t.
2. Euler map κ = (1−β)/Δt when 0<β<1; **no** κ=2 default.
3. If β≤0 (no persistence): constant-vol limit κ=1/Δt, ξ→ε (do **not** Feller-inflate ξ).
4. ξ from Euler residuals with **correct lag** v_{t-1}; numerical floor 1e-6 only; **no** ξ≤3 cap.
5. Soft Feller shrink only **reduces** ξ when 2κθ<ξ² and persistence is present.

## Unchanged (comparison identity)

| Item | Status |
|------|--------|
| Contract sample (24, seed 42) | same |
| \(S_0=\) panel `S_t` | same |
| `dte` steps, \(\Delta t=1/252\) | same |
| RN drift \(\mu\to r\) | same |
| Path seeds `42+i` | same |
| `n_paths=2000`, LSM | same |
| Lookback 6m; rolling none/monthly/daily | same |
| GBM / GARCH estimation | untouched |

## Outputs

- Study markdown + figures: `results/`
- Comparison notebooks: regenerate after the V2 run (`compare_rolling_{none,monthly,daily}.ipynb`)
- Runner: `scripts/run_optimal_stopping_study.py` (data root = `../research/data`)
