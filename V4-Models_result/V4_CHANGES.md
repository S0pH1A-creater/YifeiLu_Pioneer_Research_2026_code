# V4 methodological change — Duan (1995) GARCH LRNVR

Copied from `V3-Models_result/` (notebooks, scripts, config; not `results/`). V1 / V2 / V3 are not modified.

## What changed

V3 GARCH was a constant-mean physical-measure model, and LSM “risk-neutralized” it by the ad hoc replacement \(\mu\to r\). That is not Duan’s Locally Risk-Neutral Valuation Relationship (LRNVR).

**V4 GARCH(1,1):** estimate Duan’s GARCH-in-mean under \(P\) from lookback returns and FRED DGS3MO, then price American calls with the unique LRNVR \(Q\)-dynamics. Implementation: `scripts/garch_duan_lrnvr.py`.

| Object | Under \(P\) | Under \(Q\) |
|--------|-------------|-------------|
| \(\omega,\alpha,\beta,\sigma_0\) | MLE | unchanged |
| \(\lambda\) (unit risk premium) | MLE in the mean | same number, used as \((\xi_t-\lambda)\) in the variance |
| Log-return mean | \(r_{f,t}+\lambda\sigma_t-\tfrac12\sigma_t^2\) | \(r_{f,t}-\tfrac12\sigma_t^2\) |

**V4 Heston / Merton / Heston–Merton:** \(P\) from returns; volatility and jump-size premia from listed calls using Pan (2002) on the Bates (1996) SVJ class. LSM uses the resulting \(Q\) paths. GBM stays \(\mu\to r_f\). GARCH stays Duan LRNVR. GARCH–Merton is unchanged. Details: `P_TO_Q_RISK_PRICING.md`. Implementation: `scripts/pq_risk_premium.py`.

## Studies that were run

1. **GARCH-only documentation** (left in place). `Results_In_Short/V4/V4 1.5-year monthly GARCH LRNVR/`.

2. **Seven-model 10k LSM**, V3 monthly layout. Same Monday ATM sample, 18-month lookback, monthly rolling, four 1-year regimes, SPY/AAPL/MSFT/AMZN. 112/112 cells.

Models and grouping match V3 (split by how **P** is estimated, not by the P→Q map):

- Return-based: GBM, Modified GBM, GARCH, Merton, GARCH–Merton
- Option-implied: Heston, Heston–Merton

V4 changes inside those groups: GARCH uses Duan LRNVR; Merton / Heston / Heston–Merton use Pan (2002) premia for LSM; GBM and Modified GBM stay \(\mu\to r_f\); GARCH–Merton is unchanged.

Outputs: `results/empirical_study_1p5y_monthly_10000/` and `Results_In_Short/V4/V4 1.5-year monthly empirical study 10000 paths/` (PDFs at the folder root, notebooks in `Notebooks/`).

3. **Seven-model 10k LSM, fixed 1.5-year calibration.** Same Monday ATM sample, 18-month lookback, four 1-year regimes, SPY/AAPL/MSFT/AMZN, and the same seven models / grouping as study 2. Rolling is none: one calibration at the first session of each regime, held for the window.

Outputs: `results/empirical_study_1p5y_fixed_10000/` and `Results_In_Short/V4/V4 1.5-year fixed empirical study 10000 paths/` (PDFs at the folder root, notebooks in `Notebooks/`).

4. **Modified GBM v2** (not in the finished 10k studies). Lognormal sizes (Way B), a calm/wild Markov selector, and size-only \(Q\) (\(R^Q=\lambda R^P\)). Notebooks: `modified gbm v2 notebook/`. Builder: `scripts/build_modified_gbm_v2_notebooks.py`.

5. **Modified GBM v3** (not in the finished 10k studies). Same split-normal sizes as original Modified GBM. Sign chain is order-3: eight states UUU, …, DDD. Notebooks: `modified gbm v3 notebook/`. Builder: `scripts/build_modified_gbm_v3_notebooks.py`.
