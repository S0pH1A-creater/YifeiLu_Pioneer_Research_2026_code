# Study

Return-based models only, **1-year observation windows**, **18-month lookback**, **monthly rolling**.

## Models

1. **GBM** — constant \(\mu,\sigma\); Q is an additive shift so \(E[e^R]=e^{r_f\Delta t}\).
2. **MD-GBM** — 1-lag up/down Markov chain; folded-normal sizes matched to sample mean of \(|R|\); same additive Q shift.
3. **GARCH(1,1)** — Duan (1995) LRNVR.
4. **Merton** — 3σ residual jumps under P; Pan (2002) jump-size premium \(\mu_J^*\) from listed calls for Q.
5. **GARCH–Merton** — Duan LRNVR on the GARCH block plus Pan \(\mu_J^*\) on jumps.

## Design

| Item | Setting |
|------|---------|
| Companies | SPY, AAPL, MSFT, AMZN |
| Crisis | 2008-08-01 → 2009-07-31 |
| Normal | 2014-01-01 → 2014-12-31 |
| Late-cycle | 2018-10-01 → 2019-09-30 |
| COVID | 2019-09-01 → 2020-08-31 |
| Lookback | 18 months, monthly updates |
| Option sample | Monday nearest-ATM American calls, DTE 7–60, near ATM \|S/K−1\| ≤ 10% |
| Paths | 10,000, seed 42 |
| LSM ranking | percentage RMSE vs listed mid |
| Stock ranking | percentage RMSE of the p50 P-path vs realized adj-close |

P-measure stock paths keep \(\mu\). LSM uses Q.

Results: `results/1p5y_monthly_return_based/`. Code: `code/`.
