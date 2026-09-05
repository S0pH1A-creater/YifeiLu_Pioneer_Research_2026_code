# P → Q for the remaining models

LSM uses **Q**. Stock-path reports use **P**.

| Model | P | Q |
|-------|---|---|
| GBM | lookback \(\mu,\sigma\) | additive shift so \(E[e^R]=e^{r_f\Delta t}\) |
| MD-GBM | 1-lag signs + folded-normal sizes | same additive shift |
| GARCH | Duan GARCH-in-mean MLE | Duan LRNVR (`garch_duan_lrnvr.py`) |
| Merton | returns + 3σ jumps | Pan \(\mu_J^*\) from listed calls (`pq_risk_premium.py`) |
| GARCH–Merton | GARCH block + jumps | Duan LRNVR + Pan \(\mu_J^*\) (`garch_merton_pq.py`) |

No premium is filled in by hand. If listed quotes are missing, Merton Q jump fields stay empty and LSM should not run that cell.

§ notebook Monte Carlo vs history is P. LSM is Q.
