# What this code version is

This tree is only the **1.5-year monthly return-based** study.

**GARCH:** Duan (1995) LRNVR, not an ad hoc \(\mu\to r\) swap.

**Merton / GARCH–Merton:** Pan (2002) jump-size premium \(\mu_J^*\) from listed calls on top of return-based P jumps.

**GBM / MD-GBM:** additive shift so \(E[e^R]=e^{r_f\Delta t}\).

MD-GBM is 1-lag Markov Directional GBM with folded-normal means matched to sample \(|R|\).

Published reports: `results/1p5y_monthly_return_based/`.
Cache: `code/results/empirical_study_1p5y_monthly_10000/`.
