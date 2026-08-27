# Modified GBM v2 — model specification

**This notebook is the documentation source of truth.**

Modified GBM v2 sits beside the original Modified GBM. Direction under \(P\) is unchanged. Three upgrades:

1. **Way B sizes.** \(\mathrm{size}=\exp(N(\mu,\sigma^2))\), always positive. \(\mu,\sigma\) are set so \(E[\mathrm{size}]=m\) and \(\mathrm{Var}(\mathrm{size})=v\), with \(m,v\) the mean and variance of \(|R|\) in that bucket.
2. **Calm / wild memory.** A second 1-lag Markov chain on whether the last \(|R|\) was above the lookback median. That chain picks which lognormal \((\mu,\sigma)\) is used. It does not change the up/down coins.
3. **Size-only \(Q\).** LSM does not add a constant to every return. It scales the drawn sizes by one \(\lambda>0\) so \(E[e^{R}]=e^{r_f\Delta t}\) and the sign of each path-step is unchanged.

## Symbols

| Symbol | Meaning |
|--------|---------|
| \(R_t=\ln(S_t/S_{t-1})\) | log-return |
| \(U,D\) | up / down |
| \(L,H\) | calm / wild |
| \(P(U\mid U),\ P(D\mid U),\ P(U\mid D),\ P(D\mid D)\) | sign coins |
| \(P(H\mid H),\ P(L\mid H),\ P(H\mid L),\ P(L\mid L)\) | calm/wild coins |
| \(m,v\) | mean and variance of \(\|R\|\) in one bucket |
| \(\mu,\sigma\) | mean and SD of \(\log(\mathrm{size})\) |

## Equations

\[
\sigma^2=\log(1+v/m^2),\qquad \mu=\log(m)-\tfrac12\sigma^2,\qquad
\mathrm{size}=\exp\bigl(N(\mu,\sigma^2)\bigr)
\]

\[
R_t=\begin{cases}+\mathrm{size}&U\\-\mathrm{size}&D\end{cases},\qquad S_{t+1}=S_t e^{R_t}
\]

Under \(Q\), \(R^Q=\lambda R^P\) with \(\lambda\) solved from \(\widehat{\mathbb{E}}[e^{R^Q}]=e^{r_f\Delta t}\). No additive shift and no direction premium. If a step has no up move, the old additive fallback is used.

## Code

`V4-Models_result/modified gbm v2 notebook/20*_modified_gbm_v2.ipynb`

Functions: `estimate_modified_gbm`, `calibrate_ticker`, `simulate_modified_gbm_rolling`.
