# Modified GBM v3 — model specification

**This notebook is the documentation source of truth.**

Modified GBM v3 sits beside the original Modified GBM and v2. Sizes are unchanged from v1 (\(|N(\mu,\sigma)|\)). The only change is the sign chain: last three signs, not last one.

## Symbols

| Symbol | Meaning |
|--------|---------|
| \(R_t=\ln(S_t/S_{t-1})\) | log-return |
| \(U,D\) | up / down |
| \(s\in\{\mathrm{UUU},\ldots,\mathrm{DDD}\}\) | last three non-zero signs (8 states) |
| \(P(U\mid s)\) | next-bar up coin given \(s\) |
| \(\mu_U,\sigma_U,\mu_D,\sigma_D\) | split-normal size parameters |

## State update

Encode \(U=1\), \(D=0\). Pack \(s=4s_{t-2}+2s_{t-1}+s_t\). After drawing \(X\in\{0,1\}\),

\[
s \leftarrow (2s+X)\bmod 8.
\]

Laplace: \(\hat P(U\mid s)=(n_{s\to U}+0.5)/(n_s+1)\).

## Price

\[
R_t=\begin{cases}+|N(\mu_U,\sigma_U^2)| & U\\-|N(\mu_D,\sigma_D^2)| & D\end{cases},\qquad S_{t+1}=S_t e^{R_t}
\]

\(Q\) still shifts each step so \(E[e^{R_t}]=e^{r_f\Delta t}\).

## Code

`V4-Models_result/modified gbm v3 notebook/20*_modified_gbm_v3.ipynb`

Functions: `estimate_modified_gbm`, `calibrate_ticker`, `simulate_modified_gbm_rolling`.
