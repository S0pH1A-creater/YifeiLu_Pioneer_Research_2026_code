# Modified GBM AI — model specification

Same price law as Modified GBM: Markov up/down, split |N| sizes, \(S_{t+1}=S_t e^{r_t}\).

**What changes:** parameters are not the lookback counts. Each monthly (or daily) window:

1. Start at those counts (\(P(U\mid U), P(D\mid D), \mu_U, \sigma_U, \mu_D, \sigma_D\)).
2. Simulate one-step returns in **PyTorch**.
3. MSE vs the window’s moments (transition rates, up/down sizes, mean/std of \(r\)).
4. **Adam** moves the parameters toward smaller error.
5. `last_up` stays the last observed sign (not learned).

Q-measure paths still use the additive shift so \(E[e^{r}]=e^{r_f\Delta t}\). LSM is unchanged.

Functions: `estimate_modified_gbm` (Adam), `calibrate_ticker`, `simulate_modified_gbm_rolling`.
