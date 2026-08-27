# V4 P→Q risk pricing (Heston, Merton, Heston–Merton)

Status: **implemented in V4 notebooks and scripts. Seven-model 1.5-year monthly 10k LSM has been run**, grouped as in V3 (return-based vs option-implied).

V1 / V2 / V3 are unchanged. GBM is unchanged (\(\mu\to r_f\)). GARCH is unchanged (Duan 1995 LRNVR in `scripts/garch_duan_lrnvr.py`). GARCH–Merton is unchanged.

## Why this map (not “the best”)

The maps are taken from papers this project was asked to use. They are **not** claimed to be uniquely optimal.

| Paper | What it supplies |
|-------|------------------|
| Heston (1993) | Affine volatility risk premium \(\lambda(S,v,t)=\lambda v\) |
| Bates (1996) | SV + lognormal jumps; Q stock drift \(r-\lambda\kappa^Q\) |
| Pan (2002), JFE | Bates/SVJ pricing kernel: \(\eta_v\) in the variance drift; jump-size premium \(\mu_J^*\) vs \(\mu_J\); \(\lambda^*=\lambda\) because timing and size are not separately identified from returns (Pan p. 8 and Appendix D: \(\kappa^*=\kappa-\eta_v\), \(\bar v^*=\kappa\bar v/\kappa^*\)) |
| Christoffersen, Heston, Jacobs (WP 2002/2003; JE 2006) | Discrete-time inverse-Gaussian GARCH pricing kernel. **Not** used as the continuous-time Heston/Merton map |

V3 already fitted Heston/Bates **Q** parameters from options and then still replaced \(\mu\to r\) while leaving jump parameters at **P**. That prices neither volatility risk nor jump risk as a \(P\to Q\) change.

## Identifiability

| Object | Identified from |
|--------|-----------------|
| \(P\) Heston \((\mu,\kappa,\theta,\xi,\rho,v_0)\) | lookback returns (Method A) |
| \(P\) Merton \((\mu,\sigma,\lambda,\mu_J,\sigma_J)\) | lookback returns (3σ jumps) |
| \(\eta_v\) | listed calls (vol is not traded) |
| \(\mu_J^*\) | listed calls (jump risk is not spanned by \(S\)) |
| \(\xi,\rho,\sigma,\sigma_J,\lambda^*\) | held at \(P\) (Girsanov / Pan identification) |

No premium is filled in by hand. If \(\lambda=0\) there are no jumps to reprice (\(\mu_J^*=\mu_J\) is then a statement that the \(P\) model has no jump risk). If quotes are missing, Q fields stay NaN and LSM refuses to run.

Method A can clip \(\kappa\) or \(\xi\) on short windows. That is a **\(P\)** limitation; \(\eta_v\) is still fitted from options given that \(P\), not typed in.

## Pipeline (each rolling window)

Historical returns + listed calls \(\to\) \(P\) calibration \(\to\) option NLS for premia \(\to\) \(Q\) simulation \(\to\) LSM.

§4 graphs \(P\). §5 Monte Carlo is \(P\) (vs history). §6 LSM uses \(Q\) only.

## Files

- `scripts/pq_risk_premium.py` — estimators, Pan maps, P/Q simulators
- `scripts/check_pq_risk_premium.py` — small Q-mean check (not a study)
- `scripts/patch_v4_pq_risk_premium.py` — notebook patcher
- `heston notebook/`, `merton notebook/`, `heston merton notebook/`
