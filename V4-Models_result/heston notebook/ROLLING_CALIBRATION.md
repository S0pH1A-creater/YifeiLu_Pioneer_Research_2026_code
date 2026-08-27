# Heston notebooks — P from returns, Q via Pan (2002)

Heston-only: **no jumps**. V1/V2/V3 notebooks are not modified.

**\(P\):** Method A realized-variance moments on lookback returns \((\mu,\kappa,\theta,\xi,\rho,v_0)\).

**Risk premium:** \(\eta_v\) from listed calls (Pan 2002). \(\xi,\rho\) held at \(P\). Returns cannot identify \(\eta_v\).

**\(Q\):** \(\kappa^*=\kappa-\eta_v\), \(\bar v^*=\kappa\bar v/\kappa^*\), stock drift \(r_f\). Euler uses the Pan variance drift \(\kappa(\theta-v)+\eta_v v\).

§4 graphs \(P\). §5 Monte Carlo is \(P\). §6 LSM is \(Q\) only (`simulate_heston_q`).

Shared code: `V4-Models_result/scripts/pq_risk_premium.py`. Notes: `V4-Models_result/P_TO_Q_RISK_PRICING.md`.
