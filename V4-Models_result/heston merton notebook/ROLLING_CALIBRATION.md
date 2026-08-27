# Heston–Merton notebooks — Bates / Pan (2002) P→Q

V4 only. One SVJ framework (Bates 1996 dynamics + Pan 2002 premia). Do **not** glue a standalone Heston map to a standalone Merton map.

**\(P\):** Method A \((\mu,\kappa,\theta,\xi,\rho,v_0)\) plus 3σ jumps \((\lambda,\mu_J,\sigma_J)\).

**Premia from listed Bates/Fourier calls:** \(\eta_v\) and \(\mu_J^*\). \(\lambda^*=\lambda\); \(\xi,\rho,\sigma_J\) from \(P\).

**\(Q\) Euler:** stock drift \(r_f-\lambda\kappa^Q\); variance drift \(\kappa(\theta-v)+\eta_v v\).

§4 graphs \(P\). §5 is \(P\). §6 LSM is \(Q\) (`simulate_bates_q`).

Shared code: `V4-Models_result/scripts/pq_risk_premium.py`. Notes: `V4-Models_result/P_TO_Q_RISK_PRICING.md`.
