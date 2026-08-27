#!/usr/bin/env python3
"""Parameter-estimation report for the V4 1.5y monthly 10k study.

Reads rolling CSVs written during LSM. No re-pricing. P, premium, and Q
fields are listed separately for Heston / Merton / Heston–Merton.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_parameter_estimation as pe  # noqa: E402
import run_v4_1p5y_10k_monthly_empirical_study as v4  # noqa: E402

PDF_NAME = "V4_1p5y_monthly_parameter_estimation.pdf"
NB_NAME = "V4_1p5y_monthly_parameter_estimation.ipynb"

V4_PARAM_SPEC = {
    "GBM": [("mu", "μ (P)"), ("sigma", "σ")],
    "Modified GBM": [
        ("p_uu", "P(U|U)"),
        ("p_dd", "P(D|D)"),
        ("p_ud", "P(U|D)"),
        ("p_du", "P(D|U)"),
        ("mu_u", "μ_U"),
        ("sig_u", "σ_U"),
        ("mu_d", "μ_D"),
        ("sig_d", "σ_D"),
    ],
    "Modified GBM meanfix": [
        ("p_uu", "P(U|U)"),
        ("p_dd", "P(D|D)"),
        ("p_ud", "P(U|D)"),
        ("p_du", "P(D|U)"),
        ("mu_u", "μ_U"),
        ("sig_u", "σ_U"),
        ("mu_d", "μ_D"),
        ("sig_d", "σ_D"),
    ],
    "MD-GBM": [
        ("p_uu", "P(U|U)"),
        ("p_dd", "P(D|D)"),
        ("p_ud", "P(U|D)"),
        ("p_du", "P(D|U)"),
        ("mu_u", "μ_U"),
        ("sig_u", "σ_U"),
        ("mu_d", "μ_D"),
        ("sig_d", "σ_D"),
    ],
    "Modified GBM v2": [
        ("p_uu", "P(U|U)"),
        ("p_dd", "P(D|D)"),
        ("p_ud", "P(U|D)"),
        ("p_du", "P(D|U)"),
        ("p_hh", "P(H|H)"),
        ("p_ll", "P(L|L)"),
        ("mu_u_l", "μ_U,L"),
        ("sig_u_l", "σ_U,L"),
        ("mu_u_h", "μ_U,H"),
        ("sig_u_h", "σ_U,H"),
        ("mu_d_l", "μ_D,L"),
        ("sig_d_l", "σ_D,L"),
        ("mu_d_h", "μ_D,H"),
        ("sig_d_h", "σ_D,H"),
    ],
    "Modified GBM v3": [
        ("p_u_uuu", "P(U|UUU)"),
        ("p_u_uud", "P(U|UUD)"),
        ("p_u_udu", "P(U|UDU)"),
        ("p_u_udd", "P(U|UDD)"),
        ("p_u_duu", "P(U|DUU)"),
        ("p_u_dud", "P(U|DUD)"),
        ("p_u_ddu", "P(U|DDU)"),
        ("p_u_ddd", "P(U|DDD)"),
        ("mu_u", "μ_U"),
        ("sig_u", "σ_U"),
        ("mu_d", "μ_D"),
        ("sig_d", "σ_D"),
    ],
    "GARCH": [
        ("lambda", "λ"),
        ("omega", "ω"),
        ("alpha", "α"),
        ("beta", "β"),
        ("sigma0", "σ₀"),
        ("persist_p", "persist P"),
        ("persist_q", "persist Q"),
    ],
    "Heston": [
        ("mu", "μ (P)"),
        ("kappa", "κ (P)"),
        ("theta", "θ (P)"),
        ("xi", "ξ"),
        ("rho", "ρ"),
        ("v0", "v₀ (P)"),
        ("eta_v", "η_v (premium)"),
        ("kappa_q", "κ* (Q)"),
        ("theta_q", "θ* (Q)"),
        ("v0_q", "v₀^Q"),
        ("n_quotes", "n quotes"),
    ],
    "Merton": [
        ("mu", "μ (P)"),
        ("sigma", "σ"),
        ("lam", "λ* = λ"),
        ("mu_j", "μ_J (P)"),
        ("sigma_j", "σ_J"),
        ("kappa", "κ (P)"),
        ("mu_j_q", "μ_J* (Q)"),
        ("kappa_q", "κ^Q"),
        ("n_quotes", "n quotes"),
    ],
    "GARCH–Merton": [
        ("mu", "μ"),
        ("omega", "ω"),
        ("alpha", "α"),
        ("beta", "β"),
        ("sigma0", "σ₀"),
        ("lam", "λ"),
        ("mu_j", "μ_J"),
        ("sigma_j", "σ_J"),
        ("kappa", "κ"),
    ],
    "Heston–Merton": [
        ("mu", "μ (P)"),
        ("kappa", "κ (P)"),
        ("theta", "θ (P)"),
        ("xi", "ξ"),
        ("rho", "ρ"),
        ("v0", "v₀ (P)"),
        ("lam", "λ* = λ"),
        ("mu_j", "μ_J (P)"),
        ("sigma_j", "σ_J"),
        ("kappa_j", "κ_J (P)"),
        ("eta_v", "η_v (premium)"),
        ("kappa_q", "κ* (Q)"),
        ("theta_q", "θ* (Q)"),
        ("v0_q", "v₀^Q"),
        ("mu_j_q", "μ_J* (Q)"),
        ("kappa_j_q", "κ_J^Q"),
        ("n_quotes", "n quotes"),
    ],
}


def apply_config() -> None:
    v4.patch_v3_wrap()
    pe.PDF_NAME = PDF_NAME
    pe.NB_NAME = NB_NAME
    pe.PARAM_SPEC = {m: V4_PARAM_SPEC[m] for m in v4.STUDY_MODELS}


def main() -> int:
    apply_config()
    pe.run(recompute=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
