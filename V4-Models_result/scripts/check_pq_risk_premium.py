#!/usr/bin/env python3
"""Small P→Q checks for Heston, Merton, and Bates. Not a full LSM study.

Checks:
  1. P, risk-premium, and Q parameters are reported separately.
  2. Premia come from listed calls (or are flagged unidentified), not invented.
  3. E^Q[S_{t+nΔt}/S_t] ≈ exp(r n Δt) on a small Monte Carlo cloud.
  4. LSM helpers simulate_heston_q / simulate_merton_q / simulate_bates_q exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from heston_option_calibration import load_calls_panel, select_quotes_asof  # noqa: E402
from pq_risk_premium import (  # noqa: E402
    calibrate_bates_q,
    calibrate_heston_q,
    calibrate_merton_q,
    estimate_bates_p,
    estimate_heston_p,
    estimate_merton_p,
    q_mean_check,
    report_bates_pq,
    report_heston_pq,
    report_merton_pq,
    simulate_bates_q,
    simulate_heston_q,
    simulate_merton_q,
)

REPO = SCRIPTS.parents[1]
DATA = REPO / "research" / "data"
OUT = SCRIPTS.parents[0] / "results" / "pq_risk_premium_check"
N_DAYS = 252
N_PATHS = 80_000
SEED = 42
R_FALLBACK = 0.02


def _window():
    prices = pd.read_csv(DATA / "equity" / "prices_clean.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    end = pd.Timestamp("2018-09-28")
    start = end - pd.DateOffset(months=18)
    rets = np.log(prices["SPY"]).diff().loc[start:end].dropna()
    panel = load_calls_panel(DATA, "SPY")
    quotes = select_quotes_asof(panel, end, pd.DateOffset(months=18))
    r_ann = float(quotes["r"].median()) if len(quotes) and "r" in quotes.columns else R_FALLBACK
    if not np.isfinite(r_ann):
        r_ann = R_FALLBACK
    return rets, quotes, r_ann


def _full(n, val):
    return np.full(n, float(val), dtype=float)


def check_heston(p, q, r_ann) -> dict:
    eta = float(q["eta_v"])
    v0 = float(q["v0_q"]) if np.isfinite(q.get("v0_q", np.nan)) else float(p["v0"])
    steps1 = {
        "rf": _full(1, r_ann), "kappa": _full(1, p["kappa"]), "theta": _full(1, p["theta"]),
        "xi": _full(1, p["xi"]), "rho": _full(1, p["rho"]), "v0": _full(1, v0),
        "v0_q": _full(1, v0), "eta_v": _full(1, eta),
    }
    paths1 = simulate_heston_q(steps1, 100.0, N_PATHS, SEED, n_days=N_DAYS)
    steps21 = {k: np.full(21, v[0]) for k, v in steps1.items()}
    paths21 = simulate_heston_q(steps21, 100.0, N_PATHS, SEED + 1, n_days=N_DAYS)
    return {"one_step": q_mean_check(paths1, r_ann, N_DAYS, 1), "days_21": q_mean_check(paths21, r_ann, N_DAYS, 21)}


def check_merton(p, q, r_ann) -> dict:
    steps1 = {
        "rf": _full(1, r_ann), "sigma": _full(1, p["sigma"]), "lam": _full(1, p["lam"]),
        "mu_j": _full(1, p["mu_j"]), "mu_j_q": _full(1, q["mu_j_q"]),
        "sigma_j": _full(1, p["sigma_j"]), "kappa": _full(1, p["kappa"]),
        "kappa_q": _full(1, q["kappa_q"]),
    }
    paths1 = simulate_merton_q(steps1, 100.0, N_PATHS, SEED, n_days=N_DAYS)
    steps21 = {k: np.full(21, v[0]) for k, v in steps1.items()}
    paths21 = simulate_merton_q(steps21, 100.0, N_PATHS, SEED + 1, n_days=N_DAYS)
    return {"one_step": q_mean_check(paths1, r_ann, N_DAYS, 1), "days_21": q_mean_check(paths21, r_ann, N_DAYS, 21)}


def check_bates(p, q, r_ann) -> dict:
    v0 = float(q["v0_q"]) if np.isfinite(q.get("v0_q", np.nan)) else float(p["v0"])
    steps1 = {
        "rf": _full(1, r_ann), "kappa": _full(1, p["kappa"]), "theta": _full(1, p["theta"]),
        "xi": _full(1, p["xi"]), "rho": _full(1, p["rho"]), "v0": _full(1, v0),
        "v0_q": _full(1, v0), "eta_v": _full(1, q["eta_v"]),
        "lam": _full(1, p["lam"]), "mu_j": _full(1, p["mu_j"]),
        "mu_j_q": _full(1, q["mu_j_q"]), "sigma_j": _full(1, p["sigma_j"]),
        "kappa_j": _full(1, p["kappa_j"]), "kappa_j_q": _full(1, q["kappa_j_q"]),
    }
    paths1 = simulate_bates_q(steps1, 100.0, N_PATHS, SEED, n_days=N_DAYS)
    steps21 = {k: np.full(21, v[0]) for k, v in steps1.items()}
    paths21 = simulate_bates_q(steps21, 100.0, N_PATHS, SEED + 1, n_days=N_DAYS)
    return {"one_step": q_mean_check(paths1, r_ann, N_DAYS, 1), "days_21": q_mean_check(paths21, r_ann, N_DAYS, 21)}


def _print_block(title, p_tbl, prem, q_tbl, chk):
    print(f"\n===== {title} =====")
    print("P")
    print(pd.Series(p_tbl).to_string())
    print("\nRisk premium")
    print(pd.Series(prem).to_string())
    print("\nQ")
    print(pd.Series(q_tbl).to_string())
    for name, row in chk.items():
        print(
            f"\n{name}: E[S_T/S_0]={row['E_S']:.6f}  vs  exp(rT)={row['exp_rf']:.6f}  "
            f"rel err={row['mean_rel_err']:.4%}"
        )


def _pass(chk, lim=0.03) -> bool:
    return all(abs(row["mean_rel_err"]) <= lim for row in chk.values())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rets, quotes, r_ann = _window()
    print(f"SPY window {rets.index.min().date()} → {rets.index.max().date()}  n={len(rets)}")
    print(f"quotes={len(quotes)}  r_ann={r_ann:.4f}  paths={N_PATHS}")

    hp = estimate_heston_p(rets, N_DAYS)
    hq = calibrate_heston_q(hp, quotes)
    h_chk = check_heston(hp, hq, r_ann)
    h_p, h_prem, h_q = report_heston_pq({**hp, **hq})
    _print_block("Heston (Pan η_v)", h_p, h_prem, h_q, h_chk)

    mp = estimate_merton_p(rets, N_DAYS)
    mq = calibrate_merton_q(mp, quotes)
    m_chk = check_merton(mp, mq, r_ann)
    m_p, m_prem, m_q = report_merton_pq({**mp, **mq})
    _print_block("Merton (Pan μ*)", m_p, m_prem, m_q, m_chk)

    bp = estimate_bates_p(rets, N_DAYS)
    bq = calibrate_bates_q(bp, quotes)
    b_chk = check_bates(bp, bq, r_ann)
    b_p, b_prem, b_q = report_bates_pq({**bp, **bq})
    _print_block("Heston–Merton / Bates (Pan η_v and μ*)", b_p, b_prem, b_q, b_chk)

    payload = {
        "window": {"start": str(rets.index.min().date()), "end": str(rets.index.max().date()), "n": int(len(rets))},
        "n_quotes": int(len(quotes)),
        "r_ann": r_ann,
        "heston": {"P": h_p, "premium": h_prem, "Q": h_q, "check": h_chk, "q_fit": hq},
        "merton": {"P": m_p, "premium": m_prem, "Q": m_q, "check": m_chk, "q_fit": mq},
        "bates": {"P": b_p, "premium": b_prem, "Q": b_q, "check": b_chk, "q_fit": bq},
    }
    dest = OUT / "spy_1p5y_pq_check.json"
    dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {dest}")

    ok = _pass(h_chk) and _pass(m_chk) and _pass(b_chk)
    fitted = hq.get("success") and mq.get("success") and bq.get("success")
    if not fitted:
        print("FAIL: a Q premium was not identified from option quotes")
        return 1
    invented = (
        not np.isfinite(hq.get("eta_v", np.nan))
        or (mp["lam"] > 1e-12 and not mq.get("jump_premium_identified", False))
    )
    if invented:
        print("FAIL: a risk premium looks unidentified / invented")
        return 1
    if not ok:
        print("FAIL: Q mean check rel err > 3%")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
