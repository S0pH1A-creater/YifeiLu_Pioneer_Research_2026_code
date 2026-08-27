#!/usr/bin/env python3
"""Moment-match and rolling-mode check for Modified GBM v2. No LSM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import build_modified_gbm_v2_notebooks as bld  # noqa: E402
import run_optimal_stopping_study as os_study  # noqa: E402


def _exec_estimate():
    ns = {"np": np, "pd": pd}
    exec(bld.ESTIMATE_FN, ns)
    return ns["estimate_modified_gbm"]


def _exec_simulate():
    ns = {"np": np, "N_DAYS": 252}
    exec(bld.SIMULATE_FN, ns)
    return ns["simulate_modified_gbm_rolling"]


def _lognormal_mean(mu: float, sig: float) -> float:
    return float(np.exp(mu + 0.5 * sig * sig))


def check_way_b_identity() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    # Two size levels so median split is clean.
    small = rng.lognormal(mean=np.log(0.005), sigma=0.25, size=n // 2)
    large = rng.lognormal(mean=np.log(0.020), sigma=0.25, size=n - n // 2)
    mag = np.concatenate([small, large])
    signs = rng.choice([-1.0, 1.0], size=n)
    r = pd.Series(signs * mag)
    est = _exec_estimate()(r)
    if est is None:
        raise RuntimeError("estimate_modified_gbm returned None")
    needed = (
        "p_uu",
        "p_dd",
        "p_ud",
        "p_du",
        "p_hh",
        "p_ll",
        "mu_u_l",
        "sig_u_l",
        "mu_u_h",
        "sig_u_h",
        "mu_d_l",
        "sig_d_l",
        "mu_d_h",
        "sig_d_h",
        "last_up",
        "last_wild",
    )
    missing = [k for k in needed if k not in est]
    if missing:
        raise RuntimeError(f"missing keys: {missing}")

    x = r.dropna().to_numpy()
    x = x[x != 0.0]
    mag_x = np.abs(x)
    up = x > 0.0
    wild = mag_x > float(np.median(mag_x))
    buckets = {
        "u_l": (up & ~wild, est["mu_u_l"], est["sig_u_l"]),
        "u_h": (up & wild, est["mu_u_h"], est["sig_u_h"]),
        "d_l": ((~up) & ~wild, est["mu_d_l"], est["sig_d_l"]),
        "d_h": ((~up) & wild, est["mu_d_h"], est["sig_d_h"]),
    }
    for name, (mask, mu, sig) in buckets.items():
        arr = mag_x[mask]
        if arr.size < 2:
            continue
        m = float(arr.mean())
        implied = _lognormal_mean(mu, sig)
        rel = abs(implied - m) / m
        if rel > 1e-10:
            raise RuntimeError(f"{name}: E[size]={implied:.6g} vs m={m:.6g} rel={rel:.3g}")
        draws = np.exp(rng.normal(mu, sig, size=80_000))
        if float(np.min(draws)) <= 0:
            raise RuntimeError(f"{name}: drew a non-positive size")
        draw_mean = float(np.mean(draws))
        rel_mc = abs(draw_mean - m) / m
        if rel_mc > 0.03:
            raise RuntimeError(f"{name}: MC mean {draw_mean:.6g} vs m={m:.6g} rel={rel_mc:.3g}")
        print(f"  {name}: n={arr.size} m={m:.5f} implied={implied:.5f} mc={draw_mean:.5f} ok")


def _q_steps():
    est = {
        "p_uu": 0.55,
        "p_ud": 0.48,
        "p_hh": 0.60,
        "p_ll": 0.60,
        "mu_u_l": np.log(0.005) - 0.5 * 0.04,
        "sig_u_l": 0.2,
        "mu_u_h": np.log(0.02) - 0.5 * 0.04,
        "sig_u_h": 0.2,
        "mu_d_l": np.log(0.005) - 0.5 * 0.04,
        "sig_d_l": 0.2,
        "mu_d_h": np.log(0.02) - 0.5 * 0.04,
        "sig_d_h": 0.2,
        "last_up": 1.0,
        "last_wild": 0.0,
    }
    n_steps = 25
    return {k: np.full(n_steps, float(v)) for k, v in est.items()}


def check_simulate_positive() -> None:
    steps = _q_steps()
    sim = _exec_simulate()
    paths = sim(steps, 100.0, 200, 1)
    rets = np.diff(np.log(paths), axis=1)
    if np.any(~np.isfinite(rets)):
        raise RuntimeError("non-finite simulated returns")
    if np.any(np.abs(rets) <= 0):
        raise RuntimeError("zero or negative simulated sizes")
    print(f"  simulate: paths={paths.shape} min|R|={np.min(np.abs(rets)):.3e} ok")


def check_size_only_q() -> None:
    steps = _q_steps()
    sim = _exec_simulate()
    rf = 0.04
    p_paths = sim(steps, 100.0, 4000, 7, rf=None)
    q_paths = sim(steps, 100.0, 4000, 7, rf=rf)
    r_p = np.diff(np.log(p_paths), axis=1)
    r_q = np.diff(np.log(q_paths), axis=1)
    if np.any(np.sign(r_p) != np.sign(r_q)):
        n_flip = int(np.sum(np.sign(r_p) != np.sign(r_q)))
        raise RuntimeError(f"size-only Q flipped {n_flip} signs")
    target = float(np.exp(rf / 252.0))
    for j in range(r_q.shape[1]):
        mx = float(np.mean(np.exp(r_q[:, j])))
        if abs(np.log(mx) - np.log(target)) > 1e-10:
            raise RuntimeError(f"step {j}: E[e^R]={mx:.12g} vs e^{{rf Δt}}={target:.12g}")
    # Additive Q would flip some small down moves. Confirm P and Q share signs
    # and |R^Q| / |R^P| is one scale per step.
    scales = np.median(np.abs(r_q) / np.maximum(np.abs(r_p), 1e-16), axis=0)
    rel = np.abs(np.abs(r_q) - scales * np.abs(r_p)) / np.maximum(np.abs(r_q), 1e-16)
    if float(np.max(rel)) > 1e-8:
        raise RuntimeError("Q sizes are not a single per-step scale of P sizes")
    print(f"  size-only Q: no sign flips, E[e^R]=e^{{rf Δt}}, median λ={float(np.median(scales)):.3f} ok")


def check_notebook_none() -> None:
    nb = (
        bld.DST_DIR / "2008-2009_modified_gbm_v2.ipynb"
    )
    os_study._install_notebook_stubs()
    g = os_study._load_ns(nb)
    wo = g.get("WINDOW_OPTIONS")
    if isinstance(wo, dict) and "1.5 years" not in wo:
        wo["1.5 years"] = pd.DateOffset(months=18)
    ticker = "SPY" if "SPY" in g["TICKERS"] else g["TICKERS"][0]
    cal = g["calibrate_ticker"](ticker, "1.5 years", "none")
    if cal is None or len(cal) != 1:
        raise RuntimeError(f"expected 1 update under rolling=none, got {0 if cal is None else len(cal)}")
    row = cal.iloc[0]
    if float(row["last_up"]) not in (0.0, 1.0) or float(row["last_wild"]) not in (0.0, 1.0):
        raise RuntimeError("last_up / last_wild not 0/1")
    monthly = g["calibrate_ticker"](ticker, "1.5 years", "monthly")
    if monthly is None or len(monthly) < 2:
        raise RuntimeError(f"expected several monthly updates, got {0 if monthly is None else len(monthly)}")
    print(f"  {ticker} none updates={len(cal)} monthly updates={len(monthly)} ok")


def main() -> int:
    print("Way B identity + positive draws")
    check_way_b_identity()
    print("Simulator")
    check_simulate_positive()
    print("Size-only Q")
    check_size_only_q()
    print("Notebook calibrate_ticker")
    check_notebook_none()
    print("Modified GBM v2 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
