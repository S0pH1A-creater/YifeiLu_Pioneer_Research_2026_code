#!/usr/bin/env python3
"""Order-3 sign-chain and rolling-mode check for Modified GBM v3. No LSM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import build_modified_gbm_v3_notebooks as bld  # noqa: E402
import run_optimal_stopping_study as os_study  # noqa: E402

P_KEYS = (
    "p_u_ddd",
    "p_u_ddu",
    "p_u_dud",
    "p_u_duu",
    "p_u_udd",
    "p_u_udu",
    "p_u_uud",
    "p_u_uuu",
)


def _exec_estimate():
    ns = {"np": np, "pd": pd}
    exec(bld.ESTIMATE_FN, ns)
    return ns["estimate_modified_gbm"]


def _exec_simulate():
    ns = {"np": np, "N_DAYS": 252}
    exec(bld.SIMULATE_FN, ns)
    return ns["simulate_modified_gbm_rolling"]


def check_run_persistence() -> None:
    ups = np.full(80, 0.01)
    downs = np.full(80, -0.01)
    r = pd.Series(np.concatenate([ups, downs, ups]))
    est = _exec_estimate()(r)
    if est is None:
        raise RuntimeError("estimate_modified_gbm returned None")
    missing = [k for k in (*P_KEYS, "mu_u", "sig_u", "mu_d", "sig_d", "last_state") if k not in est]
    if missing:
        raise RuntimeError(f"missing keys: {missing}")
    if not (0.0 <= float(est["last_state"]) <= 7.0):
        raise RuntimeError(f"last_state out of range: {est['last_state']}")
    if abs(float(est["last_state"]) - 7.0) > 1e-9:
        raise RuntimeError(f"expected last_state=UUU=7, got {est['last_state']}")
    if float(est["p_u_uuu"]) < 0.95:
        raise RuntimeError(f"P(U|UUU) too low on a long up run: {est['p_u_uuu']}")
    if float(est["p_u_ddd"]) > 0.05:
        raise RuntimeError(f"P(U|DDD) too high on a long down run: {est['p_u_ddd']}")
    print(f"  P(U|UUU)={est['p_u_uuu']:.3f}  P(U|DDD)={est['p_u_ddd']:.3f}  last_state={est['last_state']:.0f} ok")


def check_simulate_stays_in_states() -> None:
    est = {k: 0.5 for k in P_KEYS}
    est["p_u_uuu"] = 1.0
    est["p_u_ddd"] = 0.0
    est.update({"mu_u": 0.01, "sig_u": 0.002, "mu_d": 0.01, "sig_d": 0.002, "last_state": 7.0})
    n_steps = 40
    steps = {k: np.full(n_steps, float(v)) for k, v in est.items()}
    sim = _exec_simulate()
    paths = sim(steps, 100.0, 300, 1)
    rets = np.diff(np.log(paths), axis=1)
    if np.any(~np.isfinite(rets)):
        raise RuntimeError("non-finite simulated returns")
    if np.any(np.abs(rets) <= 0):
        raise RuntimeError("zero simulated sizes")
    if float(np.min(rets)) <= 0:
        raise RuntimeError("expected all-up paths from last_state=UUU and P(U|UUU)=1")
    print(f"  simulate: paths={paths.shape} all-up from UUU ok")


def check_notebook_none() -> None:
    nb = bld.DST_DIR / "2008-2009_modified_gbm_v3.ipynb"
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
    st = float(row["last_state"])
    if st < 0 or st > 7:
        raise RuntimeError(f"last_state not in 0..7: {st}")
    for k in P_KEYS:
        p = float(row[k])
        if not (0.0 <= p <= 1.0):
            raise RuntimeError(f"{k}={p} not a probability")
    monthly = g["calibrate_ticker"](ticker, "1.5 years", "monthly")
    if monthly is None or len(monthly) < 2:
        raise RuntimeError(f"expected several monthly updates, got {0 if monthly is None else len(monthly)}")
    print(f"  {ticker} none updates={len(cal)} monthly updates={len(monthly)} ok")


def main() -> int:
    print("Order-3 coins on a U/D/U block")
    check_run_persistence()
    print("Simulator")
    check_simulate_stays_in_states()
    print("Notebook calibrate_ticker")
    check_notebook_none()
    print("Modified GBM v3 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
