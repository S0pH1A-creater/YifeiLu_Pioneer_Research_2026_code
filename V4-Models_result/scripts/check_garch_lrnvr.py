#!/usr/bin/env python3
"""One-step LRNVR check: E^Q[S_t/S_{t-1}] = exp(r Δt) and Var^Q[log return] = h."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from garch_duan_lrnvr import (  # noqa: E402
    estimate_garch_params,
    load_rf_annual,
    lrnvr_one_step_check,
    report_p_and_q,
)

REPO = SCRIPTS.parents[1]
DATA = REPO / "research" / "data"
OUT = SCRIPTS.parents[0] / "results" / "garch_lrnvr_check"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(DATA / "equity" / "prices_clean.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    end = pd.Timestamp("2018-09-28")
    start = end - pd.DateOffset(months=18)
    rets = np.log(prices["SPY"]).diff().loc[start:end].dropna()
    rf = load_rf_annual(DATA)
    hat = estimate_garch_params(rets, rf, n_days=252, min_window=60)
    if hat is None:
        raise SystemExit("GARCH-in-mean MLE failed on SPY 1.5y lookback")
    p_tbl, q_tbl = report_p_and_q(hat)
    rf_mean = float(hat["rf_mean"])
    chk = lrnvr_one_step_check(
        lam=hat["lambda"],
        omega=hat["omega"],
        alpha=hat["alpha"],
        beta=hat["beta"],
        sigma0=hat["sigma0"],
        r_annual=rf_mean,
        n_days=252,
        n_paths=250_000,
        seed=42,
    )
    payload = {
        "window": {"start": str(rets.index.min().date()), "end": str(rets.index.max().date()), "n": hat["n"]},
        "P": p_tbl,
        "Q": q_tbl,
        "lrnvr_check": chk,
        "mean_rel_err": chk["mean_err"] / chk["exp_rf"],
        "var_rel_err": chk["var_err"] / chk["h"],
    }
    path = OUT / "spy_1p5y_one_step.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("P-measure parameters")
    print(pd.Series(p_tbl).to_string())
    print("\nQ-dynamics (Duan LRNVR)")
    print(pd.Series(q_tbl).to_string())
    print("\nOne-step LRNVR check (250k paths)")
    print(
        f"  E[S_t/S_{{t-1}}] = {chk['E_S']:.8f}  vs  exp(rΔt) = {chk['exp_rf']:.8f}  "
        f"rel err = {payload['mean_rel_err']:.4%}"
    )
    print(
        f"  Var[ln(S_t/S_{{t-1}})] = {chk['Var_log']:.8e}  vs  h = {chk['h']:.8e}  "
        f"rel err = {payload['var_rel_err']:.4%}"
    )
    print(f"wrote {path}")
    if abs(payload["mean_rel_err"]) > 0.02 or abs(payload["var_rel_err"]) > 0.05:
        print("FAIL: LRNVR one-step errors too large", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
