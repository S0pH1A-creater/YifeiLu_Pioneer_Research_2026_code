#!/usr/bin/env python3
"""Compare Heston path / LSM / OS results before vs after the Euler-order fix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BEFORE = ROOT / "results" / "euler_fix_compare" / "before"
AFTER_PATH = REPO / "research" / "data" / "equity" / "synthetic" / "heston_10y.csv"
AFTER_METRICS = ROOT / "results" / "synthetic_heston" / "metrics.json"
OUT = ROOT / "results" / "euler_fix_compare" / "compare.json"


def _path_stats(csv: Path) -> dict:
    df = pd.read_csv(csv, parse_dates=["date"])
    r = df["log_return"].dropna().to_numpy(dtype=float)
    v = df["v_t"].to_numpy(dtype=float)
    s = df["S_t"].to_numpy(dtype=float)
    dv = np.diff(v)
    return {
        "S_T": float(s[-1]),
        "v_T": float(v[-1]),
        "mu_hat": float(r.mean() * 252.0),
        "realized_vol": float(r.std(ddof=1) * np.sqrt(252.0)),
        "mean_sqrt_v": float(np.sqrt(v).mean()),
        "leverage_corr_r_dv": float(np.corrcoef(r, dv)[0, 1]),
        "acf1_r2": float(np.corrcoef(r[1:] ** 2, r[:-1] ** 2)[0, 1]),
    }


def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _os_metrics(csv: Path) -> dict:
    df = pd.read_csv(csv)
    err = df["error"].to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "early": float(df["early_ex_frac"].mean()),
        "n": int(len(df)),
    }


def main() -> int:
    before_path = _path_stats(BEFORE / "path" / "heston_10y.csv")
    after_path = _path_stats(AFTER_PATH)
    before_m = json.loads((BEFORE / "synthetic_heston" / "metrics.json").read_text())
    after_m = json.loads(AFTER_METRICS.read_text())

    def _lsm_map(payload):
        return {row["model"]: row for row in payload["lsm"]}

    b_lsm, a_lsm = _lsm_map(before_m), _lsm_map(after_m)
    lsm_cmp = []
    for model in ("Heston", "Heston–Merton", "GBM", "Merton", "GARCH", "GARCH–Merton"):
        if model not in b_lsm or model not in a_lsm:
            continue
        lsm_cmp.append({
            "model": model,
            "rmse_before": b_lsm[model]["rmse_lsm"],
            "rmse_after": a_lsm[model]["rmse_lsm"],
            "bias_before": b_lsm[model]["bias_lsm"],
            "bias_after": a_lsm[model]["bias_lsm"],
            "early_before": b_lsm[model]["early"],
            "early_after": a_lsm[model]["early"],
        })

    os_rows = []
    before_os = BEFORE / "os"
    if before_os.is_dir():
        for old in before_os.rglob("*_contracts.csv"):
            rel = old.relative_to(before_os)
            new = ROOT / "results" / rel
            if not new.is_file():
                continue
            parts = rel.parts  # ticker/figures/stem/mode_contracts.csv
            if parts[0] == "figures":
                continue
            ticker = parts[0]
            stem = parts[-2]
            mode = old.stem.replace("_contracts", "")
            b, a = _os_metrics(old), _os_metrics(new)
            os_rows.append({
                "ticker": ticker,
                "stem": stem,
                "mode": mode,
                "rmse_before": b["rmse"],
                "rmse_after": a["rmse"],
                "mae_before": b["mae"],
                "mae_after": a["mae"],
                "bias_before": b["bias"],
                "bias_after": a["bias"],
                "early_before": b["early"],
                "early_after": a["early"],
            })
    os_rows.sort(key=lambda r: (r["stem"], r["ticker"], r["mode"]))

    payload = {
        "path": {"before": before_path, "after": after_path},
        "synthetic_lsm": lsm_cmp,
        "synthetic_S_T": {"before": before_m["S_T"], "after": after_m["S_T"]},
        "os": os_rows,
        "os_n": len(os_rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print("PATH")
    for k in before_path:
        print(f"  {k:20s}  {before_path[k]: .6f}  →  {after_path[k]: .6f}")
    print("SYNTHETIC LSM")
    for row in lsm_cmp:
        print(f"  {row['model']:16s} RMSE {row['rmse_before']:.4f} → {row['rmse_after']:.4f}  "
              f"bias {row['bias_before']:+.4f} → {row['bias_after']:+.4f}")
    if os_rows:
        print(f"OS ({len(os_rows)} tables)")
        for row in os_rows:
            print(f"  {row['ticker']:4s} {row['stem']:28s} {row['mode']:8s} "
                  f"RMSE {row['rmse_before']:.4f} → {row['rmse_after']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
