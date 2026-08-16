#!/usr/bin/env python3
"""One-path 10-year daily Heston synthetic equity (for calibration / LSM tests).

Euler + full truncation matches the V3 Heston notebooks: the stock
increment uses the current v_t, then variance is updated. Drift is the
risk-free rate (Q-measure), which is the right input for American pricing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "data" / "equity" / "synthetic" / "heston_10y.csv"

S0 = 100.0
V0 = 0.04
R = 0.05
KAPPA = 2.0
THETA = 0.04
XI = 0.3
RHO = -0.7
DT = 1.0 / 252.0
N_YEARS = 10
N_STEPS = N_YEARS * 252
SEED = 42
START = pd.Timestamp("2014-01-02")


def simulate_heston_path(
    *,
    s0: float = S0,
    v0: float = V0,
    r: float = R,
    kappa: float = KAPPA,
    theta: float = THETA,
    xi: float = XI,
    rho: float = RHO,
    dt: float = DT,
    n_steps: int = N_STEPS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rho, -0.999, 0.999))
    s = np.empty(n_steps + 1, dtype=float)
    v = np.empty(n_steps + 1, dtype=float)
    s[0] = s0
    v[0] = v0
    vt = float(v0)

    for i in range(n_steps):
        z_v = float(rng.standard_normal())
        z_s = rho * z_v + np.sqrt(max(1.0 - rho * rho, 0.0)) * float(rng.standard_normal())
        v_pos = max(vt, 0.0)
        s[i + 1] = s[i] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s)
        vt = vt + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * z_v
        vt = max(vt, 0.0)
        v[i + 1] = vt
    return s, v


def main() -> int:
    s, v = simulate_heston_path()
    dates = pd.bdate_range(START, periods=len(s), freq="B")
    log_ret = np.concatenate([[np.nan], np.diff(np.log(s))])
    df = pd.DataFrame(
        {
            "date": dates,
            "S_t": s,
            "v_t": v,
            "log_return": log_ret,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.10g")

    r = log_ret[1:]
    dv = np.diff(v)
    sq = r * r
    acf1 = float(np.corrcoef(sq[1:], sq[:-1])[0, 1])
    lev = float(np.corrcoef(r, dv)[0, 1])
    print(f"wrote {OUT}")
    print(f"rows={len(df)}  {dates[0].date()} → {dates[-1].date()}")
    print(f"S_T={s[-1]:.4f}  mean sqrt(v)={np.sqrt(v).mean():.4f}  "
          f"realized vol={r.std(ddof=1)*np.sqrt(252):.4f}")
    print(f"vol clustering acf1(r^2)={acf1:.3f}  leverage corr(r,Δv)={lev:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
