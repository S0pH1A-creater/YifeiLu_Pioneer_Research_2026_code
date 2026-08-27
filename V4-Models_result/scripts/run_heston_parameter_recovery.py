#!/usr/bin/env python3
"""Heston parameter-recovery study (implementation check).

1. Simulate daily stock prices / returns from known Heston parameters.
2. Hide the true parameters (and, for Method A, the latent variance path).
3. Recalibrate from the simulated data only.
4. Repeat over independent random seeds.
5. Report absolute / relative errors and whether recovery indicates a bug.

Estimators
----------
- method_a: V1 realized-variance moments from returns only (r_t, r_t²).
- oracle_cir: CIR Euler OLS on the hidden variance path (SDE diagnostic).
- option_nls: V3 Fourier NLS on a synthetic call surface priced at the
  true parameters, started from the production default x0 (not the truth).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_heston_synthetic import simulate_heston_path  # noqa: E402
from heston_option_calibration import (  # noqa: E402
    _LO,
    _HI,
    _default_x0,
    calibrate_heston_from_quotes,
    heston_call_prices,
)

OUT = ROOT / "results" / "heston_parameter_recovery"

TRUE = dict(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
S0 = 100.0
R = 0.05
DT = 1.0 / 252.0
N_DAYS = 252
N_STEPS = 10 * N_DAYS
SEEDS = (42, 7, 123, 2024, 99, 314, 2718, 8675309)
PARAM_KEYS = ("mu", "kappa", "theta", "xi", "rho", "v0")
MONEYNESS = (0.85, 0.92, 1.00, 1.08, 1.15)
DTE = (21, 63, 126, 252)
PRICE_NOISE = 0.005  # 50 bp relative quote noise for option_nls


def _rel(hat: float, tru: float) -> float:
    den = abs(tru) if abs(tru) > 1e-12 else 1.0
    return float((hat - tru) / den)


def _err_row(hat: dict, tru: dict) -> dict:
    out = {}
    for k in PARAM_KEYS:
        if k not in hat or not np.isfinite(hat[k]):
            out[f"{k}_abs"] = np.nan
            out[f"{k}_rel"] = np.nan
            continue
        out[f"{k}_abs"] = float(hat[k] - tru[k])
        out[f"{k}_rel"] = _rel(float(hat[k]), float(tru[k]))
    return out


def estimate_method_a(log_rets: np.ndarray) -> dict:
    """V1 Method A: moments of r_t and r_t². No access to v_t or true params."""
    x = pd.Series(np.asarray(log_rets, dtype=float)).dropna()
    n = int(x.shape[0])
    nan = {k: np.nan for k in PARAM_KEYS}
    if n < 60:
        return {**nan, "n": n, "success": False}

    sigma_day = float(x.std(ddof=1))
    if not np.isfinite(sigma_day) or sigma_day <= 0:
        return {**nan, "n": n, "success": False}

    mu = float(x.mean() * N_DAYS)
    rv = x**2
    theta = float(rv.mean() * N_DAYS)
    recent = rv.iloc[-min(21, n) :]
    v0 = float(recent.mean() * N_DAYS)
    if not np.isfinite(theta) or theta <= 0:
        return {**nan, "n": n, "success": False}
    if not np.isfinite(v0) or v0 <= 0:
        v0 = theta

    rho1 = rv.autocorr(lag=1)
    if rho1 is None or not np.isfinite(rho1) or rho1 <= 1e-6:
        kappa = 2.0
    elif rho1 >= 0.999:
        kappa = 0.05
    else:
        kappa = float(-np.log(float(rho1)) / DT)
    kappa = float(np.clip(kappa, 0.05, 20.0))

    v_ann = (rv * N_DAYS).astype(float)
    dv = v_ann.diff().dropna()
    v_lag = v_ann.loc[dv.index]
    drift = kappa * (theta - v_lag.values) * DT
    resid = dv.values - drift
    mean_v = float(np.mean(v_lag.values))
    var_resid = float(np.var(resid, ddof=1)) if resid.size >= 2 else np.nan
    if mean_v > 0 and np.isfinite(var_resid) and var_resid > 0:
        xi = float(np.sqrt(var_resid / (mean_v * DT)))
    else:
        xi = 0.5
    xi = float(np.clip(xi, 0.05, 3.0))

    aligned = pd.concat([x.rename("r"), v_ann.diff().rename("dv")], axis=1).dropna()
    if len(aligned) >= 5:
        rho = float(aligned["r"].corr(aligned["dv"]))
    else:
        rho = -0.5
    if not np.isfinite(rho):
        rho = -0.5
    rho = float(np.clip(rho, -0.99, 0.99))

    return {
        "mu": mu,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "v0": v0,
        "n": n,
        "success": True,
        "hit_bound": bool(kappa in (0.05, 20.0) or xi in (0.05, 3.0)),
    }


def estimate_oracle_cir(x: np.ndarray, v: np.ndarray) -> dict:
    """CIR Euler OLS on the latent variance path (not available in production)."""
    mu = float(np.mean(x) * N_DAYS)
    dv = np.diff(v)
    vlag = v[:-1]
    design = np.column_stack([np.ones(len(vlag)), vlag])
    coef, *_ = np.linalg.lstsq(design, dv, rcond=None)
    kappa = float(-coef[1] / DT)
    kappa = float(np.clip(kappa, 1e-6, 20.0))
    theta = float(coef[0] / (kappa * DT)) if kappa > 1e-8 else float(np.mean(v))
    theta = float(max(theta, 1e-6))
    resid = dv - (coef[0] + coef[1] * vlag)
    scale = np.sqrt(np.maximum(vlag, 1e-8) * DT)
    xi = float(np.sqrt(np.mean((resid / scale) ** 2)))
    xi = float(np.clip(xi, 1e-6, 4.0))
    z_v = resid / (xi * scale)
    # x is the return series (len = len(v) - 1), aligned with each Euler step.
    z_s = (x - (mu - 0.5 * vlag) * DT) / np.sqrt(np.maximum(vlag, 1e-8) * DT)
    rho = float(np.corrcoef(z_s, z_v)[0, 1])
    if not np.isfinite(rho):
        rho = -0.5
    rho = float(np.clip(rho, -0.99, 0.99))
    return {
        "mu": mu,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "v0": float(max(v[0], 1e-6)),
        "success": True,
        "hit_bound": bool(abs(kappa - 20.0) < 1e-12 or abs(xi - 4.0) < 1e-12),
    }


def _synthetic_quotes(true_p: dict, *, noise_seed: int | None) -> pd.DataFrame:
    rows = []
    for dte in DTE:
        T = dte / 365.0
        for m in MONEYNESS:
            K = S0 * m
            rows.append({"S_t": S0, "K": K, "T_years": T, "dte": dte, "r": R, "moneyness": m})
    q = pd.DataFrame(rows)
    prices = heston_call_prices(
        q["S_t"], q["K"], q["T_years"], q["r"],
        true_p["kappa"], true_p["theta"], true_p["xi"], true_p["rho"], true_p["v0"],
    )
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        prices = prices * (1.0 + PRICE_NOISE * rng.standard_normal(len(prices)))
        intrinsic = np.maximum(S0 - q["K"].to_numpy() * np.exp(-R * q["T_years"].to_numpy()), 0.0)
        prices = np.maximum(prices, intrinsic + 0.01)
    q["option_price"] = prices
    q["trading_date"] = pd.Timestamp("2024-01-02")
    return q


def estimate_option_nls(true_p: dict, *, noise_seed: int | None) -> dict:
    """V3 option-implied NLS. Optimizer is not given the true parameters."""
    quotes = _synthetic_quotes(true_p, noise_seed=noise_seed)
    x0 = _default_x0(quotes)
    # Sanity: model prices at the truth should fit the (noiseless) surface.
    truth_fit = heston_call_prices(
        quotes["S_t"], quotes["K"], quotes["T_years"], quotes["r"],
        true_p["kappa"], true_p["theta"], true_p["xi"], true_p["rho"], true_p["v0"],
    )
    truth_rmse = float(np.sqrt(np.mean((truth_fit - quotes["option_price"]) ** 2)))
    cal = calibrate_heston_from_quotes(quotes, x0=x0, max_nfev=120)
    cal["mu"] = float(true_p["mu"])  # μ is P-measure; not identified by Q-quotes
    cal["x0"] = [float(v) for v in x0]
    cal["truth_rmse"] = truth_rmse
    cal["n_quotes"] = int(len(quotes))
    cal["hit_bound"] = any(
        abs(cal[k] - lo) < 1e-8 or abs(cal[k] - hi) < 1e-8
        for k, lo, hi in zip(("kappa", "theta", "xi", "rho", "v0"), _LO, _HI)
    )
    return cal


def _summarize(rows: list[dict], prefix: str) -> dict:
    summary = {}
    for k in PARAM_KEYS:
        hats = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))], dtype=float)
        if hats.size == 0:
            continue
        tru = float(TRUE[k])
        abs_err = hats - tru
        rel_err = abs_err / (abs(tru) if abs(tru) > 1e-12 else 1.0)
        summary[k] = {
            "true": tru,
            "mean": float(hats.mean()),
            "median": float(np.median(hats)),
            "std": float(hats.std(ddof=1)) if hats.size > 1 else 0.0,
            "bias": float(abs_err.mean()),
            "mae": float(np.mean(np.abs(abs_err))),
            "rmse": float(np.sqrt(np.mean(abs_err**2))),
            "mape": float(np.mean(np.abs(rel_err))),
            "median_abs_rel": float(np.median(np.abs(rel_err))),
        }
    summary["n_seeds"] = len(rows)
    summary["n_success"] = int(sum(1 for r in rows if r.get("success")))
    summary["n_hit_bound"] = int(sum(1 for r in rows if r.get("hit_bound")))
    return summary


def _verdict(summary: dict, *, tight: dict, loose: dict) -> str:
    """tight/loose are median |rel err| thresholds per parameter."""
    flags = []
    for k, lim in tight.items():
        if k not in summary:
            continue
        med = summary[k]["median_abs_rel"]
        if med > loose.get(k, 0.80):
            flags.append(f"{k}:not-recovered")
        elif med > lim:
            flags.append(f"{k}:biased")
    if not flags:
        return "recovered"
    if any(s.endswith("not-recovered") for s in flags):
        return "not-recovered"
    return "partial"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    method_a_rows, oracle_rows, option_rows = [], [], []

    print("True parameters (hidden from calibrators):")
    print("  " + "  ".join(f"{k}={TRUE[k]}" for k in PARAM_KEYS), flush=True)
    print(f"  Feller 2κθ−ξ² = {2 * TRUE['kappa'] * TRUE['theta'] - TRUE['xi'] ** 2:.4f}")
    print(f"  path: {N_STEPS} daily steps, {len(SEEDS)} seeds\n", flush=True)

    for seed in SEEDS:
        s, v = simulate_heston_path(
            s0=S0, v0=TRUE["v0"], r=TRUE["mu"],
            kappa=TRUE["kappa"], theta=TRUE["theta"], xi=TRUE["xi"], rho=TRUE["rho"],
            dt=DT, n_steps=N_STEPS, seed=int(seed),
        )
        log_ret = np.diff(np.log(s))

        ma = estimate_method_a(log_ret)
        ma.update(_err_row(ma, TRUE))
        ma["seed"] = int(seed)
        method_a_rows.append(ma)

        oc = estimate_oracle_cir(log_ret, v)
        oc.update(_err_row(oc, TRUE))
        oc["seed"] = int(seed)
        oracle_rows.append(oc)

        op = estimate_option_nls(TRUE, noise_seed=int(seed))
        op.update(_err_row(op, TRUE))
        op["seed"] = int(seed)
        option_rows.append(op)

        def _fmt(d, keys=PARAM_KEYS):
            return "  ".join(f"{k}={d[k]:.4f}" for k in keys if np.isfinite(d.get(k, np.nan)))

        print(f"seed {seed}")
        print(f"  method_a   {_fmt(ma)}")
        print(f"  oracle_cir {_fmt(oc)}")
        print(f"  option_nls {_fmt(op)}  rmse={op.get('rmse', np.nan):.4f}  "
              f"truth_rmse={op.get('truth_rmse', np.nan):.4f}", flush=True)

    # Exact (zero-noise) option surface, still started away from the truth.
    exact = estimate_option_nls(TRUE, noise_seed=None)
    exact.update(_err_row(exact, TRUE))
    exact["seed"] = None
    print("\noption_nls exact surface (no quote noise)")
    print("  " + "  ".join(f"{k}={exact[k]:.4f}" for k in PARAM_KEYS if k != "mu"))
    print(f"  rmse={exact['rmse']:.6f}  truth_rmse={exact['truth_rmse']:.6f}  "
          f"success={exact['success']}", flush=True)

    summaries = {
        "method_a": _summarize(method_a_rows, "method_a"),
        "oracle_cir": _summarize(oracle_rows, "oracle_cir"),
        "option_nls": _summarize(option_rows, "option_nls"),
    }
    summaries["method_a"]["verdict"] = _verdict(
        summaries["method_a"],
        tight={"mu": 0.80, "theta": 0.25, "v0": 0.60, "kappa": 0.40, "xi": 0.40, "rho": 0.40},
        loose={"mu": 1.50, "theta": 0.60, "v0": 1.20, "kappa": 1.00, "xi": 1.50, "rho": 1.00},
    )
    summaries["oracle_cir"]["verdict"] = _verdict(
        summaries["oracle_cir"],
        tight={"mu": 0.80, "theta": 0.15, "v0": 0.05, "kappa": 0.20, "xi": 0.15, "rho": 0.10},
        loose={"mu": 1.50, "theta": 0.40, "v0": 0.15, "kappa": 0.50, "xi": 0.40, "rho": 0.30},
    )
    summaries["option_nls"]["verdict"] = _verdict(
        summaries["option_nls"],
        tight={"kappa": 0.25, "theta": 0.20, "xi": 0.25, "rho": 0.20, "v0": 0.25},
        loose={"kappa": 0.60, "theta": 0.50, "xi": 0.60, "rho": 0.50, "v0": 0.60},
    )

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            x = float(obj)
            return None if not np.isfinite(x) else x
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    payload = {
        "true": TRUE,
        "design": {
            "s0": S0,
            "r": R,
            "dt": DT,
            "n_steps": N_STEPS,
            "n_years": N_STEPS / N_DAYS,
            "seeds": list(SEEDS),
            "feller": float(2 * TRUE["kappa"] * TRUE["theta"] - TRUE["xi"] ** 2),
            "option_moneyness": list(MONEYNESS),
            "option_dte": list(DTE),
            "option_price_noise": PRICE_NOISE,
            "option_x0_rule": "production default from ATM variance, not the truth",
        },
        "summaries": summaries,
        "method_a": method_a_rows,
        "oracle_cir": oracle_rows,
        "option_nls": option_rows,
        "option_nls_exact": exact,
    }
    out_json = OUT / "recovery.json"
    out_json.write_text(json.dumps(_clean(payload), indent=2))
    print(f"\nwrote {out_json}")
    for name, sm in summaries.items():
        print(f"  {name}: verdict={sm['verdict']}  success={sm['n_success']}/{sm['n_seeds']}")
        for k in PARAM_KEYS:
            if k not in sm:
                continue
            s = sm[k]
            print(f"    {k:6s} true={s['true']:<8} mean={s['mean']:.4f}  "
                  f"mae={s['mae']:.4f}  mape={s['mape']:.1%}  "
                  f"med|rel|={s['median_abs_rel']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
