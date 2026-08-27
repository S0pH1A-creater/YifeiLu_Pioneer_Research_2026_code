#!/usr/bin/env python3
"""GARCH(1,1) Duan GARCH-in-mean parameter-recovery study.

Simulate daily prices from known (λ, ω, α, β, σ0) under Duan P-measure,
hide the true parameters, and recover them with `garch_duan_lrnvr` MLE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "garch_parameter_recovery"

from garch_duan_lrnvr import estimate_garch_params as _estimate_duan  # noqa: E402
from garch_duan_lrnvr import simulate_garch_p  # noqa: E402

N_DAYS = 252
DT = 1.0 / N_DAYS
S0 = 100.0
N_STEPS = 10 * N_DAYS
SEEDS = (42, 7, 123, 2024, 99, 314, 2718, 8675309)
RF_ANNUAL = 0.03

TRUE = dict(
    lam=0.05,
    omega=0.20**2 / N_DAYS * (1.0 - 0.10 - 0.85),
    alpha=0.10,
    beta=0.85,
)
TRUE["sigma0"] = float(np.sqrt(TRUE["omega"] / (1.0 - TRUE["alpha"] - TRUE["beta"])))
TRUE["persist"] = TRUE["alpha"] + TRUE["beta"]
TRUE["uncond_var"] = TRUE["omega"] / (1.0 - TRUE["alpha"] - TRUE["beta"])
TRUE["uncond_vol"] = float(np.sqrt(TRUE["uncond_var"] * N_DAYS))

PARAM_KEYS = ("lambda", "omega", "alpha", "beta")


def simulate_garch_path(lam, omega, alpha, beta, sigma0, *, seed: int) -> np.ndarray:
    n = N_STEPS
    steps = {
        "lambda": np.full(n, float(lam)),
        "omega": np.full(n, float(omega)),
        "alpha": np.full(n, float(alpha)),
        "beta": np.full(n, float(beta)),
        "sigma0": np.full(n, float(sigma0)),
        "rf": np.full(n, float(RF_ANNUAL)),
    }
    return simulate_garch_p(steps, S0, 1, seed, n_days=N_DAYS)[0]


def fit_garch11(x: np.ndarray):
    """Duan GARCH-in-mean fit in the V3 diagnostics tuple shape.

    Returns ``(mean_path, omega, alpha, beta, sigma_path, success)`` so
    ``(x - mean_path) / sigma_path`` is the Duan innovation ε_t.
    """
    from garch_duan_lrnvr import fit_garch11_duan

    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return None
    rf_step = np.full(x.shape, float(RF_ANNUAL) / float(N_DAYS))
    fit = fit_garch11_duan(x, rf_step)
    if fit is None:
        return None
    _lam, omega, alpha, beta, sigma, eps, ok = fit
    mean_path = x - eps
    return mean_path, float(omega), float(alpha), float(beta), sigma, bool(ok)


def estimate_garch_params(log_rets: np.ndarray) -> dict:
    idx = pd.bdate_range("2000-01-03", periods=len(log_rets))
    s = pd.Series(np.asarray(log_rets, dtype=float), index=idx)
    rf = pd.Series(RF_ANNUAL, index=idx)
    hat = _estimate_duan(s, rf, n_days=N_DAYS, min_window=60)
    nan = {k: np.nan for k in (*PARAM_KEYS, "sigma0", "persist", "uncond_var", "uncond_vol")}
    if hat is None:
        return {**nan, "n": int(len(log_rets)), "success": False}
    return {
        "lambda": float(hat["lambda"]),
        "omega": float(hat["omega"]),
        "alpha": float(hat["alpha"]),
        "beta": float(hat["beta"]),
        "sigma0": float(hat["sigma0"]),
        "persist": float(hat["persist_p"]),
        "uncond_var": float(hat["uncond_var_p"]) if np.isfinite(hat["uncond_var_p"]) else np.nan,
        "uncond_vol": float(np.sqrt(hat["uncond_var_p"] * N_DAYS)) if np.isfinite(hat["uncond_var_p"]) and hat["uncond_var_p"] > 0 else np.nan,
        "n": int(hat["n"]),
        "success": bool(hat["success"]),
    }


def _rel(hat: float, tru: float) -> float:
    den = abs(tru) if abs(tru) > 1e-12 else 1.0
    return float((hat - tru) / den)


def _summ(hats, tru):
    hats = np.asarray(hats, dtype=float)
    err = hats - tru
    rel = err / (abs(tru) if abs(tru) > 1e-12 else 1.0)
    return {
        "true": float(tru),
        "mean": float(hats.mean()),
        "median": float(np.median(hats)),
        "std": float(hats.std(ddof=1)),
        "bias": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(rel))),
        "median_abs_rel": float(np.median(np.abs(rel))),
    }


def _ok(s, tight, loose):
    med = s["median_abs_rel"]
    if med <= tight:
        return "recovered"
    if med <= loose:
        return "partial"
    return "not-recovered"


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import scipy  # noqa: F401

        opt = "scipy L-BFGS-B (notebook)"
    except Exception:
        opt = "numpy Nelder-Mead fallback (SciPy missing)"

    rows = []
    print(
        "True (hidden): λ={lam}  ω={omega:.6e}  α={alpha}  β={beta}  "
        "σ0={sigma0:.6f}  persist={persist}  uncond vol={uncond_vol:.4f}".format(**TRUE)
    )
    print(f"  {N_STEPS} daily steps, {len(SEEDS)} seeds, optimizer={opt}\n")

    for seed in SEEDS:
        s = simulate_garch_path(
            TRUE["lam"], TRUE["omega"], TRUE["alpha"], TRUE["beta"], TRUE["sigma0"],
            seed=int(seed),
        )
        r = np.diff(np.log(s))
        hat = estimate_garch_params(r)
        row = {
            "seed": int(seed),
            "S_T": float(s[-1]),
            "success": hat["success"],
            "lambda_hat": hat["lambda"],
            "omega_hat": hat["omega"],
            "alpha_hat": hat["alpha"],
            "beta_hat": hat["beta"],
            "sigma0_hat": hat["sigma0"],
            "persist_hat": hat["persist"],
            "uncond_var_hat": hat["uncond_var"],
            "uncond_vol_hat": hat["uncond_vol"],
            "lambda_abs": hat["lambda"] - TRUE["lam"],
            "lambda_rel": _rel(hat["lambda"], TRUE["lam"]),
            "omega_abs": hat["omega"] - TRUE["omega"],
            "omega_rel": _rel(hat["omega"], TRUE["omega"]),
            "alpha_abs": hat["alpha"] - TRUE["alpha"],
            "alpha_rel": _rel(hat["alpha"], TRUE["alpha"]),
            "beta_abs": hat["beta"] - TRUE["beta"],
            "beta_rel": _rel(hat["beta"], TRUE["beta"]),
            "persist_abs": hat["persist"] - TRUE["persist"],
            "persist_rel": _rel(hat["persist"], TRUE["persist"]),
            "uncond_vol_abs": hat["uncond_vol"] - TRUE["uncond_vol"],
            "uncond_vol_rel": _rel(hat["uncond_vol"], TRUE["uncond_vol"]),
        }
        rows.append(row)
        print(
            f"seed {seed:8d}  λ̂={hat['lambda']:+.4f}  ω̂={hat['omega']:.3e}  "
            f"α̂={hat['alpha']:.3f}  β̂={hat['beta']:.3f}  "
            f"α̂+β̂={hat['persist']:.3f}  σ̄̂={hat['uncond_vol']:.4f}  "
            f"σ̂0={hat['sigma0']:.4f}  S_T={s[-1]:.2f}"
        )

    summary = {
        "lambda": _summ([r["lambda_hat"] for r in rows], TRUE["lam"]),
        "omega": _summ([r["omega_hat"] for r in rows], TRUE["omega"]),
        "alpha": _summ([r["alpha_hat"] for r in rows], TRUE["alpha"]),
        "beta": _summ([r["beta_hat"] for r in rows], TRUE["beta"]),
        "persist": _summ([r["persist_hat"] for r in rows], TRUE["persist"]),
        "uncond_vol": _summ([r["uncond_vol_hat"] for r in rows], TRUE["uncond_vol"]),
        "n_seeds": len(rows),
        "optimizer": opt,
    }
    # σ0 is last filtered vol, not the simulation start — report scatter only.
    summary["sigma0_last_filter"] = {
        "mean": float(np.mean([r["sigma0_hat"] for r in rows])),
        "std": float(np.std([r["sigma0_hat"] for r in rows], ddof=1)),
        "note": "last conditional σ from the MLE filter, not the t=0 start",
    }

    summary["verdict_lambda"] = _ok(summary["lambda"], 0.20, 0.50)
    summary["verdict_omega"] = _ok(summary["omega"], 0.25, 0.60)
    summary["verdict_alpha"] = _ok(summary["alpha"], 0.25, 0.60)
    summary["verdict_beta"] = _ok(summary["beta"], 0.10, 0.25)
    summary["verdict_persist"] = _ok(summary["persist"], 0.05, 0.15)
    summary["verdict_uncond_vol"] = _ok(summary["uncond_vol"], 0.10, 0.25)

    payload = {
        "true": TRUE,
        "design": {
            "s0": S0,
            "dt": DT,
            "n_steps": N_STEPS,
            "n_years": N_STEPS / N_DAYS,
            "seeds": list(SEEDS),
            "estimator": "Duan GARCH-in-mean Gaussian MLE (garch_duan_lrnvr)",
            "simulator": "S ← S exp(r_f + λσ − ½σ² + σ ε); then σ² ← ω + α (σε)² + β σ²",
            "optimizer": opt,
            "x0": "notebook default (sample mean, log(0.05 var), logit_a=-1.5, logit_bfrac=2)",
        },
        "summary": summary,
        "seeds": rows,
    }
    out = OUT / "recovery.json"
    out.write_text(json.dumps(_clean(payload), indent=2))
    print(f"\nwrote {out}")
    for name, key in (
        ("λ", "lambda"),
        ("ω", "omega"),
        ("α", "alpha"),
        ("β", "beta"),
        ("α+β", "persist"),
        ("uncond vol", "uncond_vol"),
    ):
        s = summary[key]
        v = summary[f"verdict_{key}"]
        print(
            f"  {name:10s}  {v:16s}  mean={s['mean']:.6g}  "
            f"mae={s['mae']:.6g}  med|rel|={s['median_abs_rel']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
