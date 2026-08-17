#!/usr/bin/env python3
"""GARCH(1,1) parameter-recovery study (implementation check).

Simulate daily prices from known (μ, ω, α, β, σ0) with the V3 notebook Euler
step, hide the true parameters, and recover them with the notebook MLE:

    r_t = μ_day + ε_t,  ε_t = σ_t Z_t
    σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²
    μ̂ = μ̂_day × 252

Optimizer matches the notebooks when SciPy is present (L-BFGS-B); otherwise
Nelder–Mead on the same unconstrained parameterization. Repeat over seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "garch_parameter_recovery"

N_DAYS = 252
DT = 1.0 / N_DAYS
S0 = 100.0
N_STEPS = 10 * N_DAYS
SEEDS = (42, 7, 123, 2024, 99, 314, 2718, 8675309)

# Stationary GARCH(1,1) with ~20% annual uncond. vol (same scale as GBM test).
# ω / (1−α−β) = 0.20² / 252  ⇒  ω = 7.9365e-6
TRUE = dict(
    mu=0.08,
    omega=0.20**2 / N_DAYS * (1.0 - 0.10 - 0.85),
    alpha=0.10,
    beta=0.85,
)
TRUE["sigma0"] = float(np.sqrt(TRUE["omega"] / (1.0 - TRUE["alpha"] - TRUE["beta"])))
TRUE["persist"] = TRUE["alpha"] + TRUE["beta"]
TRUE["uncond_var"] = TRUE["omega"] / (1.0 - TRUE["alpha"] - TRUE["beta"])
TRUE["uncond_vol"] = float(np.sqrt(TRUE["uncond_var"] * N_DAYS))

PARAM_KEYS = ("mu", "omega", "alpha", "beta")


def simulate_garch_path(mu, omega, alpha, beta, sigma0, *, seed: int) -> np.ndarray:
    """Same step as V3 `simulate_garch_rolling` (one path, fixed params)."""
    rng = np.random.default_rng(seed)
    s = np.empty(N_STEPS + 1, dtype=float)
    s[0] = S0
    var = max(float(sigma0), 1e-12) ** 2
    for i in range(N_STEPS):
        sigma = np.sqrt(max(var, 1e-16))
        eps = sigma * float(rng.standard_normal())
        s[i + 1] = s[i] * np.exp(mu * DT + eps)
        var = omega + alpha * eps**2 + beta * var
        var = max(var, 1e-16)
    return s


def _unpack_garch(params):
    """Map unconstrained params → (μ_day, ω, α, β) with ω>0, α≥0, β≥0, α+β<1."""
    mu, log_omega, logit_a, logit_bfrac = params
    omega = float(np.exp(log_omega))
    alpha = float(1.0 / (1.0 + np.exp(-logit_a)) * 0.999)
    bfrac = float(1.0 / (1.0 + np.exp(-logit_bfrac)))
    beta = float(bfrac * max(1.0 - alpha - 1e-6, 1e-8))
    return mu, omega, alpha, beta


def _garch_variance_path(eps, omega, alpha, beta):
    n = len(eps)
    var = np.empty(n, dtype=float)
    v0 = float(np.var(eps)) if n else 1e-6
    var[0] = v0 if np.isfinite(v0) and v0 > 0 else 1e-6
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    return var


def _garch_nll(params, x):
    mu, omega, alpha, beta = _unpack_garch(params)
    eps = x - mu
    var = _garch_variance_path(eps, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(var) + eps**2 / var))


def _nelder_mead(fun, x0, maxiter=600):
    """Derivative-free minimizer used when SciPy is unavailable."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    simplex = np.tile(x0, (n + 1, 1))
    for i in range(n):
        simplex[i + 1, i] += 0.15 if abs(x0[i]) < 1e-8 else 0.15 * max(abs(x0[i]), 0.15)
    fvals = np.array([fun(p) for p in simplex], dtype=float)
    alpha_r, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(maxiter):
        order = np.argsort(fvals)
        simplex, fvals = simplex[order], fvals[order]
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha_r * (centroid - simplex[-1])
        fr = fun(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = fun(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        xc = centroid + rho * (simplex[-1] - centroid)
        fc = fun(xc)
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            fvals[i] = fun(simplex[i])
    order = np.argsort(fvals)
    return simplex[order[0]], float(fvals[order[0]])


def fit_garch11(x: np.ndarray):
    """MLE fit matching V3 notebooks (`fit_garch11` / `estimate_garch_params`)."""
    x = np.asarray(x, dtype=float)
    v = float(np.var(x))
    if not np.isfinite(v) or v <= 0:
        return None
    x0 = np.array(
        [float(np.mean(x)), np.log(max(v * 0.05, 1e-12)), -1.5, 2.0],
        dtype=float,
    )
    nll = lambda p: _garch_nll(p, x)
    success = False
    try:
        from scipy.optimize import minimize

        res = minimize(nll, x0, method="L-BFGS-B")
        p = res.x
        success = bool(res.success)
    except Exception:
        p, _ = _nelder_mead(nll, x0, maxiter=700)
        success = True
    mu, omega, alpha, beta = _unpack_garch(p)
    eps = x - mu
    var = _garch_variance_path(eps, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return None
    sigma = np.sqrt(var)
    return mu, omega, alpha, beta, sigma, success


def estimate_garch_params(log_rets: np.ndarray) -> dict:
    fit = fit_garch11(log_rets)
    nan = {k: np.nan for k in (*PARAM_KEYS, "sigma0", "persist", "uncond_var", "uncond_vol")}
    if fit is None:
        return {**nan, "n": int(len(log_rets)), "success": False}
    mu_day, omega, alpha, beta, sigma, ok = fit
    persist = float(alpha + beta)
    den = 1.0 - persist
    uncond_var = float(omega / den) if den > 1e-8 else np.nan
    uncond_vol = float(np.sqrt(uncond_var * N_DAYS)) if np.isfinite(uncond_var) and uncond_var > 0 else np.nan
    sigma0 = float(sigma[-1])
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.sqrt(uncond_var)) if np.isfinite(uncond_var) and uncond_var > 0 else np.nan
    return {
        "mu": float(mu_day * N_DAYS),
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma0": sigma0,
        "persist": persist,
        "uncond_var": uncond_var,
        "uncond_vol": uncond_vol,
        "n": int(len(log_rets)),
        "success": bool(ok),
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
        "True (hidden): μ={mu}  ω={omega:.6e}  α={alpha}  β={beta}  "
        "σ0={sigma0:.6f}  persist={persist}  uncond vol={uncond_vol:.4f}".format(**TRUE)
    )
    print(f"  {N_STEPS} daily steps, {len(SEEDS)} seeds, optimizer={opt}\n")

    for seed in SEEDS:
        s = simulate_garch_path(
            TRUE["mu"], TRUE["omega"], TRUE["alpha"], TRUE["beta"], TRUE["sigma0"],
            seed=int(seed),
        )
        r = np.diff(np.log(s))
        hat = estimate_garch_params(r)
        row = {
            "seed": int(seed),
            "S_T": float(s[-1]),
            "success": hat["success"],
            "mu_hat": hat["mu"],
            "omega_hat": hat["omega"],
            "alpha_hat": hat["alpha"],
            "beta_hat": hat["beta"],
            "sigma0_hat": hat["sigma0"],
            "persist_hat": hat["persist"],
            "uncond_var_hat": hat["uncond_var"],
            "uncond_vol_hat": hat["uncond_vol"],
            "mu_abs": hat["mu"] - TRUE["mu"],
            "mu_rel": _rel(hat["mu"], TRUE["mu"]),
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
            f"seed {seed:8d}  μ̂={hat['mu']:+.4f}  ω̂={hat['omega']:.3e}  "
            f"α̂={hat['alpha']:.3f}  β̂={hat['beta']:.3f}  "
            f"α̂+β̂={hat['persist']:.3f}  σ̄̂={hat['uncond_vol']:.4f}  "
            f"σ̂0={hat['sigma0']:.4f}  S_T={s[-1]:.2f}"
        )

    summary = {
        "mu": _summ([r["mu_hat"] for r in rows], TRUE["mu"]),
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

    summary["verdict_mu"] = _ok(summary["mu"], 0.20, 0.50)
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
            "estimator": "Gaussian GARCH(1,1) MLE on log returns (notebook fit_garch11)",
            "simulator": "S ← S exp(μ Δt + σ_t Z); then σ² ← ω + α ε² + β σ²",
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
        ("μ", "mu"),
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
