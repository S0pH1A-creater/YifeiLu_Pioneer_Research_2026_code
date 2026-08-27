#!/usr/bin/env python3
"""Ground-truth Heston synthetic bake-off of the six V3 models.

Calibrate from the stock path only (full sample, no rolling). Score
variance vs true v_t and American LSM vs a true-Heston LSM benchmark.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
from american_lsm import lsm_american_call  # noqa: E402

CSV = REPO / "research" / "data" / "equity" / "synthetic" / "heston_10y.csv"
OUT = ROOT / "results" / "synthetic_heston"

TRUE = dict(S0=100.0, v0=0.04, r=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
N_DAYS = 252
DT = 1.0 / N_DAYS
JUMP_THRESH = 3.0
SEED = 42
N_PATHS = 1200
MNY = (0.90, 1.00, 1.10)
DTE = (21, 63, 126)
MODELS = ("GBM", "GARCH", "Heston", "Merton", "Heston–Merton", "GARCH–Merton")


def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _mae(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m])))


# ----- estimators (same rules as the V3 / V2 notebooks) -----

def estimate_gbm(x: np.ndarray) -> dict:
    mu = float(x.mean() * N_DAYS)
    sig = float(x.std(ddof=1) * np.sqrt(N_DAYS))
    return {"mu": mu, "sigma": sig}


def estimate_merton(x: np.ndarray) -> dict:
    sigma_day = float(x.std(ddof=1))
    jump_mask = np.abs(x) > JUMP_THRESH * sigma_day
    jumps = x[jump_mask]
    normal = x[~jump_mask]
    base = normal if len(normal) >= 2 else x
    mu = float(base.mean() * N_DAYS)
    years = len(x) / float(N_DAYS)
    n_j = int(jump_mask.sum())
    lam = float(n_j / years) if years > 0 else 0.0
    if n_j >= 2:
        mu_j, sj = float(jumps.mean()), float(jumps.std(ddof=1))
    elif n_j == 1:
        mu_j, sj = float(jumps[0]), 0.0
    else:
        mu_j, sj = 0.0, 0.0
    kap = float(np.exp(mu_j + 0.5 * sj**2) - 1.0)
    var_tot = float(x.var(ddof=1) * N_DAYS)
    jump_var = lam * (mu_j**2 + sj**2)
    sig = float(np.sqrt(max(var_tot - jump_var, 1e-8)))
    return {"mu": mu, "sigma": sig, "lam": lam, "mu_j": mu_j, "sigma_j": sj, "kappa": kap}


def estimate_heston_from_variance(x: np.ndarray, v: np.ndarray) -> dict:
    """CIR Euler OLS on the observed variance path (synthetic ground truth)."""
    mu = float(x.mean() * N_DAYS)
    dv = np.diff(v)
    vlag = v[:-1]
    X = np.column_stack([np.ones(len(vlag)), vlag])
    coef, *_ = np.linalg.lstsq(X, dv, rcond=None)
    kappa = float(-coef[1] / DT)
    kappa = float(np.clip(kappa, 1e-6, 20.0))
    theta = float(coef[0] / (kappa * DT)) if kappa > 1e-8 else float(np.mean(v))
    theta = float(max(theta, 1e-6))
    resid = dv - (coef[0] + coef[1] * vlag)
    scale = np.sqrt(np.maximum(vlag, 1e-8) * DT)
    xi = float(np.sqrt(np.mean((resid / scale) ** 2)))
    xi = float(np.clip(xi, 1e-6, 4.0))
    z_v = resid / (xi * scale)
    z_s = (x[1:] - (mu - 0.5 * vlag) * DT) / np.sqrt(np.maximum(vlag, 1e-8) * DT)
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
    }


def _unpack_garch(params):
    mu, log_omega, logit_a, logit_bfrac = params
    omega = float(np.exp(log_omega))
    alpha = float(1.0 / (1.0 + np.exp(-logit_a)) * 0.999)
    bfrac = float(1.0 / (1.0 + np.exp(-logit_bfrac)))
    beta = float(bfrac * max(1.0 - alpha - 1e-6, 1e-8))
    return mu, omega, alpha, beta


def _garch_var(eps, omega, alpha, beta):
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
    var = _garch_var(eps, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(var) + eps**2 / var))


def estimate_garch(x: np.ndarray) -> dict:
    mu0 = float(x.mean())
    var0 = float(max(x.var(), 1e-8))
    x0 = np.array([mu0, np.log(var0 * 0.05), 0.0, 1.5], dtype=float)  # α~0.5, β high
    try:
        from scipy.optimize import minimize

        res = minimize(_garch_nll, x0, args=(x,), method="Nelder-Mead", options={"maxiter": 400})
        p = res.x
    except Exception:
        p = x0.copy()
        best = _garch_nll(p, x)
        rng = np.random.default_rng(0)
        for _ in range(80):
            cand = p + rng.normal(0, 0.15, size=4)
            val = _garch_nll(cand, x)
            if val < best:
                best, p = val, cand
    mu_d, omega, alpha, beta = _unpack_garch(p)
    eps = x - mu_d
    var = _garch_var(eps, omega, alpha, beta)
    sigma0 = float(np.sqrt(max(var[-1], 1e-16)))
    return {
        "mu": float(mu_d * N_DAYS),
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "sigma0": sigma0,
        "var_path": var,
    }


def estimate_garch_merton(x: np.ndarray) -> dict:
    g = estimate_garch(x)
    var = g["var_path"]
    sigma = np.sqrt(np.maximum(var, 1e-16))
    mu_d = g["mu"] / N_DAYS
    z = (x - mu_d) / sigma
    jump_mask = np.abs(z) > JUMP_THRESH
    jumps = (x - mu_d)[jump_mask]
    years = len(x) / float(N_DAYS)
    n_j = int(jump_mask.sum())
    lam = float(n_j / years) if years > 0 else 0.0
    if n_j >= 2:
        mu_j, sj = float(jumps.mean()), float(jumps.std(ddof=1))
    elif n_j == 1:
        mu_j, sj = float(jumps[0]), 0.0
    else:
        mu_j, sj = 0.0, 0.0
    kap = float(np.exp(mu_j + 0.5 * sj**2) - 1.0)
    g.update({"lam": lam, "mu_j": mu_j, "sigma_j": sj, "kappa": kap})
    return g


def heston_filter(r: np.ndarray, p: dict) -> np.ndarray:
    """One-step CIR predictor blended with realized variance (returns-only)."""
    n = len(r)
    v = np.empty(n, dtype=float)
    v[0] = float(p["v0"])
    k, th, xi = float(p["kappa"]), float(p["theta"]), float(p["xi"])
    w = float(np.clip(xi / (xi + 1.0), 0.05, 0.35))
    for t in range(1, n):
        pred = v[t - 1] + k * (th - v[t - 1]) * DT
        rv = (r[t] ** 2) / DT
        v[t] = max((1.0 - w) * pred + w * rv, 1e-8)
    return v


# ----- simulators (risk-neutral, μ → r) -----

def sim_gbm(S0, n_steps, n_paths, seed, sigma, r):
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp((r - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * z)
    return paths


def sim_merton(S0, n_steps, n_paths, seed, sigma, lam, mu_j, sj, kap, r):
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_j = rng.poisson(max(lam, 0.0) * DT, size=n_paths)
        js = np.zeros(n_paths)
        m = n_j > 0
        if m.any():
            js[m] = n_j[m] * mu_j + np.sqrt(n_j[m]) * max(sj, 0.0) * rng.standard_normal(int(m.sum()))
        paths[:, i + 1] = paths[:, i] * np.exp(
            (r - 0.5 * sigma**2 - lam * kap) * DT + sigma * np.sqrt(DT) * z + js
        )
    return paths


def sim_heston(S0, n_steps, n_paths, seed, kappa, theta, xi, rho, v0, r):
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rho, -0.999, 0.999))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    v = np.full(n_paths, float(v0))
    for i in range(n_steps):
        z_v = rng.standard_normal(n_paths)
        z_s = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        paths[:, i + 1] = paths[:, i] * np.exp((r - 0.5 * v_pos) * DT + np.sqrt(v_pos * DT) * z_s)
        v = v + kappa * (theta - v_pos) * DT + xi * np.sqrt(v_pos) * np.sqrt(DT) * z_v
        v = np.maximum(v, 0.0)
    return paths


def sim_heston_merton(S0, n_steps, n_paths, seed, kappa, theta, xi, rho, v0, lam, mu_j, sj, kap, r):
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rho, -0.999, 0.999))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    v = np.full(n_paths, float(v0))
    for i in range(n_steps):
        z_v = rng.standard_normal(n_paths)
        z_s = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        n_j = rng.poisson(max(lam, 0.0) * DT, size=n_paths)
        js = np.zeros(n_paths)
        m = n_j > 0
        if m.any():
            js[m] = n_j[m] * mu_j + np.sqrt(n_j[m]) * max(sj, 0.0) * rng.standard_normal(int(m.sum()))
        paths[:, i + 1] = paths[:, i] * np.exp(
            (r - 0.5 * v_pos - lam * kap) * DT + np.sqrt(v_pos * DT) * z_s + js
        )
        v = v + kappa * (theta - v_pos) * DT + xi * np.sqrt(v_pos) * np.sqrt(DT) * z_v
        v = np.maximum(v, 0.0)
    return paths


def sim_garch(S0, n_steps, n_paths, seed, omega, alpha, beta, sigma0, r, lam=0.0, mu_j=0.0, sj=0.0, kap=0.0):
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    var = np.full(n_paths, max(float(sigma0), 1e-8) ** 2)
    for i in range(n_steps):
        sigma = np.sqrt(np.maximum(var, 1e-16))
        z = rng.standard_normal(n_paths)
        eps = sigma * z
        js = np.zeros(n_paths)
        if lam > 0:
            n_j = rng.poisson(max(lam, 0.0) * DT, size=n_paths)
            m = n_j > 0
            if m.any():
                js[m] = n_j[m] * mu_j + np.sqrt(n_j[m]) * max(sj, 0.0) * rng.standard_normal(int(m.sum()))
        paths[:, i + 1] = paths[:, i] * np.exp(r * DT + eps + js - lam * kap * DT)
        var = omega + alpha * eps**2 + beta * var
    return paths


def rn_paths(model: str, cal: dict, S0, n_steps, seed, r, v0_heston):
    if model == "GBM":
        return sim_gbm(S0, n_steps, N_PATHS, seed, cal["sigma"], r)
    if model == "Merton":
        return sim_merton(S0, n_steps, N_PATHS, seed, cal["sigma"], cal["lam"], cal["mu_j"], cal["sigma_j"], cal["kappa"], r)
    if model == "Heston":
        return sim_heston(S0, n_steps, N_PATHS, seed, cal["kappa"], cal["theta"], cal["xi"], cal["rho"], v0_heston, r)
    if model == "Heston–Merton":
        return sim_heston_merton(
            S0, n_steps, N_PATHS, seed, cal["kappa"], cal["theta"], cal["xi"], cal["rho"], v0_heston,
            cal["lam"], cal["mu_j"], cal["sigma_j"], cal["kappa_j"], r,
        )
    if model == "GARCH":
        return sim_garch(S0, n_steps, N_PATHS, seed, cal["omega"], cal["alpha"], cal["beta"], cal["sigma0"], r)
    return sim_garch(
        S0, n_steps, N_PATHS, seed, cal["omega"], cal["alpha"], cal["beta"], cal["sigma0"], r,
        cal["lam"], cal["mu_j"], cal["sigma_j"], cal["kappa"],
    )


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV, parse_dates=["date"])
    r = df["log_return"].to_numpy(dtype=float)
    v_true = df["v_t"].to_numpy(dtype=float)
    s = df["S_t"].to_numpy(dtype=float)
    x = r[1:]  # drop NaN at t0
    v_true_x = v_true[1:]
    print(f"loaded {len(df)} rows  S_T={s[-1]:.3f}  v_T={v_true[-1]:.4f}", flush=True)

    print("calibrate…", flush=True)
    gbm = estimate_gbm(x)
    merton = estimate_merton(x)
    heston = estimate_heston_from_variance(x, v_true_x)
    hm_j = estimate_merton(x)
    hm = {**heston, "lam": hm_j["lam"], "mu_j": hm_j["mu_j"], "sigma_j": hm_j["sigma_j"], "kappa_j": hm_j["kappa"]}
    garch = estimate_garch(x)
    gm = estimate_garch_merton(x)
    cals = {
        "GBM": gbm,
        "GARCH": garch,
        "Heston": heston,
        "Merton": merton,
        "Heston–Merton": hm,
        "GARCH–Merton": gm,
    }

    v_hat = {
        "GBM": np.full_like(v_true_x, gbm["sigma"] ** 2),
        "Merton": np.full_like(v_true_x, merton["sigma"] ** 2),
        "GARCH": garch["var_path"] * N_DAYS,
        "GARCH–Merton": gm["var_path"] * N_DAYS,
        "Heston": heston_filter(x, heston),
        "Heston–Merton": heston_filter(x, heston),
    }

    vol_rows = []
    for m in MODELS:
        vol_rows.append({
            "model": m,
            "rmse_v": _rmse(v_hat[m], v_true_x),
            "mae_v": _mae(v_hat[m], v_true_x),
        })
    vol_rows.sort(key=lambda z: z["rmse_v"])

    print("params:", flush=True)
    print(f"  TRUE  μ={TRUE['r']} κ={TRUE['kappa']} θ={TRUE['theta']} ξ={TRUE['xi']} ρ={TRUE['rho']} v0={TRUE['v0']}", flush=True)
    print(f"  GBM   μ={gbm['mu']:.4f} σ={gbm['sigma']:.4f}  (true σ≈{TRUE['theta']**0.5:.2f})", flush=True)
    print(f"  Heston κ={heston['kappa']:.3f} θ={heston['theta']:.4f} ξ={heston['xi']:.3f} ρ={heston['rho']:.3f} v0={heston['v0']:.4f}", flush=True)
    print(f"  Merton λ={merton['lam']:.3f} σ={merton['sigma']:.4f}  (true λ=0)", flush=True)
    print(f"  GARCH  ω={garch['omega']:.2e} α={garch['alpha']:.3f} β={garch['beta']:.3f}", flush=True)

    # American grid at the last date vs true-Heston LSM
    S0 = float(s[-1])
    r_rf = TRUE["r"]
    v0_true = float(v_true[-1])
    v0_h = v0_true
    contracts = []
    k = 0
    print("LSM benchmark + models…", flush=True)
    for dte in DTE:
        for mny in MNY:
            K = round(S0 * mny, 4)
            paths_true = sim_heston(
                S0, dte, N_PATHS, SEED + k,
                TRUE["kappa"], TRUE["theta"], TRUE["xi"], TRUE["rho"], v0_true, r_rf,
            )
            bench = lsm_american_call(paths_true, K=K, r=r_rf, dt=DT)
            rec = {
                "dte": dte, "K": K, "mny": mny,
                "bench": bench.price, "bench_early": bench.early_exercise_frac,
                "bench_ex": bench.mean_exercise_step,
            }
            for model in MODELS:
                paths = rn_paths(model, cals[model], S0, dte, SEED + 1000 + k, r_rf, v0_h)
                res = lsm_american_call(paths, K=K, r=r_rf, dt=DT)
                rec[f"{model}_px"] = res.price
                rec[f"{model}_err"] = res.price - bench.price
                rec[f"{model}_early"] = res.early_exercise_frac
                rec[f"{model}_ex"] = res.mean_exercise_step
            contracts.append(rec)
            k += 1
            print(f"  dte={dte} K/S={mny:.2f} bench={bench.price:.3f}", flush=True)

    cdf = pd.DataFrame(contracts)
    lsm_rows = []
    for m in MODELS:
        err = cdf[f"{m}_err"].to_numpy()
        lsm_rows.append({
            "model": m,
            "rmse_lsm": float(np.sqrt(np.mean(err**2))),
            "mae_lsm": float(np.mean(np.abs(err))),
            "bias_lsm": float(np.mean(err)),
            "early": float(cdf[f"{m}_early"].mean()),
            "ex_mae": float(np.mean(np.abs(cdf[f"{m}_ex"] - cdf["bench_ex"]))),
            "early_mae": float(np.mean(np.abs(cdf[f"{m}_early"] - cdf["bench_early"]))),
        })
    lsm_rows.sort(key=lambda z: z["rmse_lsm"])

    recov = []
    recov.append({"item": "GBM μ vs r", "true": TRUE["r"], "hat": gbm["mu"], "abs_err": abs(gbm["mu"] - TRUE["r"])})
    recov.append({"item": "GBM σ vs √θ", "true": TRUE["theta"] ** 0.5, "hat": gbm["sigma"], "abs_err": abs(gbm["sigma"] - TRUE["theta"] ** 0.5)})
    recov.append({"item": "Heston κ", "true": TRUE["kappa"], "hat": heston["kappa"], "abs_err": abs(heston["kappa"] - TRUE["kappa"])})
    recov.append({"item": "Heston θ", "true": TRUE["theta"], "hat": heston["theta"], "abs_err": abs(heston["theta"] - TRUE["theta"])})
    recov.append({"item": "Heston ξ", "true": TRUE["xi"], "hat": heston["xi"], "abs_err": abs(heston["xi"] - TRUE["xi"])})
    recov.append({"item": "Heston ρ", "true": TRUE["rho"], "hat": heston["rho"], "abs_err": abs(heston["rho"] - TRUE["rho"])})
    recov.append({"item": "Merton λ vs 0", "true": 0.0, "hat": merton["lam"], "abs_err": abs(merton["lam"])})

    payload = {
        "n_days": int(len(df)),
        "S_T": S0,
        "v_T": v0_true,
        "true": TRUE,
        "cals": {m: {k: (float(v) if np.isscalar(v) else None) for k, v in c.items() if k != "var_path"} for m, c in cals.items()},
        "vol": vol_rows,
        "lsm": lsm_rows,
        "recovery": recov,
        "contracts": contracts,
        "bench_early": float(cdf["bench_early"].mean()),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    cdf.to_csv(OUT / "contracts.csv", index=False)

    # plots
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    t = np.arange(len(v_true_x))
    ax.plot(t, v_true_x, color="black", lw=1.1, label="true v_t")
    colors = {"GBM": "#4C72B0", "GARCH": "#55A868", "Heston": "#C44E52", "Merton": "#8172B2", "Heston–Merton": "#CCB974", "GARCH–Merton": "#64B5CD"}
    for m in ("Heston", "GARCH", "GBM"):
        ax.plot(t, v_hat[m], color=colors[m], lw=0.9, alpha=0.9, label=m)
    ax.set_title("Annual variance: true Heston v_t vs calibrated filters")
    ax.set_xlabel("trading day")
    ax.set_ylabel("variance")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "variance_paths.png", dpi=120)
    plt.close(fig)

    def _bar(rows, key, title, fname, xlabel):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        names = [z["model"] for z in rows]
        vals = [z[key] for z in rows]
        y = np.arange(len(names))
        ax.barh(y, vals, color=[colors[n] for n in names])
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=120)
        plt.close(fig)

    _bar(vol_rows, "rmse_v", "Volatility estimation RMSE vs true v_t (lower is better)", "rank_vol_rmse.png", "RMSE (variance)")
    _bar(lsm_rows, "rmse_lsm", "American LSM RMSE vs true-Heston benchmark (lower is better)", "rank_lsm_rmse.png", "RMSE (option price)")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = MODELS
    y = np.arange(len(names))
    ax.barh(y - 0.18, [float(cdf[f"{m}_early"].mean()) for m in names], height=0.35, color="#4C72B0", label="model")
    ax.barh(y + 0.18, [payload["bench_early"]] * len(names), height=0.35, color="#C44E52", label="true Heston LSM")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("mean early-exercise fraction")
    ax.set_title("Optimal stopping: early exercise vs true-Heston LSM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "stopping_early.png", dpi=120)
    plt.close(fig)

    print("\nVOL RANK", flush=True)
    for z in vol_rows:
        print(f"  {z['model']:16s} RMSE_v={z['rmse_v']:.4f}  MAE_v={z['mae_v']:.4f}", flush=True)
    print("LSM RANK", flush=True)
    for z in lsm_rows:
        print(f"  {z['model']:16s} RMSE={z['rmse_lsm']:.4f}  bias={z['bias_lsm']:.4f}  early={z['early']:.3f}", flush=True)
    print(f"wrote {OUT}  ({payload['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
