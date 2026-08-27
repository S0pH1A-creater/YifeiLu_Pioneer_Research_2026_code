"""GARCH–Merton P→Q: Duan (1995) LRNVR + Pan (2002) jump-size premium.

Physical block: Duan GARCH-in-mean on the continuous part; Merton 3σ jumps
from standardized GARCH residuals.

Risk-neutral block: Duan LRNVR for (ω,α,β,λ_duan,σ0) plus Pan μ_J* with
λ*=λ^P, σ_J*=σ_J^P. Jump-size NLS uses the GARCH unconditional diffusion
vol as the constant-σ proxy in the Merton European formula (same Pan
identifier as standalone Merton).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from garch_duan_lrnvr import (
    align_rf_step,
    estimate_garch_params,
    fit_garch11_duan,
    load_rf_annual,
    q_persist,
)
from pq_risk_premium import calibrate_merton_q, kappa_j_from_mu

N_DAYS_DEFAULT = 252
MIN_WINDOW_DEFAULT = 60
JUMP_THRESH_DEFAULT = 3.0


def estimate_garch_merton_p(
    log_rets: pd.Series,
    rf_annual: pd.Series,
    *,
    n_days: int = N_DAYS_DEFAULT,
    min_window: int = MIN_WINDOW_DEFAULT,
    jump_thresh: float = JUMP_THRESH_DEFAULT,
    warm=None,
) -> Optional[dict[str, Any]]:
    """Duan GARCH-in-mean + 3σ jumps on standardized residuals."""
    x_s = log_rets.dropna()
    n = int(x_s.shape[0])
    if n < int(min_window):
        return None
    rf_step = align_rf_step(x_s.index, rf_annual, n_days)
    fit = fit_garch11_duan(x_s.to_numpy(dtype=float), rf_step, warm=warm)
    if fit is None:
        return None
    lam_duan, omega, alpha, beta, sigma, eps, ok = fit
    sigma0 = float(sigma[-1])
    den_p = 1.0 - alpha - beta
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = (
            float(np.sqrt(omega / den_p))
            if den_p > 1e-8
            else float(np.std(x_s.to_numpy(dtype=float), ddof=1))
        )

    std_resid = eps / np.maximum(sigma, 1e-12)
    jump_mask = np.abs(std_resid) > float(jump_thresh)
    jumps = x_s.to_numpy(dtype=float)[jump_mask]
    n_jumps = int(jump_mask.sum())
    years = n / float(n_days)
    lam = float(n_jumps / years) if years > 0 else 0.0
    if n_jumps >= 2:
        mu_j = float(np.mean(jumps))
        sigma_j = float(np.std(jumps, ddof=1))
    elif n_jumps == 1:
        mu_j = float(jumps[0])
        sigma_j = 0.0
    else:
        mu_j = 0.0
        sigma_j = 0.0
    if not np.isfinite(sigma_j) or sigma_j < 0:
        sigma_j = 0.0
    kappa = float(np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)

    # Annualized diffusion σ proxy for Pan NLS (unconditional GARCH var).
    uncond_var_day = float(omega / den_p) if den_p > 1e-8 else float(np.var(eps))
    sigma_ann = float(np.sqrt(max(uncond_var_day, 1e-16) * float(n_days)))
    persist_q = q_persist(alpha, beta, lam_duan)
    mean_p = rf_step + lam_duan * sigma - 0.5 * sigma**2
    return {
        "lambda": float(lam_duan),
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma0": float(sigma0),
        "lam": float(lam),
        "mu_j": float(mu_j),
        "sigma_j": float(sigma_j),
        "kappa": float(kappa),
        "sigma": float(sigma_ann),
        "n": n,
        "success": bool(ok),
        "mu_p": float(np.mean(mean_p) * n_days),
        "rf_mean": float(np.mean(rf_step) * n_days),
        "persist_p": float(alpha + beta),
        "persist_q": float(persist_q),
        "uncond_var_p": float(uncond_var_day),
        "uncond_var_q": float(omega / (1.0 - persist_q)) if persist_q < 1.0 - 1e-8 else np.nan,
        "q_stationary": bool(persist_q < 1.0 - 1e-8),
    }


def calibrate_garch_merton_q(p: Mapping[str, Any], quotes: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Pan μ_J* on the jump block; Duan numbers are unchanged under Q."""
    merton_p = {
        "success": bool(p.get("success", True)),
        "lam": float(p["lam"]),
        "sigma": float(p["sigma"]),
        "sigma_j": float(p["sigma_j"]),
        "mu_j": float(p["mu_j"]),
        "kappa": float(p["kappa"]),
    }
    q = calibrate_merton_q(merton_p, quotes)
    return {
        "mu_j_q": q.get("mu_j_q", np.nan),
        "kappa_q": q.get("kappa_q", np.nan),
        "n_quotes": int(q.get("n_quotes", 0) or 0),
        "rmse": q.get("rmse", np.nan),
        "success": bool(q.get("success", False)),
        "jump_premium_identified": bool(q.get("jump_premium_identified", False)),
    }


def fit_garch_merton_window(
    log_rets: pd.Series,
    rf_annual: pd.Series,
    quotes: Optional[pd.DataFrame],
    *,
    n_days: int = N_DAYS_DEFAULT,
    min_window: int = MIN_WINDOW_DEFAULT,
    jump_thresh: float = JUMP_THRESH_DEFAULT,
    warm=None,
) -> Optional[dict[str, Any]]:
    p = estimate_garch_merton_p(
        log_rets,
        rf_annual,
        n_days=n_days,
        min_window=min_window,
        jump_thresh=jump_thresh,
        warm=warm,
    )
    if p is None:
        return None
    q = calibrate_garch_merton_q(p, quotes)
    out = dict(p)
    out.update(q)
    return out


def report_garch_merton_pq(row: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    p_tbl = {
        "measure": "P",
        "lambda": float(row["lambda"]),
        "omega": float(row["omega"]),
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "sigma0": float(row["sigma0"]),
        "lam": float(row["lam"]),
        "mu_j": float(row["mu_j"]),
        "sigma_j": float(row["sigma_j"]),
        "kappa": float(row["kappa"]),
        "sigma": float(row["sigma"]) if "sigma" in row and pd.notna(row.get("sigma")) else np.nan,
        "mean_equation": "r_f + λ σ − ½ σ² + J",
        "variance_shock": "ε_t",
    }
    prem = {
        "mu_j_q": float(row["mu_j_q"]) if "mu_j_q" in row and pd.notna(row.get("mu_j_q")) else np.nan,
        "n_quotes": int(row["n_quotes"]) if "n_quotes" in row and pd.notna(row.get("n_quotes")) else 0,
        "jump_premium_identified": bool(row.get("jump_premium_identified", False)),
    }
    q_tbl = {
        "measure": "Q (Duan LRNVR + Pan μ_J*)",
        "lambda": float(row["lambda"]),
        "omega": float(row["omega"]),
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "sigma0": float(row["sigma0"]),
        "lam": float(row["lam"]),
        "mu_j_q": prem["mu_j_q"],
        "sigma_j": float(row["sigma_j"]),
        "kappa_q": float(row["kappa_q"]) if "kappa_q" in row and pd.notna(row.get("kappa_q")) else np.nan,
        "mean_equation": "r_f − λ κ* − ½ σ² + J*",
        "variance_shock": "ξ_t − λ",
        "q_stationary": bool(row["q_stationary"]) if "q_stationary" in row else True,
    }
    return p_tbl, prem, q_tbl


def _steps_len(steps: Mapping[str, np.ndarray]) -> int:
    return int(len(steps["omega"]))


def simulate_garch_merton_p(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    """Physical paths: Duan GARCH-in-mean + compound-Poisson jumps."""
    rng = np.random.default_rng(seed)
    n_steps = _steps_len(steps)
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    var = np.full(n_paths, max(float(steps["sigma0"][0]), 1e-12) ** 2, dtype=float)
    for i in range(n_steps):
        omega = float(steps["omega"][i])
        alpha = float(steps["alpha"][i])
        beta = float(steps["beta"][i])
        lam_duan = float(steps["lambda"][i])
        rf_step = float(steps["rf"][i]) * dt
        lam = float(steps["lam"][i])
        mu_j = float(steps["mu_j"][i])
        sigma_j = float(steps["sigma_j"][i])
        kappa = float(steps["kappa"][i])
        sigma = np.sqrt(np.maximum(var, 1e-16))
        z = rng.standard_normal(n_paths)
        eps = sigma * z
        n_jumps = rng.poisson(max(lam, 0.0) * dt, size=n_paths)
        jump_sizes = np.zeros(n_paths, dtype=float)
        mask = n_jumps > 0
        if mask.any():
            jump_sizes[mask] = (
                n_jumps[mask] * mu_j
                + np.sqrt(n_jumps[mask]) * max(sigma_j, 0.0) * rng.standard_normal(int(mask.sum()))
            )
        log_r = rf_step + lam_duan * sigma - 0.5 * var - lam * kappa * dt + eps + jump_sizes
        paths[:, i + 1] = paths[:, i] * np.exp(log_r)
        var = omega + alpha * eps**2 + beta * var
        var = np.maximum(var, 1e-16)
    return paths


def simulate_garch_merton_q(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    """Q paths: Duan LRNVR continuous block + Pan μ_J* jumps."""
    rng = np.random.default_rng(seed)
    n_steps = _steps_len(steps)
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    var = np.full(n_paths, max(float(steps["sigma0"][0]), 1e-12) ** 2, dtype=float)
    mu_j = steps.get("mu_j_q", steps["mu_j"])
    kap = steps.get("kappa_q", steps["kappa"])
    warned = False
    for i in range(n_steps):
        omega = float(steps["omega"][i])
        alpha = float(steps["alpha"][i])
        beta = float(steps["beta"][i])
        lam_duan = float(steps["lambda"][i])
        if not warned and q_persist(alpha, beta, lam_duan) >= 1.0 - 1e-8:
            print(
                "warning: Duan Q-GARCH not covariance-stationary "
                f"(α(1+λ²)+β ≥ 1) at step {i}; leaving (ω,α,β,λ) unchanged",
                flush=True,
            )
            warned = True
        rf_step = float(steps["rf"][i]) * dt
        lam = float(steps["lam"][i])
        mu_j_i = float(mu_j[i])
        sigma_j = float(steps["sigma_j"][i])
        kappa_i = float(kap[i])
        sigma = np.sqrt(np.maximum(var, 1e-16))
        xi = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(max(lam, 0.0) * dt, size=n_paths)
        jump_sizes = np.zeros(n_paths, dtype=float)
        mask = n_jumps > 0
        if mask.any():
            jump_sizes[mask] = (
                n_jumps[mask] * mu_j_i
                + np.sqrt(n_jumps[mask]) * max(sigma_j, 0.0) * rng.standard_normal(int(mask.sum()))
            )
        log_r = rf_step - lam * kappa_i * dt - 0.5 * var + sigma * xi + jump_sizes
        paths[:, i + 1] = paths[:, i] * np.exp(log_r)
        shock = sigma * (xi - lam_duan)
        var = omega + alpha * shock**2 + beta * var
        var = np.maximum(var, 1e-16)
    return paths


__all__ = [
    "estimate_garch_merton_p",
    "calibrate_garch_merton_q",
    "fit_garch_merton_window",
    "report_garch_merton_pq",
    "simulate_garch_merton_p",
    "simulate_garch_merton_q",
    "load_rf_annual",
    "estimate_garch_params",
    "kappa_j_from_mu",
]
