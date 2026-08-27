"""Duan (1995) GARCH(1,1)-in-mean MLE and LRNVR P/Q simulators.

Physical measure (Duan eq. for GARCH-in-mean):

    ln(S_t / S_{t-1}) = r_{f,t} + λ σ_t − ½ σ_t² + σ_t ε_t
    σ_t² = ω + α σ_{t-1}² ε_{t-1}² + β σ_{t-1}²

    ε_t | F_{t-1} ~ N(0,1) under P.

LRNVR: E^Q[S_t/S_{t-1} | F_{t-1}] = exp(r_{f,t}) and
Var^Q[ln(S_t/S_{t-1}) | F_{t-1}] = σ_t². Then ξ_t = ε_t + λ and

    ln(S_t / S_{t-1}) = r_{f,t} − ½ σ_t² + σ_t ξ_t
    σ_t² = ω + α σ_{t-1}² (ξ_{t-1} − λ)² + β σ_{t-1}²

    ξ_t | F_{t-1} ~ N(0,1) under Q.

λ is estimated from lookback log returns and observed DGS3MO (not from options).
ω, α, β, λ, σ_0 are the same numbers under Q; only the mean and the GARCH shock change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

MIN_WINDOW_DEFAULT = 60
_RF_CACHE: dict[tuple[str, bool], pd.Series] = {}


def load_rf_annual(data_dir, *, short_interval: bool = False) -> pd.Series:
    """Annualized DGS3MO as a decimal yield, ffill-ready DatetimeIndex."""
    data_dir = Path(data_dir)
    key = (str(data_dir.resolve()), bool(short_interval))
    if key in _RF_CACHE:
        return _RF_CACHE[key]
    name = "risk_free_dgs3mo_short_interval.csv" if short_interval else "risk_free_dgs3mo.csv"
    path = data_dir / "rates" / name
    rf = pd.read_csv(path)
    date_col = rf.columns[0]
    val_col = rf.columns[1]
    s = pd.to_numeric(rf[val_col], errors="coerce")
    idx = pd.to_datetime(rf[date_col])
    out = pd.Series(s.values, index=idx, name="rf").sort_index()
    out = out.dropna() / 100.0
    _RF_CACHE[key] = out
    return out


def align_rf_step(index, rf_annual: pd.Series, n_days: int) -> np.ndarray:
    """One-period risk-free rate r_{f,t} = R_t / n_days on `index`."""
    idx = pd.DatetimeIndex(index)
    rf = rf_annual.reindex(idx, method="ffill")
    if rf.isna().any():
        rf = rf.bfill()
    if rf.isna().any():
        rf = rf.fillna(float(rf_annual.dropna().iloc[-1]) if len(rf_annual.dropna()) else 0.0)
    n_days = float(n_days)
    return (rf.to_numpy(dtype=float) / n_days).astype(float)


def unpack_duan(params) -> tuple[float, float, float, float]:
    """Unconstrained → (λ, ω, α, β) with ω>0, α≥0, β≥0, α+β<1."""
    lam, log_omega, logit_a, logit_bfrac = np.asarray(params, dtype=float)
    omega = float(np.exp(log_omega))
    alpha = float(1.0 / (1.0 + np.exp(-logit_a)) * 0.999)
    bfrac = float(1.0 / (1.0 + np.exp(-logit_bfrac)))
    beta = float(bfrac * max(1.0 - alpha - 1e-6, 1e-8))
    return float(lam), omega, alpha, beta


def pack_duan(lam: float, omega: float, alpha: float, beta: float) -> np.ndarray:
    a0 = min(max(float(alpha), 1e-4), 0.5)
    b0 = min(max(float(beta), 1e-4), 0.98)
    if a0 + b0 >= 0.999:
        b0 = max(0.5, 0.998 - a0)
    a_frac = a0 / max(0.999 - a0, 1e-8)
    b_frac = (b0 / max(1.0 - a0, 1e-8)) / max(1.0 - b0 / max(1.0 - a0, 1e-8), 1e-8)
    return np.array(
        [
            float(lam),
            float(np.log(max(omega, 1e-12))),
            float(np.log(max(a_frac, 1e-12))),
            float(np.log(max(b_frac, 1e-12))),
        ],
        dtype=float,
    )


def _filter_path(x: np.ndarray, rf_step: np.ndarray, lam, omega, alpha, beta):
    n = len(x)
    var = np.empty(n, dtype=float)
    eps = np.empty(n, dtype=float)
    v0 = float(np.var(x - rf_step)) if n else 1e-6
    var[0] = v0 if np.isfinite(v0) and v0 > 0 else 1e-6
    for t in range(n):
        sig = float(np.sqrt(max(var[t], 1e-16)))
        mean = float(rf_step[t]) + float(lam) * sig - 0.5 * var[t]
        eps[t] = x[t] - mean
        if t + 1 < n:
            nxt = float(omega) + float(alpha) * eps[t] ** 2 + float(beta) * var[t]
            var[t + 1] = nxt if np.isfinite(nxt) and nxt > 0 else 1e-16
    return var, eps


def duan_nll(params, x: np.ndarray, rf_step: np.ndarray) -> float:
    lam, omega, alpha, beta = unpack_duan(params)
    if not np.all(np.isfinite([lam, omega, alpha, beta])):
        return 1e12
    var, eps = _filter_path(x, rf_step, lam, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(var) + eps**2 / var))


def fit_garch11_duan(x: np.ndarray, rf_step: np.ndarray, warm=None):
    """MLE of (λ, ω, α, β). Returns lam, omega, alpha, beta, sigma_path, eps, success."""
    x = np.asarray(x, dtype=float)
    rf_step = np.asarray(rf_step, dtype=float)
    if rf_step.shape != x.shape:
        raise ValueError("rf_step must match returns length")
    n = len(x)
    if n < 2:
        return None
    v = float(np.var(x))
    if not np.isfinite(v) or v <= 0:
        return None
    sig = float(np.sqrt(v))
    rf_mean = float(np.mean(rf_step))
    lam0 = float((np.mean(x) - rf_mean + 0.5 * v) / max(sig, 1e-8))
    if warm is not None:
        lam0, omega0, a0, b0 = warm
        x0 = pack_duan(lam0, omega0, a0, b0)
    else:
        x0 = pack_duan(lam0, max(v * 0.05, 1e-12), 0.08, 0.85)

    def nll(p):
        return duan_nll(p, x, rf_step)

    success = False
    p = x0
    try:
        from scipy.optimize import minimize

        # Nelder–Mead: L-BFGS-B can segfault on this NLL (unconstrained ω).
        res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-8, "fatol": 1e-8})
        p = res.x
        success = bool(res.success) or np.isfinite(res.fun)
    except Exception:
        p, success = x0, False

    lam, omega, alpha, beta = unpack_duan(p)
    var, eps = _filter_path(x, rf_step, lam, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return None
    sigma = np.sqrt(np.maximum(var, 1e-16))
    return lam, omega, alpha, beta, sigma, eps, bool(success)


def q_persist(alpha: float, beta: float, lam: float) -> float:
    return float(alpha) * (1.0 + float(lam) ** 2) + float(beta)


def estimate_garch_params(
    log_rets: pd.Series,
    rf_annual: pd.Series,
    *,
    n_days: int,
    min_window: int = MIN_WINDOW_DEFAULT,
    warm=None,
) -> Optional[dict[str, Any]]:
    """Fit Duan GARCH-in-mean on a lookback window. None if too short / failed."""
    x_s = log_rets.dropna()
    n = int(x_s.shape[0])
    if n < int(min_window):
        return None
    rf_step = align_rf_step(x_s.index, rf_annual, n_days)
    fit = fit_garch11_duan(x_s.to_numpy(dtype=float), rf_step, warm=warm)
    if fit is None:
        return None
    lam, omega, alpha, beta, sigma, eps, ok = fit
    sigma0 = float(sigma[-1])
    den_p = 1.0 - alpha - beta
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.sqrt(omega / den_p)) if den_p > 1e-8 else float(np.std(x_s.to_numpy(), ddof=1))
    mean_p = rf_step + lam * sigma - 0.5 * sigma**2
    persist_q = q_persist(alpha, beta, lam)
    out = {
        "lambda": float(lam),
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma0": float(sigma0),
        "n": n,
        "success": bool(ok),
        "mu_p": float(np.mean(mean_p) * n_days),
        "rf_mean": float(np.mean(rf_step) * n_days),
        "persist_p": float(alpha + beta),
        "persist_q": float(persist_q),
        "uncond_var_p": float(omega / den_p) if den_p > 1e-8 else np.nan,
        "uncond_var_q": float(omega / (1.0 - persist_q)) if persist_q < 1.0 - 1e-8 else np.nan,
        "q_stationary": bool(persist_q < 1.0 - 1e-8),
    }
    return out


def report_p_and_q(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two separate dicts: physical parameters vs risk-neutral dynamics."""
    lam = float(row["lambda"])
    omega = float(row["omega"])
    alpha = float(row["alpha"])
    beta = float(row["beta"])
    sigma0 = float(row["sigma0"])
    persist_p = float(row["persist_p"]) if "persist_p" in row and pd.notna(row["persist_p"]) else alpha + beta
    persist_q = float(row["persist_q"]) if "persist_q" in row and pd.notna(row["persist_q"]) else q_persist(alpha, beta, lam)
    p_tbl = {
        "measure": "P",
        "lambda": lam,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "sigma0": sigma0,
        "mean_equation": "r_f + λ σ − ½ σ²",
        "variance_shock": "ε_t",
        "persist": persist_p,
        "uncond_var": float(row["uncond_var_p"]) if "uncond_var_p" in row else np.nan,
        "mu_p": float(row["mu_p"]) if "mu_p" in row and pd.notna(row.get("mu_p")) else np.nan,
        "rf_mean": float(row["rf_mean"]) if "rf_mean" in row and pd.notna(row.get("rf_mean")) else np.nan,
    }
    q_tbl = {
        "measure": "Q (LRNVR)",
        "lambda": lam,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "sigma0": sigma0,
        "mean_equation": "r_f − ½ σ²",
        "variance_shock": "ξ_t − λ",
        "persist": persist_q,
        "uncond_var": float(row["uncond_var_q"]) if "uncond_var_q" in row else np.nan,
        "q_stationary": bool(row["q_stationary"]) if "q_stationary" in row else persist_q < 1.0,
    }
    return p_tbl, q_tbl


def _steps_len(steps: Mapping[str, np.ndarray]) -> int:
    if "omega" in steps:
        return int(len(steps["omega"]))
    return int(len(next(iter(steps.values()))))


def simulate_garch_p(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    """Physical-measure Monte Carlo (Duan GARCH-in-mean). `steps['rf']` is annualized."""
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
        lam = float(steps["lambda"][i])
        rf_step = float(steps["rf"][i]) * dt
        sigma = np.sqrt(np.maximum(var, 1e-16))
        z = rng.standard_normal(n_paths)
        log_r = rf_step + lam * sigma - 0.5 * var + sigma * z
        paths[:, i + 1] = paths[:, i] * np.exp(log_r)
        eps = sigma * z
        var = omega + alpha * eps**2 + beta * var
        var = np.maximum(var, 1e-16)
    return paths


def simulate_garch_q(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    """Risk-neutral Monte Carlo (Duan LRNVR). `steps['rf']` is annualized."""
    rng = np.random.default_rng(seed)
    n_steps = _steps_len(steps)
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    var = np.full(n_paths, max(float(steps["sigma0"][0]), 1e-12) ** 2, dtype=float)
    warned = False
    for i in range(n_steps):
        omega = float(steps["omega"][i])
        alpha = float(steps["alpha"][i])
        beta = float(steps["beta"][i])
        lam = float(steps["lambda"][i])
        if not warned and q_persist(alpha, beta, lam) >= 1.0 - 1e-8:
            print(
                "warning: Duan Q-GARCH not covariance-stationary "
                f"(α(1+λ²)+β ≥ 1) at step {i}; leaving (ω,α,β,λ) unchanged",
                flush=True,
            )
            warned = True
        rf_step = float(steps["rf"][i]) * dt
        sigma = np.sqrt(np.maximum(var, 1e-16))
        xi = rng.standard_normal(n_paths)
        log_r = rf_step - 0.5 * var + sigma * xi
        paths[:, i + 1] = paths[:, i] * np.exp(log_r)
        shock = sigma * (xi - lam)
        var = omega + alpha * shock**2 + beta * var
        var = np.maximum(var, 1e-16)
    return paths


def q_step_contract_steps(p: Mapping[str, Any], r_annual: float, n_steps: int) -> dict[str, np.ndarray]:
    """Constant-parameter Q schedule for one listed option (LSM)."""
    n_steps = int(n_steps)
    return {
        "lambda": np.full(n_steps, float(p["lambda"]), dtype=float),
        "omega": np.full(n_steps, float(p["omega"]), dtype=float),
        "alpha": np.full(n_steps, float(p["alpha"]), dtype=float),
        "beta": np.full(n_steps, float(p["beta"]), dtype=float),
        "sigma0": np.full(n_steps, float(p["sigma0"]), dtype=float),
        "rf": np.full(n_steps, float(r_annual), dtype=float),
    }


def lrnvr_one_step_check(
    *,
    lam: float,
    omega: float,
    alpha: float,
    beta: float,
    sigma0: float,
    r_annual: float,
    n_days: int = 252,
    n_paths: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """Monte Carlo check of LRNVR (1)–(2) over a single step."""
    steps = q_step_contract_steps(
        {"lambda": lam, "omega": omega, "alpha": alpha, "beta": beta, "sigma0": sigma0},
        r_annual,
        1,
    )
    paths = simulate_garch_q(steps, 1.0, n_paths, seed, n_days=n_days)
    dt = 1.0 / float(n_days)
    rf_step = float(r_annual) * dt
    h = max(float(sigma0), 1e-12) ** 2
    ratio = paths[:, 1]
    log_r = np.log(ratio)
    return {
        "E_S": float(np.mean(ratio)),
        "exp_rf": float(np.exp(rf_step)),
        "mean_err": float(np.mean(ratio) - np.exp(rf_step)),
        "Var_log": float(np.var(log_r)),
        "h": float(h),
        "var_err": float(np.var(log_r) - h),
    }
