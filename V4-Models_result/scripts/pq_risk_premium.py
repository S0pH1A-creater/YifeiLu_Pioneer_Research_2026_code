"""P → Q risk pricing for Heston, Merton, and Bates (Heston–Merton).

Literature (equations used as published; not invented):

* Heston (1993): volatility risk premium of the form λ(S,v,t) = λ v.
* Pan (2002), JFE, eqs. (2.5) and Appendix D: under Q,
      dV = [κ_v (v̄ − V) + η_v V] dt + σ_v √V dW^Q,
      κ*_v = κ_v − η_v,   v̄* = κ_v v̄ / κ*_v.
  Jump-size premium: μ*_J ≠ μ_J. Jump-timing: Pan sets λ* = λ because
  timing and size premia are not separately identified from returns (p. 8).
* Bates (1996): stochastic volatility + lognormal jumps; Q stock drift is
  r − λ κ^Q with κ^Q = E^Q[e^J − 1]. Heston–Merton uses this Bates class
  with Pan’s premium identification (one consistent SVJ framework).

Christoffersen, Heston, Jacobs (working paper 2002/2003; JE 2006) is a
discrete-time inverse-Gaussian GARCH pricing-kernel paper. It is not used
as the continuous-time Heston/Merton map.

Identifiability
---------------
* P dynamics (μ, CIR, diffusion σ, jump λ, μ_J, σ_J): returns.
* η_v (vol risk premium) and μ*_J (jump-size premium): listed option quotes.
  Volatility and jump risk are not traded assets; returns alone do not
  identify those premia (Pan 2002). Diffusion coefficients (ξ, ρ, σ, σ_J)
  are held at their P values (Girsanov changes Brownian drifts, not
  instantaneous covariances).

GARCH is not in this module (Duan 1995 LRNVR stays in garch_duan_lrnvr.py).
GBM is not in this module (standard μ → r_f).
"""
from __future__ import annotations

from math import erf, exp, log, sqrt
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from heston_option_calibration import (
    _atm_variance,
    _fit_nls,
    heston_call_prices,
    merton_jump_params,
    physical_mu,
)

N_DAYS_DEFAULT = 252
MIN_WINDOW = 60


def kappa_j_from_mu(mu_j: float, sigma_j: float) -> float:
    return float(np.exp(float(mu_j) + 0.5 * float(sigma_j) ** 2) - 1.0)


def pan_q_cir(kappa_p: float, theta_p: float, eta_v: float) -> tuple[float, float]:
    """Pan (2002) Appendix D: κ* = κ − η_v, v̄* = κ v̄ / κ*."""
    kappa_q = float(kappa_p) - float(eta_v)
    if abs(kappa_q) < 1e-6:
        kappa_q = 1e-6 if kappa_q >= 0.0 else -1e-6
    theta_q = float(kappa_p) * float(theta_p) / kappa_q
    return float(kappa_q), float(theta_q)


# ---------------------------------------------------------------------------
# P-measure (returns)
# ---------------------------------------------------------------------------

def estimate_heston_p(log_rets: pd.Series, n_days: float = N_DAYS_DEFAULT) -> dict[str, Any]:
    """Method A: realized-variance moments on all lookback returns (V1/V2)."""
    x = pd.Series(np.asarray(log_rets, dtype=float)).dropna()
    n = int(x.shape[0])
    nan = dict(
        mu=np.nan, kappa=np.nan, theta=np.nan, xi=np.nan, rho=np.nan, v0=np.nan,
        n=n, success=False,
    )
    if n < MIN_WINDOW:
        return nan
    sigma_day = float(x.std(ddof=1))
    if not np.isfinite(sigma_day) or sigma_day <= 0:
        return nan
    dt = 1.0 / float(n_days)
    mu = float(x.mean() * n_days)
    rv = x**2
    theta = float(rv.mean() * n_days)
    recent = rv.iloc[-min(21, n) :]
    v0 = float(recent.mean() * n_days)
    if not np.isfinite(theta) or theta <= 0:
        return nan
    if not np.isfinite(v0) or v0 <= 0:
        v0 = theta
    rho1 = rv.autocorr(lag=1)
    if rho1 is None or not np.isfinite(rho1) or rho1 <= 1e-6:
        kappa = 2.0
    elif rho1 >= 0.999:
        kappa = 0.05
    else:
        kappa = float(-np.log(float(rho1)) / dt)
    kappa = float(np.clip(kappa, 0.05, 20.0))
    v_ann = (rv * n_days).astype(float)
    dv = v_ann.diff().dropna()
    v_lag = v_ann.loc[dv.index]
    drift = kappa * (theta - v_lag.values) * dt
    resid = dv.values - drift
    mean_v = float(np.mean(v_lag.values))
    var_resid = float(np.var(resid, ddof=1)) if resid.size >= 2 else np.nan
    if mean_v > 0 and np.isfinite(var_resid) and var_resid > 0:
        xi = float(np.sqrt(var_resid / (mean_v * dt)))
    else:
        xi = 0.5
    xi = float(np.clip(xi, 0.05, 3.0))
    aligned = pd.concat([x.rename("r"), v_ann.diff().rename("dv")], axis=1).dropna()
    rho = float(aligned["r"].corr(aligned["dv"])) if len(aligned) >= 5 else -0.5
    if not np.isfinite(rho):
        rho = -0.5
    rho = float(np.clip(rho, -0.99, 0.99))
    return dict(
        mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
        n=n, success=True,
    )


def estimate_merton_p(
    log_rets: pd.Series,
    n_days: float = N_DAYS_DEFAULT,
    jump_thresh: float = 3.0,
) -> dict[str, Any]:
    """Existing V2/V4 return estimator: 3σ jumps + residual diffusion variance."""
    x = pd.Series(np.asarray(log_rets, dtype=float)).dropna()
    n = int(x.shape[0])
    nan = dict(
        mu=np.nan, sigma=np.nan, lam=np.nan, mu_j=np.nan, sigma_j=np.nan, kappa=np.nan,
        n=n, success=False,
    )
    if n < 2:
        return nan
    sigma_day = float(x.std(ddof=1))
    if not np.isfinite(sigma_day) or sigma_day <= 0:
        return nan
    jump_mask = np.abs(x.to_numpy(dtype=float)) > float(jump_thresh) * sigma_day
    jumps = x.iloc[jump_mask]
    normal = x.iloc[~jump_mask]
    base = normal if int(normal.shape[0]) >= 2 else x
    mu = float(base.mean() * n_days)
    years = n / float(n_days)
    n_jumps = int(jump_mask.sum())
    lam = float(n_jumps / years) if years > 0 else 0.0
    if n_jumps >= 2:
        mu_j = float(jumps.mean())
        sigma_j = float(jumps.std(ddof=1))
    elif n_jumps == 1:
        mu_j = float(jumps.iloc[0])
        sigma_j = 0.0
    else:
        mu_j = 0.0
        sigma_j = 0.0
    if not np.isfinite(sigma_j) or sigma_j < 0:
        sigma_j = 0.0
    kappa = kappa_j_from_mu(mu_j, sigma_j)
    var_total_ann = float(x.var(ddof=1) * n_days)
    jump_var_ann = float(max(lam, 0.0) * (mu_j**2 + sigma_j**2))
    sigma2 = var_total_ann - jump_var_ann
    eps = 1e-12
    if not np.isfinite(sigma2) or sigma2 < eps:
        sigma2 = max(var_total_ann, eps)
    return dict(
        mu=mu, sigma=float(np.sqrt(sigma2)), lam=lam, mu_j=mu_j,
        sigma_j=sigma_j, kappa=kappa, n=n, success=True,
    )


def estimate_bates_p(
    log_rets: pd.Series,
    n_days: float = N_DAYS_DEFAULT,
    jump_thresh: float = 3.0,
) -> dict[str, Any]:
    """Heston Method A + Merton 3σ jumps (one Bates/SVJ P-vector)."""
    h = estimate_heston_p(log_rets, n_days)
    lam, mu_j, sigma_j, kappa_j = merton_jump_params(log_rets, n_days, jump_thresh)
    h.update(lam=lam, mu_j=mu_j, sigma_j=sigma_j, kappa_j=kappa_j)
    return h


# ---------------------------------------------------------------------------
# Q-measure (options identify premia that returns cannot)
# ---------------------------------------------------------------------------

def _quotes_ok(quotes: Optional[pd.DataFrame], min_n: int = 4) -> bool:
    return quotes is not None and not quotes.empty and len(quotes) >= min_n


def merton_call_price(
    S: float, K: float, T: float, r: float,
    sigma: float, lam: float, mu_j: float, sigma_j: float,
    n_max: int = 40,
) -> float:
    """Merton (1976) Poisson mixture of Black–Scholes, under Q."""
    S, K, T, r = float(S), float(K), float(T), float(r)
    sigma = max(float(sigma), 1e-8)
    lam = max(float(lam), 0.0)
    sigma_j = max(float(sigma_j), 0.0)
    if T <= 1.0 / 365.0 or S <= 0.0 or K <= 0.0:
        return float(max(S - K * np.exp(-r * max(T, 0.0)), 0.0))
    kappa = kappa_j_from_mu(mu_j, sigma_j)
    if kappa <= -0.999:
        kappa = -0.999
    lam_p = lam * (1.0 + kappa)
    log_1k = log(max(1.0 + kappa, 1e-12))
    p = exp(-lam_p * T)
    price = 0.0
    for n in range(n_max + 1):
        if n > 0:
            p *= lam_p * T / n
        if p < 1e-16:
            break
        sigma_n = sqrt(sigma * sigma + n * sigma_j * sigma_j / T)
        r_n = r - lam * kappa + n * log_1k / T
        vs = sigma_n * sqrt(T)
        d1 = (log(S / K) + (r_n + 0.5 * sigma_n * sigma_n) * T) / vs
        d2 = d1 - vs
        n1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
        n2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
        price += p * (S * n1 - K * exp(-r_n * T) * n2)
    return float(max(price, 0.0))


def calibrate_heston_q(p: Mapping[str, Any], quotes: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Fit Pan η_v and v0^Q from listed calls; ξ, ρ held at P."""
    empty = dict(
        eta_v=np.nan, kappa_q=np.nan, theta_q=np.nan, v0_q=np.nan,
        n_quotes=0, rmse=np.nan, success=False, q_explosive=False,
    )
    if not p.get("success") or not _quotes_ok(quotes):
        return empty
    kappa_p, theta_p = float(p["kappa"]), float(p["theta"])
    xi, rho = float(p["xi"]), float(p["rho"])
    v0_p = float(p["v0"])
    S = quotes["S_t"].to_numpy(dtype=float)
    K = quotes["K"].to_numpy(dtype=float)
    r = quotes["r"].to_numpy(dtype=float)
    mkt = quotes["option_price"].to_numpy(dtype=float)
    T = (
        quotes["T_years"].to_numpy(dtype=float)
        if "T_years" in quotes.columns
        else quotes["dte"].to_numpy(dtype=float) / 365.0
    )
    ok = np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(mkt)
    ok &= (S > 0) & (K > 0) & (T > 1.0 / 365.0) & (mkt > 0)
    if int(ok.sum()) < 4:
        return empty
    S, K, T, r, mkt = S[ok], K[ok], T[ok], r[ok], mkt[ok]
    w = 1.0 / np.maximum(mkt, 0.25)
    v_atm = float(np.clip(_atm_variance(quotes), 1e-4, 1.0))
    x0 = np.array([0.0, float(np.clip(v0_p if np.isfinite(v0_p) else v_atm, 1e-4, 3.0))])
    lo = np.array([-40.0, 1e-4])
    hi = np.array([40.0, 3.0])

    def resid(x):
        eta_v, v0_q = float(x[0]), float(x[1])
        kappa_q, theta_q = pan_q_cir(kappa_p, theta_p, eta_v)
        model = heston_call_prices(
            S, K, T, r, kappa_q, theta_q, xi, rho, v0_q, q=0.0,
        )
        err = (model - mkt) * w
        return np.where(np.isfinite(err), err, 1e3)

    x_hat, success = _fit_nls(resid, x0, lo, hi, max_nfev=40)
    eta_v, v0_q = float(x_hat[0]), float(x_hat[1])
    kappa_q, theta_q = pan_q_cir(kappa_p, theta_p, eta_v)
    model = heston_call_prices(S, K, T, r, kappa_q, theta_q, xi, rho, v0_q)
    rmse = float(np.sqrt(np.mean((model - mkt) ** 2))) if np.isfinite(model).all() else np.nan
    return dict(
        eta_v=eta_v, kappa_q=kappa_q, theta_q=theta_q, v0_q=v0_q,
        n_quotes=int(mkt.size), rmse=rmse,
        success=bool(success and np.isfinite(rmse)),
        q_explosive=bool(kappa_q <= 0.0),
    )


def calibrate_merton_q(p: Mapping[str, Any], quotes: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Fit Pan μ*_J from listed calls; λ* = λ^P, σ^Q = σ^P, σ_J^Q = σ_J^P."""
    empty = dict(
        mu_j_q=np.nan, kappa_q=np.nan, n_quotes=0, rmse=np.nan, success=False,
        jump_premium_identified=False,
    )
    if not p.get("success"):
        return empty
    lam, sigma, sigma_j = float(p["lam"]), float(p["sigma"]), float(p["sigma_j"])
    mu_j_p = float(p["mu_j"])
    if lam <= 1e-12:
        kappa_p = float(p["kappa"])
        return dict(
            mu_j_q=mu_j_p, kappa_q=kappa_p, n_quotes=0, rmse=np.nan, success=True,
            jump_premium_identified=False,
        )
    if not _quotes_ok(quotes):
        return empty
    S = quotes["S_t"].to_numpy(dtype=float)
    K = quotes["K"].to_numpy(dtype=float)
    r = quotes["r"].to_numpy(dtype=float)
    mkt = quotes["option_price"].to_numpy(dtype=float)
    T = (
        quotes["T_years"].to_numpy(dtype=float)
        if "T_years" in quotes.columns
        else quotes["dte"].to_numpy(dtype=float) / 365.0
    )
    ok = np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(mkt)
    ok &= (S > 0) & (K > 0) & (T > 1.0 / 365.0) & (mkt > 0)
    if int(ok.sum()) < 4:
        return empty
    S, K, T, r, mkt = S[ok], K[ok], T[ok], r[ok], mkt[ok]
    w = 1.0 / np.maximum(mkt, 0.25)
    lo, hi = np.array([-2.0]), np.array([1.0])
    x0 = np.array([float(np.clip(mu_j_p, -2.0, 1.0))])

    def resid(x):
        mu_j_q = float(x[0])
        model = np.array([
            merton_call_price(S[i], K[i], T[i], r[i], sigma, lam, mu_j_q, sigma_j)
            for i in range(S.size)
        ], dtype=float)
        err = (model - mkt) * w
        return np.where(np.isfinite(err), err, 1e3)

    x_hat, success = _fit_nls(resid, x0, lo, hi, max_nfev=35)
    mu_j_q = float(x_hat[0])
    kappa_q = kappa_j_from_mu(mu_j_q, sigma_j)
    model = np.array([
        merton_call_price(S[i], K[i], T[i], r[i], sigma, lam, mu_j_q, sigma_j)
        for i in range(S.size)
    ], dtype=float)
    rmse = float(np.sqrt(np.mean((model - mkt) ** 2))) if np.isfinite(model).all() else np.nan
    return dict(
        mu_j_q=mu_j_q, kappa_q=kappa_q, n_quotes=int(mkt.size), rmse=rmse,
        success=bool(success and np.isfinite(rmse)),
        jump_premium_identified=True,
    )


def calibrate_bates_q(p: Mapping[str, Any], quotes: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Joint Pan η_v, v0^Q, μ*_J on the Bates CF; λ* = λ^P; ξ, ρ from P."""
    empty = dict(
        eta_v=np.nan, kappa_q=np.nan, theta_q=np.nan, v0_q=np.nan,
        mu_j_q=np.nan, kappa_j_q=np.nan, n_quotes=0, rmse=np.nan,
        success=False, q_explosive=False, jump_premium_identified=False,
    )
    if not p.get("success") or not _quotes_ok(quotes):
        return empty
    kappa_p, theta_p = float(p["kappa"]), float(p["theta"])
    xi, rho = float(p["xi"]), float(p["rho"])
    v0_p = float(p["v0"])
    lam, sigma_j, mu_j_p = float(p["lam"]), float(p["sigma_j"]), float(p["mu_j"])
    S = quotes["S_t"].to_numpy(dtype=float)
    K = quotes["K"].to_numpy(dtype=float)
    r = quotes["r"].to_numpy(dtype=float)
    mkt = quotes["option_price"].to_numpy(dtype=float)
    T = (
        quotes["T_years"].to_numpy(dtype=float)
        if "T_years" in quotes.columns
        else quotes["dte"].to_numpy(dtype=float) / 365.0
    )
    ok = np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(mkt)
    ok &= (S > 0) & (K > 0) & (T > 1.0 / 365.0) & (mkt > 0)
    if int(ok.sum()) < 4:
        return empty
    S, K, T, r, mkt = S[ok], K[ok], T[ok], r[ok], mkt[ok]
    w = 1.0 / np.maximum(mkt, 0.25)
    v_atm = float(np.clip(_atm_variance(quotes), 1e-4, 1.0))
    x0 = np.array([
        0.0,
        float(np.clip(v0_p if np.isfinite(v0_p) else v_atm, 1e-4, 3.0)),
        float(np.clip(mu_j_p, -2.0, 1.0)),
    ])
    lo = np.array([-40.0, 1e-4, -2.0])
    hi = np.array([40.0, 3.0, 1.0])

    def resid(x):
        eta_v, v0_q, mu_j_q = float(x[0]), float(x[1]), float(x[2])
        kappa_q, theta_q = pan_q_cir(kappa_p, theta_p, eta_v)
        model = heston_call_prices(
            S, K, T, r, kappa_q, theta_q, xi, rho, v0_q, q=0.0,
            lam=lam, mu_j=mu_j_q, sigma_j=sigma_j,
        )
        err = (model - mkt) * w
        return np.where(np.isfinite(err), err, 1e3)

    x_hat, success = _fit_nls(resid, x0, lo, hi, max_nfev=50)
    eta_v, v0_q, mu_j_q = float(x_hat[0]), float(x_hat[1]), float(x_hat[2])
    kappa_q, theta_q = pan_q_cir(kappa_p, theta_p, eta_v)
    kappa_j_q = kappa_j_from_mu(mu_j_q, sigma_j)
    model = heston_call_prices(
        S, K, T, r, kappa_q, theta_q, xi, rho, v0_q, q=0.0,
        lam=lam, mu_j=mu_j_q, sigma_j=sigma_j,
    )
    rmse = float(np.sqrt(np.mean((model - mkt) ** 2))) if np.isfinite(model).all() else np.nan
    return dict(
        eta_v=eta_v, kappa_q=kappa_q, theta_q=theta_q, v0_q=v0_q,
        mu_j_q=mu_j_q, kappa_j_q=kappa_j_q, n_quotes=int(mkt.size), rmse=rmse,
        success=bool(success and np.isfinite(rmse)),
        q_explosive=bool(kappa_q <= 0.0),
        jump_premium_identified=bool(lam > 1e-12),
    )


def fit_heston_window(log_rets, quotes, n_days: float = N_DAYS_DEFAULT) -> Optional[dict]:
    p = estimate_heston_p(log_rets, n_days)
    if not p["success"]:
        return None
    q = calibrate_heston_q(p, quotes)
    return {**p, **q, "mu": physical_mu(log_rets, n_days) if not np.isfinite(p["mu"]) else p["mu"]}


def fit_merton_window(
    log_rets, quotes, n_days: float = N_DAYS_DEFAULT, jump_thresh: float = 3.0,
) -> Optional[dict]:
    p = estimate_merton_p(log_rets, n_days, jump_thresh)
    if not p["success"]:
        return None
    q = calibrate_merton_q(p, quotes)
    return {**p, **q}


def fit_bates_window(
    log_rets, quotes, n_days: float = N_DAYS_DEFAULT, jump_thresh: float = 3.0,
) -> Optional[dict]:
    p = estimate_bates_p(log_rets, n_days, jump_thresh)
    if not p["success"]:
        return None
    q = calibrate_bates_q(p, quotes)
    return {**p, **q}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_heston_pq(row: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    p = {
        "measure": "P",
        "mu": float(row["mu"]),
        "kappa": float(row["kappa"]),
        "theta": float(row["theta"]),
        "xi": float(row["xi"]),
        "rho": float(row["rho"]),
        "v0": float(row["v0"]),
        "variance_drift": "κ(θ − v)",
    }
    prem = {
        "eta_v": float(row["eta_v"]) if pd.notna(row.get("eta_v")) else np.nan,
        "source": "listed calls (Pan 2002 η_v); not from returns",
        "q_explosive": bool(row.get("q_explosive", False)),
        "n_quotes": int(row["n_quotes"]) if pd.notna(row.get("n_quotes")) else 0,
    }
    q = {
        "measure": "Q (Pan)",
        "kappa_q": float(row["kappa_q"]) if pd.notna(row.get("kappa_q")) else np.nan,
        "theta_q": float(row["theta_q"]) if pd.notna(row.get("theta_q")) else np.nan,
        "xi": float(row["xi"]),
        "rho": float(row["rho"]),
        "v0_q": float(row["v0_q"]) if pd.notna(row.get("v0_q")) else np.nan,
        "mean_equation": "r_f − ½ v",
        "variance_drift": "κ(θ − v) + η_v v",
    }
    return p, prem, q


def report_merton_pq(row: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    p = {
        "measure": "P",
        "mu": float(row["mu"]),
        "sigma": float(row["sigma"]),
        "lam": float(row["lam"]),
        "mu_j": float(row["mu_j"]),
        "sigma_j": float(row["sigma_j"]),
        "kappa": float(row["kappa"]),
    }
    prem = {
        "mu_j_minus_mu_j_q": (
            float(row["mu_j"]) - float(row["mu_j_q"])
            if pd.notna(row.get("mu_j_q")) else np.nan
        ),
        "source": "listed calls (Pan 2002 μ*); λ* = λ (timing not separately identified)",
        "jump_premium_identified": bool(row.get("jump_premium_identified", False)),
        "n_quotes": int(row["n_quotes"]) if pd.notna(row.get("n_quotes")) else 0,
    }
    q = {
        "measure": "Q (Pan)",
        "sigma": float(row["sigma"]),
        "lam": float(row["lam"]),
        "mu_j_q": float(row["mu_j_q"]) if pd.notna(row.get("mu_j_q")) else np.nan,
        "sigma_j": float(row["sigma_j"]),
        "kappa_q": float(row["kappa_q"]) if pd.notna(row.get("kappa_q")) else np.nan,
        "mean_equation": "r_f − λ κ^Q − ½ σ²",
    }
    return p, prem, q


def report_bates_pq(row: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    p, prem_h, q_h = report_heston_pq(row)
    p.update(lam=float(row["lam"]), mu_j=float(row["mu_j"]),
             sigma_j=float(row["sigma_j"]), kappa_j=float(row["kappa_j"]))
    prem = {
        **prem_h,
        "mu_j_minus_mu_j_q": (
            float(row["mu_j"]) - float(row["mu_j_q"])
            if pd.notna(row.get("mu_j_q")) else np.nan
        ),
        "jump_premium_identified": bool(row.get("jump_premium_identified", False)),
    }
    q_h.update(
        lam=float(row["lam"]),
        mu_j_q=float(row["mu_j_q"]) if pd.notna(row.get("mu_j_q")) else np.nan,
        sigma_j=float(row["sigma_j"]),
        kappa_j_q=float(row["kappa_j_q"]) if pd.notna(row.get("kappa_j_q")) else np.nan,
        mean_equation="r_f − λ κ^Q − ½ v",
    )
    return p, prem, q_h


# ---------------------------------------------------------------------------
# Simulators (Euler; stock uses current v, then v updates — no look-ahead)
# ---------------------------------------------------------------------------

def _heston_euler(rng, paths, v, mu, kappa, theta, xi, rho, eta_v, dt, n_paths):
    z_v = rng.standard_normal(n_paths)
    z_s = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(n_paths)
    v_pos = np.maximum(v, 0.0)
    log_r = (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s
    v_new = v + (kappa * (theta - v_pos) + eta_v * v_pos) * dt + xi * np.sqrt(v_pos * dt) * z_v
    return paths * np.exp(log_r), np.maximum(v_new, 0.0)


def simulate_heston_p(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(len(steps["mu"]))
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    v = np.full(n_paths, float(steps["v0"][0]), dtype=float)
    for i in range(n_steps):
        paths[:, i + 1], v = _heston_euler(
            rng, paths[:, i], v,
            float(steps["mu"][i]), float(steps["kappa"][i]), float(steps["theta"][i]),
            float(steps["xi"][i]), float(np.clip(steps["rho"][i], -0.999, 0.999)),
            0.0, dt, n_paths,
        )
    return paths


def simulate_heston_q(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(len(steps["rf"]))
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    v0 = steps.get("v0_q", steps["v0"])
    v = np.full(n_paths, float(v0[0]), dtype=float)
    kappa = steps.get("kappa", steps.get("kappa_p"))
    theta = steps.get("theta", steps.get("theta_p"))
    for i in range(n_steps):
        paths[:, i + 1], v = _heston_euler(
            rng, paths[:, i], v,
            float(steps["rf"][i]), float(kappa[i]), float(theta[i]),
            float(steps["xi"][i]), float(np.clip(steps["rho"][i], -0.999, 0.999)),
            float(steps["eta_v"][i]), dt, n_paths,
        )
    return paths


def _merton_step(rng, paths, mu, sigma, lam, mu_j, sigma_j, kappa, dt, n_paths):
    z = rng.standard_normal(n_paths)
    n_jumps = rng.poisson(max(lam, 0.0) * dt, size=n_paths)
    jump_sizes = np.zeros(n_paths, dtype=float)
    mask = n_jumps > 0
    if mask.any():
        jump_sizes[mask] = (
            n_jumps[mask] * mu_j
            + np.sqrt(n_jumps[mask]) * max(sigma_j, 0.0) * rng.standard_normal(int(mask.sum()))
        )
    return paths * np.exp(
        (mu - 0.5 * sigma**2 - lam * kappa) * dt + sigma * np.sqrt(dt) * z + jump_sizes
    )


def simulate_merton_p(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(len(steps["mu"]))
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    for i in range(n_steps):
        paths[:, i + 1] = _merton_step(
            rng, paths[:, i],
            float(steps["mu"][i]), float(steps["sigma"][i]), float(steps["lam"][i]),
            float(steps["mu_j"][i]), float(steps["sigma_j"][i]), float(steps["kappa"][i]),
            dt, n_paths,
        )
    return paths


def simulate_merton_q(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(len(steps["rf"]))
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    mu_j = steps.get("mu_j_q", steps["mu_j"])
    kap = steps.get("kappa_q", steps["kappa"])
    for i in range(n_steps):
        paths[:, i + 1] = _merton_step(
            rng, paths[:, i],
            float(steps["rf"][i]), float(steps["sigma"][i]), float(steps["lam"][i]),
            float(mu_j[i]), float(steps["sigma_j"][i]), float(kap[i]),
            dt, n_paths,
        )
    return paths


def simulate_bates_p(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(len(steps["mu"]))
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    v = np.full(n_paths, float(steps["v0"][0]), dtype=float)
    for i in range(n_steps):
        rho = float(np.clip(steps["rho"][i], -0.999, 0.999))
        z_v = rng.standard_normal(n_paths)
        z_s = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        lam = max(float(steps["lam"][i]), 0.0)
        mu_j = float(steps["mu_j"][i])
        sigma_j = max(float(steps["sigma_j"][i]), 0.0)
        kappa_j = float(steps["kappa_j"][i])
        n_jumps = rng.poisson(lam * dt, size=n_paths)
        jump_sizes = np.zeros(n_paths, dtype=float)
        mask = n_jumps > 0
        if mask.any():
            jump_sizes[mask] = (
                n_jumps[mask] * mu_j
                + np.sqrt(n_jumps[mask]) * sigma_j * rng.standard_normal(int(mask.sum()))
            )
        mu = float(steps["mu"][i])
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * v_pos - lam * kappa_j) * dt + np.sqrt(v_pos * dt) * z_s + jump_sizes
        )
        v = v + float(steps["kappa"][i]) * (float(steps["theta"][i]) - v_pos) * dt
        v = v + float(steps["xi"][i]) * np.sqrt(v_pos * dt) * z_v
        v = np.maximum(v, 0.0)
    return paths


def simulate_bates_q(steps, S0, n_paths, seed, *, n_days: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(len(steps["rf"]))
    dt = 1.0 / float(n_days)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    v0 = steps.get("v0_q", steps["v0"])
    v = np.full(n_paths, float(v0[0]), dtype=float)
    mu_j = steps.get("mu_j_q", steps["mu_j"])
    kapj = steps.get("kappa_j_q", steps.get("kappa_j"))
    for i in range(n_steps):
        rho = float(np.clip(steps["rho"][i], -0.999, 0.999))
        z_v = rng.standard_normal(n_paths)
        z_s = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        lam = max(float(steps["lam"][i]), 0.0)
        n_jumps = rng.poisson(lam * dt, size=n_paths)
        jump_sizes = np.zeros(n_paths, dtype=float)
        mask = n_jumps > 0
        if mask.any():
            jump_sizes[mask] = (
                n_jumps[mask] * float(mu_j[i])
                + np.sqrt(n_jumps[mask]) * max(float(steps["sigma_j"][i]), 0.0)
                * rng.standard_normal(int(mask.sum()))
            )
        rf = float(steps["rf"][i])
        paths[:, i + 1] = paths[:, i] * np.exp(
            (rf - 0.5 * v_pos - lam * float(kapj[i])) * dt
            + np.sqrt(v_pos * dt) * z_s + jump_sizes
        )
        eta_v = float(steps["eta_v"][i])
        v = v + (float(steps["kappa"][i]) * (float(steps["theta"][i]) - v_pos) + eta_v * v_pos) * dt
        v = v + float(steps["xi"][i]) * np.sqrt(v_pos * dt) * z_v
        v = np.maximum(v, 0.0)
    return paths


def q_mean_check(paths: np.ndarray, r_annual: float, n_days: int, n_steps: int) -> dict:
    """E^Q[S_{t+nΔt}/S_t] vs exp(r n Δt)."""
    dt = 1.0 / float(n_days)
    ratio = paths[:, n_steps] / paths[:, 0]
    e_hat = float(np.mean(ratio))
    e_true = float(np.exp(r_annual * dt * n_steps))
    return {
        "E_S": e_hat,
        "exp_rf": e_true,
        "mean_err": e_hat - e_true,
        "mean_rel_err": (e_hat - e_true) / e_true if e_true else np.nan,
        "n_paths": int(paths.shape[0]),
        "n_steps": int(n_steps),
    }
