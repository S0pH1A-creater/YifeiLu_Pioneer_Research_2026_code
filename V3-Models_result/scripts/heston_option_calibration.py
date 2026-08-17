"""Option-implied Heston / Bates calibration (V3).

Calibrates (κ, θ, ξ, ρ, v0) by matching market call prices with the
semi-closed Fourier (characteristic-function) European formula.
Jump intensity / size for Heston–Merton (Bates) are taken from a
return-threshold estimator and held fixed while the Heston block is
fitted.

Monte Carlo SDEs and LSM are unchanged; this module is used only in §4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from option_filters import apply_estimation_filters

# Fourier grid for the Heston inversion (Gatheral / Albrecher little-trap).
_PHI = np.linspace(1e-5, 200.0, 192)
_DPHI = float(_PHI[1] - _PHI[0])


def _trapz(y, dx: float) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    return float(dx * (0.5 * y[0] + 0.5 * y[-1] + y[1:-1].sum()))

# Calibration bounds: κ, θ, ξ, ρ, v0
_LO = np.array([0.05, 1e-4, 0.02, -0.99, 1e-4], dtype=float)
_HI = np.array([20.0, 3.0, 4.0, 0.99, 3.0], dtype=float)

_CALLS_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.complex128)


def heston_log_cf(
    u,
    T: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    r: float,
    q: float,
    S0: float,
    lam: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0,
):
    """Characteristic function of ln S_T (risk-neutral, little-Heston-trap).

    Optional Bates / Merton jumps: log-normal jumps with compensator
    κ_J = exp(μ_J + σ_J²/2) − 1.
    """
    u = _as_float_array(u)
    i = 1j
    xi = max(float(xi), 1e-8)
    kappa = max(float(kappa), 1e-8)
    theta = max(float(theta), 1e-10)
    v0 = max(float(v0), 1e-10)
    rho = float(np.clip(rho, -0.999, 0.999))
    T = max(float(T), 1e-8)

    b = kappa - rho * xi * i * u
    d = np.sqrt(b * b + xi * xi * (i * u + u * u))
    denom = b + d
    denom = np.where(np.abs(denom) < 1e-16, 1e-16 + 0j, denom)
    g = (b - d) / denom
    # Albrecher et al. (2007): switch to the representation with |g|≤1
    swap = np.abs(g) > 1.0
    g_safe = np.where(np.abs(g) < 1e-16, 1e-16 + 0j, g)
    g = np.where(swap, 1.0 / g_safe, g)
    bmd = np.where(swap, b + d, b - d)

    exp_dt = np.exp(-d * T)
    one_g = 1.0 - g
    one_ge = 1.0 - g * exp_dt
    # Guard log / division
    one_g = np.where(np.abs(one_g) < 1e-14, 1e-14 + 0j, one_g)
    one_ge = np.where(np.abs(one_ge) < 1e-14, 1e-14 + 0j, one_ge)

    C = (r - q) * i * u * T + (kappa * theta / (xi * xi)) * (
        bmd * T - 2.0 * np.log(one_ge / one_g)
    )
    D = (bmd / (xi * xi)) * (1.0 - exp_dt) / one_ge
    cf = np.exp(C + D * v0 + i * u * np.log(S0))

    if lam > 0.0 and T > 0.0:
        kappa_j = float(np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
        jump_mgf = np.exp(i * u * mu_j - 0.5 * sigma_j * sigma_j * u * u)
        cf = cf * np.exp(lam * T * (jump_mgf - 1.0 - i * u * kappa_j))
    return cf


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    q: float = 0.0,
    lam: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0,
) -> float:
    """European call via Heston (1993) P1/P2 inversion (Bates if jumps given)."""
    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    q = float(q)
    if not np.isfinite(S0 + K + T + r) or S0 <= 0.0 or K <= 0.0:
        return np.nan
    if T <= 1.0 / 365.0:
        return float(max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0))

    disc_k = K * np.exp(-r * T)
    fwd = S0 * np.exp(-q * T)
    log_k = np.log(K)
    phi = _PHI
    i = 1j

    cf2 = heston_log_cf(phi, T, kappa, theta, xi, rho, v0, r, q, S0, lam, mu_j, sigma_j)
    cf1 = heston_log_cf(phi - i, T, kappa, theta, xi, rho, v0, r, q, S0, lam, mu_j, sigma_j)

    integ2 = np.real(np.exp(-i * phi * log_k) * cf2 / (i * phi))
    integ1 = np.real(np.exp(-i * phi * log_k) * cf1 / (i * phi * S0 * np.exp((r - q) * T)))

    p2 = 0.5 + (1.0 / np.pi) * _trapz(integ2, _DPHI)
    p1 = 0.5 + (1.0 / np.pi) * _trapz(integ1, _DPHI)
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    price = fwd * p1 - disc_k * p2
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if not np.isfinite(price):
        return intrinsic
    return float(max(price, intrinsic * 0.5, 0.0))


def heston_call_prices(
    S0,
    K,
    T,
    r,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    q: float = 0.0,
    lam: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0,
) -> np.ndarray:
    """Vector of European call prices (loop over contracts; CF is vectorized in φ)."""
    S0 = np.asarray(S0, dtype=float).ravel()
    K = np.asarray(K, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    n = int(K.size)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = heston_call_price(
            float(S0[i] if S0.size > 1 else S0[0]),
            float(K[i]),
            float(T[i] if T.size > 1 else T[0]),
            float(r[i] if r.size > 1 else r[0]),
            kappa,
            theta,
            xi,
            rho,
            v0,
            q=q,
            lam=lam,
            mu_j=mu_j,
            sigma_j=sigma_j,
        )
    return out


def _bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    from math import erf, exp, log, sqrt

    if T <= 0.0 or sigma <= 0.0:
        return max(S * exp(-q * T) - K * exp(-r * T), 0.0)
    vs = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vs
    d2 = d1 - vs
    n1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
    n2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
    return S * exp(-q * T) * n1 - K * exp(-r * T) * n2


def implied_vol_call(S: float, K: float, T: float, r: float, price: float, q: float = 0.0) -> float:
    """Bisection Black–Scholes implied vol (NaN if the quote is unusable)."""
    if not np.isfinite(price) or price <= 0.0 or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return np.nan
    disc = S * np.exp(-q * T) - K * np.exp(-r * T)
    lo_p = max(disc, 0.0)
    hi_p = S * np.exp(-q * T)
    if price <= lo_p * 1.0001 or price >= hi_p * 0.999:
        return np.nan
    lo, hi = 1e-4, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _bs_call(S, K, T, r, mid, q) > price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def _atm_variance(quotes: pd.DataFrame) -> float:
    if quotes is None or quotes.empty:
        return 0.04
    q = quotes.copy()
    m = np.abs(q["moneyness"].to_numpy(dtype=float) - 1.0)
    i = int(np.nanargmin(m))
    row = q.iloc[i]
    T = _maturity_years(row)
    iv = implied_vol_call(
        float(row["S_t"]),
        float(row["K"]),
        T,
        float(row["r"]),
        float(row["option_price"]),
    )
    if np.isfinite(iv) and iv > 0:
        return float(np.clip(iv * iv, 1e-4, 1.0))
    return 0.04


def _maturity_years(row) -> float:
    if "T_years" in row.index and np.isfinite(row["T_years"]) and float(row["T_years"]) > 0:
        return float(row["T_years"])
    dte = float(row["dte"]) if "dte" in row.index else 30.0
    return max(dte, 1.0) / 365.0


def load_calls_panel(data_root: Path, ticker: str, *, opt_dir: Path | None = None) -> pd.DataFrame:
    """Load listed calls for one underlying (cached)."""
    data_root = Path(data_root)
    if opt_dir is None:
        opt_dir = data_root / "options" / "processed"
    key = (str(opt_dir.resolve()), ticker)
    if key in _CALLS_CACHE:
        return _CALLS_CACHE[key]

    candidates = [
        opt_dir / f"{ticker}_calls_panel.csv",
        opt_dir / f"{ticker}_options_panel.csv",
        data_root / "options" / "processed" / f"{ticker}_calls_panel.csv",
        data_root / "options" / "processed" / "short_interval" / f"{ticker}_calls_panel.csv",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        _CALLS_CACHE[key] = pd.DataFrame()
        return _CALLS_CACHE[key]

    df = pd.read_csv(path)
    if "trading_date" in df.columns:
        df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = apply_estimation_filters(df)
    _CALLS_CACHE[key] = df.reset_index(drop=True)
    return _CALLS_CACHE[key]


def _lookback_start(asof: pd.Timestamp, lookback) -> pd.Timestamp:
    asof = pd.Timestamp(asof)
    if isinstance(lookback, (pd.DateOffset, pd.Timedelta)):
        return asof - lookback
    if isinstance(lookback, (int, np.integer, float)):
        # 1-min notebooks: lookback is a bar count. Option quotes are daily,
        # so map 60 bars → 1 session and 390 bars → 1 session (padded).
        n = int(lookback)
        days = 1 if n <= 90 else max(int(np.ceil(n / 390.0)), 1)
        return asof - pd.Timedelta(days=days)
    return asof - pd.DateOffset(months=6)


def select_quotes_asof(
    panel: pd.DataFrame,
    asof,
    lookback,
    *,
    min_quotes: int = 6,
    max_quotes: int = 24,
    moneyness_band: tuple[float, float] | None = None,
    dte_band: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Option quotes in (asof − lookback, asof], subsampled for calibration.

    Quality filters are the shared V3 estimation rules (no-arbitrage,
    DTE 7–60, |S/K−1|≤10%, liquidity). ``moneyness_band`` / ``dte_band``
    are ignored; kept so existing notebooks still import.
    """
    del moneyness_band, dte_band
    if panel is None or panel.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof)
    asof_day = asof.normalize()
    start = _lookback_start(asof, lookback).normalize()

    td = pd.to_datetime(panel["trading_date"])
    sub = panel.loc[(td >= start) & (td <= asof_day)].copy()
    if sub.empty:
        earlier = panel.loc[td <= asof_day]
        if earlier.empty:
            return pd.DataFrame()
        last = pd.to_datetime(earlier["trading_date"]).max()
        sub = earlier.loc[pd.to_datetime(earlier["trading_date"]) == last].copy()

    sub = apply_estimation_filters(sub)
    if sub.empty:
        return sub

    # Prefer the latest snapshot in the window, then fill with older quotes.
    last = pd.to_datetime(sub["trading_date"]).max()
    snap = sub.loc[pd.to_datetime(sub["trading_date"]) == last]
    if len(snap) < min_quotes:
        snap = sub
    return _subsample_surface(snap, max_quotes=max_quotes)


def _subsample_surface(df: pd.DataFrame, max_quotes: int = 24) -> pd.DataFrame:
    if len(df) <= max_quotes:
        return df.reset_index(drop=True)
    work = df.copy()
    if "T_years" in work.columns:
        mat = work["T_years"].to_numpy(dtype=float)
    else:
        mat = work["dte"].to_numpy(dtype=float) / 365.0
    work["_mat_bucket"] = np.round(mat * 12.0)  # ~monthly buckets
    m = work["moneyness"].to_numpy(dtype=float) if "moneyness" in work.columns else np.ones(len(work))
    work["_m_bucket"] = pd.cut(m, bins=[0.0, 0.95, 1.05, 10.0], labels=False)
    parts = []
    n_mat = max(int(work["_mat_bucket"].nunique()), 1)
    per_bucket = max(int(np.ceil(max_quotes / max(n_mat * 3, 1))), 1)
    for _, g in work.groupby(["_mat_bucket", "_m_bucket"], dropna=False):
        g = g.assign(_atm=np.abs(g["moneyness"] - 1.0) if "moneyness" in g.columns else 0.0)
        parts.append(g.sort_values("_atm").head(per_bucket))
    out = pd.concat(parts, ignore_index=True) if parts else work
    if len(out) > max_quotes:
        out = out.assign(_atm=np.abs(out["moneyness"] - 1.0) if "moneyness" in out.columns else 0.0)
        out = out.sort_values("_atm").head(max_quotes)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore").reset_index(drop=True)


def physical_mu(log_rets: pd.Series, n_days: float) -> float:
    """P-measure drift from the lookback of stock returns (not identified by Q-calibration)."""
    x = log_rets.dropna().astype(float)
    if x.empty:
        return 0.0
    mu = float(x.mean() * float(n_days))
    return mu if np.isfinite(mu) else 0.0


def merton_jump_params(
    log_rets: pd.Series,
    n_days: float,
    jump_thresh: float = 3.0,
) -> tuple[float, float, float, float]:
    """(λ, μ_J, σ_J, κ_J) from a 3σ return threshold. Not used for Heston-only."""
    x = log_rets.dropna().astype(float)
    n = int(x.shape[0])
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    sigma = float(x.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    jump_mask = np.abs(x.to_numpy(dtype=float)) > float(jump_thresh) * sigma
    jumps = x.iloc[jump_mask]
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
    kappa_j = float(np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    return lam, mu_j, sigma_j, kappa_j


def _fit_nls(resid, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_nfev: int = 60):
    """Bounded nonlinear least squares.

    Uses SciPy TRF when available; otherwise a projected Levenberg–Marquardt
    step with finite-difference Jacobian (no extra dependencies).
    """
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    try:
        from scipy.optimize import least_squares

        res = least_squares(
            resid,
            x0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=int(max_nfev),
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            jac="2-point",
        )
        return np.clip(res.x, lo, hi), bool(res.success or np.isfinite(res.cost))
    except Exception:
        pass

    x = x0.copy()
    r = np.asarray(resid(x), dtype=float)
    best = 0.5 * float(np.dot(r, r))
    nfev = 1
    n = int(x.size)
    lam = 1e-2
    success = np.isfinite(best)

    while nfev + n + 1 <= max_nfev:
        J = np.empty((r.size, n), dtype=float)
        for j in range(n):
            step = max(1e-5 * max(abs(x[j]), 1.0), 1e-6)
            if x[j] + step > hi[j]:
                step = -step
            xj = x.copy()
            xj[j] = float(np.clip(x[j] + step, lo[j], hi[j]))
            den = xj[j] - x[j]
            if abs(den) < 1e-12:
                J[:, j] = 0.0
                continue
            rj = np.asarray(resid(xj), dtype=float)
            nfev += 1
            J[:, j] = (rj - r) / den
        jtj = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(5):
            if nfev >= max_nfev:
                break
            scale = np.diag(jtj) + 1e-8
            A = jtj + lam * np.diag(scale)
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e8)
                continue
            x_new = np.clip(x + delta, lo, hi)
            r_new = np.asarray(resid(x_new), dtype=float)
            nfev += 1
            cost = 0.5 * float(np.dot(r_new, r_new))
            if np.isfinite(cost) and cost < best * (1.0 - 1e-8):
                x, r, best = x_new, r_new, cost
                lam = max(lam / 3.0, 1e-8)
                improved = True
                success = True
                break
            lam = min(lam * 8.0, 1e8)
        if not improved:
            break
    return x, success


def _default_x0(quotes: pd.DataFrame) -> np.ndarray:
    v = _atm_variance(quotes)
    return np.array([1.5, v, 0.6, -0.5, v], dtype=float)


def calibrate_heston_from_quotes(
    quotes: pd.DataFrame,
    *,
    x0: np.ndarray | None = None,
    lam: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0,
    q: float = 0.0,
    max_nfev: int = 60,
) -> dict[str, Any]:
    """Nonlinear least squares: market calls vs Heston/Bates European prices.

    Returns dict with kappa, theta, xi, rho, v0, rmse, n_quotes, success.
    """
    empty = {
        "kappa": np.nan,
        "theta": np.nan,
        "xi": np.nan,
        "rho": np.nan,
        "v0": np.nan,
        "rmse": np.nan,
        "n_quotes": 0,
        "success": False,
    }
    if quotes is None or quotes.empty:
        return empty

    S = quotes["S_t"].to_numpy(dtype=float)
    K = quotes["K"].to_numpy(dtype=float)
    r = quotes["r"].to_numpy(dtype=float)
    mkt = quotes["option_price"].to_numpy(dtype=float)
    if "T_years" in quotes.columns:
        T = quotes["T_years"].to_numpy(dtype=float)
    else:
        T = quotes["dte"].to_numpy(dtype=float) / 365.0
    ok = np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(mkt)
    ok &= (S > 0) & (K > 0) & (T > 1.0 / 365.0) & (mkt > 0)
    if int(ok.sum()) < 4:
        return empty
    S, K, T, r, mkt = S[ok], K[ok], T[ok], r[ok], mkt[ok]
    w = 1.0 / np.maximum(mkt, 0.25)

    x_init = np.array(_default_x0(quotes) if x0 is None else x0, dtype=float).ravel()[:5]
    x_init = np.clip(x_init, _LO, _HI)

    def resid(x):
        kappa, theta, xi, rho, v0 = [float(v) for v in x]
        model = heston_call_prices(
            S, K, T, r, kappa, theta, xi, rho, v0, q=q, lam=lam, mu_j=mu_j, sigma_j=sigma_j
        )
        err = (model - mkt) * w
        return np.where(np.isfinite(err), err, 1e3)

    x_hat, success = _fit_nls(resid, x_init, _LO, _HI, max_nfev=int(max_nfev))

    model = heston_call_prices(
        S, K, T, r, *x_hat, q=q, lam=lam, mu_j=mu_j, sigma_j=sigma_j
    )
    rmse = float(np.sqrt(np.mean((model - mkt) ** 2))) if np.isfinite(model).all() else np.nan
    rel = (model - mkt) / np.maximum(np.abs(mkt), 1e-8)
    rmse_pct = float(100.0 * np.sqrt(np.mean(rel**2))) if np.isfinite(model).all() else np.nan
    return {
        "kappa": float(x_hat[0]),
        "theta": float(x_hat[1]),
        "xi": float(x_hat[2]),
        "rho": float(x_hat[3]),
        "v0": float(x_hat[4]),
        "rmse": rmse,
        "rmse_pct": rmse_pct,
        "n_quotes": int(mkt.size),
        "success": bool(success and np.isfinite(rmse)),
    }


def quotes_fingerprint(quotes: pd.DataFrame) -> tuple:
    """Identity of a calibration surface (reuse params when unchanged)."""
    if quotes is None or quotes.empty:
        return ()
    last = pd.to_datetime(quotes["trading_date"]).max() if "trading_date" in quotes.columns else None
    last_i = int(pd.Timestamp(last).value) if last is not None and pd.notna(last) else 0
    return (
        last_i,
        int(len(quotes)),
        round(float(np.nansum(quotes["option_price"])), 6),
        round(float(np.nansum(quotes["K"])), 4),
    )
