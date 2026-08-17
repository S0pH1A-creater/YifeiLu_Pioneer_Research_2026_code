"""Longstaff–Schwartz American call pricing helpers for research notebooks.

Used by §6 Optimal stopping in the GBM / Merton / Heston / Heston–Merton / GARCH / GARCH–Merton
regime notebooks. Path generation stays in each notebook (reuse §5 simulators
under risk-neutral drift μ → r); this module only does LSM + contract sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LSMResult:
    price: float
    exercise_steps: np.ndarray  # per path: step index when exercised (0..n_steps)
    early_exercise_frac: float
    mean_exercise_step: float
    n_paths: int
    n_steps: int


def lsm_american_call(
    paths: np.ndarray,
    K: float,
    r: float,
    dt: float,
    degree: int = 2,
) -> LSMResult:
    """Longstaff–Schwartz American call on a path panel.

    Parameters
    ----------
    paths : array, shape (n_paths, n_steps + 1)
    K : strike
    r : continuously compounded annual risk-free rate (decimal)
    dt : year fraction per step (e.g. 1/252)
    degree : polynomial degree in S for continuation regression (ITM only)
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps+1)")
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1

    # Store realized exercise cashflow time/value; price = mean discounted CF
    exercise_step = np.full(n_paths, n_steps, dtype=int)
    cashflow = np.maximum(paths[:, -1] - K, 0.0)

    for t in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(paths[:, t] - K, 0.0)
        itm = intrinsic > 0.0
        if np.count_nonzero(itm) < degree + 2:
            continue

        # Discount from current exercise time back to t
        tau = (exercise_step[itm] - t).astype(float)
        y = cashflow[itm] * np.exp(-r * dt * tau)
        S_itm = paths[itm, t]
        X = np.vander(S_itm, N=degree + 1, increasing=True)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        continuation = X @ beta

        exercise_now = intrinsic[itm] > continuation
        if np.any(exercise_now):
            idx = np.flatnonzero(itm)[exercise_now]
            cashflow[idx] = intrinsic[idx]
            exercise_step[idx] = t

    disc = np.exp(-r * dt * exercise_step.astype(float))
    cont0 = float(np.mean(cashflow * disc))
    intrinsic0 = float(max(float(paths[0, 0]) - K, 0.0))
    price = max(intrinsic0, cont0)
    if intrinsic0 > cont0 + 1e-12:
        exercise_step = np.zeros(n_paths, dtype=int)
        cashflow = np.full(n_paths, intrinsic0)
        price = intrinsic0

    early = float(np.mean(exercise_step < n_steps))
    return LSMResult(
        price=float(price),
        exercise_steps=exercise_step,
        early_exercise_frac=early,
        mean_exercise_step=float(np.mean(exercise_step)),
        n_paths=n_paths,
        n_steps=n_steps,
    )


def params_asof(cal_table: pd.DataFrame, asof: pd.Timestamp) -> Optional[pd.Series]:
    """Latest calibration row with date <= asof (or earliest row if all later)."""
    if cal_table is None or len(cal_table) == 0:
        return None
    cal = cal_table.sort_values("date").reset_index(drop=True)
    asof = pd.Timestamp(asof)
    dates = pd.to_datetime(cal["date"])
    usable = cal.loc[dates <= asof]
    if len(usable) == 0:
        return cal.iloc[0]
    return usable.iloc[-1]


def sample_spy_calls(
    panel: pd.DataFrame,
    period_start,
    period_end,
    n_total: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample of SPY calls in [period_start, period_end]."""
    df = panel.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    m = (df["trading_date"] >= pd.Timestamp(period_start)) & (
        df["trading_date"] <= pd.Timestamp(period_end)
    )
    df = df.loc[m].dropna(subset=["S_t", "K", "r", "dte", "option_price"])
    if df.empty:
        return df

    df = df.copy()
    df["moneyness_bucket"] = pd.cut(
        df["moneyness"],
        bins=[0.9, 0.97, 1.03, 1.1],
        labels=["OTM", "ATM", "ITM"],
        include_lowest=True,
    )
    df["dte_bucket"] = pd.cut(
        df["dte"],
        bins=[6, 20, 40, 60],
        labels=["short", "med", "long"],
        include_lowest=True,
    )
    df = df.dropna(subset=["moneyness_bucket", "dte_bucket"])

    rng = np.random.default_rng(seed)
    parts = []
    groups = list(df.groupby(["moneyness_bucket", "dte_bucket"], observed=True))
    if not groups:
        return df.sample(n=min(n_total, len(df)), random_state=seed).reset_index(drop=True)

    per = max(1, n_total // len(groups))
    for _, g in groups:
        take = min(per, len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[idx])

    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=seed)
    elif len(out) < n_total:
        leftover = df.drop(index=out.index, errors="ignore")
        # out was reset; re-sample from full df excluding chosen keys
        chosen = set(zip(out["trading_date"], out["K"], out["expiration"].astype(str)))
        mask = [
            (row.trading_date, row.K, str(row.expiration)) not in chosen
            for row in df.itertuples()
        ]
        rest = df.loc[mask]
        need = n_total - len(out)
        if len(rest) and need > 0:
            extra = rest.sample(n=min(need, len(rest)), random_state=seed)
            out = pd.concat([out, extra], ignore_index=True)

    return out.reset_index(drop=True)


def load_spy_calls(data_dir) -> pd.DataFrame:
    path = data_dir / "options" / "processed" / "SPY_calls_panel.csv"
    return pd.read_csv(path, parse_dates=["trading_date", "expiration"])


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black–Scholes European call (used as a 2023 quote substitute)."""
    import math

    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    n = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return float(S * n(d1) - K * math.exp(-r * T) * n(d2))


def make_synthetic_intraday_calls(
    px: pd.Series,
    rets: pd.Series,
    *,
    n_days: float,
    period_start,
    period_end,
    rates: pd.Series,
    ticker: str = "SPY",
    dte_days: int = 2,
    moneyness=(0.97, 1.00, 1.03),
    clock_times=((9, 30), (11, 0), (13, 0), (15, 0)),
    min_bars: int = 30,
    vol_bars: int = 60,
) -> pd.DataFrame:
    """Build 2-day ATM/OTM/ITM calls when listed quotes do not cover the window.

    Benchmark price is a Black–Scholes European call from 1-minute realized σ.
    Vol window is the last `vol_bars` returns ending at the quote time (so a
    Monday open can still use Friday's last hour).
    """
    px = px.dropna()
    rets = rets.dropna()
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    day_px = px.loc[(px.index >= start) & (px.index <= end)]
    days = pd.DatetimeIndex(day_px.index.normalize().unique()).sort_values()
    targets = []
    for d in days:
        session = day_px.loc[day_px.index.normalize() == d]
        if session.empty:
            continue
        for h, m in clock_times:
            ts = pd.Timestamp(d) + pd.Timedelta(hours=h, minutes=m)
            if ts in session.index:
                targets.append(ts)
            else:
                later = session.index[session.index >= ts]
                if len(later):
                    targets.append(later[0])
    rows = []
    T = float(dte_days) / 252.0
    for ts in pd.DatetimeIndex(sorted(set(targets))):
        S = float(px.loc[ts])
        if not np.isfinite(S) or S <= 0:
            continue
        win = rets.loc[rets.index <= ts].tail(int(vol_bars))
        if len(win) < int(min_bars):
            continue
        sigma = float(win.std(ddof=1) * np.sqrt(float(n_days)))
        r = float(rates.asof(ts.normalize()))
        if not np.isfinite(r):
            r = float(rates.dropna().iloc[-1])
        for mny in moneyness:
            K = round(S * float(mny), 2)
            rows.append(
                {
                    "underlying": ticker,
                    "trading_date": ts,
                    "S_t": S,
                    "K": K,
                    "expiration": ts + pd.Timedelta(days=int(dte_days)),
                    "dte": int(dte_days),
                    "r": r,
                    "moneyness": S / K,
                    "option_price": bs_call(S, K, T, r, sigma),
                    "sigma_bs": sigma,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)
