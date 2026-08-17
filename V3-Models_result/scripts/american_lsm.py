"""Longstaff–Schwartz American call pricing helpers for research notebooks.

Used by §6 Optimal stopping in the GBM / Merton / Heston / Heston–Merton / GARCH / GARCH–Merton
regime notebooks. Path generation stays in each notebook (reuse §5 simulators
under risk-neutral drift μ → r); this module only does LSM + contract sampling.

Contract sample is systematic (no RNG):
  * 1-year regime windows: one nearest-ATM listed call each Monday, or the next
    session if Monday is closed (~50 weeks; fewer if the panel is sparse).
  * 7-day windows: every 15 RTH minutes (~130).
  * 1-day windows: every 5 RTH minutes (~78).
Quotes are as-of the grid timestamp (no look-ahead). RMSE helpers report
percentage RMSE vs the market price.

Supports SPY (primary), AAPL, and MSFT call panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from option_filters import DTE_MAX, DTE_MIN, apply_estimation_filters

STOP_TICKERS = ("SPY", "AAPL", "MSFT")


@dataclass
class LSMResult:
    price: float
    exercise_steps: np.ndarray  # per path: step index when exercised (0..n_steps)
    early_exercise_frac: float
    mean_exercise_step: float
    n_paths: int
    n_steps: int
    path_payoffs: np.ndarray  # discounted stopping payoff, one per path


def lsm_american_call(
    paths: np.ndarray,
    K: float,
    r: float,
    dt: float,
    degree: int = 2,
) -> LSMResult:
    """Path-wise Longstaff–Schwartz American call.

    The Monte Carlo cloud is never collapsed to a mean path. Each trajectory
    keeps its own shocks. The only average is the last step: the American
    value is the mean of discounted per-path stopping payoffs.

    1. Immediate payoff on path i at date t: max(S_{i,t} − K, 0).
    2. Continuation: regress discounted *future* cashflows (under the already
       chosen later stopping policy) on in-the-money states
       (1, S, S², |Δlog S| vol proxy).
    3. Path i exercises at t iff payoff_i > continuation_i.
    4. Discount that path's stopping payoff to t = 0.
    5. Price = max(S_0 − K, mean of those discounted payoffs).
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 1:
        raise ValueError(
            "LSM needs the full path cloud of shape (n_paths, n_steps+1); "
            "do not pass an averaged stock path."
        )
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps+1)")
    n_paths, n_cols = paths.shape
    if n_paths < 8:
        raise ValueError(
            f"LSM needs many individual paths, got n_paths={n_paths}. "
            "Do not average simulated paths before optimal stopping."
        )
    n_steps = n_cols - 1
    S = paths

    # Hold to maturity on every path until a later date prefers exercise.
    exercise_step = np.full(n_paths, n_steps, dtype=int)
    cashflow = np.maximum(S[:, -1] - K, 0.0)
    dlog = np.diff(np.log(np.maximum(S, 1e-16)), axis=1)

    for t in range(n_steps - 1, 0, -1):
        payoff = np.maximum(S[:, t] - K, 0.0)
        itm = payoff > 0.0
        n_itm = int(np.count_nonzero(itm))
        if n_itm < degree + 3:
            continue

        tau = (exercise_step[itm] - t).astype(float)
        y = cashflow[itm] * np.exp(-r * dt * tau)

        St = S[itm, t]
        vol = np.abs(dlog[itm, t - 1]) / np.sqrt(max(float(dt), 1e-16))
        X = np.column_stack([np.ones(n_itm), St, St * St, vol])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        continuation = np.maximum(X @ beta, 0.0)

        exercise_now = payoff[itm] > continuation
        if np.any(exercise_now):
            idx = np.flatnonzero(itm)[exercise_now]
            cashflow[idx] = payoff[idx]
            exercise_step[idx] = t

    discounted = cashflow * np.exp(-r * dt * exercise_step.astype(float))
    continuation0 = float(np.mean(discounted))
    intrinsic0 = float(max(float(S[0, 0]) - K, 0.0))
    price = max(intrinsic0, continuation0)

    early = float(np.mean(exercise_step < n_steps))
    return LSMResult(
        price=float(price),
        exercise_steps=exercise_step,
        early_exercise_frac=early,
        mean_exercise_step=float(np.mean(exercise_step)),
        n_paths=n_paths,
        n_steps=n_steps,
        path_payoffs=discounted,
    )


MIN_DTE = DTE_MIN
MAX_DTE = DTE_MAX
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60  # exclusive: 5-min ends 15:55, 15-min ends 15:45


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


def pct_rmse(model, market, *, min_price: float = 1e-8) -> float:
    """Percentage RMSE: 100 × √mean(((ŷ − y)/y)²)."""
    yhat = np.asarray(model, dtype=float)
    y = np.asarray(market, dtype=float)
    ok = np.isfinite(yhat) & np.isfinite(y) & (np.abs(y) > float(min_price))
    if not np.any(ok):
        return float("nan")
    rel = (yhat[ok] - y[ok]) / y[ok]
    return float(100.0 * np.sqrt(np.mean(rel**2)))


def sampling_rule(period_start, period_end) -> str:
    """``monday`` (1-year regime), ``15min`` (7-day), or ``5min`` (1-day)."""
    start = pd.Timestamp(period_start).normalize()
    end = pd.Timestamp(period_end).normalize()
    span = int((end - start).days)
    if span <= 1:
        return "5min"
    if span <= 8:
        return "15min"
    return "monday"


def rth_grid(index, freq_minutes: int) -> pd.DatetimeIndex:
    """RTH bars on an n-minute grid from 09:30, excluding 16:00 close."""
    idx = pd.DatetimeIndex(index)
    tod = idx.hour * 60 + idx.minute
    from_open = tod - RTH_OPEN_MIN
    keep = (from_open >= 0) & (tod < RTH_CLOSE_MIN) & (from_open % int(freq_minutes) == 0)
    return idx[keep]


def monday_week_anchors(trading_days, period_start, period_end) -> pd.DatetimeIndex:
    """One date per week: Monday if that session exists, else the next session."""
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().unique().sort_values()
    start = pd.Timestamp(period_start).normalize()
    end = pd.Timestamp(period_end).normalize()
    days = days[(days >= start) & (days <= end)]
    out: list[pd.Timestamp] = []
    for mon in pd.date_range(start, end, freq="W-MON"):
        week = days[(days >= mon) & (days <= mon + pd.Timedelta(days=6)) & (days <= end)]
        if len(week):
            out.append(pd.Timestamp(week[0]))
    return pd.DatetimeIndex(out)


def _eligible_calls(df: pd.DataFrame) -> pd.DataFrame:
    q = apply_estimation_filters(df)
    if q.empty:
        return q
    q = q.dropna(subset=["S_t", "K", "r", "dte", "option_price"]).copy()
    q["trading_date"] = pd.to_datetime(q["trading_date"])
    return q


def _pick_atm(quotes: pd.DataFrame) -> Optional[pd.Series]:
    """Nearest-ATM listed call, then DTE closest to 30. Deterministic, no RNG."""
    q = _eligible_calls(quotes)
    if q.empty:
        return None
    mny = q["K"].astype(float) / q["S_t"].astype(float)
    q = q.assign(_atm=(mny - 1.0).abs(), _dte30=(q["dte"].astype(float) - 30.0).abs())
    q = q.sort_values(["_atm", "_dte30", "K", "expiration"], kind="mergesort")
    return q.iloc[0]


def _quotes_asof(df: pd.DataFrame, ts) -> pd.DataFrame:
    """Same-day quotes if present; else the last earlier session (no look-ahead)."""
    t = pd.Timestamp(ts)
    dates = pd.to_datetime(df["trading_date"])
    past = df.loc[dates <= t]
    if past.empty:
        return past
    day = t.normalize()
    past_days = dates.loc[past.index].dt.normalize()
    same = past.loc[past_days == day]
    if len(same):
        return same
    last = past_days.max()
    return past.loc[past_days == last]


def sample_calls(
    panel: pd.DataFrame,
    period_start,
    period_end,
    n_total: int = 24,
    seed: int = 42,
    px: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Systematic 1-year-regime sample: one ATM call on each Monday (else next session).

    ``n_total`` / ``seed`` are ignored (kept so existing notebooks still import).
    Count follows actual trading weeks in the panel (~50 in a full 1-year file).
    """
    del n_total, seed
    start = pd.Timestamp(period_start).normalize()
    end = pd.Timestamp(period_end).normalize()
    df = _eligible_calls(panel)
    df = df.loc[(df["trading_date"].dt.normalize() >= start) & (df["trading_date"].dt.normalize() <= end)]
    if df.empty:
        return df.reset_index(drop=True)

    if px is not None and len(pd.Series(px).dropna()):
        days = pd.DatetimeIndex(pd.to_datetime(pd.Series(px).dropna().index).normalize().unique())
    else:
        days = df["trading_date"].dt.normalize()
    anchors = monday_week_anchors(days, start, end)
    rows = []
    for d in anchors:
        rec = _pick_atm(_quotes_asof(df, pd.Timestamp(d) + pd.Timedelta(hours=16)))
        if rec is None:
            continue
        row = rec.to_dict()
        row["trading_date"] = pd.Timestamp(d)
        rows.append(row)
    if not rows:
        return df.iloc[0:0].reset_index(drop=True)
    out = pd.DataFrame(rows).drop(columns=["_atm", "_dte30"], errors="ignore")
    return out.sort_values("trading_date").reset_index(drop=True)


def sample_listed_minute_calls(
    panel: pd.DataFrame,
    px: pd.Series,
    period_start,
    period_end,
    n_total: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Systematic RTH sample on the 1-minute price clock.

    * 7-day window: every 15 minutes (~130 bars: 5 sessions × 26).
    * 1-day window: every 5 minutes (~78 bars: 09:30–15:55).
    * 1-year regime window: Monday close (delegates to ``sample_calls``).

    Expiration is the listed expiry. Each quote time is unique. No RNG.
    Option quotes are as-of the grid timestamp (same session, else prior session).
    """
    del n_total, seed
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    rule = sampling_rule(start, end)
    px = pd.Series(px).dropna().sort_index()
    if px.empty:
        return pd.DataFrame()

    if rule == "monday":
        return sample_calls(panel, start.normalize(), end.normalize(), px=px)

    freq = 5 if rule == "5min" else 15
    end_day = end.normalize()
    session = px.loc[(px.index >= start) & (px.index.normalize() <= end_day)]
    grid = rth_grid(session.index, freq)
    df = _eligible_calls(panel)
    rows: list[dict] = []
    for ts in grid:
        ts = pd.Timestamp(ts)
        rec = _pick_atm(_quotes_asof(df, ts))
        if rec is None:
            continue
        S = float(px.loc[ts]) if ts in px.index else float(px.asof(ts))
        if not np.isfinite(S) or S <= 0:
            continue
        row = rec.to_dict()
        row["trading_date"] = ts
        row["S_t"] = S
        row["moneyness"] = float(row["K"]) / S
        row["n_steps"] = int(row["dte"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop(columns=["_atm", "_dte30"], errors="ignore")
    out = out.sort_values("trading_date").reset_index(drop=True)
    if not out["trading_date"].is_unique:
        raise RuntimeError("systematic minute sample has duplicate trading minutes")
    floored = out["trading_date"].dt.floor("min")
    if not (floored == out["trading_date"]).all():
        raise RuntimeError("systematic minute sample is not on a 1-minute grid")
    return out


def sample_forced_expiry_minute_calls(*args, **kwargs):
    """Back-compat alias — listed/natural expiry, unique 1-minute quotes."""
    # Old signature: (px, rets, listed=..., period_start=..., ...)
    if len(args) >= 1 and not isinstance(args[0], pd.DataFrame):
        px = args[0]
        listed = kwargs.get("listed")
        if listed is None:
            raise TypeError("listed panel is required")
        return sample_listed_minute_calls(
            listed,
            px,
            kwargs["period_start"],
            kwargs["period_end"],
            n_total=kwargs.get("n_total", 24),
            seed=kwargs.get("seed", 42),
        )
    return sample_listed_minute_calls(*args, **kwargs)


def load_calls(data_dir, ticker: str = "SPY", *, panel: str | None = None) -> pd.DataFrame:
    """Load processed American-call panel for one underlying.

    panel="short_interval" reads 2022-09-30 → 2023-09-29 quotes and does not
    touch the 2008–2021 1-year regime files under options/processed/.
    """
    ticker = str(ticker).upper()
    processed = data_dir / "options" / "processed"
    if panel == "short_interval":
        path = processed / "short_interval" / f"{ticker}_calls_panel.csv"
    else:
        path = processed / f"{ticker}_calls_panel.csv"
    return apply_estimation_filters(
        pd.read_csv(path, parse_dates=["trading_date", "expiration"])
    )


# Back-compat aliases used by older notebook cells / runners
def sample_spy_calls(
    panel: pd.DataFrame,
    period_start,
    period_end,
    n_total: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    return sample_calls(panel, period_start, period_end, n_total=n_total, seed=seed)


def load_spy_calls(data_dir) -> pd.DataFrame:
    return load_calls(data_dir, "SPY")


GAP_MIN = pd.Timedelta(minutes=2)


def session_gap(idx) -> pd.Series:
    """True at the first bar of a new RTH session (overnight/weekend gap)."""
    return pd.Series(idx, index=idx).diff() > GAP_MIN


def stitch_continuous(close: pd.Series) -> pd.Series:
    """Gapless trading-time price: overnight/weekend returns are not applied."""
    s = close.dropna()
    r = np.log(s).diff()
    r = r.mask(session_gap(s.index), 0.0)
    r.iloc[0] = 0.0
    return pd.Series(float(s.iloc[0]) * np.exp(r.cumsum()), index=s.index, name=s.name)


def trading_x(idx) -> np.ndarray:
    return np.arange(len(idx))


def session_starts(idx) -> np.ndarray:
    g = session_gap(idx).fillna(False).to_numpy()
    return np.flatnonzero(g)


def n_steps_to_expiry(index, quote_ts, expiry_ts) -> int:
    """Remaining 1-minute RTH bars from quote (exclusive) to expiry (inclusive)."""
    q = pd.Timestamp(quote_ts)
    e = pd.Timestamp(expiry_ts)
    n = int(((index > q) & (index <= e)).sum())
    return max(int(n), 2)


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
    expiration=None,
    bars_per_day: int = 390,
    moneyness=(0.97, 1.00, 1.03),
    clock_times=((9, 30), (11, 0), (13, 0), (15, 0)),
    min_bars: int = 30,
    vol_bars: int = 60,
    min_steps: int = 2,
) -> pd.DataFrame:
    """ATM/OTM/ITM calls when listed quotes do not cover the window.

    If `expiration` is set (e.g. 2023-03-15 session close), every contract
    runs until that expiry: DTE and BS T use remaining RTH 1-minute bars.
    """
    px = px.dropna()
    rets = rets.dropna()
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    expiry = pd.Timestamp(expiration) if expiration is not None else end
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
    full_idx = px.index
    for ts in pd.DatetimeIndex(sorted(set(targets))):
        S = float(px.loc[ts])
        if not np.isfinite(S) or S <= 0:
            continue
        n_steps = n_steps_to_expiry(full_idx, ts, expiry)
        if n_steps < int(min_steps):
            continue
        win = rets.loc[rets.index <= ts].tail(int(vol_bars))
        if len(win) < int(min_bars):
            continue
        sigma = float(win.std(ddof=1) * np.sqrt(float(n_days)))
        r = float(rates.asof(ts.normalize()))
        if not np.isfinite(r):
            r = float(rates.dropna().iloc[-1])
        T = float(n_steps) / float(n_days)
        dte = max(1, int(np.ceil(n_steps / float(bars_per_day))))
        for mny in moneyness:
            K = round(S * float(mny), 2)
            rows.append(
                {
                    "underlying": ticker,
                    "trading_date": ts,
                    "S_t": S,
                    "K": K,
                    "expiration": expiry,
                    "dte": int(dte),
                    "n_steps": int(n_steps),
                    "r": r,
                    "moneyness": S / K,
                    "option_price": bs_call(S, K, T, r, sigma),
                    "sigma_bs": sigma,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)
