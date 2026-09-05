"""Shared listed-call filters.

Applied to the LSM contract sample (and to Merton / GARCH–Merton Q premia).
Return-based P estimators use lookback log returns, not these quotes.

Steps (in order):
  1. No-arbitrage: C ≥ max(0, S − K)
  2. Maturity: 7 ≤ DTE ≤ 60
  3. Near ATM: |S/K − 1| ≤ 10%
  4. Liquidity: valid bid/ask and relative spread ≤ 50% when those
      columns exist; otherwise drop clearly unreliable premia (price < 0.05,
      non-finite). Volume ≥ 1 is required when a volume column exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DTE_MIN = 7
DTE_MAX = 60
MONEYNESS_CUTOFF = 0.10  # |S/K - 1|
MAX_REL_SPREAD = 0.50
MIN_PREMIUM = 0.05
MIN_VOLUME = 1

MONEYNESS_BUCKETS = (
    ("deep OTM", 0.0, 0.90),
    ("OTM", 0.90, 0.98),
    ("ATM", 0.98, 1.02),
    ("ITM", 1.02, 1.10),
    ("deep ITM", 1.10, np.inf),
)


def spot_over_strike(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["S_t"], errors="coerce")
    k = pd.to_numeric(df["K"], errors="coerce")
    out = s / k
    out = out.where((k > 0) & np.isfinite(k))
    return out


def moneyness_bucket(sk: pd.Series) -> pd.Series:
    sk = pd.to_numeric(sk, errors="coerce")
    labels = [name for name, _, _ in MONEYNESS_BUCKETS]
    bins = [0.0, 0.90, 0.98, 1.02, 1.10, np.inf]
    return pd.cut(sk, bins=bins, labels=labels, right=False, include_lowest=True)


def _mid_price(df: pd.DataFrame) -> pd.Series:
    if {"bid", "ask"}.issubset(df.columns):
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        mid = 0.5 * (bid + ask)
        valid = (bid > 0) & (ask > bid) & np.isfinite(mid)
        return mid.where(valid)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _premium(df: pd.DataFrame) -> pd.Series:
    """Market call price: listed option_price, else valid mid, else mark."""
    if "option_price" in df.columns:
        px = pd.to_numeric(df["option_price"], errors="coerce")
    else:
        px = pd.Series(np.nan, index=df.index, dtype=float)
    mid = _mid_price(df)
    px = px.where(np.isfinite(px) & (px > 0), mid)
    if "mark" in df.columns:
        mark = pd.to_numeric(df["mark"], errors="coerce")
        px = px.where(np.isfinite(px) & (px > 0), mark)
    return px


def apply_estimation_filters(
    df: pd.DataFrame,
    *,
    audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the four research filters. Empty in → empty out."""
    if df is None or len(df) == 0:
        empty = df if df is not None else pd.DataFrame()
        if audit:
            return empty, [{"step": "input", "n": 0}]
        return empty

    work = df.copy()
    log: list[dict[str, Any]] = [{"step": "input", "n": int(len(work))}]

    if "option_type" in work.columns:
        work = work.loc[work["option_type"].astype(str).str.lower().eq("call")].copy()
    elif "type" in work.columns:
        work = work.loc[work["type"].astype(str).str.lower().eq("call")].copy()
    log.append({"step": "calls only", "n": int(len(work))})

    if "S_t" not in work.columns or "K" not in work.columns:
        if audit:
            log.append({"step": "missing S or K", "n": 0})
            return work.iloc[0:0], log
        return work.iloc[0:0]

    work["S_t"] = pd.to_numeric(work["S_t"], errors="coerce")
    work["K"] = pd.to_numeric(work["K"], errors="coerce")
    work["_C"] = _premium(work)
    work = work.loc[
        work["S_t"].gt(0)
        & work["K"].gt(0)
        & np.isfinite(work["S_t"])
        & np.isfinite(work["K"])
        & np.isfinite(work["_C"])
    ].copy()
    log.append({"step": "finite S, K, C", "n": int(len(work))})

    # 1. No-arbitrage: C ≥ max(0, S − K)
    intrinsic = np.maximum(0.0, work["S_t"].to_numpy(dtype=float) - work["K"].to_numpy(dtype=float))
    work = work.loc[work["_C"].to_numpy(dtype=float) + 1e-12 >= intrinsic].copy()
    log.append({"step": "1. no-arbitrage C ≥ max(0, S−K)", "n": int(len(work))})

    # 2. Maturity
    if "dte" in work.columns:
        dte = pd.to_numeric(work["dte"], errors="coerce")
    elif {"expiration", "trading_date"}.issubset(work.columns):
        dte = (
            pd.to_datetime(work["expiration"]) - pd.to_datetime(work["trading_date"])
        ).dt.days
        work["dte"] = dte
    else:
        dte = pd.Series(np.nan, index=work.index)
    work = work.loc[dte.ge(DTE_MIN) & dte.le(DTE_MAX)].copy()
    log.append({"step": "2. maturity 7–60 DTE", "n": int(len(work))})

    # 3a. Moneyness |S/K − 1| ≤ 10%
    sk = spot_over_strike(work)
    work = work.loc[np.abs(sk - 1.0) <= MONEYNESS_CUTOFF].copy()
    log.append({"step": "3a. |S/K − 1| ≤ 10%", "n": int(len(work))})

    # 3b. Liquidity / unreliable quotes
    keep = np.ones(len(work), dtype=bool)
    c = _premium(work)
    keep &= np.isfinite(c.to_numpy(dtype=float)) & (c.to_numpy(dtype=float) >= MIN_PREMIUM)
    if {"bid", "ask"}.issubset(work.columns):
        bid = pd.to_numeric(work["bid"], errors="coerce")
        ask = pd.to_numeric(work["ask"], errors="coerce")
        mid = 0.5 * (bid + ask)
        rel = (ask - bid) / mid.replace(0, np.nan)
        keep &= (bid.to_numpy(dtype=float) > 0) & (ask.to_numpy(dtype=float) > bid.to_numpy(dtype=float))
        keep &= np.isfinite(rel.to_numpy(dtype=float)) & (rel.to_numpy(dtype=float) <= MAX_REL_SPREAD)
    if "volume" in work.columns:
        vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
        keep &= vol.to_numpy(dtype=float) >= MIN_VOLUME
    work = work.loc[keep].copy()
    log.append({"step": "3b. liquidity / wide bid–ask", "n": int(len(work))})

    if "option_price" not in work.columns or work["option_price"].isna().all():
        work["option_price"] = _premium(work)
    work = work.drop(columns=["_C"], errors="ignore")
    if audit:
        return work.reset_index(drop=True), log
    return work.reset_index(drop=True)
