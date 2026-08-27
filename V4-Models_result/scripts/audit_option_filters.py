"""Funnel counts for V3 option estimation filters (raw parquet + processed panels)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from option_filters import (  # noqa: E402
    DTE_MAX,
    DTE_MIN,
    MAX_REL_SPREAD,
    MIN_PREMIUM,
    MIN_VOLUME,
    MONEYNESS_CUTOFF,
    apply_estimation_filters,
    moneyness_bucket,
    spot_over_strike,
)

RAW = REPO / "research" / "data" / "options" / "raw"
PROCESSED = REPO / "research" / "data" / "options" / "processed"
EQUITY = REPO / "research" / "data" / "equity" / "prices_clean.csv"

REGIMES = {
    "2008-2009": ("2008-08-01", "2009-07-31"),
    "2013-2014": ("2014-01-01", "2014-12-31"),
    "2018-2019": ("2018-10-01", "2019-09-30"),
    "2019-2020": ("2019-09-01", "2020-08-31"),
}
STEPS = [
    "finite S, K, C",
    "1. no-arbitrage",
    "2. maturity 7–60",
    "3a. |S/K−1|≤10%",
    "3b. liquidity",
]
BUCKETS = ["deep OTM", "OTM", "ATM", "ITM", "deep ITM"]


def _empty_counter():
    return {s: 0 for s in STEPS}


def _add(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def _normalize(df: pd.DataFrame, ticker: str, spy_px: pd.Series) -> pd.DataFrame:
    out = df.copy()
    rename = {"date": "trading_date", "strike": "K", "type": "option_type"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    out["underlying"] = ticker
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    if "expiration" in out.columns:
        out["expiration"] = pd.to_datetime(out["expiration"])
    if "underlying_last" in out.columns:
        out["S_t"] = pd.to_numeric(out["underlying_last"], errors="coerce")
    else:
        px = spy_px.rename("S_t")
        out = out.merge(px, left_on="trading_date", right_index=True, how="left")
    out["K"] = pd.to_numeric(out["K"], errors="coerce")
    if "option_type" in out.columns:
        out = out.loc[out["option_type"].astype(str).str.lower().eq("call")]
    return out


def _step_counts(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Return overall counts, counts by bucket, counts by V3 regime."""
    overall = _empty_counter()
    by_bucket = {b: _empty_counter() for b in BUCKETS}
    by_regime = {r: _empty_counter() for r in REGIMES}

    if df.empty:
        return overall, by_bucket, by_regime

    work = df.copy()
    if "option_type" in work.columns:
        work = work.loc[work["option_type"].astype(str).str.lower().eq("call")]
    C = pd.Series(np.nan, index=work.index, dtype=float)
    if {"bid", "ask"}.issubset(work.columns):
        bid = pd.to_numeric(work["bid"], errors="coerce")
        ask = pd.to_numeric(work["ask"], errors="coerce")
        mid = 0.5 * (bid + ask)
        valid = (bid > 0) & (ask > bid) & np.isfinite(mid)
        C = mid.where(valid)
    if "mark" in work.columns:
        mark = pd.to_numeric(work["mark"], errors="coerce")
        C = C.where(np.isfinite(C) & (C > 0), mark)
    if "option_price" in work.columns:
        px = pd.to_numeric(work["option_price"], errors="coerce")
        C = C.where(np.isfinite(C) & (C > 0), px)
    S = pd.to_numeric(work["S_t"], errors="coerce")
    K = pd.to_numeric(work["K"], errors="coerce")
    ok = S.gt(0) & K.gt(0) & np.isfinite(S) & np.isfinite(K) & np.isfinite(C)
    work = work.loc[ok]
    C, S, K = C.loc[ok], S.loc[ok], K.loc[ok]
    if work.empty:
        return overall, by_bucket, by_regime

    if "dte" in work.columns:
        dte = pd.to_numeric(work["dte"], errors="coerce")
    else:
        dte = (pd.to_datetime(work["expiration"]) - pd.to_datetime(work["trading_date"])).dt.days
    sk = S / K
    bucket = moneyness_bucket(sk).astype("object").fillna("deep OTM")
    dates = pd.to_datetime(work["trading_date"])

    m1 = C.to_numpy(dtype=float) + 1e-12 >= np.maximum(0.0, S.to_numpy(dtype=float) - K.to_numpy(dtype=float))
    m2 = dte.ge(DTE_MIN).to_numpy() & dte.le(DTE_MAX).to_numpy()
    m3a = np.abs(sk.to_numpy(dtype=float) - 1.0) <= MONEYNESS_CUTOFF
    m3b = np.isfinite(C.to_numpy(dtype=float)) & (C.to_numpy(dtype=float) >= MIN_PREMIUM)
    if {"bid", "ask"}.issubset(work.columns):
        bid = pd.to_numeric(work["bid"], errors="coerce")
        ask = pd.to_numeric(work["ask"], errors="coerce")
        mid = 0.5 * (bid + ask)
        rel = (ask - bid) / mid.replace(0, np.nan)
        m3b &= (bid.to_numpy(dtype=float) > 0) & (ask.to_numpy(dtype=float) > bid.to_numpy(dtype=float))
        m3b &= np.isfinite(rel.to_numpy(dtype=float)) & (rel.to_numpy(dtype=float) <= MAX_REL_SPREAD)
    if "volume" in work.columns:
        vol = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
        m3b &= vol.to_numpy(dtype=float) >= MIN_VOLUME

    masks = {
        "finite S, K, C": np.ones(len(work), dtype=bool),
        "1. no-arbitrage": m1,
        "2. maturity 7–60": m1 & m2,
        "3a. |S/K−1|≤10%": m1 & m2 & m3a,
        "3b. liquidity": m1 & m2 & m3a & m3b,
    }

    for step, mask in masks.items():
        overall[step] += int(mask.sum())
        for b in BUCKETS:
            by_bucket[b][step] += int(((bucket == b).to_numpy() & mask).sum())
        for name, (a, z) in REGIMES.items():
            in_r = (dates >= a) & (dates <= z)
            by_regime[name][step] += int((in_r.to_numpy() & mask).sum())
    return overall, by_bucket, by_regime


def audit_parquet(spy_px: pd.Series) -> dict:
    files = {
        "SPY": RAW / "SPY_options.parquet",
        "AAPL": RAW / "AAPL_options.parquet",
        "MSFT": RAW / "MSFT_options.parquet",
    }
    cols = {
        "SPY": ["date", "expiration", "strike", "type", "mark", "bid", "ask", "volume"],
        "AAPL": ["date", "expiration", "strike", "type", "mark", "bid", "ask", "volume", "underlying_last"],
        "MSFT": ["date", "expiration", "strike", "type", "mark", "bid", "ask", "volume", "underlying_last"],
    }
    overall = _empty_counter()
    by_ticker = {t: _empty_counter() for t in files}
    by_bucket = {b: _empty_counter() for b in BUCKETS}
    by_regime = {r: _empty_counter() for r in REGIMES}
    ticker_bucket = {t: {b: _empty_counter() for b in BUCKETS} for t in files}

    for ticker, path in files.items():
        print(f"  parquet {ticker} …", flush=True, file=sys.stderr)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=250_000, columns=cols[ticker]):
            df = _normalize(batch.to_pandas(), ticker, spy_px)
            o, bb, br = _step_counts(df)
            _add(overall, o)
            _add(by_ticker[ticker], o)
            for b, c in bb.items():
                _add(by_bucket[b], c)
                _add(ticker_bucket[ticker][b], c)
            for r, c in br.items():
                _add(by_regime[r], c)
    return {
        "overall": overall,
        "by_ticker": by_ticker,
        "by_bucket": by_bucket,
        "by_regime": by_regime,
        "ticker_bucket": ticker_bucket,
    }


def audit_processed() -> dict:
    out = {}
    for ticker in ("SPY", "AAPL", "MSFT"):
        path = PROCESSED / f"{ticker}_calls_panel.csv"
        df = pd.read_csv(path, parse_dates=["trading_date", "expiration"])
        filtered, log = apply_estimation_filters(df, audit=True)
        out[ticker] = {
            "input": int(len(df)),
            "steps": log,
            "kept": int(len(filtered)),
        }
        o, bb, br = _step_counts(df)
        out[ticker]["funnel"] = o
        out[ticker]["by_bucket"] = bb
    return out


def main() -> int:
    spy_px = (
        pd.read_csv(EQUITY, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()["SPY"]
    )
    print("processed panels …", flush=True, file=sys.stderr)
    processed = audit_processed()
    print("raw parquet …", flush=True, file=sys.stderr)
    raw = audit_parquet(spy_px)
    payload = {"processed_calls_panel": processed, "raw_parquet": raw}
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
