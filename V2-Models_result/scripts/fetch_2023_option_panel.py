#!/usr/bin/env python3
"""Build 2023-03-09 → 2023-03-15 American-call panels (listed quotes).

Sources already on disk:
  SPY  — research/data/options/raw/SPY_options.parquet (through 2025)
  AAPL/MSFT — Cobweb ToS EOD ZIPs in raw/_staging (through Jun 2024)

Does not overwrite the 2008–2020 2-year panels. Writes:
  research/data/options/processed/2023-03-09_to_2023-03-15/{TICKER}_calls_panel.csv
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from cobweb_to_parquet import FILE_RE, _parse_one_file  # noqa: E402

DATA = REPO / "research" / "data"
RAW = DATA / "options" / "raw"
STAGING = RAW / "_staging"
WINDOW_ID = "2023-03-09_to_2023-03-15"
OUT = DATA / "options" / "processed" / WINDOW_ID
INTRA = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID
RATES = INTRA / "dgs3mo.csv"

START = pd.Timestamp("2023-03-09")
END = pd.Timestamp("2023-03-15")
TICKERS = ("SPY", "AAPL", "MSFT")
ATM_BAND = 0.10
MIN_DTE = 7
MAX_DTE = 60
MIN_VOLUME = 1


def _eod_close(ticker: str) -> pd.Series:
    p = pd.read_csv(INTRA / f"{ticker}.csv", parse_dates=["Datetime"])
    s = p.set_index("Datetime")["Close"].sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.groupby(s.index.normalize()).last().rename("S_t")


def _rates() -> pd.Series:
    r = pd.read_csv(RATES, parse_dates=["observation_date"])
    s = r.set_index("observation_date")["DGS3MO"].astype(float) / 100.0
    s.index = pd.to_datetime(s.index).normalize()
    s.name = "r"
    return s.sort_index()


def _spy_raw() -> pd.DataFrame:
    path = RAW / "SPY_options.parquet"
    table = pq.read_table(
        path,
        filters=[
            ("date", ">=", START.to_datetime64()),
            ("date", "<=", END.to_datetime64()),
            ("type", "==", "call"),
        ],
        columns=["date", "expiration", "strike", "type", "mark", "bid", "ask", "volume", "symbol"],
    )
    df = table.to_pandas()
    df["trading_date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["K"] = df["strike"].astype(float)
    df["option_price"] = pd.to_numeric(df["mark"], errors="coerce")
    df["option_type"] = "call"
    df["underlying"] = "SPY"
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    return df


def _cobweb_week(ticker: str) -> pd.DataFrame:
    zip_path = STAGING / f"{ticker}.zip"
    rows: list[dict] = []
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = FILE_RE.match(name)
            if not m or m.group("ticker").upper() != ticker:
                continue
            trading_date = pd.Timestamp(m.group("date"))
            if trading_date < START or trading_date > END:
                continue
            n += 1
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
            rows.extend(_parse_one_file(text, ticker, trading_date))
    print(f"  {ticker} cobweb: {n} files, {len(rows):,} raw rows", flush=True)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "expiration", "strike", "mark", "type"])
    df = df[df["type"].astype(str).str.lower().eq("call")]
    df["trading_date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["K"] = df["strike"].astype(float)
    df["option_price"] = pd.to_numeric(df["mark"], errors="coerce")
    df["option_type"] = "call"
    df["underlying"] = ticker
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    if "underlying_last" in df.columns:
        df["S_broker"] = pd.to_numeric(df["underlying_last"], errors="coerce")
    return df


def _to_panel(df: pd.DataFrame, ticker: str, px: pd.Series, rf: pd.Series) -> pd.DataFrame:
    out = df.copy()
    if "S_broker" in out.columns and out["S_broker"].notna().any():
        out["S_t"] = out["S_broker"]
        missing = out["S_t"].isna()
        if missing.any():
            out.loc[missing, "S_t"] = out.loc[missing, "trading_date"].map(px)
    else:
        out["S_t"] = out["trading_date"].map(px)
    out["r"] = out["trading_date"].map(rf).ffill().bfill()
    out = out.dropna(subset=["option_price", "K", "trading_date", "expiration", "S_t", "r"])
    out = out[out["option_price"] > 0]
    out = out[out["volume"].fillna(0) >= MIN_VOLUME]
    if {"bid", "ask"}.issubset(out.columns):
        out = out[(out["bid"].fillna(0) > 0) & (out["ask"] > out["bid"])]
    out["dte"] = (out["expiration"] - out["trading_date"]).dt.days
    out = out[(out["dte"] >= MIN_DTE) & (out["dte"] <= MAX_DTE)]
    out["T_years"] = out["dte"] / 365.25
    out["moneyness"] = out["K"] / out["S_t"]
    out = out[out["moneyness"].between(1 - ATM_BAND, 1 + ATM_BAND)]
    intrinsic = np.maximum(out["S_t"] - out["K"], 0.0)
    out = out[out["option_price"] >= 0.5 * intrinsic]
    out = out[out["option_price"] >= 0.05]
    out["style"] = "American"
    out["regime"] = WINDOW_ID
    cols = [
        "underlying", "trading_date", "S_t", "K", "expiration", "T_years", "dte",
        "option_type", "option_price", "r", "moneyness", "regime", "style",
    ]
    return (
        out[cols]
        .sort_values(["trading_date", "expiration", "K"])
        .reset_index(drop=True)
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rf = _rates()
    print(f"rates {rf.index.min().date()} → {rf.index.max().date()}", flush=True)
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===", flush=True)
        px = _eod_close(ticker)
        if ticker == "SPY":
            raw = _spy_raw()
            print(f"  SPY parquet calls in window: {len(raw):,}", flush=True)
        else:
            raw = _cobweb_week(ticker)
        panel = _to_panel(raw, ticker, px, rf)
        dest = OUT / f"{ticker}_calls_panel.csv"
        panel.to_csv(dest, index=False)
        dates = sorted(panel["trading_date"].dt.date.unique()) if len(panel) else []
        print(
            f"  wrote {dest.relative_to(REPO)}  n={len(panel):,}  "
            f"dates={dates}  dte={panel['dte'].min() if len(panel) else '—'}–"
            f"{panel['dte'].max() if len(panel) else '—'}",
            flush=True,
        )
        if len(panel):
            print(panel.head(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
