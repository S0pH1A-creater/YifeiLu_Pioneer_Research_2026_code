#!/usr/bin/env python3
"""Fill the short-interval (2022-09-30 → 2023-09-29) research gap.

Does not overwrite 2008–2020 files. Writes:

  research/data/equity/short_interval/
  research/data/rates/risk_free_dgs3mo_short_interval.csv
  research/data/options/processed/short_interval/
  research/data/options/raw/short_interval/   (AAPL/MSFT year slices)
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from cobweb_to_parquet import FILE_RE, _parse_one_file  # noqa: E402

DATA = REPO / "research" / "data"
EQUITY = DATA / "equity" / "short_interval"
RATES_PATH = DATA / "rates" / "risk_free_dgs3mo_short_interval.csv"
OPT_PROC = DATA / "options" / "processed" / "short_interval"
OPT_RAW = DATA / "options" / "raw" / "short_interval"
SOURCE_1MIN = DATA / "equity" / "intraday" / "source_1min"
COBWEB = DATA / "options" / "raw" / "_staging"
SPY_PARQUET = DATA / "options" / "raw" / "SPY_options.parquet"

TICKERS = ("SPY", "AAPL", "MSFT")
# 1-min sample on disk; 30 Sep 2023 is Saturday
EVAL_START = pd.Timestamp("2022-09-30")
EVAL_END = pd.Timestamp("2023-09-29")
# 6-month lookback for daily calibration, matching the 2-year study
DAILY_START = pd.Timestamp("2022-03-30")
FRED_START = "2022-03-01"
FRED_END = "2023-09-29"

ATM_BAND = 0.10
MIN_DTE = 7
MAX_DTE = 60
MIN_VOLUME = 1
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
SESSION_CLOSE = pd.Timedelta(hours=16)
GAP_MIN = pd.Timedelta(minutes=2)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})


def _get(url: str, **kw) -> requests.Response:
    resp = SESSION.get(url, timeout=kw.pop("timeout", 120), **kw)
    resp.raise_for_status()
    return resp


def fetch_fred() -> pd.Series:
    RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"FRED DGS3MO {FRED_START} → {FRED_END}", flush=True)
    if not RATES_PATH.exists() or RATES_PATH.stat().st_size < 500:
        raise RuntimeError(f"Missing {RATES_PATH}; write DGS3MO CSV first")
    rf = pd.read_csv(RATES_PATH)
    rf.columns = ["date", "dgs3mo"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["dgs3mo"] = pd.to_numeric(rf["dgs3mo"], errors="coerce")
    s = rf.dropna().set_index("date")["dgs3mo"] / 100.0
    s = s.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="B")
    s = s.reindex(full).ffill().bfill()
    s.index.name = "trading_date"
    s.name = "r"
    # rewrite as clean decimal series plus percent column for FRED-style cache
    out = pd.DataFrame({"observation_date": s.index, "DGS3MO": s.values * 100.0})
    out.to_csv(RATES_PATH, index=False)
    print(f"  {len(s)} business days  {s.index.min().date()} → {s.index.max().date()}", flush=True)
    return s


def fetch_daily_prices() -> pd.DataFrame:
    dest = EQUITY / "prices_daily.csv"
    ret_path = EQUITY / "log_returns_daily.csv"
    if dest.exists() and ret_path.exists():
        prices = pd.read_csv(dest, index_col=0, parse_dates=True)
        print(f"  using cached daily prices {len(prices)}  {prices.index.min().date()} → {prices.index.max().date()}", flush=True)
        return prices
    frames = []
    for ticker in TICKERS:
        series = None
        url = f"https://raw.githubusercontent.com/dieperdev/yfinance-stock-data/main/data/{ticker}.csv"
        try:
            print(f"  GitHub YF mirror {ticker}…", flush=True)
            resp = _get(url, timeout=60)
            df = pd.read_csv(io.StringIO(resp.text))
            df["Date"] = pd.to_datetime(df["Date"])
            series = df.set_index("Date")["Adj Close"].astype(float).sort_index()
        except Exception as exc:  # noqa: BLE001
            print(f"    mirror failed ({exc}); Yahoo…", flush=True)
            period1 = int(datetime(2022, 3, 1, tzinfo=timezone.utc).timestamp())
            period2 = int(datetime(2023, 10, 1, tzinfo=timezone.utc).timestamp())
            last = None
            for host in ("query1", "query2"):
                try:
                    yurl = (
                        f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
                        f"?period1={period1}&period2={period2}&interval=1d"
                    )
                    block = _get(yurl, timeout=40).json()["chart"]["result"][0]
                    ts = pd.to_datetime(block["timestamp"], unit="s").tz_localize(None).normalize()
                    adj = (block["indicators"].get("adjclose") or [{}])[0].get("adjclose")
                    close = (block["indicators"].get("quote") or [{}])[0].get("close")
                    series = pd.Series(adj or close, index=ts, dtype=float).dropna()
                    last = None
                    break
                except Exception as exc2:  # noqa: BLE001
                    last = exc2
                    time.sleep(2)
            if series is None:
                raise RuntimeError(f"no daily prices for {ticker}: {last}")
        series = series[(series.index >= DAILY_START) & (series.index <= EVAL_END)].dropna()
        series.name = ticker
        print(f"    {ticker}: {len(series)}  {series.index.min().date()} → {series.index.max().date()}", flush=True)
        frames.append(series)
        time.sleep(0.3)
    prices = pd.concat(frames, axis=1).sort_index().ffill().dropna()
    prices.index.name = "Date"
    dest = EQUITY / "prices_daily.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(dest)
    rets = np.log(prices / prices.shift(1)).dropna()
    rets.to_csv(EQUITY / "log_returns_daily.csv")
    print(f"  daily prices {len(prices)}  returns {len(rets)}", flush=True)
    return prices


def rth_1min() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    dest_dir = EQUITY / "prices_1min_rth"
    ret_dir = EQUITY / "log_returns_1min_rth"
    dest_dir.mkdir(parents=True, exist_ok=True)
    ret_dir.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        df = pd.read_csv(SOURCE_1MIN / f"{ticker}.csv", parse_dates=["Datetime"])
        tod = df["Datetime"] - df["Datetime"].dt.normalize()
        df = df.loc[(tod >= SESSION_OPEN) & (tod < SESSION_CLOSE)].copy()
        df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
        df = df.loc[(df["Datetime"] >= EVAL_START) & (df["Datetime"] <= EVAL_END + pd.Timedelta(hours=16))]
        df.to_csv(dest_dir / f"{ticker}.csv", index=False)
        px = df.set_index("Datetime")["Close"].astype(float)
        lr = np.log(px / px.shift(1))
        gap = pd.Series(px.index, index=px.index).diff() > GAP_MIN
        lr.loc[gap] = np.nan
        lr.name = ticker
        lr.to_csv(ret_dir / f"{ticker}.csv", header=True)
        out[ticker] = df
        print(
            f"  1-min RTH {ticker}: {len(df)} bars  {df['Datetime'].min()} → {df['Datetime'].max()}  "
            f"sessions={df['Datetime'].dt.normalize().nunique()}",
            flush=True,
        )
    return out


def _eod_from_1min(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = []
    for ticker, df in frames.items():
        s = df.set_index("Datetime")["Close"].astype(float)
        eod = s.groupby(s.index.normalize()).last().rename(ticker)
        cols.append(eod)
    return pd.concat(cols, axis=1).sort_index()


def cobweb_year(ticker: str) -> pd.DataFrame:
    cached = OPT_RAW / f"{ticker}_options.parquet"
    if cached.exists() and cached.stat().st_size > 100_000:
        print(f"  using cached {cached.name}", flush=True)
        return pd.read_parquet(cached)
    zip_path = COBWEB / f"{ticker}.zip"
    rows: list[dict] = []
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = FILE_RE.match(name)
            if not m or m.group("ticker").upper() != ticker:
                continue
            trading_date = pd.Timestamp(m.group("date"))
            if trading_date < EVAL_START or trading_date > EVAL_END:
                continue
            n += 1
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
            rows.extend(_parse_one_file(text, ticker, trading_date))
            if n % 50 == 0:
                print(f"    {ticker} parsed {n} days, {len(rows):,} rows", flush=True)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "expiration", "strike", "mark", "type"])
    OPT_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, index=False)
    print(f"  {ticker} cobweb year: {n} files, {len(df):,} rows → {cached.name}", flush=True)
    return df


def spy_year() -> pd.DataFrame:
    table = pq.read_table(
        SPY_PARQUET,
        filters=[
            ("date", ">=", EVAL_START.to_datetime64()),
            ("date", "<=", EVAL_END.to_datetime64()),
        ],
        columns=[
            "date", "expiration", "strike", "type", "mark", "bid", "ask",
            "volume", "symbol",
        ],
    )
    df = table.to_pandas()
    print(f"  SPY parquet year: {len(df):,} rows", flush=True)
    return df


def _normalize_raw(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["trading_date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"]).dt.normalize()
    if "strike" in out.columns:
        out["K"] = out["strike"].astype(float)
    out["option_price"] = pd.to_numeric(out.get("mark", out.get("option_price")), errors="coerce")
    out["option_type"] = (
        out["type"].astype(str).str.lower().str.strip()
        .replace({"c": "call", "p": "put"})
    )
    out["underlying"] = ticker
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0) if "volume" in out.columns else 1.0
    if "bid" in out.columns:
        out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    if "ask" in out.columns:
        out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    if "underlying_last" in out.columns:
        out["S_broker"] = pd.to_numeric(out["underlying_last"], errors="coerce")
    return out


def build_panel(raw: pd.DataFrame, ticker: str, eod: pd.Series, rf: pd.Series) -> pd.DataFrame:
    df = _normalize_raw(raw, ticker)
    df = df[df["option_type"].isin(["call", "put"])]
    df = df.dropna(subset=["option_price", "K", "trading_date", "expiration"])
    df = df[df["option_price"] > 0]
    df = df[df["volume"].fillna(0) >= MIN_VOLUME]
    if {"bid", "ask"}.issubset(df.columns):
        df = df[(df["bid"].fillna(0) > 0) & (df["ask"] > df["bid"])]
    if "S_broker" in df.columns and df["S_broker"].notna().any():
        df["S_t"] = df["S_broker"]
        missing = df["S_t"].isna()
        if missing.any():
            df.loc[missing, "S_t"] = df.loc[missing, "trading_date"].map(eod)
    else:
        df["S_t"] = df["trading_date"].map(eod)
    df["r"] = df["trading_date"].map(rf)
    df["r"] = df["r"].ffill().bfill()
    df = df.dropna(subset=["S_t", "r"])
    df["dte"] = (df["expiration"] - df["trading_date"]).dt.days
    df = df[(df["dte"] >= MIN_DTE) & (df["dte"] <= MAX_DTE)]
    df["T_years"] = df["dte"] / 365.25
    df["moneyness"] = df["K"] / df["S_t"]
    df = df[df["moneyness"].between(1 - ATM_BAND, 1 + ATM_BAND)]
    intrinsic = np.where(
        df["option_type"] == "call",
        np.maximum(df["S_t"] - df["K"], 0.0),
        np.maximum(df["K"] - df["S_t"], 0.0),
    )
    df = df[df["option_price"] >= 0.5 * intrinsic]
    df = df[df["option_price"] >= 0.05]
    df["style"] = "American"
    df["regime"] = "short_interval"
    cols = [
        "underlying", "trading_date", "S_t", "K", "expiration", "T_years", "dte",
        "option_type", "option_price", "r", "moneyness", "regime", "style",
    ]
    return df[cols].sort_values(["trading_date", "expiration", "option_type", "K"]).reset_index(drop=True)


def write_panels(panels: dict[str, pd.DataFrame]) -> None:
    OPT_PROC.mkdir(parents=True, exist_ok=True)
    summary = []
    for ticker, panel in panels.items():
        panel.to_csv(OPT_PROC / f"{ticker}_options_panel.csv", index=False)
        calls = panel[panel["option_type"] == "call"]
        calls.to_csv(OPT_PROC / f"{ticker}_calls_panel.csv", index=False)
        print(
            f"  {ticker}: {len(panel):,} quotes  {int((panel.option_type=='call').sum()):,} calls  "
            f"{panel['trading_date'].nunique()} dates  "
            f"{panel['trading_date'].min().date()} → {panel['trading_date'].max().date()}",
            flush=True,
        )
        summary.append({
            "underlying": ticker,
            "n_quotes": len(panel),
            "n_calls": int((panel["option_type"] == "call").sum()),
            "n_puts": int((panel["option_type"] == "put").sum()),
            "n_dates": int(panel["trading_date"].nunique()),
            "start": str(panel["trading_date"].min().date()),
            "end": str(panel["trading_date"].max().date()),
            "mean_S": float(panel["S_t"].mean()),
            "mean_option_price": float(panel["option_price"].mean()),
            "mean_dte": float(panel["dte"].mean()),
        })
    combined = pd.concat(panels.values(), ignore_index=True)
    combined.to_csv(OPT_PROC / "options_panel_all.csv", index=False)
    combined[combined["option_type"] == "call"].to_csv(OPT_PROC / "calls_panel_all.csv", index=False)
    pd.DataFrame(summary).to_csv(OPT_PROC / "options_summary.csv", index=False)


def main() -> int:
    t0 = time.time()
    EQUITY.mkdir(parents=True, exist_ok=True)
    print("=== 1. FRED rates ===", flush=True)
    rf = fetch_fred()
    print("=== 2. Daily adj close + log returns ===", flush=True)
    daily = fetch_daily_prices()
    print("=== 3. RTH 1-minute prices + log returns ===", flush=True)
    intra = rth_1min()
    eod = _eod_from_1min(intra)
    print("=== 4. Listed American options ===", flush=True)
    panels = {}
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---", flush=True)
        raw = spy_year() if ticker == "SPY" else cobweb_year(ticker)
        px = eod[ticker] if ticker in eod.columns else daily[ticker]
        panels[ticker] = build_panel(raw, ticker, px, rf)
    write_panels(panels)

    meta = {
        "eval_window": f"{EVAL_START.date()}_to_{EVAL_END.date()}",
        "daily_lookback_start": str(DAILY_START.date()),
        "tickers": list(TICKERS),
        "filters": {"atm_band": ATM_BAND, "min_dte": MIN_DTE, "max_dte": MAX_DTE, "min_volume": MIN_VOLUME},
        "does_not_overwrite": [
            "equity/prices_clean.csv",
            "rates/risk_free_dgs3mo.csv",
            "options/processed/*_calls_panel.csv",
        ],
        "seconds": round(time.time() - t0, 1),
    }
    (EQUITY / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {time.time()-t0:.0f}s → {EQUITY.relative_to(REPO)} and {OPT_PROC.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
