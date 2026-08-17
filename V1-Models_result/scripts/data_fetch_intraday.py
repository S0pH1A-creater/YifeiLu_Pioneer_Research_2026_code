"""
Free intraday bars for SPY / AAPL / MSFT.

Yahoo's 7d/60d/2y windows are attempted first. If Yahoo rate-limits (typical),
fall back to FirstRate's free 1-minute sample (~1 year, currently 2022-09 → 2023-09)
and derive the other intervals by resampling.

Layout under research/data/equity/intraday/:

  source_1min/   full free 1-minute source (extended hours kept)
  7d_1min/       last 8 regular sessions, 1-minute  → USED by 7d_1min_*.ipynb
                 (8th day is lookback buffer; notebooks evaluate the last 7)
  60d_2min/      last 60 regular sessions, 2-minute  (stored, unused)
  60d_5min/      last 60 regular sessions, 5-minute  (stored, unused)
  60d_15min/     last 60 regular sessions, 15-minute (stored, unused)
  60d_30min/     last 60 regular sessions, 30-minute (stored, unused)
  1h/            1-hour bars from the free sample    (stored, unused)
                 labelled 1h not 2y unless Yahoo 2y actually arrived

Regular session = 09:30–16:00 America/New_York. Timestamps are tz-naive ET.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "research" / "data" / "equity" / "intraday"

TICKERS = ["SPY", "AAPL", "MSFT"]
FRD_ZIP = "https://frd001.s3-us-east-2.amazonaws.com/{t}_1min_sample_firstratedata.zip"

SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
SESSION_CLOSE = pd.Timedelta(hours=16)


def _curl_session():
    from curl_cffi import requests as cr

    return cr.Session(impersonate="chrome")


def _to_naive_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, "tz", None) is None:
        return pd.DatetimeIndex(idx)
    return idx.tz_convert("America/New_York").tz_localize(None)


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    tod = df["Datetime"] - df["Datetime"].dt.normalize()
    out = df.loc[(tod >= SESSION_OPEN) & (tod < SESSION_CLOSE)].copy()
    return out.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).reset_index(drop=True)


def _write_ticker_frames(folder: Path, frames: dict[str, pd.DataFrame]) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    info = {}
    for t, df in frames.items():
        path = folder / f"{t}.csv"
        df.to_csv(path, index=False)
        info[t] = {
            "path": str(path.relative_to(ROOT)),
            "n_bars": int(len(df)),
            "start": str(df["Datetime"].min()) if len(df) else None,
            "end": str(df["Datetime"].max()) if len(df) else None,
        }
    return info


def try_yahoo() -> dict[str, dict[str, pd.DataFrame]] | None:
    """Return {label: {ticker: df}} or None if Yahoo is blocked/empty."""
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        print(f"yfinance unavailable: {exc}", flush=True)
        return None

    specs = {
        "7d_1min": ("1m", "7d"),
        "60d_2min": ("2m", "60d"),
        "60d_5min": ("5m", "60d"),
        "60d_15min": ("15m", "60d"),
        "60d_30min": ("30m", "60d"),
        "1h": ("1h", "2y"),
    }
    out: dict[str, dict[str, pd.DataFrame]] = {k: {} for k in specs}
    try:
        for label, (interval, period) in specs.items():
            for ticker in TICKERS:
                print(f"  Yahoo {ticker} {interval}/{period}...", flush=True)
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    prepost=False,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if raw is None or raw.empty:
                    raise ValueError(f"empty {ticker} {interval}")
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.rename(columns=str.title)
                df = pd.DataFrame(
                    {
                        "Datetime": _to_naive_et(pd.DatetimeIndex(raw.index)),
                        "Open": raw["Open"].to_numpy(),
                        "High": raw["High"].to_numpy(),
                        "Low": raw["Low"].to_numpy(),
                        "Close": raw["Close"].to_numpy(),
                        "Volume": raw["Volume"].to_numpy() if "Volume" in raw.columns else 0,
                    }
                ).dropna(subset=["Close"])
                df = regular_hours(df)
                if df.empty:
                    raise ValueError(f"empty after RTH {ticker} {interval}")
                out[label][ticker] = df
                print(f"    ✓ {len(df)} bars {df['Datetime'].min()} → {df['Datetime'].max()}", flush=True)
                time.sleep(1.0)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"Yahoo path failed ({exc}); will use FirstRate free 1-min sample.", flush=True)
        return None


def download_firstrate_1min() -> dict[str, pd.DataFrame]:
    s = _curl_session()
    frames = {}
    for t in TICKERS:
        url = FRD_ZIP.format(t=t)
        print(f"  FirstRate {t} 1min sample...", flush=True)
        resp = s.get(url, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"{t} HTTP {resp.status_code}")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            raw = zf.read(name)
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip().title() if c.strip().lower() != "timestamp" else "Datetime" for c in df.columns]
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.dropna(subset=["Close"]).drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
        frames[t] = df.reset_index(drop=True)
        print(
            f"    ✓ {t}: {len(df)} bars  {df['Datetime'].min()} → {df['Datetime'].max()}",
            flush=True,
        )
    return frames


def last_n_sessions(df: pd.DataFrame, n: int) -> pd.DataFrame:
    days = df["Datetime"].dt.normalize().drop_duplicates().sort_values()
    keep = set(days.iloc[-n:])
    return df.loc[df["Datetime"].dt.normalize().isin(keep)].copy().reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("Datetime").sort_index()
    out = g.resample(rule, label="right", closed="right").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna(subset=["Close"])
    return out.reset_index()


def from_firstrate(source: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    rth = {t: regular_hours(df) for t, df in source.items()}
    out: dict[str, dict[str, pd.DataFrame]] = {
        "source_1min": source,
        "7d_1min": {t: last_n_sessions(rth[t], 8) for t in TICKERS},
        "60d_2min": {},
        "60d_5min": {},
        "60d_15min": {},
        "60d_30min": {},
        "1h": {},
    }
    for t in TICKERS:
        d60 = last_n_sessions(rth[t], 60)
        out["60d_2min"][t] = resample_ohlcv(d60, "2min")
        out["60d_5min"][t] = resample_ohlcv(d60, "5min")
        out["60d_15min"][t] = resample_ohlcv(d60, "15min")
        out["60d_30min"][t] = resample_ohlcv(d60, "30min")
        out["1h"][t] = resample_ohlcv(rth[t], "1h")
        print(
            f"  derived {t}: 7d_1min={len(out['7d_1min'][t])}  "
            f"60d_5min={len(out['60d_5min'][t])}  1h={len(out['1h'][t])}",
            flush=True,
        )
    return out


def fetch_all() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Trying Yahoo free windows...", flush=True)
    yahoo = try_yahoo()
    if yahoo is not None:
        source_name = "Yahoo Finance via yfinance (regular session)"
        bundles = yahoo
        unused = ["60d_2min", "60d_5min", "60d_15min", "60d_30min", "1h"]
        used = ["7d_1min"]
    else:
        print("Downloading FirstRate free 1-minute samples...", flush=True)
        source = download_firstrate_1min()
        bundles = from_firstrate(source)
        source_name = (
            "FirstRate free 1-minute sample (Yahoo 7d/60d/2y blocked). "
            "7d/60d/1h derived from that 1-minute file; 1h is ~1 year, not 2 years."
        )
        unused = ["source_1min", "60d_2min", "60d_5min", "60d_15min", "60d_30min", "1h"]
        used = ["7d_1min"]

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source_name,
        "session": "regular hours 09:30–16:00 ET (source_1min keeps extended hours)",
        "timezone": "America/New_York (tz-naive)",
        "tickers": TICKERS,
        "used_by_notebooks": used,
        "stored_unused": unused,
        "datasets": {},
    }
    for label, frames in bundles.items():
        meta["datasets"][label] = _write_ticker_frames(DATA_DIR / label, frames)
    meta_path = DATA_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    readme = DATA_DIR / "README.md"
    readme.write_text(
        "# Intraday equity bars\n\n"
        f"- Source: {source_name}\n"
        "- Tickers: SPY (primary), AAPL, MSFT (same names as the daily notebooks)\n"
        "- **Used now:** `7d_1min/` — 1-minute bars, last 8 regular sessions "
        "(notebooks evaluate the last 7; the extra session is the 1-day lookback buffer)\n"
        "- **Stored, not used yet:** `60d_2min`, `60d_5min`, `60d_15min`, `60d_30min`, `1h` "
        "(and `source_1min` when FirstRate is the source)\n"
        "- See `metadata.json` for bar counts and timestamps.\n"
    )
    print(f"\nWrote {meta_path}", flush=True)
    return meta


if __name__ == "__main__":
    fetch_all()
