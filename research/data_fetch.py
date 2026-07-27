"""
Step 1 — Data acquisition for jump-diffusion research.

Downloads daily close prices covering 2007–2018, cleans missing values,
and saves research/data/prices_clean.csv.

Required tickers (always fetched):
  SPY  — State Street official NAV history
  AAPL — Twelve Data (demo key works)

Optional secondary tickers (JPM, XOM):
  Fetched when TWELVEDATA_API_KEY is set, or when Yahoo Finance is available.
  Get a free key in ~10s: https://twelvedata.com

  export TWELVEDATA_API_KEY=your_key
  python data_fetch.py
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path(__file__).resolve().parent
DATA_DIR = RESEARCH_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_clean.csv"

REQUIRED_TICKERS = ["SPY", "AAPL"]
OPTIONAL_TICKERS = ["JPM", "XOM"]
TICKERS = REQUIRED_TICKERS + OPTIONAL_TICKERS  # target set for research

START = "2007-01-01"
END = "2018-12-31"

SSGA_SPY_URL = (
    "https://www.ssga.com/us/en/intermediaries/etfs/library-content/"
    "products/fund-data/etfs/us/navhist-us-en-spy.xlsx"
)
TWELVE_URL = "https://api.twelvedata.com/time_series"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def _filter_window(series: pd.Series, start: str = START, end: str = END) -> pd.Series:
    out = series.sort_index()
    out = out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    out = out.dropna()
    if out.empty:
        raise ValueError("Empty series after date filter")
    return out


def _assert_full_sample(series: pd.Series, ticker: str) -> pd.Series:
    """Reject truncated series that miss the 2007–2009 crisis regime."""
    if series.index.min() > pd.Timestamp("2007-06-01"):
        raise ValueError(
            f"{ticker} starts {series.index.min().date()} — missing crisis window"
        )
    if len(series) < 2500:
        raise ValueError(
            f"{ticker} too short ({len(series)} rows); need ~3000 for 2007–2018"
        )
    return series


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------
def download_spy_ssga(start: str = START, end: str = END) -> pd.Series:
    """Official State Street SPY NAV history."""
    print("  SSGA NAV history for SPY...")
    resp = SESSION.get(SSGA_SPY_URL, timeout=60)
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content), header=None)
    body = raw.iloc[3:, [0, 1]].copy()
    body.columns = ["Date", "SPY"]
    body["Date"] = pd.to_datetime(body["Date"], format="mixed", errors="coerce")
    body["SPY"] = pd.to_numeric(body["SPY"], errors="coerce")
    body = body.dropna().set_index("Date")["SPY"].astype(float)
    body.name = "SPY"
    series = _assert_full_sample(_filter_window(body, start, end), "SPY")
    print(f"    ✓ SPY: {len(series)} rows via SSGA")
    return series


def download_twelve_data(
    ticker: str,
    start: str = START,
    end: str = END,
) -> pd.Series:
    """Twelve Data daily closes. Demo key supports AAPL; env key unlocks others."""
    key = os.environ.get("TWELVEDATA_API_KEY") or "demo"
    key_label = "env" if os.environ.get("TWELVEDATA_API_KEY") else "demo"
    print(f"  Twelve Data {ticker} (key={key_label})...")
    params = {
        "symbol": ticker,
        "interval": "1day",
        "start_date": start,
        "end_date": end,
        "apikey": key,
        "format": "CSV",
        "outputsize": 5000,
    }
    resp = SESSION.get(TWELVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    text = resp.text.strip()
    if text.startswith("{"):
        raise ValueError(text[:200])
    df = pd.read_csv(io.StringIO(text), sep=";")
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Unexpected Twelve Data columns: {list(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    series = df.set_index("datetime")["close"].astype(float).sort_index()
    series.name = ticker
    series = _assert_full_sample(_filter_window(series, start, end), ticker)
    print(f"    ✓ {ticker}: {len(series)} rows via Twelve Data")
    return series


def download_yahoo_chart(
    ticker: str,
    start: str = START,
    end: str = END,
    max_retries: int = 3,
) -> pd.Series:
    """Yahoo Finance chart API v8 (often rate-limited; retries with backoff)."""
    period1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Yahoo chart {ticker} (attempt {attempt}/{max_retries})...")
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            result = (resp.json().get("chart") or {}).get("result")
            if not result:
                raise ValueError(f"Yahoo empty for {ticker}")
            block = result[0]
            timestamps = block.get("timestamp") or []
            indicators = block.get("indicators") or {}
            adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
            close = (indicators.get("quote") or [{}])[0].get("close")
            values = adj if adj is not None else close
            dates = pd.to_datetime(timestamps, unit="s").tz_localize(None).normalize()
            series = pd.Series(values, index=dates, name=ticker, dtype=float).dropna()
            series = _assert_full_sample(_filter_window(series, start, end), ticker)
            print(f"    ✓ {ticker}: {len(series)} rows via Yahoo")
            return series
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"    Yahoo failed: {exc}")
            time.sleep(8 * attempt)
    raise RuntimeError(f"Yahoo failed for {ticker}") from last_error


def download_one(ticker: str, required: bool) -> pd.Series | None:
    """Try sources in priority order. Optional tickers return None on failure."""
    errors: list[str] = []

    if ticker == "SPY":
        try:
            return download_spy_ssga()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SSGA: {exc}")

    try:
        return download_twelve_data(ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"TwelveData: {exc}")

    try:
        return download_yahoo_chart(ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Yahoo: {exc}")

    message = f"Could not download {ticker}. Tried: {errors}."
    if required:
        raise RuntimeError(message)
    print(f"  ⚠ Skipping optional {ticker}: {message}")
    print("    Tip: export TWELVEDATA_API_KEY=... then re-run to add JPM/XOM.")
    return None


def download_prices() -> pd.DataFrame:
    series_list: list[pd.Series] = []

    for ticker in REQUIRED_TICKERS:
        series_list.append(download_one(ticker, required=True))  # type: ignore[arg-type]
        time.sleep(0.3)

    for ticker in OPTIONAL_TICKERS:
        series = download_one(ticker, required=False)
        if series is not None:
            series_list.append(series)
        time.sleep(0.3)

    prices = pd.concat(series_list, axis=1)
    return _normalize_index(prices)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then drop remaining NaNs."""
    return _normalize_index(prices.ffill().dropna())


def _cache_is_valid(cached: pd.DataFrame) -> bool:
    if not all(t in cached.columns for t in REQUIRED_TICKERS):
        return False
    if len(cached) < 2500:
        return False
    if cached.index.min() > pd.Timestamp("2007-06-01"):
        return False
    return True


def load_or_download(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if PRICES_PATH.exists() and not force:
        print(f"Loading cached prices from {PRICES_PATH}")
        cached = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        if _cache_is_valid(cached):
            print(f"  Cache OK: {len(cached)} rows, {list(cached.columns)}")
            return cached
        print("  Cache incomplete — re-downloading")

    print(f"Downloading prices from {START} to {END}...")
    print(f"  Required: {REQUIRED_TICKERS}")
    print(f"  Optional: {OPTIONAL_TICKERS}")
    prices = clean_prices(download_prices())
    prices.to_csv(PRICES_PATH)
    print(f"✓ Saved: {PRICES_PATH}")
    missing_optional = [t for t in OPTIONAL_TICKERS if t not in prices.columns]
    if missing_optional:
        print(f"  Note: optional tickers not included yet: {missing_optional}")
    return prices


def main() -> None:
    print("=" * 70)
    print("STEP 1 — DATA FETCH (prices)")
    print("=" * 70)

    prices = load_or_download(force=False)

    print("\nPrice summary:")
    print(f"  Tickers: {list(prices.columns)}")
    print(f"  Date range: {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"  Observations: {len(prices)}")
    print(f"  Missing values: {int(prices.isna().sum().sum())}")
    print("\n  First 5 rows:")
    print(prices.head())
    print("\n  Last 5 rows:")
    print(prices.tail())
    print("\n✓ Data fetch complete")


if __name__ == "__main__":
    main()
