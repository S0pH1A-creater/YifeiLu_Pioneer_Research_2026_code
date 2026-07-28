"""
Step 1 — Equity price acquisition (DataCollection §1 / ResearchProposal-v2).

Downloads adjusted closes for:
  SPY (primary), AAPL + MSFT (secondary), JPM + XOM (auxiliary / retained)

Window: post-2000 through the end of the high-vol evaluation regime
(2000-01-01 → 2018-12-31). Regime subsets are carved out in data_prepare.py.

Sources (tried in order per ticker):
  SPY  — State Street NAV, then Yahoo chart API
  Others — GitHub YF mirror, Yahoo chart API, then Twelve Data
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
EQUITY_DIR = DATA_DIR / "equity"
PRICES_PATH = EQUITY_DIR / "prices_clean.csv"
# Backward-compatible alias used by older scripts
LEGACY_PRICES_PATH = DATA_DIR / "prices_clean.csv"

# Primary research set: SPY + AAPL/MSFT. JPM/XOM kept for completeness.
PRIMARY_TICKER = "SPY"
SECONDARY_TICKERS = ["AAPL", "MSFT"]
AUXILIARY_TICKERS = ["JPM", "XOM"]
TICKERS = [PRIMARY_TICKER, *SECONDARY_TICKERS, *AUXILIARY_TICKERS]

START = "2000-01-01"
END = "2018-12-31"

SSGA_SPY_URL = (
    "https://www.ssga.com/us/en/intermediaries/etfs/library-content/"
    "products/fund-data/etfs/us/navhist-us-en-spy.xlsx"
)
TWELVE_URL = "https://api.twelvedata.com/time_series"
# Offline Yahoo mirror (Unlicense): daily OHLCV since ~2000 for US tickers
GITHUB_YF_MIRROR = (
    "https://raw.githubusercontent.com/dieperdev/yfinance-stock-data/main/data"
)

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
    """Reject truncated series that miss the crisis evaluation window."""
    # Prefer post-2000; hard-require coverage from before the 2007–09 crisis regime.
    if series.index.min() > pd.Timestamp("2007-01-01"):
        raise ValueError(
            f"{ticker} starts {series.index.min().date()} — missing crisis window"
        )
    if len(series) < 2500:
        raise ValueError(
            f"{ticker} too short ({len(series)} rows); need full crisis→high-vol coverage"
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
    """Twelve Data daily closes. Demo key is limited; env key unlocks more."""
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


def download_github_yf_mirror(
    ticker: str,
    start: str = START,
    end: str = END,
) -> pd.Series:
    """Adj Close from dieperdev/yfinance-stock-data (covers post-2000 US equities)."""
    url = f"{GITHUB_YF_MIRROR}/{ticker.upper()}.csv"
    print(f"  GitHub YF mirror {ticker}...")
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if "Date" not in df.columns or "Adj Close" not in df.columns:
        raise ValueError(f"Unexpected mirror columns: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    series = df.set_index("Date")["Adj Close"].astype(float).sort_index()
    series.name = ticker
    series = _assert_full_sample(_filter_window(series, start, end), ticker)
    print(f"    ✓ {ticker}: {len(series)} rows via GitHub YF mirror")
    return series


def download_yahoo_chart(
    ticker: str,
    start: str = START,
    end: str = END,
    max_retries: int = 6,
) -> pd.Series:
    """Yahoo Finance chart API v8 — preferred free source for adj closes."""
    period1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400
    hosts = ["query1", "query2"]
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        host = hosts[(attempt - 1) % len(hosts)]
        url = (
            f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={period1}&period2={period2}&interval=1d"
        )
        try:
            print(f"  Yahoo chart {ticker} (attempt {attempt}/{max_retries}, {host})...")
            resp = SESSION.get(url, timeout=40)
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
            time.sleep(10 * attempt)
    raise RuntimeError(f"Yahoo failed for {ticker}") from last_error


def download_one(ticker: str) -> pd.Series:
    """Try sources in priority order; all four tickers are required."""
    errors: list[str] = []

    # Stable GitHub mirror first (avoids live Yahoo rate limits).
    try:
        return download_github_yf_mirror(ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"GitHubMirror: {exc}")

    if ticker == "SPY":
        try:
            return download_spy_ssga()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SSGA: {exc}")

    try:
        return download_yahoo_chart(ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Yahoo: {exc}")

    try:
        return download_twelve_data(ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"TwelveData: {exc}")

    raise RuntimeError(f"Could not download {ticker}. Tried: {errors}.")


def download_prices() -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for ticker in TICKERS:
        series_list.append(download_one(ticker))
        time.sleep(0.4)
    prices = pd.concat(series_list, axis=1, sort=True)
    prices = prices[[t for t in TICKERS if t in prices.columns]]
    return _normalize_index(prices)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then drop remaining NaNs (align common trading calendar)."""
    return _normalize_index(prices.ffill().dropna())


def _cache_is_valid(cached: pd.DataFrame) -> bool:
    if not all(t in cached.columns for t in TICKERS):
        return False
    if len(cached) < 2500:
        return False
    if cached.index.min() > pd.Timestamp("2007-01-01"):
        return False
    if cached.index.max() < pd.Timestamp("2018-06-01"):
        return False
    return True


def _resolve_prices_path() -> Path:
    """Prefer organized equity/ path; fall back to legacy flat path."""
    if PRICES_PATH.exists():
        return PRICES_PATH
    if LEGACY_PRICES_PATH.exists():
        return LEGACY_PRICES_PATH
    return PRICES_PATH


def load_or_download(force: bool = False) -> pd.DataFrame:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    path = _resolve_prices_path()

    if path.exists() and not force:
        print(f"Loading cached prices from {path}")
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if _cache_is_valid(cached):
            print(f"  Cache OK: {len(cached)} rows, {list(cached.columns)}")
            # Migrate legacy location if needed
            if path != PRICES_PATH:
                cached.to_csv(PRICES_PATH)
                print(f"  Migrated cache → {PRICES_PATH}")
            return cached
        print("  Cache incomplete — re-downloading")

    print(f"Downloading prices from {START} to {END}...")
    print(f"  Tickers (all required): {TICKERS}")
    prices = clean_prices(download_prices())
    prices.to_csv(PRICES_PATH)
    print(f"✓ Saved: {PRICES_PATH}")
    return prices


def main() -> None:
    print("=" * 70)
    print("STEP 1 — DATA FETCH (equity prices)")
    print("=" * 70)

    prices = load_or_download(force=True)

    print("\nPrice summary:")
    print(f"  Tickers: {list(prices.columns)}")
    print(f"  Date range: {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"  Observations: {len(prices)}")
    print(f"  Missing values: {int(prices.isna().sum().sum())}")
    print("\n  First 5 rows:")
    print(prices.head())
    print("\n  Last 5 rows:")
    print(prices.tail())
    print("\n✓ Equity data fetch complete")


if __name__ == "__main__":
    main()
