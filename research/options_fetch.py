"""
Step 1 — American option panels (DataCollection §2 / ResearchProposal-v2).

Tickers: SPY (primary), AAPL, JPM, XOM (secondary when raw files available).

Research fields:
  underlying, trading_date, S_t, K, expiration, T_years, option_type,
  option_price, r, moneyness, dte, regime, style (= American)

Risk-free rate: FRED 3-Month Treasury (DGS3MO), decimal form.

Coverage note: open options history starts 2008-01-02 (crisis window
uses 2008–2009 rather than calendar 2007).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from data_fetch import DATA_DIR, END, EQUITY_DIR, START, load_or_download

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OPTIONS_DIR = DATA_DIR / "options"
RAW_DIR = OPTIONS_DIR / "raw"
PROCESSED_DIR = OPTIONS_DIR / "processed"
RATES_DIR = DATA_DIR / "rates"
RISK_FREE_PATH = RATES_DIR / "risk_free_dgs3mo.csv"
LEGACY_RISK_FREE_PATH = DATA_DIR / "risk_free_dgs3mo.csv"

SPY_RELEASE_URL = (
    "https://github.com/lambdaclass/options_portfolio_backtester/"
    "releases/download/data-v1/SPY_options.parquet"
)
# Secondary tickers: same CDN layout as philippdubach/options-data (when online)
OPTIONS_CDN = "https://static.philippdubach.com/data/options"

OPTION_TICKERS = ["SPY", "AAPL", "JPM", "XOM"]

OPTIONS_START = "2008-01-01"
OPTIONS_END = END
ATM_BAND = 0.10
MIN_DTE = 7
MAX_DTE = 60
MIN_VOLUME = 1
SAMPLE_EVERY_N_DAYS = 5

REGIMES = {
    "crisis": ("2008-01-01", "2009-12-31"),
    "normal": ("2013-01-01", "2014-12-31"),
    "high_vol": ("2017-01-01", "2018-12-31"),
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def assign_regime(date: pd.Timestamp) -> str | None:
    for name, (a, b) in REGIMES.items():
        if pd.Timestamp(a) <= date <= pd.Timestamp(b):
            return name
    return None


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------
def fetch_risk_free(force: bool = False) -> pd.Series:
    """FRED DGS3MO → decimal annual rate, forward-filled to business days."""
    RATES_DIR.mkdir(parents=True, exist_ok=True)

    src = RISK_FREE_PATH if RISK_FREE_PATH.exists() else LEGACY_RISK_FREE_PATH
    if src.exists() and not force:
        print(f"Loading cached risk-free rates from {src}")
        if src != RISK_FREE_PATH:
            RISK_FREE_PATH.write_bytes(src.read_bytes())
            print(f"  Migrated → {RISK_FREE_PATH}")
    else:
        print("Downloading FRED DGS3MO (3M T-bill)...")
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id=DGS3MO&cosd={START}&coed={END}"
        )
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
        resp.raise_for_status()
        RISK_FREE_PATH.write_bytes(resp.content)
        print(f"✓ Saved: {RISK_FREE_PATH}")

    rf = pd.read_csv(RISK_FREE_PATH)
    rf.columns = ["date", "dgs3mo"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["dgs3mo"] = pd.to_numeric(rf["dgs3mo"], errors="coerce")
    rf = rf.dropna().set_index("date")["dgs3mo"] / 100.0
    rf = rf.sort_index()
    full_idx = pd.date_range(rf.index.min(), rf.index.max(), freq="B")
    rf = rf.reindex(full_idx).ffill().bfill()
    rf.index.name = "trading_date"
    rf.name = "r"
    return rf


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_file_parallel(url: str, dest: Path, n_parts: int = 8) -> Path:
    """Multi-range parallel download for large GitHub release assets."""
    import concurrent.futures as cf

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=60) as probe:
        probe.raise_for_status()
        final = probe.url
        size = int(probe.headers["Content-Length"])

    part_dir = dest.parent / f"_{dest.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = size // n_parts
    ranges = [
        (i, i * chunk, size - 1 if i == n_parts - 1 else (i + 1) * chunk - 1)
        for i in range(n_parts)
    ]

    def fetch_part(item: tuple[int, int, int]) -> Path:
        i, start, end = item
        part = part_dir / f"part_{i:02d}"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return part
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
        with requests.get(final, headers=headers, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for block in resp.iter_content(1 << 20):
                    if block:
                        f.write(block)
        return part

    print(f"  Parallel download ({n_parts} parts, {size / 1e6:.0f} MB)...")
    with cf.ThreadPoolExecutor(max_workers=n_parts) as ex:
        parts = list(ex.map(fetch_part, ranges))

    with open(dest, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    for p in part_dir.glob("part_*"):
        p.unlink()
    part_dir.rmdir()
    return dest


def download_file_stream(url: str, dest: Path) -> Path:
    """Single-stream download (CDN / smaller files)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=180) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        written = 0
        with open(dest, "wb") as f:
            for block in resp.iter_content(1 << 20):
                if block:
                    f.write(block)
                    written += len(block)
        if total and written < total * 0.95:
            raise RuntimeError(f"Incomplete download: {written}/{total} bytes")
    return dest


def download_spy_options_raw(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "SPY_options.parquet"
    if path.exists() and path.stat().st_size > 100_000_000 and not force:
        print(f"Using cached raw SPY options: {path} ({path.stat().st_size / 1e6:.0f} MB)")
        return path

    print("Downloading SPY options parquet (~600 MB) from GitHub release...")
    download_file_parallel(SPY_RELEASE_URL, path)
    print(f"✓ Saved: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return path


def try_download_secondary_options(ticker: str) -> Path | None:
    """Attempt CDN download for AAPL/JPM/XOM. Returns path or None if unavailable."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{ticker}_options.parquet"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Using cached raw {ticker} options: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    url = f"{OPTIONS_CDN}/{ticker.lower()}/options.parquet"
    print(f"  Trying CDN for {ticker}: {url}")
    try:
        download_file_stream(url, dest)
        if dest.stat().st_size < 1_000_000:
            dest.unlink(missing_ok=True)
            print(f"  ⚠ CDN file too small / invalid for {ticker}")
            return None
        print(f"  ✓ Saved: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        print(f"  ⚠ CDN unavailable for {ticker}: {exc}")
        return None


def _normalize_option_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common source column names to a canonical set."""
    rename = {}
    cols = {c.lower(): c for c in df.columns}
    mapping = {
        "date": "trading_date",
        "quote_date": "trading_date",
        "expiration": "expiration",
        "expiry": "expiration",
        "strike": "K",
        "type": "option_type",
        "option_type": "option_type",
        "call_put": "option_type",
        "mark": "option_price",
        "mid": "option_price",
        "last": "last_price",
        "bid": "bid",
        "ask": "ask",
        "volume": "volume",
        "open_interest": "open_interest",
        "symbol": "underlying",
        "underlying": "underlying",
        "ticker": "underlying",
    }
    for src, dst in mapping.items():
        if src in cols and dst not in df.columns:
            rename[cols[src]] = dst
    out = df.rename(columns=rename).copy()

    if "option_price" not in out.columns:
        if {"bid", "ask"}.issubset(out.columns):
            out["option_price"] = (out["bid"].astype(float) + out["ask"].astype(float)) / 2.0
        elif "last_price" in out.columns:
            out["option_price"] = out["last_price"]
        else:
            raise ValueError(f"Cannot infer option_price from columns: {list(out.columns)}")

    out["trading_date"] = pd.to_datetime(out["trading_date"])
    out["expiration"] = pd.to_datetime(out["expiration"])
    out["K"] = out["K"].astype(float)
    out["option_price"] = pd.to_numeric(out["option_price"], errors="coerce")
    out["option_type"] = (
        out["option_type"].astype(str).str.lower().str.strip()
        .replace({"c": "call", "p": "put", "call": "call", "put": "put"})
    )
    return out


def load_raw_options(ticker: str) -> pd.DataFrame | None:
    """Load raw options for a ticker from RAW_DIR if present."""
    candidates = [
        RAW_DIR / f"{ticker}_options.parquet",
        RAW_DIR / f"{ticker.lower()}_options.parquet",
        RAW_DIR / f"{ticker}_options.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        print(f"Loading raw options: {path.name}")
        if path.suffix == ".parquet":
            schema = pq.read_schema(path)
            names = set(schema.names)
            wanted = [
                c
                for c in [
                    "date",
                    "expiration",
                    "strike",
                    "type",
                    "mark",
                    "bid",
                    "ask",
                    "last",
                    "volume",
                    "open_interest",
                    "symbol",
                ]
                if c in names
            ]
            filters = [
                ("date", ">=", pd.Timestamp(OPTIONS_START).to_datetime64()),
                ("date", "<=", pd.Timestamp(OPTIONS_END).to_datetime64()),
            ]
            try:
                table = pq.read_table(path, columns=wanted or None, filters=filters)
                df = table.to_pandas()
            except Exception as exc:  # noqa: BLE001
                print(f"  Filter read failed ({exc}); scanning with pandas...")
                df = pd.read_parquet(path, columns=wanted or None)
                df["date"] = pd.to_datetime(df["date"])
                df = df[
                    (df["date"] >= pd.Timestamp(OPTIONS_START))
                    & (df["date"] <= pd.Timestamp(OPTIONS_END))
                ]
        else:
            df = pd.read_csv(path)
        df = _normalize_option_columns(df)
        if "underlying" not in df.columns:
            df["underlying"] = ticker
        else:
            df["underlying"] = df["underlying"].astype(str).str.upper()
            df = df[df["underlying"] == ticker]
        return df
    return None


def build_panel(
    options: pd.DataFrame,
    underlying_prices: pd.Series,
    risk_free: pd.Series,
    ticker: str,
) -> pd.DataFrame:
    """Join S_t and r; filter ATM / DTE; tag regimes."""
    df = options.copy()
    df = df[
        (df["trading_date"] >= pd.Timestamp(OPTIONS_START))
        & (df["trading_date"] <= pd.Timestamp(OPTIONS_END))
    ]
    df = df[df["option_type"].isin(["call", "put"])]
    df = df.dropna(subset=["option_price", "K", "trading_date", "expiration"])
    df = df[df["option_price"] > 0]

    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= MIN_VOLUME]

    if {"bid", "ask"}.issubset(df.columns):
        df = df[(df["bid"].fillna(0) > 0) & (df["ask"] > df["bid"])]

    px = underlying_prices.rename("S_t")
    px.index = pd.to_datetime(px.index)
    df = df.merge(px, left_on="trading_date", right_index=True, how="inner")

    rf = risk_free.rename("r")
    df = df.merge(rf, left_on="trading_date", right_index=True, how="left")
    df["r"] = df["r"].ffill().bfill()

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

    dates = np.sort(df["trading_date"].unique())
    keep_dates = set(dates[::SAMPLE_EVERY_N_DAYS])
    df = df[df["trading_date"].isin(keep_dates)]

    df["style"] = "American"
    df["regime"] = df["trading_date"].map(lambda d: assign_regime(pd.Timestamp(d)))
    df = df[df["regime"].notna()]

    out = df[
        [
            "underlying",
            "trading_date",
            "S_t",
            "K",
            "expiration",
            "T_years",
            "dte",
            "option_type",
            "option_price",
            "r",
            "moneyness",
            "regime",
            "style",
        ]
    ].sort_values(["trading_date", "expiration", "option_type", "K"])
    out = out.reset_index(drop=True)
    out["underlying"] = ticker
    return out


def process_ticker(
    ticker: str,
    prices: pd.DataFrame,
    risk_free: pd.Series,
) -> pd.DataFrame | None:
    raw = load_raw_options(ticker)
    if raw is None:
        print(f"  ⚠ No raw options file for {ticker} in {RAW_DIR}")
        return None
    if ticker not in prices.columns:
        print(f"  ⚠ No underlying price series for {ticker}")
        return None
    panel = build_panel(raw, prices[ticker], risk_free, ticker)
    print(
        f"  ✓ {ticker}: {len(panel):,} option quotes | "
        f"{panel['trading_date'].nunique()} dates | "
        f"regimes={sorted(panel['regime'].unique())}"
    )
    return panel


def write_organized(panels: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, panel in panels.items():
        path = PROCESSED_DIR / f"{ticker}_options_panel.csv"
        panel.to_csv(path, index=False)
        print(f"✓ Saved: {path.relative_to(DATA_DIR)}")

        # DataCollection focuses on American calls for optimal stopping
        calls = panel[panel["option_type"] == "call"]
        call_path = PROCESSED_DIR / f"{ticker}_calls_panel.csv"
        calls.to_csv(call_path, index=False)
        print(f"✓ Saved: {call_path.relative_to(DATA_DIR)} ({len(calls):,} calls)")

    ordered = [t for t in OPTION_TICKERS if t in panels]
    combined = pd.concat([panels[t] for t in ordered], ignore_index=True)
    combined_path = PROCESSED_DIR / "options_panel_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"✓ Saved: {combined_path.relative_to(DATA_DIR)} ({len(combined):,} rows)")

    calls_all = combined[combined["option_type"] == "call"]
    calls_all_path = PROCESSED_DIR / "calls_panel_all.csv"
    calls_all.to_csv(calls_all_path, index=False)
    print(f"✓ Saved: {calls_all_path.relative_to(DATA_DIR)} ({len(calls_all):,} calls)")

    for regime in REGIMES:
        chunk = combined[combined["regime"] == regime]
        path = PROCESSED_DIR / f"options_panel_{regime}.csv"
        chunk.to_csv(path, index=False)
        print(f"✓ Saved: {path.relative_to(DATA_DIR)} ({len(chunk):,} rows)")

        call_chunk = chunk[chunk["option_type"] == "call"]
        call_path = PROCESSED_DIR / f"calls_panel_{regime}.csv"
        call_chunk.to_csv(call_path, index=False)
        print(f"✓ Saved: {call_path.relative_to(DATA_DIR)} ({len(call_chunk):,} calls)")

    summary_rows = []
    for ticker, panel in panels.items():
        for regime, g in panel.groupby("regime"):
            summary_rows.append(
                {
                    "underlying": ticker,
                    "regime": regime,
                    "n_quotes": len(g),
                    "n_dates": g["trading_date"].nunique(),
                    "n_calls": int((g["option_type"] == "call").sum()),
                    "n_puts": int((g["option_type"] == "put").sum()),
                    "mean_S": float(g["S_t"].mean()),
                    "mean_option_price": float(g["option_price"].mean()),
                    "mean_r": float(g["r"].mean()),
                    "mean_dte": float(g["dte"].mean()),
                    "start": g["trading_date"].min().date().isoformat(),
                    "end": g["trading_date"].max().date().isoformat(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = PROCESSED_DIR / "options_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path.relative_to(DATA_DIR)}")


def main() -> None:
    print("=" * 70)
    print("STEP 1 — OPTIONS + RISK-FREE DATA")
    print("=" * 70)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RATES_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)

    risk_free = fetch_risk_free(force=False)
    prices = load_or_download(force=False)

    download_spy_options_raw(force=False)

    # Attempt secondary option downloads (best-effort; CDN may be offline)
    for ticker in ["AAPL", "JPM", "XOM"]:
        print(f"\nEnsuring raw options for {ticker}...")
        try_download_secondary_options(ticker)

    panels: dict[str, pd.DataFrame] = {}
    for ticker in OPTION_TICKERS:
        print(f"\nProcessing {ticker}...")
        panel = process_ticker(ticker, prices, risk_free)
        if panel is not None and len(panel) > 0:
            panels[ticker] = panel

    if "SPY" not in panels:
        raise SystemExit("SPY options panel is required (primary) but was not built.")

    write_organized(panels)

    missing = [t for t in OPTION_TICKERS if t not in panels]
    if missing:
        print("\nMissing secondary equity option panels:", ", ".join(missing))
        print(
            "Open CDN currently unavailable. Place files named "
            f"{{TICKER}}_options.parquet in {RAW_DIR} and re-run.\n"
            "Expected columns: date, expiration, strike, type, mark|bid|ask, volume."
        )

    print("\n✓ Options Step 1 complete")
    print(f"  Primary panel: {PROCESSED_DIR / 'SPY_options_panel.csv'}")
    print(f"  Calls (pricing): {PROCESSED_DIR / 'calls_panel_all.csv'}")
    print(f"  Risk-free:       {RISK_FREE_PATH}")


if __name__ == "__main__":
    main()
