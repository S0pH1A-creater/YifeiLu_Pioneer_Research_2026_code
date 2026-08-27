#!/usr/bin/env python3
"""Download AMZN equity + American options into the existing research/data layout.

AMZN is the extra 1y-study name: Cobweb ToS EOD 2003–2024 (crisis window intact),
same fields as AAPL/MSFT. Does not rewrite combined SPY/AAPL/MSFT panels.
Does not run the 1y LSM study.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if not hasattr(asyncio, "coroutine"):
    asyncio.coroutine = lambda fn: fn  # type: ignore[attr-defined]

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from american_lsm import sample_calls  # noqa: E402
from cobweb_to_parquet import convert_cobweb_zip  # noqa: E402
from data_fetch import (  # noqa: E402
    EQUITY_DIR,
    PRICES_PATH,
    download_one,
)
from data_prepare import (  # noqa: E402
    LOG_RETURNS_ALL,
    LOG_RETURNS_BY_REGIME,
    REGIME_ORDER,
    SUMMARY_STATS,
    build_regime_long,
    compute_log_returns,
    summary_statistics,
)
from options_fetch import (  # noqa: E402
    COBWEB_MEGA,
    PROCESSED_DIR,
    RAW_DIR,
    fetch_risk_free,
    process_ticker,
)

TICKER = "AMZN"
EVAL_WINDOWS = {
    "2008-2009": ("2008-08-01", "2009-07-31"),
    "2013-2014": ("2014-01-01", "2014-12-31"),
    "2018-2019": ("2018-10-01", "2019-09-30"),
    "2019-2020": ("2019-09-01", "2020-08-31"),
}
LOOKBACK = pd.DateOffset(months=18)
PEERS = ("AAPL", "MSFT")


def merge_equity() -> pd.DataFrame:
    print("=" * 70)
    print("AMZN equity → prices_clean.csv + log returns")
    print("=" * 70)
    series = download_one(TICKER)
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    aligned = series.reindex(prices.index)
    n_missing = int(aligned.isna().sum())
    if n_missing:
        aligned = aligned.ffill().bfill()
        n_missing = int(aligned.isna().sum())
    if n_missing:
        raise SystemExit(f"AMZN still has {n_missing} missing dates on the common calendar")
    if aligned.index.min() > pd.Timestamp("2007-01-01"):
        raise SystemExit(f"AMZN starts {aligned.index.min().date()} — missing crisis lookback")
    prices[TICKER] = aligned.astype(float)
    prices.to_csv(PRICES_PATH)
    print(
        f"✓ {PRICES_PATH.relative_to(EQUITY_DIR.parent)}  "
        f"{prices.index.min().date()} → {prices.index.max().date()}  "
        f"n={len(prices)}  cols={list(prices.columns)}"
    )

    returns = compute_log_returns(prices)
    returns.to_csv(LOG_RETURNS_ALL)
    regime_long = build_regime_long(returns)
    regime_long.to_csv(LOG_RETURNS_BY_REGIME, index=False)
    stats = summary_statistics(regime_long)
    # summary_statistics categoricals drop tickers not in data_fetch.TICKERS
    if TICKER not in set(stats["ticker"]):
        extra = _stats_unrestricted(regime_long[regime_long["ticker"] == TICKER])
        stats = pd.concat([stats, extra], ignore_index=True)
    stats.to_csv(SUMMARY_STATS, index=False)
    print(f"✓ updated {LOG_RETURNS_ALL.name}, {LOG_RETURNS_BY_REGIME.name}, {SUMMARY_STATS.name}")
    return prices


def _stats_unrestricted(regime_long: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (ticker, regime), group in regime_long.groupby(["ticker", "regime"]):
        r = group["log_return"]
        records.append(
            {
                "ticker": ticker,
                "regime": regime,
                "n_obs": int(len(r)),
                "start_date": group["date"].min().date().isoformat(),
                "end_date": group["date"].max().date().isoformat(),
                "mean": float(r.mean()),
                "std": float(r.std()),
                "ann_vol": float(r.std() * np.sqrt(252)),
                "skewness": float(r.skew()),
                "kurtosis": float(r.kurtosis()),
                "min": float(r.min()),
                "max": float(r.max()),
            }
        )
    return pd.DataFrame(records)


def _download_mega_zip(url: str, dest_zip: Path) -> Path:
    """Resume-friendly Mega public-file download (mega.py's stream often drops)."""
    import concurrent.futures as cf
    import threading
    import time

    import requests
    from Crypto.Cipher import AES
    from Crypto.Util import Counter
    from mega import Mega
    from mega.crypto import a32_to_str, base64_to_a32, base64_url_decode, decrypt_attr

    if not hasattr(asyncio, "coroutine"):
        asyncio.coroutine = lambda fn: fn  # type: ignore[attr-defined]

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    enc_path = dest_zip.with_suffix(dest_zip.suffix + ".enc")

    print("  Mega login (anonymous)…", flush=True)
    mega = Mega()
    mega.login()
    file_id, file_key = mega._parse_url(url).split("!")
    file_key = base64_to_a32(file_key)
    k = (
        file_key[0] ^ file_key[4],
        file_key[1] ^ file_key[5],
        file_key[2] ^ file_key[6],
        file_key[3] ^ file_key[7],
    )
    iv = file_key[4:6] + (0, 0)

    file_data = mega._api_request({"a": "g", "g": 1, "p": file_id})
    file_size = int(file_data["s"])
    attribs = decrypt_attr(base64_url_decode(file_data["at"]), k)
    name = attribs.get("n") if isinstance(attribs, dict) else dest_zip.name
    print(f"  Mega file {name}  {file_size / 1e6:.1f} MB", flush=True)

    n_parts = 8
    part_dir = dest_zip.parent / f"_{dest_zip.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = file_size // n_parts
    ranges = [
        (i, i * chunk, file_size - 1 if i == n_parts - 1 else (i + 1) * chunk - 1)
        for i in range(n_parts)
    ]
    url_lock = threading.Lock()

    def fresh_url() -> str:
        with url_lock:
            return mega._api_request({"a": "g", "g": 1, "p": file_id})["g"]

    def fetch_part(item: tuple[int, int, int]) -> Path:
        i, start, end = item
        part = part_dir / f"part_{i:02d}"
        expected = end - start + 1
        while True:
            have = part.stat().st_size if part.exists() else 0
            if have > expected:
                part.unlink()
                have = 0
            if have == expected:
                return part
            try:
                file_url = fresh_url()
                headers = {
                    "Range": f"bytes={start + have}-{end}",
                    "User-Agent": "Mozilla/5.0",
                }
                with requests.get(file_url, headers=headers, stream=True, timeout=60) as resp:
                    if resp.status_code not in {200, 206}:
                        raise RuntimeError(f"part {i} HTTP {resp.status_code}")
                    mode = "ab"
                    if have > 0 and resp.status_code == 200:
                        mode = "wb"
                        have = 0
                    with open(part, mode) as fh:
                        for block in resp.iter_content(1 << 20):
                            if block:
                                fh.write(block)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ part {i} interrupted ({exc}); retry", flush=True)
                time.sleep(3)

    print(f"  parallel Mega download ({n_parts} parts, {file_size / 1e6:.0f} MB)…", flush=True)
    with cf.ThreadPoolExecutor(max_workers=n_parts) as ex:
        parts = list(ex.map(fetch_part, ranges))
    with open(enc_path, "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
            p.unlink()
    part_dir.rmdir()
    print(f"  assembled encrypted file {enc_path.stat().st_size / 1e6:.1f} MB", flush=True)

    print("  decrypting Mega AES-CTR → zip…", flush=True)
    k_str = a32_to_str(k)
    counter = Counter.new(128, initial_value=((iv[0] << 32) + iv[1]) << 64)
    aes = AES.new(k_str, AES.MODE_CTR, counter=counter)
    with open(enc_path, "rb") as src, open(dest_zip, "wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(aes.decrypt(chunk))
    enc_path.unlink(missing_ok=True)
    print(f"  ✓ {dest_zip.name} ({dest_zip.stat().st_size / 1e6:.0f} MB)", flush=True)
    return dest_zip


def fetch_options(prices: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("AMZN American options (Cobweb ToS EOD → parquet → processed panels)")
    print("=" * 70)
    dest = RAW_DIR / f"{TICKER}_options.parquet"
    zip_path = RAW_DIR / "_staging" / f"{TICKER}.zip"
    if not (dest.exists() and dest.stat().st_size > 1_000_000):
        if not (zip_path.exists() and zip_path.stat().st_size > 50_000_000):
            _download_mega_zip(COBWEB_MEGA[TICKER], zip_path)
        print(f"  converting {zip_path.name} (regime windows)…", flush=True)
        convert_cobweb_zip(zip_path, TICKER, dest)
    else:
        print(f"  using cached {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
    rf = fetch_risk_free(force=False)
    panel = process_ticker(TICKER, prices, rf)
    if panel is None or panel.empty:
        raise SystemExit("AMZN processed option panel is empty")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    opt_path = PROCESSED_DIR / f"{TICKER}_options_panel.csv"
    call_path = PROCESSED_DIR / f"{TICKER}_calls_panel.csv"
    panel.to_csv(opt_path, index=False)
    calls = panel[panel["option_type"] == "call"].copy()
    calls.to_csv(call_path, index=False)
    print(f"✓ {opt_path.name}: {len(panel):,} quotes")
    print(f"✓ {call_path.name}: {len(calls):,} calls")

    summary_path = PROCESSED_DIR / "options_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    summary = summary[summary["underlying"] != TICKER]
    rows = []
    for regime, g in panel.groupby("regime"):
        rows.append(
            {
                "underlying": TICKER,
                "role": "extra_1y",
                "regime": regime,
                "n_quotes": len(g),
                "n_dates": g["trading_date"].nunique(),
                "n_calls": int((g["option_type"] == "call").sum()),
                "n_puts": int((g["option_type"] == "put").sum()),
                "mean_S": float(g["S_t"].mean()),
                "mean_option_price": float(g["option_price"].mean()),
                "mean_r": float(g["r"].mean()),
                "mean_dte": float(g["dte"].mean()),
                "start": pd.Timestamp(g["trading_date"].min()).date().isoformat(),
                "end": pd.Timestamp(g["trading_date"].max()).date().isoformat(),
            }
        )
    summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"✓ appended {TICKER} rows to {summary_path.name} (combined SPY/AAPL/MSFT panels unchanged)")
    return calls


def audit(prices: pd.DataFrame, calls: pd.DataFrame) -> int:
    print("=" * 70)
    print("Coverage audit vs AAPL / MSFT (1y study requirements)")
    print("=" * 70)
    required_cols = [
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
    missing_cols = [c for c in required_cols if c not in calls.columns]
    if missing_cols:
        print("FAIL missing call-panel columns:", missing_cols)
        return 1

    equity = prices[TICKER]
    print(
        f"Equity {TICKER}: {equity.index.min().date()} → {equity.index.max().date()}  "
        f"n={equity.notna().sum()}  nan={int(equity.isna().sum())}"
    )
    ok = True
    print(
        f"{'window':<12} {'eval':<26} {'lookback from':<12} "
        f"{'eq n':>6} {'calls':>7} {'dates':>6} {'Mon ATM':>8}  peers Mon ATM"
    )
    for name, (a, b) in EVAL_WINDOWS.items():
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        look_start = end - LOOKBACK
        eq = equity.loc[look_start:end]
        sub = calls[
            (pd.to_datetime(calls["trading_date"]) >= start)
            & (pd.to_datetime(calls["trading_date"]) <= end)
        ]
        sample = sample_calls(calls, start, end, px=eq.loc[start:end])
        n_atm = 0 if sample is None else len(sample)
        peer_ns = []
        for peer in PEERS:
            p = pd.read_csv(
                PROCESSED_DIR / f"{peer}_calls_panel.csv",
                parse_dates=["trading_date", "expiration"],
            )
            peer_ns.append(str(len(sample_calls(p, start, end))))
        eq_ok = len(eq.dropna()) >= 200 and eq.index.min() <= look_start + pd.Timedelta(days=10)
        call_ok = n_atm >= 40
        flag = "OK" if eq_ok and call_ok else "GAP"
        if flag != "OK":
            ok = False
        print(
            f"{name:<12} {a}→{b}  {look_start.date()}  "
            f"{len(eq.dropna()):6d} {len(sub):7d} {sub['trading_date'].nunique():6d} "
            f"{n_atm:8d}  {', '.join(peer_ns):>12}  {flag}"
        )

    regimes = sorted(pd.Series(calls["regime"]).dropna().unique())
    print(f"Processed call regimes: {regimes}")
    print(f"Call style: {sorted(calls['style'].dropna().unique())}")
    print(f"option_type: {sorted(calls['option_type'].dropna().unique())}")
    if not ok:
        print("\nAMZN is missing a 1y-study piece (Monday ATM sample < 40 or equity lookback gap).")
        return 1
    print("\nAMZN has the same 1y-study pieces as AAPL/MSFT: daily equity (18m lookback),")
    print("DGS3MO rates (shared), raw options parquet, processed calls/options panels,")
    print("and a Monday ATM DTE 7–60 sample in every evaluation window.")
    return 0


def main() -> int:
    prices = merge_equity()
    calls = fetch_options(prices)
    return audit(prices, calls)


if __name__ == "__main__":
    raise SystemExit(main())
