"""
Convert Cobweb thinkorswim EOD option ZIPs → canonical options parquet.

Source: https://cobwebscripts.com/data/toseodoptiondata.html (free ToS EOD dumps)
Output schema matches philippdubach/options-data style used by options_fetch.py:
  date, expiration, strike, type, mark, bid, ask, last, volume, open_interest, symbol,
  underlying_last  (raw broker close; prefer over split-adjusted equity for moneyness)
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "options" / "raw"
STAGING_DIR = RAW_DIR / "_staging"

# Evaluation regimes (filename trading dates)
REGIME_WINDOWS = [
    ("2008-01-01", "2009-12-31"),
    ("2013-01-01", "2014-12-31"),
    ("2018-01-01", "2019-12-31"),
]

FILE_RE = re.compile(
    r"^(?:.*/)?(?P<ticker>[A-Z.]+)/(?P<date>\d{4}-\d{2}-\d{2})-StockAndOptionQuoteFor"
)
EXP_HEADER_RE = re.compile(
    r"^(?P<label>\d{1,2}\s+[A-Z]{3}\s+\d{2})\s+\((?P<dte>-?\d+)\)\s+(?P<mult>\d+)"
)

def _in_regimes(d: pd.Timestamp) -> bool:
    for a, b in REGIME_WINDOWS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return True
    return False


def _to_float(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace('"', "")
    if s == "" or s.lower() in {"<empty>", "na", "nan", ".", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_exp(label: str, trading_date: pd.Timestamp) -> pd.Timestamp | None:
    """Parse '19 SEP 08' using trading year century."""
    label = label.strip()
    try:
        dt = pd.to_datetime(label, format="%d %b %y")
    except Exception:
        try:
            dt = pd.to_datetime(label)
        except Exception:
            return None
    # Guard against century edge cases around year 2000
    if dt.year < 2000 and trading_date.year >= 2000:
        dt = dt.replace(year=dt.year + 100)
    return pd.Timestamp(dt)


def _parse_one_file(text: str, ticker: str, trading_date: pd.Timestamp) -> list[dict]:
    lines = text.replace("\ufeff", "").splitlines()
    underlying_last = None
    rows: list[dict] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.upper() == "UNDERLYING" and i + 2 < len(lines):
            vals = lines[i + 2].split(",")
            underlying_last = _to_float(vals[0] if vals else None)
            i += 3
            continue

        m = EXP_HEADER_RE.match(line)
        if m:
            mult = int(m.group("mult"))
            exp = _parse_exp(m.group("label"), trading_date)
            # skip minis / non-standard multipliers
            i += 1
            if i < len(lines) and all(h in lines[i] for h in ("Strike", "Mark")):
                i += 1  # column header
            if mult != 100 or exp is None:
                # skip block body until next blank/header
                while i < len(lines) and lines[i].strip() and not EXP_HEADER_RE.match(lines[i].strip()):
                    i += 1
                continue

            while i < len(lines):
                body = lines[i].strip()
                if not body:
                    i += 1
                    break
                if EXP_HEADER_RE.match(body) or body.upper() == "UNDERLYING":
                    break
                parts = body.split(",")
                # Expected: empty, empty, call fields..., Exp, Strike, put fields...
                # Find strike index by locating the Exp label column near center.
                # Robust approach: find token matching expiration label month pattern
                strike_idx = None
                for j, tok in enumerate(parts):
                    if j + 1 < len(parts) and _to_float(parts[j + 1]) is not None:
                        # Exp column often looks like '19 SEP 08'
                        if re.search(r"\d{1,2}\s+[A-Z]{3}\s+\d{2}", tok.strip(), re.I):
                            strike_idx = j + 1
                            break
                if strike_idx is None:
                    # fallback: classic layout Exp at -16 from end-ish; use known width
                    # call: 2 blanks + 13 fields, then Exp, Strike => strike at index 15
                    if len(parts) > 16:
                        strike_idx = 15
                    else:
                        i += 1
                        continue

                strike = _to_float(parts[strike_idx])
                if strike is None:
                    i += 1
                    continue

                # Call side immediately before Exp: ... Mark, Bid, Ask, Exp, Strike
                # Indices relative to strike_idx: Exp=strike_idx-1, Ask=c-2, Bid=c-3, Mark=c-4
                # LAST at strike_idx-14 in full layout (2 blanks + LAST ...)
                def get(idx: int):
                    return parts[idx] if 0 <= idx < len(parts) else None

                c_mark = _to_float(get(strike_idx - 4))
                c_bid = _to_float(get(strike_idx - 3))
                c_ask = _to_float(get(strike_idx - 2))
                c_last = _to_float(get(strike_idx - 14))
                c_vol = _to_float(get(strike_idx - 12))
                c_oi = _to_float(get(strike_idx - 11))

                # Put side after strike: Bid, Ask, LAST, LX, Volume, Open.Int, ..., Mark
                p_bid = _to_float(get(strike_idx + 1))
                p_ask = _to_float(get(strike_idx + 2))
                p_last = _to_float(get(strike_idx + 3))
                p_vol = _to_float(get(strike_idx + 5))
                p_oi = _to_float(get(strike_idx + 6))
                p_mark = _to_float(get(strike_idx + 13))

                def price(mark, bid, ask, last):
                    if mark is not None and mark > 0:
                        return mark
                    if bid is not None and ask is not None and ask >= bid and bid > 0:
                        return 0.5 * (bid + ask)
                    if last is not None and last > 0:
                        return last
                    return None

                call_px = price(c_mark, c_bid, c_ask, c_last)
                put_px = price(p_mark, p_bid, p_ask, p_last)

                base = {
                    "date": trading_date,
                    "expiration": exp,
                    "strike": strike,
                    "symbol": ticker,
                    "underlying_last": underlying_last,
                }
                if call_px is not None:
                    rows.append(
                        {
                            **base,
                            "type": "call",
                            "mark": call_px,
                            "bid": c_bid,
                            "ask": c_ask,
                            "last": c_last,
                            "volume": int(c_vol) if c_vol is not None else 0,
                            "open_interest": int(c_oi) if c_oi is not None else 0,
                        }
                    )
                if put_px is not None:
                    rows.append(
                        {
                            **base,
                            "type": "put",
                            "mark": put_px,
                            "bid": p_bid,
                            "ask": p_ask,
                            "last": p_last,
                            "volume": int(p_vol) if p_vol is not None else 0,
                            "open_interest": int(p_oi) if p_oi is not None else 0,
                        }
                    )
                i += 1
            continue
        i += 1
    return rows


def convert_cobweb_zip(zip_path: Path, ticker: str, out_path: Path | None = None) -> Path:
    out_path = out_path or (RAW_DIR / f"{ticker}_options.parquet")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = FILE_RE.match(name)
            if not m:
                continue
            if m.group("ticker").upper() != ticker.upper():
                continue
            trading_date = pd.Timestamp(m.group("date"))
            if not _in_regimes(trading_date):
                continue
            n_files += 1
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()
            all_rows.extend(_parse_one_file(text, ticker.upper(), trading_date))
            if n_files % 100 == 0:
                print(f"  parsed {n_files} regime files → {len(all_rows):,} rows", flush=True)

    if not all_rows:
        raise RuntimeError(f"No option rows parsed from {zip_path} for {ticker}")

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["date", "expiration", "strike", "mark", "type"])
    df = df.sort_values(["date", "expiration", "type", "strike"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(
        f"✓ {ticker}: {len(df):,} quotes from {n_files} files → {out_path} "
        f"({out_path.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"  dates {df['date'].min().date()} → {df['date'].max().date()} | "
        f"types={sorted(df['type'].unique())}"
    )
    return out_path


def main() -> None:
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "MSFT").upper()
    zip_path = STAGING_DIR / f"{ticker}.zip"
    if not zip_path.exists():
        raise SystemExit(
            f"Missing {zip_path}; download Cobweb {ticker}.zip first "
            f"(or: python scripts/options_fetch.py)."
        )
    print(f"Converting {zip_path} (regime windows only)...")
    convert_cobweb_zip(zip_path, ticker)


if __name__ == "__main__":
    main()
