"""Cut a dated 7-day 1-minute window from source_1min/ and archive loose 7d files.

Default window: calendar week 2023-03-09 → 2023-03-15
  RTH sessions: 9, 10, 13, 14, 15 Mar (weekend 11–12 skipped)
  Lookback buffer: previous RTH session (2023-03-08)

Layout:
  research/data/equity/intraday/7d_1min/<YYYY-MM-DD_to_YYYY-MM-DD>/{SPY,AAPL,MSFT}.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTRA = ROOT / "research" / "data" / "equity" / "intraday"
SOURCE = INTRA / "source_1min"
OUT_ROOT = INTRA / "7d_1min"
TICKERS = ["SPY", "AAPL", "MSFT"]

EVAL_START = pd.Timestamp("2023-03-09 09:30:00")
EVAL_END = pd.Timestamp("2023-03-15 15:59:00")
LOOKBACK_START = pd.Timestamp("2023-03-08 09:30:00")
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
SESSION_CLOSE = pd.Timedelta(hours=16)


def rth(df: pd.DataFrame) -> pd.DataFrame:
    tod = df["Datetime"] - df["Datetime"].dt.normalize()
    out = df.loc[(tod >= SESSION_OPEN) & (tod < SESSION_CLOSE)].copy()
    return out.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).reset_index(drop=True)


def window_id(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.date()}_to_{end.date()}"


def slice_window(
    eval_start: pd.Timestamp = EVAL_START,
    eval_end: pd.Timestamp = EVAL_END,
    lookback_start: pd.Timestamp = LOOKBACK_START,
) -> Path:
    wid = window_id(eval_start, eval_end)
    dest = OUT_ROOT / wid
    dest.mkdir(parents=True, exist_ok=True)
    info = {}
    for t in TICKERS:
        src = pd.read_csv(SOURCE / f"{t}.csv", parse_dates=["Datetime"])
        src = rth(src)
        sl = src.loc[(src["Datetime"] >= lookback_start) & (src["Datetime"] <= eval_end)].copy()
        if sl.empty:
            raise RuntimeError(f"{t}: no bars in {lookback_start} → {eval_end}")
        sl.to_csv(dest / f"{t}.csv", index=False)
        days = sorted({d.date() for d in sl["Datetime"]})
        eval_days = [d for d in days if d >= eval_start.date()]
        info[t] = {
            "n_bars": int(len(sl)),
            "start": str(sl["Datetime"].min()),
            "end": str(sl["Datetime"].max()),
            "sessions_on_file": [str(d) for d in days],
            "eval_sessions": [str(d) for d in eval_days],
        }
        print(f"  {t}: {len(sl)} bars  {sl['Datetime'].min()} → {sl['Datetime'].max()}  sessions={days}", flush=True)

    readme = dest / "README.md"
    spy = info["SPY"]
    readme.write_text(
        f"""# 7-day 1-minute window `{wid}`

- **Evaluation (calendar):** {eval_start.date()} → {eval_end.date()}
- **RTH sessions evaluated:** {", ".join(spy["eval_sessions"])} ({len(spy["eval_sessions"])} sessions; weekend skipped)
- **Lookback buffer (1 day):** {lookback_start.date()} (included in CSVs, not in the evaluation window)
- **Bars:** regular hours 09:30–16:00 ET, 1-minute
- **Source:** `source_1min/` (FirstRate free sample)

This folder is named by evaluation dates so other 7-day cuts can sit beside it.
""",
        encoding="utf-8",
    )
    (dest / "window.json").write_text(json.dumps({"window_id": wid, "tickers": info}, indent=2), encoding="utf-8")
    return dest


def archive_loose_top_level() -> None:
    """Move legacy 7d_1min/*.csv (Sep 2023 last-8-sessions cut) into a dated folder."""
    loose = [OUT_ROOT / f"{t}.csv" for t in TICKERS]
    if not all(p.exists() for p in loose):
        return
    spy = pd.read_csv(OUT_ROOT / "SPY.csv", parse_dates=["Datetime"])
    days = sorted({d.date() for d in pd.to_datetime(spy["Datetime"])})
    if len(days) < 2:
        return
    # last 7 sessions are the evaluation window; first session is lookback
    eval_days = days[-7:] if len(days) >= 7 else days[1:]
    dest = OUT_ROOT / f"{eval_days[0]}_to_{eval_days[-1]}"
    if dest.exists() and (dest / "SPY.csv").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        src = OUT_ROOT / f"{t}.csv"
        if src.exists():
            src.replace(dest / f"{t}.csv")
    (dest / "README.md").write_text(
        f"""# 7-day 1-minute window `{dest.name}`

Legacy top-level `7d_1min/*.csv` (last 8 RTH sessions of the FirstRate sample).
Lookback buffer = first session on file; notebooks originally evaluated the last 7.
""",
        encoding="utf-8",
    )
    print(f"archived loose CSVs → {dest.relative_to(ROOT)}", flush=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_loose_top_level()
    dest = slice_window()
    print(f"wrote {dest.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
