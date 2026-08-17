#!/usr/bin/env python3
"""Run Heston-only on the Friday 2023-03-17 1h study and merge into the existing JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_20221021_1h_tables as study

EXISTING = study.ROOT / "results" / "rmse_report_20230317_1h.json"
TMP = study.ROOT / "results" / "rmse_report_20230317_1h_heston.json"

HESTON_NBS = {
    "7-day": [("Heston", "heston notebook", "7d_1min_heston.ipynb")],
    "1-day": [("Heston", "heston notebook", "1d_1min_heston.ipynb")],
}


def main() -> int:
    study.OUT = TMP
    study.rpt.WINDOWS = [
        {
            "name": "7-day",
            "period_start": pd.Timestamp("2023-03-13 09:30:00"),
            "period_end": pd.Timestamp("2023-03-17 15:59:00"),
            "nbs": HESTON_NBS["7-day"],
        },
        {
            "name": "1-day",
            "period_start": pd.Timestamp("2023-03-17 09:30:00"),
            "period_end": pd.Timestamp("2023-03-17 15:59:00"),
            "nbs": HESTON_NBS["1-day"],
        },
    ]
    print(
        "2023-03-17 expiry | Heston-only | lookback=1 hour | hourly & minutely | "
        "1-day 2023-03-17 and 5-weekday 2023-03-13→17",
        flush=True,
    )
    rc = study.main()
    if rc:
        return rc

    old = json.loads(EXISTING.read_text(encoding="utf-8")) if EXISTING.exists() else []
    new = json.loads(TMP.read_text(encoding="utf-8"))
    kept = [r for r in old if r.get("model") != "Heston"]
    merged = kept + new
    EXISTING.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"merged {len(new)} Heston rows into {EXISTING}  (total {len(merged)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
