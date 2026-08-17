#!/usr/bin/env python3
"""Same 1h lookback study as 2022-10-21, for Friday 2023-03-17 monthly expiry."""
from __future__ import annotations

import pandas as pd

import run_20221021_1h_tables as study

study.OUT = study.ROOT / "results" / "rmse_report_20230317_1h.json"
study.rpt.WINDOWS = [
    {
        "name": "7-day",
        "period_start": pd.Timestamp("2023-03-13 09:30:00"),
        "period_end": pd.Timestamp("2023-03-17 15:59:00"),
        "nbs": study.rpt.WINDOWS[0]["nbs"],
    },
    {
        "name": "1-day",
        "period_start": pd.Timestamp("2023-03-17 09:30:00"),
        "period_end": pd.Timestamp("2023-03-17 15:59:00"),
        "nbs": study.rpt.WINDOWS[1]["nbs"],
    },
]


def main() -> int:
    # reuse the same engine; only dates / output path change
    orig = study.main

    def wrapped() -> int:
        print(
            "2023-03-17 expiry | lookback=1 hour | hourly & minutely | "
            "1-day 2023-03-17 and 5-weekday 2023-03-13→17",
            flush=True,
        )
        return orig()

    return wrapped()


if __name__ == "__main__":
    raise SystemExit(main())
