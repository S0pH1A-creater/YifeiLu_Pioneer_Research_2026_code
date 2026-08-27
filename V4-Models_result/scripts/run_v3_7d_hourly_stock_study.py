#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3_intraday_hourly_empirical_study as intra  # noqa: E402
import run_v3_intraday_hourly_stock_study as stock  # noqa: E402

intra.configure_study("7d")


def main(argv=None) -> int:
    intra.configure_study("7d")
    return stock.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
