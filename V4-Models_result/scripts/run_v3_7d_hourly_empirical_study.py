#!/usr/bin/env python3
"""V3 7-day hourly empirical study wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3_intraday_hourly_empirical_study as intra  # noqa: E402

intra.configure_study("7d")


def main(argv=None) -> int:
    intra.configure_study("7d")
    return intra.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
