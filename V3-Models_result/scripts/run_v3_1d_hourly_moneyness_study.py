#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3_5y_monthly_moneyness_study as mx  # noqa: E402
import run_v3_intraday_hourly_empirical_study as intra  # noqa: E402

intra.configure_study("1d")
mx.main = intra
mx.NB_NAME = "V3_1d_hourly_moneyness_performance.ipynb"
mx.PDF_NAME = "V3_1d_hourly_moneyness_performance.pdf"
mx.MONEYNESS_IMPORT = "run_v3_1d_hourly_moneyness_study"


def main() -> int:
    intra.configure_study("1d")
    mx.main = intra
    mx.NB_NAME = "V3_1d_hourly_moneyness_performance.ipynb"
    mx.PDF_NAME = "V3_1d_hourly_moneyness_performance.pdf"
    mx.MONEYNESS_IMPORT = "run_v3_1d_hourly_moneyness_study"
    mx.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
