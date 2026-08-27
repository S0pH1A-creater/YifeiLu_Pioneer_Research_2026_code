#!/usr/bin/env python3
"""Moneyness report for the 1.5-year monthly empirical study."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_monthly_empirical_study as emp  # noqa: E402
import run_v3_5y_monthly_moneyness_study as mx  # noqa: E402

emp.apply_1p5y_config()
mx.NB_NAME = "V3_1p5y_monthly_moneyness_performance.ipynb"
mx.PDF_NAME = "V3_1p5y_monthly_moneyness_performance.pdf"
mx.MONEYNESS_IMPORT = "run_v3_1p5y_monthly_moneyness_study"


def main() -> int:
    emp.apply_1p5y_config()
    mx.NB_NAME = "V3_1p5y_monthly_moneyness_performance.ipynb"
    mx.PDF_NAME = "V3_1p5y_monthly_moneyness_performance.pdf"
    mx.MONEYNESS_IMPORT = "run_v3_1p5y_monthly_moneyness_study"
    mx.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
