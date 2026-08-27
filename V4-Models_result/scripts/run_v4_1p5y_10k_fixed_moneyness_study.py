#!/usr/bin/env python3
"""Moneyness report for the V4 1.5-year fixed 10k study."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_moneyness_study as mx  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study as v4  # noqa: E402


def apply_config() -> None:
    v4.patch_v3_wrap()
    mx.NB_NAME = "V4_1p5y_fixed_moneyness_performance.ipynb"
    mx.PDF_NAME = "V4_1p5y_fixed_moneyness_performance.pdf"
    mx.MONEYNESS_IMPORT = "run_v4_1p5y_10k_fixed_moneyness_study"


def main() -> int:
    apply_config()
    mx.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
