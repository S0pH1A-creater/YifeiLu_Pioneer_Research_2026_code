#!/usr/bin/env python3
"""Realized-return report for the V4 1.5-year fixed 10k windows."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_return_analysis as ra  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study as v4  # noqa: E402


def apply_config() -> None:
    v4.patch_v3_wrap()
    ra.PDF_NAME = "V4_1p5y_fixed_return_analysis.pdf"
    ra.NB_NAME = "V4_1p5y_fixed_return_analysis.ipynb"
    ra.REPORT_TITLE = "V4 return analysis"
    ra.STUDY_NAME = "V4 1.5-year fixed empirical study 10000 paths"


def main() -> int:
    apply_config()
    ra.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
