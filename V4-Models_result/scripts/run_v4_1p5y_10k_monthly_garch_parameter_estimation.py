#!/usr/bin/env python3
"""GARCH-only parameter-estimation report for the V4 1.5y monthly 10k study.

Reads the rolling GARCH.csv files already written during LSM. No re-pricing.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_1p5y_10k_monthly_parameter_estimation as pe  # noqa: E402
import run_v4_1p5y_10k_monthly_garch_study as v4  # noqa: E402

PDF_NAME = "V4_1p5y_monthly_garch_lrnvr_parameter_estimation.pdf"
NB_NAME = "V4_1p5y_monthly_garch_lrnvr_parameter_estimation.ipynb"


def apply_config() -> None:
    v4.apply_v4_garch_config()
    pe.PDF_NAME = PDF_NAME
    pe.NB_NAME = NB_NAME
    wrap.apply_1p5y_10k_config = apply_config


def main() -> int:
    apply_config()
    pe.run(recompute=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
