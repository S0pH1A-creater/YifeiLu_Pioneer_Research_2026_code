#!/usr/bin/env python3
"""Parameter-estimation report for the V4 1.5y fixed 10k study.

Reads calibration CSVs written during LSM. No re-pricing. P, premium, and Q
fields are listed separately for Heston / Merton / Heston–Merton.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_parameter_estimation as pe  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study as v4  # noqa: E402
from run_v4_1p5y_10k_monthly_parameter_estimation import V4_PARAM_SPEC  # noqa: E402

PDF_NAME = "V4_1p5y_fixed_parameter_estimation.pdf"
NB_NAME = "V4_1p5y_fixed_parameter_estimation.ipynb"


def apply_config() -> None:
    v4.patch_v3_wrap()
    pe.PDF_NAME = PDF_NAME
    pe.NB_NAME = NB_NAME
    pe.PARAM_SPEC = {m: V4_PARAM_SPEC[m] for m in v4.STUDY_MODELS}


def main() -> int:
    apply_config()
    pe.run(recompute=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
