#!/usr/bin/env python3
"""Model-diagnostic report for the V4 1.5-year monthly 10k study."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_model_diagnostics as diag  # noqa: E402
import run_v4_1p5y_10k_monthly_empirical_study as v4  # noqa: E402


def apply_config() -> None:
    v4.patch_v3_wrap()
    diag.PDF_NAME = "V4_1p5y_monthly_model_diagnostics.pdf"
    diag.NB_NAME = "V4_1p5y_monthly_model_diagnostics.ipynb"
    diag.REPORT_TITLE = "V4 model diagnostic verification"
    diag.STUDY_NAME = "V4 1.5-year monthly empirical study 10000 paths"
    _orig = diag.build_payload

    def build_payload():
        payload = _orig()
        payload["meta"]["garch"] = "Duan GARCH-in-mean standardized residuals"
        return payload

    diag.build_payload = build_payload


def main() -> int:
    apply_config()
    diag.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
