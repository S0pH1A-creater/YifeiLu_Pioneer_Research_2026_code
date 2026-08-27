#!/usr/bin/env python3
"""Assemble the V4 1.5-year fixed 10k study the way V4 monthly is laid out.

Runs missing LSM cells, then grouped reports, stock, PE, moneyness,
diagnostics, return analysis, and Modified GBM spec.

Calibration is fixed (rolling=none): one 18-month lookback at the first
session of each 1-year window. Same Monday ATM sample as the monthly 10k
study. Does not write into that monthly cache or Results_In_Short folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study as v4  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study_groups as groups  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip = {a[7:] for a in argv if a.startswith("--skip=")}
    extra = [a for a in argv if not a.startswith("--")]

    v4.patch_v3_wrap()
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    emp.CACHE.mkdir(parents=True, exist_ok=True)

    if "decision" not in skip:
        print("===== V4 10k LSM (fixed 1.5y calibration) =====", flush=True)
        rc = v4.main(extra)
        if rc != 0:
            return rc
    if "groups" not in skip:
        print("===== grouped reports =====", flush=True)
        groups.main()
    if "stock" not in skip:
        print("===== stock paths =====", flush=True)
        import run_v4_1p5y_10k_fixed_stock_study as stock

        rc = stock.main()
        if rc != 0:
            return rc
    if "params" not in skip:
        print("===== parameter estimation =====", flush=True)
        import run_v4_1p5y_10k_fixed_parameter_estimation as pe

        pe.main()
    if "moneyness" not in skip:
        print("===== moneyness =====", flush=True)
        import run_v4_1p5y_10k_fixed_moneyness_study as mx

        mx.main()
    if "diagnostics" not in skip:
        print("===== model diagnostics =====", flush=True)
        import run_v4_1p5y_10k_fixed_model_diagnostics as diag

        diag.main()
    if "returns" not in skip:
        print("===== return analysis =====", flush=True)
        import run_v4_1p5y_10k_fixed_return_analysis as ra

        ra.main()
    if "spec" not in skip:
        print("===== Modified GBM spec =====", flush=True)
        v4.copy_modified_gbm_spec()
    v4.layout_short_outputs()
    print(f"outputs in {emp.SHORT}", flush=True)
    print(f"cache in {emp.CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
