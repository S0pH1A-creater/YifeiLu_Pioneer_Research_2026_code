#!/usr/bin/env python3
"""Assemble the 1.5-year monthly 10k return-based study.

Runs missing LSM cells, then the LSM report, stock-path report, and MD-GBM spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v4_1p5y_10k_monthly_empirical_study as v4  # noqa: E402
import run_v4_1p5y_10k_monthly_empirical_study_groups as groups  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip = {a[7:] for a in argv if a.startswith("--skip=")}
    extra = [a for a in argv if not a.startswith("--")]

    v4.patch_v3_wrap()
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    emp.CACHE.mkdir(parents=True, exist_ok=True)
    v4.swap_modified_gbm_meanfix_cache()

    if "decision" not in skip:
        print("===== LSM =====", flush=True)
        rc = v4.main(extra)
        if rc != 0:
            return rc
    if "groups" not in skip:
        print("===== LSM report =====", flush=True)
        groups.main()
    if "stock" not in skip:
        print("===== stock paths =====", flush=True)
        import run_v4_1p5y_10k_monthly_stock_study as stock

        rc = stock.main()
        if rc != 0:
            return rc
    if "spec" not in skip:
        print("===== MD-GBM spec =====", flush=True)
        v4.copy_modified_gbm_spec()
    v4.layout_short_outputs()
    print(f"outputs in {emp.SHORT}", flush=True)
    print(f"cache in {emp.CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
