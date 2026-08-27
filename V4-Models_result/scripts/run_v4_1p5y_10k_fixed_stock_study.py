#!/usr/bin/env python3
"""P-measure stock-path report for the V4 1.5-year fixed 10k study."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_stock_study as stock  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study as v4  # noqa: E402


def apply_config() -> None:
    v4.patch_v3_wrap()

    def _paths():
        cache = emp.CACHE / "stock"
        short = emp.SHORT
        return (
            cache,
            short / "V4_1p5y_fixed_stock_price.pdf",
            short / "V4_1p5y_fixed_stock_price.ipynb",
        )

    stock._paths = _paths


def main(argv=None) -> int:
    apply_config()
    return stock.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
