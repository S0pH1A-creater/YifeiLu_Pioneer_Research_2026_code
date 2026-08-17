#!/usr/bin/env python3
"""V3 1.5-year monthly empirical study with 10,000 LSM paths.

Same Monday ATM sample, 18-month lookback, monthly rolling, and four
regimes as the original 1.5-year run. Writes a new folder so the 2000-path
reports are not overwritten.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as study  # noqa: E402

SOURCE_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_1p5y_monthly" / "shared_contracts.json"
)
FIVE_Y_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_5y_monthly" / "shared_contracts.json"
)


def apply_1p5y_10k_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000",
        short_name="V3 1.5-year monthly empirical study 10000 paths",
        pdf_name="V3_1p5y_monthly_empirical_study.pdf",
        nb_name="V3_1p5y_monthly_empirical_study.ipynb",
        notebook_import="run_v3_1p5y_10k_monthly_empirical_study",
        engine_script="run_v3_1p5y_10k_monthly_empirical_study.py",
        n_paths=10000,
    )


apply_1p5y_10k_config()


def _seed_shared_contracts() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if study.CONTRACTS_JSON.exists():
        return
    src = SOURCE_CONTRACTS if SOURCE_CONTRACTS.exists() else FIVE_Y_CONTRACTS
    if not src.exists():
        raise FileNotFoundError(f"Need a frozen Monday sample at {SOURCE_CONTRACTS} or {FIVE_Y_CONTRACTS}")
    shutil.copy2(src, study.CONTRACTS_JSON)
    print(f"copied shared contracts from {src}", flush=True)


def main(argv: list[str] | None = None) -> int:
    apply_1p5y_10k_config()
    _seed_shared_contracts()
    return study.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
