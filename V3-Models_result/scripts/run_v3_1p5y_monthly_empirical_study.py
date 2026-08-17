#!/usr/bin/env python3
"""V3 empirical study with a 1.5-year lookback (otherwise identical to the 5-year run)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as study  # noqa: E402

FIVE_Y_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_5y_monthly" / "shared_contracts.json"
)


def apply_1p5y_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly",
        short_name="V3 1.5-year monthly empirical study",
        pdf_name="V3_1p5y_monthly_empirical_study.pdf",
        nb_name="V3_1p5y_monthly_empirical_study.ipynb",
        notebook_import="run_v3_1p5y_monthly_empirical_study",
        engine_script="run_v3_1p5y_monthly_empirical_study.py",
    )


apply_1p5y_config()


def _seed_shared_contracts() -> None:
    """Reuse the frozen 5-year Monday sample so only the lookback changes."""
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if study.CONTRACTS_JSON.exists():
        return
    if not FIVE_Y_CONTRACTS.exists():
        raise FileNotFoundError(
            f"Need the 5-year shared sample at {FIVE_Y_CONTRACTS} so both studies price the same contracts."
        )
    shutil.copy2(FIVE_Y_CONTRACTS, study.CONTRACTS_JSON)
    print(f"copied shared contracts from {FIVE_Y_CONTRACTS}", flush=True)


def main(argv: list[str] | None = None) -> int:
    apply_1p5y_config()
    _seed_shared_contracts()
    return study.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
