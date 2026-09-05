#!/usr/bin/env python3
"""V3 1.5-year monthly empirical study with 10,000 LSM paths.

Same Monday ATM sample, 18-month lookback, monthly rolling, and four
regimes as the original 1.5-year run. Writes a new folder so the 2000-path
reports are not overwritten.

Underlyings: SPY, AAPL, MSFT, AMZN.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_5y_monthly_empirical_study as study  # noqa: E402

STUDY_TICKERS = ("SPY", "AAPL", "MSFT", "AMZN")
STUDY_MODELS = (
    "GBM",
    "MD-GBM",
    "GARCH",
    "Merton",
    "GARCH–Merton",
)
SOURCE_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_1p5y_monthly" / "shared_contracts.json"
)
FIVE_Y_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_5y_monthly" / "shared_contracts.json"
)
AMZN_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_1p5y_monthly_10000_modified_gbm_amzn" / "shared_contracts.json"
)
MGBM_CACHE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000_modified_gbm"


def apply_1p5y_10k_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000",
        short_name="results/1p5y_monthly_return_based",
        pdf_name="V3_1p5y_monthly_empirical_study.pdf",
        nb_name="V3_1p5y_monthly_empirical_study.ipynb",
        notebook_import="run_v3_1p5y_10k_monthly_empirical_study",
        engine_script="run_v3_1p5y_10k_monthly_empirical_study.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}


apply_1p5y_10k_config()


def _merge_amzn_contracts(dest: Path) -> None:
    """Keep frozen SPY/AAPL/MSFT samples; add AMZN from the dedicated Monday draw."""
    if not dest.exists() or not AMZN_CONTRACTS.exists():
        return
    src = json.loads(AMZN_CONTRACTS.read_text(encoding="utf-8"))
    dst = json.loads(dest.read_text(encoding="utf-8"))
    changed = False
    for regime, block in src.items():
        amzn = block.get("tickers", {}).get("AMZN")
        if not amzn or regime not in dst:
            continue
        dst[regime].setdefault("tickers", {})
        if "AMZN" not in dst[regime]["tickers"]:
            dst[regime]["tickers"]["AMZN"] = amzn
            changed = True
    if changed:
        dest.write_text(json.dumps(dst, indent=2), encoding="utf-8")
        print("merged AMZN Monday sample into shared_contracts.json", flush=True)


def _copy_missing(src: Path, dst: Path, pattern: str) -> int:
    if not src.exists():
        return 0
    n = 0
    for path in src.rglob(pattern):
        if not path.is_file():
            continue
        dest = dst / path.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
            n += 1
    return n


def merge_modified_gbm_cache() -> None:
    """Copy already-run Modified GBM cells into the seven-model 10k cache."""
    src, dst = MGBM_CACHE, study.CACHE
    if not src.exists():
        return
    n = 0
    n += _copy_missing(src / "partial", dst / "partial", "*.json")
    n += _copy_missing(src / "contracts", dst / "contracts", "*modified_gbm.csv")
    n += _copy_missing(src / "stock" / "partial", dst / "stock" / "partial", "*.json")
    n += _copy_missing(src / "stock" / "series", dst / "stock" / "series", "*modified_gbm.csv")
    n += _copy_missing(
        src / "parameter_estimation" / "raw",
        dst / "parameter_estimation" / "raw",
        "Modified_GBM.csv",
    )
    if n:
        print(f"merged {n} Modified GBM artifacts from {src.name}", flush=True)


def _seed_shared_contracts() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if not study.CONTRACTS_JSON.exists():
        src = SOURCE_CONTRACTS if SOURCE_CONTRACTS.exists() else FIVE_Y_CONTRACTS
        if not src.exists():
            raise FileNotFoundError(f"Need a frozen Monday sample at {SOURCE_CONTRACTS} or {FIVE_Y_CONTRACTS}")
        shutil.copy2(src, study.CONTRACTS_JSON)
        print(f"copied shared contracts from {src}", flush=True)
    _merge_amzn_contracts(study.CONTRACTS_JSON)
    merge_modified_gbm_cache()


def main(argv: list[str] | None = None) -> int:
    apply_1p5y_10k_config()
    _seed_shared_contracts()
    return study.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
