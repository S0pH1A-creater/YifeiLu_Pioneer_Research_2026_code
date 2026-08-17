#!/usr/bin/env python3
"""V3 1.5-year fixed empirical study with 10,000 LSM paths.

Same Monday ATM sample, 18-month lookback, seven models, and four names as
the official monthly 10k run. Rolling is none: one calibration at the first
session of each 1-year regime, held for the whole window.

Writes a new cache and Results_In_Short folder so the monthly 10k reports
are not overwritten.
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
import run_v3_1p5y_10k_monthly_empirical_study as monthly  # noqa: E402
import run_v3_5y_monthly_empirical_study as study  # noqa: E402

STUDY_TICKERS = monthly.STUDY_TICKERS
STUDY_MODELS = monthly.STUDY_MODELS
SOURCE_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_1p5y_monthly_10000" / "shared_contracts.json"
)
SHORT_NAME = "V3 1.5-year fixed 10000 paths"
CACHE_NAME = "empirical_study_1p5y_fixed_10000"

GROUPS = {
    "option_implied": {
        "models": ("Heston", "Heston–Merton"),
        "pdf_name": "V3_1p5y_fixed_empirical_study_option_implied.pdf",
        "nb_name": "V3_1p5y_fixed_empirical_study_option_implied.ipynb",
        "payload_name": "payload_option_implied.json",
        "banner_extra": "option-implied Heston family",
        "notebook_title_suffix": " — option-implied Heston family",
        "group_intro": [
            "Companion to the seven-model 10,000-path fixed-calibration study. This report keeps only Heston and Heston–Merton, which are calibrated to listed calls (Fourier NLS). Rankings are within this pair; they are not compared here with return-based models.",
            "Companies are SPY, AAPL, MSFT, and AMZN. No new calibration or LSM. Every number is the stored cell from V3_1p5y_fixed_empirical_study.",
            "Return-based companion: V3_1p5y_fixed_empirical_study_return_based.pdf.",
        ],
    },
    "return_based": {
        "models": ("GBM", "Modified GBM", "GARCH", "Merton", "GARCH–Merton"),
        "pdf_name": "V3_1p5y_fixed_empirical_study_return_based.pdf",
        "nb_name": "V3_1p5y_fixed_empirical_study_return_based.ipynb",
        "payload_name": "payload_return_based.json",
        "banner_extra": "return-based calibration",
        "notebook_title_suffix": " — return-based models",
        "group_intro": [
            "Companion to the seven-model 10,000-path fixed-calibration study. This report keeps GBM, Modified GBM, GARCH, Merton, and GARCH–Merton, which are calibrated from lookback returns. Rankings are within this group; they are not compared here with option-implied Heston models.",
            "Companies are SPY, AAPL, MSFT, and AMZN. No new calibration or LSM. Every number is the stored cell from V3_1p5y_fixed_empirical_study.",
            "Option-implied companion: V3_1p5y_fixed_empirical_study_option_implied.pdf.",
        ],
    },
}


def apply_fixed_10k_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name=CACHE_NAME,
        short_name=SHORT_NAME,
        pdf_name="V3_1p5y_fixed_empirical_study.pdf",
        nb_name="V3_1p5y_fixed_empirical_study.ipynb",
        notebook_import="run_v3_1p5y_10k_fixed_empirical_study",
        engine_script="run_v3_1p5y_10k_fixed_empirical_study.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
        rolling_mode="none",
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}
    monthly.apply_1p5y_10k_config = apply_fixed_10k_config


def patch_companion_outputs() -> None:
    """Point stock / params / moneyness / diagnostics / returns at this folder."""
    import run_v3_1p5y_10k_monthly_model_diagnostics as diag
    import run_v3_1p5y_10k_monthly_moneyness_study as mx_wrap
    import run_v3_1p5y_10k_monthly_parameter_estimation as pe
    import run_v3_1p5y_10k_monthly_return_analysis as ra
    import run_v3_1p5y_10k_monthly_stock_study as stock
    import run_v3_5y_monthly_moneyness_study as mx

    apply_fixed_10k_config()
    stock._paths = lambda: (
        study.CACHE / "stock",
        study.SHORT / "V3_1p5y_fixed_stock_price.pdf",
        study.SHORT / "V3_1p5y_fixed_stock_price.ipynb",
    )
    pe.PDF_NAME = "V3_1p5y_fixed_parameter_estimation.pdf"
    pe.NB_NAME = "V3_1p5y_fixed_parameter_estimation.ipynb"
    mx.PDF_NAME = "V3_1p5y_fixed_moneyness_performance.pdf"
    mx.NB_NAME = "V3_1p5y_fixed_moneyness_performance.ipynb"
    mx.MONEYNESS_IMPORT = "run_v3_1p5y_10k_fixed_empirical_study"
    mx_wrap.emp.apply_1p5y_10k_config = apply_fixed_10k_config
    diag.PDF_NAME = "V3_1p5y_fixed_model_diagnostics.pdf"
    diag.NB_NAME = "V3_1p5y_fixed_model_diagnostics.ipynb"
    ra.PDF_NAME = "V3_1p5y_fixed_return_analysis.pdf"
    ra.NB_NAME = "V3_1p5y_fixed_return_analysis.ipynb"


def _seed_shared_contracts() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    study.SHORT.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if study.CONTRACTS_JSON.exists():
        return
    if not SOURCE_CONTRACTS.exists():
        raise FileNotFoundError(
            f"Need the frozen Monday sample at {SOURCE_CONTRACTS}"
        )
    shutil.copy2(SOURCE_CONTRACTS, study.CONTRACTS_JSON)
    print(f"copied shared contracts from {SOURCE_CONTRACTS}", flush=True)


def write_groups(full: dict | None = None) -> None:
    apply_fixed_10k_config()
    orig_pdf = study.SHORT / study.PDF_NAME
    orig_nb = study.SHORT / study.NB_NAME
    before = {
        orig_pdf: orig_pdf.stat().st_mtime if orig_pdf.exists() else None,
        orig_nb: orig_nb.stat().st_mtime if orig_nb.exists() else None,
    }
    full = full if full is not None else study.load_payload()
    n = len(full.get("cells", {}))
    n_needed = study._n_expected_cells()
    if n != n_needed:
        raise RuntimeError(f"Need the frozen {n_needed}-cell payload, found {n}")
    for gid, spec in GROUPS.items():
        extra = {
            "grouped_report": gid,
            "banner_extra": spec["banner_extra"],
            "notebook_title_suffix": spec["notebook_title_suffix"],
            "group_intro": spec["group_intro"],
            "pdf_name": spec["pdf_name"],
            "nb_name": spec["nb_name"],
        }
        payload = study.slice_payload_models(full, spec["models"], extra_meta=extra)
        sliced_json = study.CACHE / spec["payload_name"]
        sliced_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pdf_path = study.SHORT / spec["pdf_name"]
        nb_path = study.SHORT / spec["nb_name"]
        with study.use_models(spec["models"]):
            study.write_pdf(payload, path=pdf_path)
            study.build_notebook(payload, path=nb_path)
        print(f"{gid}: {pdf_path.name}  {nb_path.name}", flush=True)
    after_pdf = orig_pdf.stat().st_mtime if orig_pdf.exists() else None
    after_nb = orig_nb.stat().st_mtime if orig_nb.exists() else None
    if before[orig_pdf] != after_pdf or before[orig_nb] != after_nb:
        raise RuntimeError("Original full-study PDF/notebook timestamps changed")


def copy_modified_gbm_spec() -> None:
    candidates = [
        study.REPO / "Results_In_Short" / "V3 1.5-year monthly Modified GBM",
        study.REPO / "Results_In_Short" / "V3 1.5-year monthly empirical study 10000 paths",
    ]
    study.SHORT.mkdir(parents=True, exist_ok=True)
    for name in ("V3_modified_gbm_model.pdf", "V3_modified_gbm_model.ipynb"):
        dest = study.SHORT / name
        for src_dir in candidates:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dest)
                print(f"copied {name} from {src_dir.name}", flush=True)
                break
        else:
            print(f"missing {name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    apply_fixed_10k_config()
    _seed_shared_contracts()
    return study.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
