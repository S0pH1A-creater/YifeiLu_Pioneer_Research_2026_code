#!/usr/bin/env python3
"""Same 1.5y monthly 10k return-based OS study as v2, for Modified GBM meanfix.

Copies GBM / Modified GBM / GARCH / Merton / GARCH–Merton cells. Prices only
meanfix. Cache only; does not write Results_In_Short.
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
import run_v4_1p5y_10k_monthly_empirical_study as v4  # noqa: E402

STUDY_TICKERS = ("SPY", "AAPL", "MSFT", "AMZN")
STUDY_MODELS = (
    "GBM",
    "Modified GBM",
    "Modified GBM meanfix",
    "GARCH",
    "Merton",
    "GARCH–Merton",
)
REUSE_MODELS = ("GBM", "Modified GBM", "GARCH", "Merton", "GARCH–Merton")
SOURCE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000"
CACHE_NAME = "empirical_study_1p5y_monthly_10000_mgbm_meanfix"


def apply_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name=CACHE_NAME,
        short_name="V4/_do_not_use_results_in_short_mgbm_meanfix",
        pdf_name="unused.pdf",
        nb_name="unused.ipynb",
        notebook_import="run_v4_1p5y_10k_return_based_mgbm_meanfix",
        engine_script="run_v4_1p5y_10k_return_based_mgbm_meanfix.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
        rolling_mode="monthly",
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    study.SHORT = study.CACHE
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}
    v4.patch_v3_wrap = lambda: apply_config()


def _safe(model: str) -> str:
    return model.replace("–", "-").replace(" ", "_")


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def seed_from_monthly() -> None:
    apply_config()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    _copy_file(SOURCE / "shared_contracts.json", study.CONTRACTS_JSON)
    _copy_file(SOURCE / "filter_funnel.json", study.FILTER_JSON)
    for model in REUSE_MODELS:
        for regime in study.REGIME_ORDER:
            name = f"{regime}_{_safe(model)}.json"
            _copy_file(SOURCE / "partial" / name, study.CACHE / "partial" / name)
    if (SOURCE / "contracts").exists():
        for path in (SOURCE / "contracts").rglob("*.csv"):
            stem = path.stem
            if "modified_gbm_v2" in stem or "heston" in stem or "meanfix" in stem:
                continue
            if any(
                stem.endswith(s.lower()) or stem.endswith(s)
                for s in ("gbm", "modified_gbm", "garch", "merton", "garch_merton")
            ):
                _copy_file(path, study.CACHE / "contracts" / path.relative_to(SOURCE / "contracts"))
    print(f"seeded reusable cells from {SOURCE.name} → {study.CACHE.name}", flush=True)


def _print_option_rmse(payload: dict) -> None:
    print("\n=== Return-based option RMSE%  ·  1.5y lookback  ·  monthly  ·  n_paths=10000 ===")
    print(f"cache: {study.CACHE}")
    print(f"cells: {len(payload.get('cells', {}))}  failures: {payload['meta'].get('failures')}")
    print("\nOverall (mean RMSE% over company × regime)")
    print(f"{'Rank':<6}{'Model':<24}{'Mean RMSE%':>12}{'Median':>10}{'# best':>8}")
    for i, rec in enumerate(payload["summary_overall"], start=1):
        print(
            f"{i:<6}{rec['model']:<24}{rec['mean_rmse_pct']:12.2f}"
            f"{rec['median_rmse_pct']:10.2f}{rec['n_best']:8d}"
        )
    print("\nBy company × regime")
    print(f"{'Ticker':<6}{'Regime':<12}{'Model':<24}{'RMSE%':>10}")
    for ticker in study.TICKERS:
        for regime in study.REGIME_ORDER:
            for model in study.TABLE_MODELS:
                rec = payload.get("cells", {}).get(f"{ticker}|{regime}|{model}")
                if rec:
                    print(f"{ticker:<6}{regime:<12}{model:<24}{rec['rmse_pct']:10.2f}")


def main() -> int:
    apply_config()
    seed_from_monthly()
    payload = study.run_or_load(recompute=False)
    study.PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_option_rmse(payload)
    leaked = study.REPO / "Results_In_Short" / "V4" / "_do_not_use_results_in_short_mgbm_meanfix"
    if leaked.exists():
        shutil.rmtree(leaked)
    print(f"\ncache only (no Results_In_Short): {study.CACHE}", flush=True)
    n = len(payload.get("cells", {}))
    n_needed = study._n_expected_cells()
    if n != n_needed or payload["meta"]["failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
