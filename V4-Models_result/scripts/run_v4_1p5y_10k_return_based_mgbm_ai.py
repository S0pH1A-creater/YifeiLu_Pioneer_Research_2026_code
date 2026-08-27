#!/usr/bin/env python3
"""1.5-year monthly 10k LSM for Modified GBM AI only.

Same Monday ATM sample, 18-month lookback, monthly rolling, four names,
four 1-year regimes as the V4 monthly 10k study. Cache only; no Results_In_Short.
"""
from __future__ import annotations

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
STUDY_MODELS = ("Modified GBM AI",)
CACHE_NAME = "empirical_study_1p5y_monthly_10000_mgbm_ai"


def apply_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name=CACHE_NAME,
        short_name="V4/_do_not_use_results_in_short_mgbm_ai",
        pdf_name="unused.pdf",
        nb_name="unused.ipynb",
        notebook_import="run_v4_1p5y_10k_return_based_mgbm_ai",
        engine_script="run_v4_1p5y_10k_return_based_mgbm_ai.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
        rolling_mode="monthly",
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    study.SHORT = study.CACHE
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}
    v4.patch_v3_wrap = lambda: apply_config()


def _print_option_rmse(payload: dict) -> None:
    print("\n=== Modified GBM AI option RMSE%  ·  1.5y lookback  ·  monthly  ·  n_paths=10000 ===")
    print(f"cache: {study.CACHE}")
    print(f"cells: {len(payload.get('cells', {}))}  failures: {payload['meta'].get('failures')}")
    print("\nBy company × regime")
    print(f"{'Ticker':<6}{'Regime':<12}{'RMSE%':>10}{'n':>8}")
    for ticker in study.TICKERS:
        for regime in study.REGIME_ORDER:
            rec = payload.get("cells", {}).get(f"{ticker}|{regime}|Modified GBM AI")
            if rec:
                print(f"{ticker:<6}{regime:<12}{rec['rmse_pct']:10.2f}{rec['n']:8d}")
    recs = [
        payload.get("cells", {}).get(f"{t}|{r}|Modified GBM AI")
        for t in study.TICKERS
        for r in study.REGIME_ORDER
    ]
    recs = [x for x in recs if x]
    if recs:
        mean_rmse = sum(x["rmse_pct"] for x in recs) / len(recs)
        print(f"\nMean RMSE% over {len(recs)} cells: {mean_rmse:.2f}")


def seed_contracts() -> None:
    src = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000"
    study.CACHE.mkdir(parents=True, exist_ok=True)
    for name in ("shared_contracts.json", "filter_funnel.json"):
        a, b = src / name, study.CACHE / name
        if a.exists() and not b.exists():
            shutil.copy2(a, b)
            print(f"copied {name} from {src.name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    apply_config()
    seed_contracts()
    payload = study.run_study()
    if payload.get("cells"):
        _print_option_rmse(payload)
    leaked = study.REPO / "Results_In_Short" / "V4" / "_do_not_use_results_in_short_mgbm_ai"
    if leaked.exists():
        import shutil

        shutil.rmtree(leaked)
    print(f"\ncache only (no Results_In_Short): {study.CACHE}", flush=True)
    return 0 if not payload.get("meta", {}).get("failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
