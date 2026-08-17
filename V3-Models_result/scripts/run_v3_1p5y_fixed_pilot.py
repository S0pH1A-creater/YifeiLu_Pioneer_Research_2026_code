#!/usr/bin/env python3
"""Pilot: 1-year evaluation windows, fixed 1.5-year calibration, 2000 LSM paths.

rolling='none' — one lookback ending at the first day of each 1-year regime,
parameters held for the whole window. Six models, SPY/AAPL/MSFT. Same Monday
ATM sample as the 1.5-year monthly 2000-path study. Does not overwrite that
cache or its PDFs.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as study  # noqa: E402

SIX = ("GBM", "GARCH", "Heston", "Merton", "GARCH–Merton", "Heston–Merton")
TICKERS = ("SPY", "AAPL", "MSFT")
SOURCE_CONTRACTS = (
    study.ROOT / "results" / "empirical_study_1p5y_monthly" / "shared_contracts.json"
)


def apply_pilot_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_fixed_pilot",
        short_name="V3 1.5-year fixed pilot",
        pdf_name="V3_1p5y_fixed_pilot.pdf",
        nb_name="V3_1p5y_fixed_pilot.ipynb",
        notebook_import="run_v3_1p5y_fixed_pilot",
        engine_script="run_v3_1p5y_fixed_pilot.py",
        n_paths=2000,
        tickers=TICKERS,
    )
    study.ROLLING_MODE = "none"
    study.TABLE_MODELS = SIX
    study.MODELS = SIX
    import run_optimal_stopping_study as os_study

    os_study.ROLLING_MODES = ("none",)
    os_study.N_PATHS = 2000


def _seed() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if study.CONTRACTS_JSON.exists():
        return
    if not SOURCE_CONTRACTS.exists():
        raise FileNotFoundError(SOURCE_CONTRACTS)
    shutil.copy2(SOURCE_CONTRACTS, study.CONTRACTS_JSON)
    print(f"copied Monday sample from {SOURCE_CONTRACTS.name}", flush=True)


def _print_payload(payload: dict) -> None:
    cells = payload["cells"]
    print("\n=== PILOT  ·  1y windows  ·  fixed 1.5y calibration  ·  rolling=none  ·  n_paths=2000 ===")
    print("tickers:", ", ".join(payload["meta"]["tickers"]))
    print("models:", ", ".join(payload["meta"]["models"]))
    print("rolling:", payload["meta"]["rolling"], " lookback:", payload["meta"]["window_label"])
    print("cells:", len(cells), " failures:", payload["meta"]["failures"])
    print("\nOverall ranking (mean RMSE% over 12 company×regime cells)")
    print(f"{'Rank':<6}{'Model':<18}{'Mean RMSE%':>12}{'Median':>10}{'# best':>8}")
    for i, rec in enumerate(payload["summary_overall"], start=1):
        print(
            f"{i:<6}{rec['model']:<18}{rec['mean_rmse_pct']:12.2f}"
            f"{rec['median_rmse_pct']:10.2f}{rec['n_best']:8d}"
        )
    print("\nRMSE% by company × regime  (BEST in each 6-model cell)")
    header = f"{'Ticker':<6}{'Regime':<12}{'Window':<28}" + "".join(f"{m:>16}" for m in SIX) + f"{'BEST':>16}"
    print(header)
    for ticker in TICKERS:
        for regime in study.REGIME_ORDER:
            tab = payload["tables"][f"{ticker}|{regime}"]
            by_m = {r["model"]: r["rmse_pct"] for r in tab["rows"]}
            vals = [by_m[m] for m in SIX]
            line = f"{ticker:<6}{regime:<12}{study.REGIME_META[regime]['window']:<28}"
            line += "".join(f"{by_m[m]:16.2f}" for m in SIX)
            line += f"{tab['best_model']:>16}"
            print(line)
    print("\nMean RMSE% by company")
    by_t = {(r["ticker"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_ticker"]}
    print(f"{'Model':<18}" + "".join(f"{t:>10}" for t in TICKERS))
    for m in SIX:
        print(f"{m:<18}" + "".join(f"{by_t[(t, m)]:10.2f}" for t in TICKERS))
    print("\nMean RMSE% by regime")
    by_r = {(r["regime"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_regime"]}
    print(f"{'Model':<18}" + "".join(f"{r:>12}" for r in study.REGIME_ORDER))
    for m in SIX:
        print(f"{m:<18}" + "".join(f"{by_r[(r, m)]:12.2f}" for r in study.REGIME_ORDER))
    monthly = study.ROOT / "results" / "empirical_study_1p5y_monthly" / "payload.json"
    if monthly.exists():
        old = json.loads(monthly.read_text(encoding="utf-8"))
        print("\nVs monthly-rolling 1.5y / 2000-path study (same contracts, SPY/AAPL/MSFT)")
        print(f"{'Model':<18}{'Fixed none':>12}{'Monthly':>12}{'Δ (none−mth)':>14}")
        old_o = {r["model"]: r["mean_rmse_pct"] for r in old["summary_overall"]}
        new_o = {r["model"]: r["mean_rmse_pct"] for r in payload["summary_overall"]}
        for m in SIX:
            a, b = new_o[m], old_o[m]
            print(f"{m:<18}{a:12.2f}{b:12.2f}{a - b:14.2f}")


def main() -> int:
    apply_pilot_config()
    _seed()
    payload = study.run_or_load(recompute=False)
    n = len(payload.get("cells", {}))
    need = study._n_expected_cells()
    payload["meta"]["rolling"] = study.ROLLING_MODE
    study.PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_payload(payload)
    if n != need or payload["meta"]["failures"]:
        print(f"\nINCOMPLETE {n}/{need} failures={payload['meta']['failures']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
