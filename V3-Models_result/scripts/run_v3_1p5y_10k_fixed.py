#!/usr/bin/env python3
"""Official 1.5-year fixed 10,000-path study: decision, groups, stock, moneyness, params.

Does not write into the monthly 10k cache or Results_In_Short folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_fixed_empirical_study as fixed  # noqa: E402
import run_v3_1p5y_10k_monthly_model_diagnostics as diag  # noqa: E402
import run_v3_1p5y_10k_monthly_parameter_estimation as pe  # noqa: E402
import run_v3_1p5y_10k_monthly_return_analysis as ra  # noqa: E402
import run_v3_1p5y_10k_monthly_stock_study as stock  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v3_5y_monthly_moneyness_study as mx  # noqa: E402


def _n_needed() -> int:
    return emp._n_expected_cells()


def run_decision() -> dict:
    fixed._seed_shared_contracts()
    payload = emp.run_or_load(recompute=False)
    n = len(payload.get("cells", {}))
    n_needed = _n_needed()
    if n != n_needed or payload["meta"]["failures"]:
        raise RuntimeError(
            f"Decision study incomplete: {n}/{n_needed} cells, "
            f"failures={payload['meta']['failures']}"
        )
    emp.write_outputs(payload)
    return payload


def run_stock() -> dict:
    payload = stock.run_or_load(recompute=False)
    n = len(payload.get("cells", {}))
    n_needed = _n_needed()
    if n != n_needed or payload["meta"]["failures"]:
        raise RuntimeError(
            f"Stock study incomplete: {n}/{n_needed} cells, "
            f"failures={payload['meta']['failures']}"
        )
    stock.write_outputs(payload)
    return payload


def _print_ranking(payload: dict) -> None:
    print("\n=== OFFICIAL  ·  1y windows  ·  fixed 1.5y calibration  ·  n_paths=10000 ===")
    print("folder:", emp.SHORT)
    print("cache:", emp.CACHE)
    print("rolling:", payload["meta"].get("rolling"), " lookback:", payload["meta"].get("window_label"))
    print("tickers:", ", ".join(payload["meta"]["tickers"]))
    print("models:", ", ".join(payload["meta"]["models"]))
    print("cells:", len(payload.get("cells", {})), " failures:", payload["meta"].get("failures"))
    print("\nOverall ranking (mean RMSE% over company×regime cells)")
    print(f"{'Rank':<6}{'Model':<18}{'Mean RMSE%':>12}{'Median':>10}{'# best':>8}")
    for i, rec in enumerate(payload["summary_overall"], start=1):
        print(
            f"{i:<6}{rec['model']:<18}{rec['mean_rmse_pct']:12.2f}"
            f"{rec['median_rmse_pct']:10.2f}{rec['n_best']:8d}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    skip = {a[7:] for a in argv if a.startswith("--skip=")}
    fixed.apply_fixed_10k_config()
    fixed.patch_companion_outputs()
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    emp.CACHE.mkdir(parents=True, exist_ok=True)

    payload = None
    if "decision" not in skip:
        print("===== fixed 10k decision (LSM) =====", flush=True)
        payload = run_decision()
        _print_ranking(payload)
    if "groups" not in skip:
        print("===== grouped reports =====", flush=True)
        fixed.write_groups(full=payload)
    if "stock" not in skip:
        print("===== stock paths =====", flush=True)
        run_stock()
    if "params" not in skip:
        print("===== parameter estimation =====", flush=True)
        pe.run(recompute=False)
    if "moneyness" not in skip:
        print("===== moneyness =====", flush=True)
        mx.run()
    if "diagnostics" not in skip:
        print("===== model diagnostics =====", flush=True)
        diag.run()
    if "returns" not in skip:
        print("===== return analysis =====", flush=True)
        ra.run()
    if "spec" not in skip:
        print("===== Modified GBM spec =====", flush=True)
        fixed.copy_modified_gbm_spec()
    print(f"outputs in {emp.SHORT}", flush=True)
    print(f"cache in {emp.CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
