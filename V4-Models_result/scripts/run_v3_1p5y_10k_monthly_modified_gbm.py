#!/usr/bin/env python3
"""1.5-year monthly 10k study for Modified GBM only.

Same frozen Monday ATM sample as the six-model 10k run. Writes a separate
cache and Results_In_Short folder so those PDFs are not overwritten.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_1p5y_10k_monthly_moneyness_study as mx_wrap  # noqa: E402
import run_v3_1p5y_10k_monthly_parameter_estimation as pe  # noqa: E402
import run_v3_1p5y_10k_monthly_stock_study as stock  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v3_5y_monthly_moneyness_study as mx  # noqa: E402

PARENT_CONTRACTS = (
    emp.ROOT / "results" / "empirical_study_1p5y_monthly_10000" / "shared_contracts.json"
)
MODELS = ("Modified GBM",)


def apply_modified_gbm_config() -> None:
    emp.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000_modified_gbm",
        short_name="V3 1.5-year monthly Modified GBM",
        pdf_name="V3_1p5y_monthly_modified_gbm_empirical_study.pdf",
        nb_name="V3_1p5y_monthly_modified_gbm_empirical_study.ipynb",
        notebook_import="run_v3_1p5y_10k_monthly_modified_gbm",
        engine_script="run_v3_1p5y_10k_monthly_modified_gbm.py",
        n_paths=10000,
        tickers=("SPY", "AAPL", "MSFT", "AMZN"),
    )
    emp.CACHE.mkdir(parents=True, exist_ok=True)
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    (emp.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    wrap.apply_1p5y_10k_config = apply_modified_gbm_config
    stock._paths = lambda: (
        emp.CACHE / "stock",
        emp.SHORT / "V3_1p5y_monthly_modified_gbm_stock_price.pdf",
        emp.SHORT / "V3_1p5y_monthly_modified_gbm_stock_price.ipynb",
    )
    mx.NB_NAME = "V3_1p5y_monthly_modified_gbm_moneyness.ipynb"
    mx.PDF_NAME = "V3_1p5y_monthly_modified_gbm_moneyness.pdf"
    mx.MONEYNESS_IMPORT = "run_v3_1p5y_10k_monthly_modified_gbm"
    pe.PDF_NAME = "V3_1p5y_monthly_modified_gbm_parameter_estimation.pdf"
    pe.NB_NAME = "V3_1p5y_monthly_modified_gbm_parameter_estimation.ipynb"


def _seed_shared_contracts() -> None:
    emp.CACHE.mkdir(parents=True, exist_ok=True)
    (emp.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if not emp.CONTRACTS_JSON.exists():
        src = PARENT_CONTRACTS if PARENT_CONTRACTS.exists() else wrap.SOURCE_CONTRACTS
        if not src.exists():
            raise FileNotFoundError(f"Need frozen Monday sample at {PARENT_CONTRACTS}")
        shutil.copy2(src, emp.CONTRACTS_JSON)
        print(f"copied shared contracts from {src}", flush=True)
    wrap._merge_amzn_contracts(emp.CONTRACTS_JSON)
    _merge_amzn_partials()


def _merge_amzn_partials() -> None:
    """Copy AMZN Modified GBM cells already computed in the AMZN-only cache."""
    src = emp.ROOT / "results" / "empirical_study_1p5y_monthly_10000_modified_gbm_amzn"
    if not src.exists():
        return
    copied = 0
    for part in sorted((src / "partial").glob("*.json")):
        srow = json.loads(part.read_text(encoding="utf-8"))
        amzn = srow.get("tickers", {}).get("AMZN")
        if not amzn:
            continue
        dest = emp.CACHE / "partial" / part.name
        drow = (
            json.loads(dest.read_text(encoding="utf-8"))
            if dest.exists()
            else {k: v for k, v in srow.items() if k != "tickers"} | {"tickers": {}}
        )
        if "AMZN" in drow.get("tickers", {}):
            continue
        drow.setdefault("tickers", {})["AMZN"] = amzn
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(drow, indent=2), encoding="utf-8")
        csv_rel = amzn.get("contracts_csv")
        if csv_rel:
            src_csv = src / csv_rel
            dst_csv = emp.CACHE / csv_rel
            if src_csv.exists():
                dst_csv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_csv, dst_csv)
        copied += 1
    stock_src = src / "stock"
    stock_dst = emp.CACHE / "stock"
    if (stock_src / "partial").exists():
        for part in sorted((stock_src / "partial").glob("*.json")):
            srow = json.loads(part.read_text(encoding="utf-8"))
            amzn = srow.get("tickers", {}).get("AMZN")
            if not amzn:
                continue
            dest = stock_dst / "partial" / part.name
            drow = (
                json.loads(dest.read_text(encoding="utf-8"))
                if dest.exists()
                else {k: v for k, v in srow.items() if k != "tickers"} | {"tickers": {}}
            )
            if "AMZN" in drow.get("tickers", {}):
                continue
            drow.setdefault("tickers", {})["AMZN"] = amzn
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(drow, indent=2, default=str), encoding="utf-8")
            csv_rel = amzn.get("series_csv")
            if csv_rel:
                src_csv = stock_src / csv_rel
                dst_csv = stock_dst / csv_rel
                if src_csv.exists():
                    dst_csv.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_csv, dst_csv)
            copied += 1
    if copied:
        print(f"merged {copied} AMZN Modified GBM partials from {src.name}", flush=True)


def _n_needed() -> int:
    return len(emp.TICKERS) * len(emp.REGIME_ORDER) * len(emp.TABLE_MODELS)


def run_decision() -> dict:
    _seed_shared_contracts()
    payload = emp.run_study()
    n_needed = _n_needed()
    n = len(payload.get("cells", {}))
    if n != n_needed or payload["meta"]["failures"]:
        raise RuntimeError(
            f"Decision study incomplete: {n}/{n_needed} cells, "
            f"failures={payload['meta']['failures']}"
        )
    emp.write_outputs(payload)
    return payload


def run_stock() -> dict:
    payload = stock.run_study()
    n_needed = _n_needed()
    n = len(payload.get("cells", {}))
    if n != n_needed or payload["meta"]["failures"]:
        raise RuntimeError(
            f"Stock study incomplete: {n}/{n_needed} cells, "
            f"failures={payload['meta']['failures']}"
        )
    stock.write_outputs(payload)
    return payload


def run_parameters() -> dict:
    return pe.run(recompute=False)


def run_moneyness() -> dict:
    mx_wrap.emp.apply_1p5y_10k_config = apply_modified_gbm_config
    apply_modified_gbm_config()
    return mx.run()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_modified_gbm_config()
    with emp.use_models(MODELS):
        apply_modified_gbm_config()
        skip = {a[7:] for a in argv if a.startswith("--skip=")}
        if "decision" not in skip:
            print("===== Modified GBM decision (LSM) =====", flush=True)
            run_decision()
        if "stock" not in skip:
            print("===== Modified GBM stock paths =====", flush=True)
            run_stock()
        if "params" not in skip:
            print("===== Modified GBM parameter estimation =====", flush=True)
            run_parameters()
        if "moneyness" not in skip:
            print("===== Modified GBM moneyness =====", flush=True)
            run_moneyness()
    print(f"outputs in {emp.SHORT}", flush=True)
    print(f"cache in {emp.CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
