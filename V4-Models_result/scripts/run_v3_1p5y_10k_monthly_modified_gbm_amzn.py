#!/usr/bin/env python3
"""1.5-year monthly 10k Modified GBM study for AMZN only.

Does not overwrite the SPY/AAPL/MSFT Modified GBM cache or PDFs.
Samples AMZN Monday ATM contracts from the processed call panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_1p5y_10k_monthly_modified_gbm as mgbm  # noqa: E402
import run_v3_1p5y_10k_monthly_moneyness_study as mx_wrap  # noqa: E402
import run_v3_1p5y_10k_monthly_parameter_estimation as pe  # noqa: E402
import run_v3_1p5y_10k_monthly_stock_study as stock  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v3_5y_monthly_moneyness_study as mx  # noqa: E402

TICKER = "AMZN"
_INNER_LOAD_NS = os_study._load_ns


def _load_ns_amzn(nb_path):
    g = _INNER_LOAD_NS(nb_path)
    prices = g["prices"]
    if TICKER not in prices.columns:
        raise RuntimeError(f"{TICKER} missing from prices_clean.csv")
    tickers = [TICKER]
    g["TICKERS"] = tickers
    g["period_prices"] = prices.loc[g["PERIOD_START"] : g["PERIOD_END"], tickers].copy()
    g["log_returns_all"] = np.log(prices[tickers]).diff()
    return g


def apply_amzn_config() -> None:
    emp.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000_modified_gbm_amzn",
        short_name="V3 1.5-year monthly Modified GBM AMZN",
        pdf_name="V3_1p5y_monthly_modified_gbm_amzn_empirical_study.pdf",
        nb_name="V3_1p5y_monthly_modified_gbm_amzn_empirical_study.ipynb",
        notebook_import="run_v3_1p5y_10k_monthly_modified_gbm_amzn",
        engine_script="run_v3_1p5y_10k_monthly_modified_gbm_amzn.py",
        n_paths=10000,
    )
    emp.TICKERS = (TICKER,)
    os_study.TICKERS = (TICKER,)
    os_study.COLORS = {**os_study.COLORS, TICKER: "#C73E7B"}
    os_study._load_ns = _load_ns_amzn
    emp._load_ns = _load_ns_amzn
    # scipy.optimize import can hang in this environment; Modified GBM does not need it.
    os_study._install_scipy_minimize_fallback = lambda: None
    wrap.apply_1p5y_10k_config = apply_amzn_config
    mgbm.apply_modified_gbm_config = apply_amzn_config
    emp.CACHE.mkdir(parents=True, exist_ok=True)
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    (emp.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    stock._paths = lambda: (
        emp.CACHE / "stock",
        emp.SHORT / "V3_1p5y_monthly_modified_gbm_amzn_stock_price.pdf",
        emp.SHORT / "V3_1p5y_monthly_modified_gbm_amzn_stock_price.ipynb",
    )
    mx.NB_NAME = "V3_1p5y_monthly_modified_gbm_amzn_moneyness.ipynb"
    mx.PDF_NAME = "V3_1p5y_monthly_modified_gbm_amzn_moneyness.pdf"
    mx.MONEYNESS_IMPORT = "run_v3_1p5y_10k_monthly_modified_gbm_amzn"
    pe.PDF_NAME = "V3_1p5y_monthly_modified_gbm_amzn_parameter_estimation.pdf"
    pe.NB_NAME = "V3_1p5y_monthly_modified_gbm_amzn_parameter_estimation.ipynb"


def _banner_amzn(payload: dict | None = None) -> str:
    extra = ""
    if payload:
        extra = str(payload.get("meta", {}).get("banner_extra") or "").strip()
    base = (
        f"{emp._models_phrase()}  ·  AMZN  ·  four volatility regimes  ·  "
        f"{emp.LOOKBACK_PHRASE} monthly calibration"
    )
    return f"{base}  ·  {extra}" if extra else base


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_amzn_config()
    emp._banner_line = _banner_amzn
    mgbm.apply_modified_gbm_config = apply_amzn_config
    mgbm._seed_shared_contracts = lambda: None  # sample AMZN; do not copy SPY/AAPL/MSFT
    with emp.use_models(mgbm.MODELS):
        apply_amzn_config()
        skip = {a[7:] for a in argv if a.startswith("--skip=")}
        if "decision" not in skip:
            print("===== Modified GBM AMZN decision (LSM) =====", flush=True)
            mgbm.run_decision()
        if "stock" not in skip:
            print("===== Modified GBM AMZN stock paths =====", flush=True)
            mgbm.run_stock()
        if "params" not in skip:
            print("===== Modified GBM AMZN parameter estimation =====", flush=True)
            mgbm.run_parameters()
        if "moneyness" not in skip:
            print("===== Modified GBM AMZN moneyness =====", flush=True)
            mx_wrap.emp.apply_1p5y_10k_config = apply_amzn_config
            apply_amzn_config()
            mx.run()
    print(f"outputs in {emp.SHORT}", flush=True)
    print(f"cache in {emp.CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
