#!/usr/bin/env python3
"""V4 GARCH-only 1.5-year monthly empirical study, 10,000 LSM paths.

Duan (1995) GARCH-in-mean under P; American calls priced on LRNVR Q-paths.
Companies: SPY, AAPL, MSFT, AMZN. Four 1-year evaluation regimes.
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
from garch_duan_lrnvr import report_p_and_q  # noqa: E402

STUDY_TICKERS = ("SPY", "AAPL", "MSFT", "AMZN")
STUDY_MODELS = ("GARCH",)
V3_CONTRACTS = (
    study.REPO
    / "V3-Models_result"
    / "results"
    / "empirical_study_1p5y_monthly_10000"
    / "shared_contracts.json"
)


def apply_v4_garch_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000_garch",
        short_name="V4 1.5-year monthly GARCH LRNVR",
        pdf_name="V4_1p5y_monthly_garch_lrnvr.pdf",
        nb_name="V4_1p5y_monthly_garch_lrnvr.ipynb",
        notebook_import="run_v4_1p5y_10k_monthly_garch_study",
        engine_script="run_v4_1p5y_10k_monthly_garch_study.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}


apply_v4_garch_config()


def _seed_shared_contracts() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if study.CONTRACTS_JSON.exists():
        return
    if not V3_CONTRACTS.exists():
        raise FileNotFoundError(
            f"Need the frozen V3 Monday sample at {V3_CONTRACTS}"
        )
    shutil.copy2(V3_CONTRACTS, study.CONTRACTS_JSON)
    print(f"copied shared contracts from {V3_CONTRACTS}", flush=True)


_study_load_ns = os_study._load_ns
_orig_run_mode = os_study._run_mode


def _load_ns(nb_path):
    g = _study_load_ns(nb_path)
    g["_v4_regime"] = os_study._regime_from_name(Path(nb_path))
    return g


def _run_mode(g, rolling_mode, contracts, ticker):
    result = _orig_run_mode(g, rolling_mode, contracts, ticker)
    cal = g.get("rolling", {}).get(ticker)
    regime = g.get("_v4_regime")
    if cal is None or regime is None or len(cal) == 0:
        return result
    dest = study.CACHE / "parameter_estimation" / "raw" / regime / ticker
    dest.mkdir(parents=True, exist_ok=True)
    cal.to_csv(dest / "GARCH.csv", index=False)
    last = cal.iloc[-1]
    p_tbl, q_tbl = report_p_and_q(last)
    dest.joinpath("GARCH_P_Q.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "regime": regime,
                "n_updates": int(len(cal)),
                "asof": str(last.get("date")),
                "P": {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in p_tbl.items()},
                "Q": {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in q_tbl.items()},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return result


os_study._load_ns = _load_ns
os_study._run_mode = _run_mode


def collect_pq_tables() -> pd.DataFrame:
    rows = []
    raw = study.CACHE / "parameter_estimation" / "raw"
    if not raw.exists():
        return pd.DataFrame()
    for path in sorted(raw.glob("*/*/GARCH_P_Q.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        p, q = rec["P"], rec["Q"]
        rows.append(
            {
                "ticker": rec["ticker"],
                "regime": rec["regime"],
                "n_updates": rec["n_updates"],
                "asof": rec["asof"],
                "P_lambda": p.get("lambda"),
                "P_omega": p.get("omega"),
                "P_alpha": p.get("alpha"),
                "P_beta": p.get("beta"),
                "P_sigma0": p.get("sigma0"),
                "P_mean": p.get("mean_equation"),
                "P_shock": p.get("variance_shock"),
                "P_persist": p.get("persist"),
                "P_mu_p": p.get("mu_p"),
                "Q_lambda": q.get("lambda"),
                "Q_omega": q.get("omega"),
                "Q_alpha": q.get("alpha"),
                "Q_beta": q.get("beta"),
                "Q_sigma0": q.get("sigma0"),
                "Q_mean": q.get("mean_equation"),
                "Q_shock": q.get("variance_shock"),
                "Q_persist": q.get("persist"),
                "Q_stationary": q.get("q_stationary"),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        dest = study.CACHE / "pq_parameters.csv"
        df.to_csv(dest, index=False)
        study.SHORT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, study.SHORT / "pq_parameters.csv")
    return df


def main(argv: list[str] | None = None) -> int:
    apply_v4_garch_config()
    _seed_shared_contracts()
    rc = study.main(argv)
    df = collect_pq_tables()
    if len(df):
        print("\n=== P-measure parameters vs Q-dynamics (last monthly update) ===", flush=True)
        print(df.to_string(index=False), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
