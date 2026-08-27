#!/usr/bin/env python3
"""Re-run V4 1.5y monthly 10k LSM with GARCH–Merton Duan+Pan Q-map.

Stores under V4-Models_result/results/empirical_study_1p5y_monthly_10000_garch_merton_duan_pan/
— does **not** write Results_In_Short.

Other six models are copied from the finished monthly 10k cache so the
seven-model tables stay comparable; only GARCH–Merton cells are re-priced.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_5y_monthly_empirical_study as study  # noqa: E402
import run_v4_1p5y_10k_monthly_empirical_study as v4  # noqa: E402
from garch_merton_pq import report_garch_merton_pq  # noqa: E402

PARENT_CACHE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000"
OUT_CACHE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000_garch_merton_duan_pan"
OUT_SHORT = OUT_CACHE / "_local_reports"  # not Results_In_Short


def apply_config() -> None:
    v4.apply_v4_10k_config()
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000_garch_merton_duan_pan",
        short_name="V4/_local_garch_merton_duan_pan_monthly_10000",
        pdf_name="V4_1p5y_monthly_garch_merton_duan_pan.pdf",
        nb_name="V4_1p5y_monthly_garch_merton_duan_pan.ipynb",
        notebook_import="run_v4_1p5y_10k_monthly_garch_merton_duan_pan",
        engine_script="run_v4_1p5y_10k_monthly_garch_merton_duan_pan.py",
        n_paths=10000,
        tickers=v4.STUDY_TICKERS,
    )
    # Keep SHORT inside the results cache so Results_In_Short is untouched.
    study.SHORT = OUT_SHORT
    study.CACHE = OUT_CACHE
    study.PAYLOAD_JSON = OUT_CACHE / "payload.json"
    study.CONTRACTS_JSON = OUT_CACHE / "shared_contracts.json"
    study.FILTER_JSON = OUT_CACHE / "filter_funnel.json"
    study.TABLE_MODELS = v4.STUDY_MODELS
    study.MODELS = v4.STUDY_MODELS
    v4.PQ_REPORTERS["GARCH–Merton"] = report_garch_merton_pq


def _seed_from_parent() -> None:
    if not PARENT_CACHE.exists():
        raise FileNotFoundError(f"Need parent monthly cache at {PARENT_CACHE}")
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    (OUT_CACHE / "partial").mkdir(parents=True, exist_ok=True)
    (OUT_CACHE / "contracts").mkdir(parents=True, exist_ok=True)
    OUT_SHORT.mkdir(parents=True, exist_ok=True)

    for name in ("shared_contracts.json", "filter_funnel.json"):
        src = PARENT_CACHE / name
        if src.exists() and not (OUT_CACHE / name).exists():
            shutil.copy2(src, OUT_CACHE / name)
            print(f"copied {name}", flush=True)

    # Copy every non-GARCH–Merton partial so assemble_payload can build 7-model tables.
    src_partial = PARENT_CACHE / "partial"
    for path in sorted(src_partial.glob("*.json")):
        if "GARCH-Merton" in path.name or "GARCH–Merton" in path.name:
            continue
        dest = OUT_CACHE / "partial" / path.name
        if not dest.exists():
            shutil.copy2(path, dest)

    # Drop any stale GARCH–Merton partials so they are re-priced with Duan+Pan.
    for path in list((OUT_CACHE / "partial").glob("*GARCH*Merton*.json")):
        path.unlink()
        print(f"removed stale {path.name}", flush=True)

    # Copy non-GM contract CSVs (optional; GM will rewrite its own).
    src_contracts = PARENT_CACHE / "contracts"
    if src_contracts.exists():
        for path in src_contracts.rglob("*.csv"):
            if "garch_merton" in path.name.lower():
                continue
            dest = OUT_CACHE / "contracts" / path.relative_to(src_contracts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(path, dest)


def _print_summary(payload: dict) -> None:
    models = list(payload.get("meta", {}).get("models", []))
    print("\n===== GARCH–Merton Duan+Pan monthly 10k — RMSE% by cell =====", flush=True)
    header = f"{'ticker':<6} {'regime':<10} " + " ".join(f"{m[:12]:>12}" for m in models)
    print(header, flush=True)
    for ticker in payload["meta"]["tickers"]:
        for regime in payload["meta"]["regimes"]:
            vals = []
            for model in models:
                cell = payload["cells"].get(f"{ticker}|{regime}|{model}")
                vals.append(f"{cell['rmse_pct']:12.2f}" if cell else f"{'—':>12}")
            print(f"{ticker:<6} {regime:<10} " + " ".join(vals), flush=True)

    print("\n===== Best-in-cell (lowest RMSE%) =====", flush=True)
    wins = {m: 0 for m in models}
    for key, tab in payload.get("tables", {}).items():
        best = tab.get("best_model")
        if best in wins:
            wins[best] += 1
        print(f"  {key}: {best}", flush=True)
    print("\nWin counts:", flush=True)
    for m, n in sorted(wins.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {m}: {n}", flush=True)

    # GARCH–Merton alone vs prior cache
    prior = PARENT_CACHE / "payload.json"
    if prior.exists():
        old = json.loads(prior.read_text(encoding="utf-8"))
        print("\n===== GARCH–Merton RMSE% — old (μ→r_f) vs new (Duan+Pan) =====", flush=True)
        print(f"{'ticker':<6} {'regime':<10} {'old':>10} {'new':>10} {'Δ':>10}", flush=True)
        for ticker in payload["meta"]["tickers"]:
            for regime in payload["meta"]["regimes"]:
                k = f"{ticker}|{regime}|GARCH–Merton"
                o = old.get("cells", {}).get(k, {}).get("rmse_pct")
                n = payload.get("cells", {}).get(k, {}).get("rmse_pct")
                if o is None or n is None:
                    continue
                print(f"{ticker:<6} {regime:<10} {o:10.2f} {n:10.2f} {n-o:10.2f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_config()
    _seed_from_parent()
    apply_config()

    # V4 hooks already wrap os_study at import; keep them, then add WINDOW/TICKERS.
    _after_v4 = os_study._load_ns

    def _load_ns(nb_path):
        g = _after_v4(nb_path)
        wo = g.get("WINDOW_OPTIONS")
        if isinstance(wo, dict) and study.WINDOW_LABEL not in wo:
            wo[study.WINDOW_LABEL] = study.LOOKBACK_OFFSET
        g["WINDOW_LABEL"] = study.WINDOW_LABEL
        prices = g.get("prices")
        if prices is not None:
            tickers = [t for t in study.TICKERS if t in prices.columns]
            if tickers:
                g["TICKERS"] = tickers
                g["period_prices"] = prices.loc[g["PERIOD_START"] : g["PERIOD_END"], tickers].copy()
                g["log_returns_all"] = np.log(prices[tickers]).diff()
        return g

    os_study._load_ns = _load_ns

    print(f"cache → {OUT_CACHE}", flush=True)
    print(f"Results_In_Short untouched; local notes → {OUT_SHORT}", flush=True)

    only = [a for a in argv if not a.startswith("--")] or ["garch-merton"]
    recompute = "--recompute" in argv
    study.run_or_load(recompute=recompute, only=only)

    shared = study.sample_shared_contracts()
    funnel = study.filter_funnel()
    completed = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((OUT_CACHE / "partial").glob("*.json"))
    ]
    payload = study.enrich_bias_pct(study.assemble_payload(shared, funnel, completed, [], 0.0))
    payload["meta"]["banner_extra"] = (
        "GARCH–Merton Q-map: Duan (1995) LRNVR + Pan (2002) μ_J*. "
        "Other models copied from empirical_study_1p5y_monthly_10000."
    )
    payload["meta"]["cache"] = str(OUT_CACHE)
    study.PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    df = v4.collect_pq_tables()
    if len(df):
        dest = OUT_CACHE / "pq_parameters.csv"
        df.to_csv(dest, index=False)
        print(f"wrote {dest}", flush=True)

    _print_summary(payload)
    print(f"\nStored at: {OUT_CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
