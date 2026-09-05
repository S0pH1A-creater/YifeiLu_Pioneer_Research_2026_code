#!/usr/bin/env python3
"""V4 return-based study on 1-minute data: 5-day lookback, hourly rolling.

Same return-based model set as the main 1.5-year monthly study:
MD-GBM, GBM, GARCH, Merton, GARCH–Merton
(Duan LRNVR for GARCH; Duan + Pan for GARCH–Merton; Pan for Merton).

Evaluation windows: same 12 Friday-before-expiry weeks as the V3/V4
7-day hourly study (five RTH sessions ending that Friday).

Stores under V4-Models_result/results/ only — does NOT write Results_In_Short.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_intraday_hourly_empirical_study as intra  # noqa: E402

STUDY_MODELS = ("MD-GBM", "GBM", "GARCH", "Merton", "GARCH–Merton")
STUDY_TICKERS = ("SPY", "AAPL", "MSFT")
CACHE_NAME = "empirical_study_5d_hourly_return_based"
LOOKBACK_LABEL = "5 days"
LOOKBACK_BARS = 5 * 390  # 5 RTH days × 390 one-minute bars

_CONFIGURED = False


def _apply_5d_settings() -> None:
    intra.WINDOW_LABEL = LOOKBACK_LABEL
    intra.LOOKBACK_BARS = LOOKBACK_BARS
    intra.LOOKBACK_PHRASE = "5-day"
    intra.ROLLING_MODE = "hourly"
    intra.TICKERS = STUDY_TICKERS
    intra.TABLE_MODELS = STUDY_MODELS
    intra.MODELS = STUDY_MODELS
    intra.CACHE = ROOT / "results" / CACHE_NAME
    intra.SHORT = ROOT / "results" / CACHE_NAME / "_no_results_in_short"
    intra.PDF_NAME = "DO_NOT_WRITE_TO_RESULTS_IN_SHORT.pdf"
    intra.NB_NAME = "DO_NOT_WRITE_TO_RESULTS_IN_SHORT.ipynb"
    intra.NOTEBOOK_IMPORT = "run_v4_5d_hourly_return_based_study"
    intra.ENGINE_SCRIPT = "run_v4_5d_hourly_return_based_study.py"
    intra.PAYLOAD_JSON = intra.CACHE / "payload.json"
    intra.CONTRACTS_JSON = intra.CACHE / "shared_contracts.json"
    intra.FILTER_JSON = intra.CACHE / "filter_funnel.json"
    intra.MODEL_STEM = {
        "GBM": "gbm",
        "MD-GBM": "modified_gbm_meanfix",
        "GARCH": "garch",
        "Merton": "merton",
        "GARCH–Merton": "garch_merton",
        "Heston": "heston",
        "Heston–Merton": "heston_merton",
        "Modified GBM": "modified_gbm",
    }


def apply_config() -> None:
    global _CONFIGURED
    intra.configure_study("7d")
    _apply_5d_settings()
    for kind in ("7d", "1d"):
        nb = dict(intra.MODEL_NOTEBOOKS[kind])
        stem = "7d_1min" if kind == "7d" else "1d_1min"
        nb["MD-GBM"] = (
            "md_gbm",
            f"{stem}_modified_gbm_meanfix.ipynb",
        )
        intra.MODEL_NOTEBOOKS[kind] = nb

    if not _CONFIGURED:
        _orig_configure = intra.configure_study

        def _configure_study(kind: str) -> None:
            _orig_configure(kind)
            _apply_5d_settings()

        intra.configure_study = _configure_study

        _orig_load = intra._load_model_ns

        def _load_model_ns(model: str) -> dict:
            g = _orig_load(model)
            g["WINDOW_OPTIONS"] = {
                "1 hour": 60,
                "1 day": 390,
                "5 days": LOOKBACK_BARS,
                "7 days": 7 * 390,
            }
            g["WINDOW_LABEL"] = LOOKBACK_LABEL
            # Notebooks bind load_calls_panel by name; keep them on the cached panels.
            import heston_option_calibration as hoc

            if "load_calls_panel" in g:
                g["load_calls_panel"] = hoc.load_calls_panel
            return g

        intra._load_model_ns = _load_model_ns
        _install_fast_option_panel_cache()
        _CONFIGURED = True

    intra.CACHE.mkdir(parents=True, exist_ok=True)
    if "Results_In_Short" in str(intra.SHORT):
        raise RuntimeError(f"Refusing Results_In_Short path: {intra.SHORT}")


def _install_fast_option_panel_cache() -> None:
    """Prefer /tmp copies of short_interval call panels (Desktop/iCloud reads time out)."""
    import heston_option_calibration as hoc
    import pandas as pd

    tmp = Path("/tmp/v4_5d_options_panels")
    if not tmp.exists():
        return

    # Pre-warm cache from /tmp so Merton / GARCH–Merton Pan fits do not re-read Desktop.
    for ticker in STUDY_TICKERS:
        path = tmp / f"{ticker}_calls_panel.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "trading_date" in df.columns:
            df["trading_date"] = pd.to_datetime(df["trading_date"])
        from option_filters import apply_estimation_filters

        df = apply_estimation_filters(df).reset_index(drop=True)
        for opt_dir in (
            tmp,
            Path(os_study.DATA_ROOT) / "options" / "processed",
            Path(os_study.DATA_ROOT) / "options" / "processed" / "short_interval",
        ):
            key = (str(Path(opt_dir).resolve()), ticker)
            hoc._CALLS_CACHE[key] = df
        print(f"  cached option panel {ticker}: n={len(df)} from {path}", flush=True)

    _orig = hoc.load_calls_panel

    def load_calls_panel(data_root, ticker, *, opt_dir=None):
        ticker = str(ticker).upper()
        # Always serve the prewarmed short_interval panel for this study.
        for opt in (
            tmp,
            Path(data_root) / "options" / "processed" / "short_interval",
            Path(data_root) / "options" / "processed",
        ):
            key = (str(Path(opt).resolve()), ticker)
            if key in hoc._CALLS_CACHE:
                return hoc._CALLS_CACHE[key]
        return _orig(data_root, ticker, opt_dir=opt_dir)

    hoc.load_calls_panel = load_calls_panel


def print_results(payload: dict) -> None:
    cells = payload["cells"]
    models = list(payload["meta"]["models"])
    tickers = list(payload["meta"]["tickers"])
    regimes = list(payload["meta"]["regimes"])
    print("\n" + "=" * 72)
    print("V4 5-day lookback · hourly rolling · return-based models")
    print("=" * 72)
    print(
        f"Models: {', '.join(models)}\n"
        f"Tickers: {', '.join(tickers)}\n"
        f"Windows: {len(regimes)} Friday-before-expiry weeks "
        f"({regimes[0]} … {regimes[-1]})\n"
        f"Lookback: 5 RTH days ({LOOKBACK_BARS} one-minute bars)\n"
        f"Rolling: hourly · n_paths={payload['meta']['n_paths']} · "
        f"Δt={payload['meta']['step_minutes']} min\n"
        f"Cells: {len(cells)} · elapsed {payload['meta'].get('elapsed_sec', 0)/60:.1f} min\n"
        f"Cache: {intra.CACHE}"
    )
    print("\n--- Overall ranking (mean RMSE% over all company × window cells) ---")
    for i, rec in enumerate(payload["summary_overall"], 1):
        print(
            f"  {i}. {rec['model']:14s}  mean RMSE%={rec['mean_rmse_pct']:7.2f}  "
            f"median={rec['median_rmse_pct']:7.2f}  "
            f"wins={rec['n_best']}/{rec['n_cells']}"
        )

    print("\n--- SPY asset · mean RMSE% by model (across 12 windows) ---")
    by_t = {(r["ticker"], r["model"]): r for r in payload["summary_by_ticker"]}
    for m in models:
        rec = by_t.get(("SPY", m))
        if rec:
            print(
                f"  {m:14s}  mean RMSE%={rec['mean_rmse_pct']:7.2f}  "
                f"wins={rec['n_best']}/12"
            )

    names = [t for t in tickers if t != "SPY"]
    print(f"\n--- Company asset ({', '.join(names)}) · mean of names, by model ---")
    for m in models:
        vals = [by_t[(t, m)]["mean_rmse_pct"] for t in names if (t, m) in by_t]
        wins = sum(by_t[(t, m)]["n_best"] for t in names if (t, m) in by_t)
        if vals:
            print(
                f"  {m:14s}  mean RMSE%={sum(vals)/len(vals):7.2f}  "
                f"name-wins={wins}"
            )

    print("\n--- Best model by window (equal-weight mean of SPY+AAPL+MSFT) ---")
    by_r = {
        (r["regime"], r["model"]): r["mean_rmse_pct"]
        for r in payload["summary_by_regime"]
    }
    for reg in regimes:
        pick = min(models, key=lambda m: by_r.get((reg, m), 1e9))
        title = intra.MONTH_TITLE.get(reg, reg)
        print(f"  {title:8s} ({reg}): {pick:14s}  RMSE%={by_r[(reg, pick)]:6.2f}")

    print("\n--- Sample windows: SPY + company-mean RMSE% ---")
    show = [regimes[0], regimes[3], regimes[6], regimes[11]]
    for reg in show:
        title = intra.MONTH_TITLE.get(reg, reg)
        spy_vals = {
            m: cells[f"SPY|{reg}|{m}"]["rmse_pct"]
            for m in models
            if f"SPY|{reg}|{m}" in cells
        }
        co_vals = {
            m: sum(cells[f"{t}|{reg}|{m}"]["rmse_pct"] for t in names) / len(names)
            for m in models
            if all(f"{t}|{reg}|{m}" in cells for t in names)
        }
        if spy_vals:
            best_s = min(spy_vals, key=spy_vals.get)
            print(f"  SPY asset · {title}: best {best_s} ({spy_vals[best_s]:.2f})")
            print(
                "    "
                + "  ".join(f"{m}={spy_vals[m]:.1f}" for m in models if m in spy_vals)
            )
        if co_vals:
            best_c = min(co_vals, key=co_vals.get)
            print(f"  Company asset · {title}: best {best_c} ({co_vals[best_c]:.2f})")
            print(
                "    "
                + "  ".join(f"{m}={co_vals[m]:.1f}" for m in models if m in co_vals)
            )
    print("=" * 72)


def write_readme(payload: dict) -> None:
    (intra.CACHE / "README.md").write_text(
        f"""# V4 5-day hourly return-based intraday study

- **Lookback:** 5 RTH days ({LOOKBACK_BARS} one-minute bars)
- **Rolling:** hourly (09:59–15:59 stamps)
- **Models:** {', '.join(STUDY_MODELS)}
- **Tickers:** {', '.join(STUDY_TICKERS)}
- **Windows:** 12 Friday-before-expiry evaluation weeks (Oct 2022–Sep 2023)
- **Eval clock:** five RTH sessions ending that Friday; LSM Δt = 5 minutes; n_paths = {payload['meta']['n_paths']}
- **P→Q:** GBM μ→r_f; GARCH Duan LRNVR; Merton Pan μ_J*; GARCH–Merton Duan+Pan; MD-GBM additive Q
- **Not written to** `Results_In_Short/`

Payload: `payload.json`
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    recompute = "--recompute" in argv
    only = [a for a in argv if not a.startswith("--")]
    apply_config()
    meanfix_nb = (
        ROOT / "md_gbm" / "7d_1min_modified_gbm_meanfix.ipynb"
    )
    if not meanfix_nb.exists():
        raise FileNotFoundError(meanfix_nb)

    n_needed = len(STUDY_TICKERS) * len(intra.REGIME_ORDER) * len(STUDY_MODELS)
    payload = intra.run_or_load(recompute=recompute, only=only or None)
    payload.setdefault("meta", {})
    payload["meta"].update(
        {
            "lookback_phrase": "5-day",
            "lookback_bars": LOOKBACK_BARS,
            "window_label": LOOKBACK_LABEL,
            "rolling": "hourly",
            "models": list(STUDY_MODELS),
            "tickers": list(STUDY_TICKERS),
            "study_id": CACHE_NAME,
            "results_in_short": False,
        }
    )
    intra.PAYLOAD_JSON.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    write_readme(payload)
    print_results(payload)
    n = len(payload.get("cells", {}))
    if n < n_needed or payload["meta"].get("failures"):
        print(
            f"Incomplete: {n}/{n_needed} cells, "
            f"failures={payload['meta'].get('failures')}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
