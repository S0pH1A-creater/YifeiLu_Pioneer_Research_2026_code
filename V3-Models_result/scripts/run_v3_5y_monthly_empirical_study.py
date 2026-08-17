#!/usr/bin/env python3
"""V3 empirical study: models × companies × four regimes.

Default underlyings are SPY, AAPL, MSFT. The 1.5-year 10k reports add AMZN.
Lookback and rolling are configured by the wrapper. Shared contract sample per
(ticker × regime). Percentage RMSE is the ranking metric.

The notebook in Results_In_Short is the user-facing source of truth;
this module is the computational engine it imports. The PDF is written
from the same payload dict so every number matches.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
from american_lsm import load_calls, pct_rmse, sample_calls  # noqa: E402


def pct_bias(model, market, *, min_price: float = 1e-8) -> float:
    """Percentage bias: 100 × mean((ŷ − y)/y). Companion to percentage RMSE."""
    yhat = np.asarray(model, dtype=float)
    y = np.asarray(market, dtype=float)
    ok = np.isfinite(yhat) & np.isfinite(y) & (np.abs(y) > float(min_price))
    if not np.any(ok):
        return float("nan")
    return float(100.0 * np.mean((yhat[ok] - y[ok]) / y[ok]))
from option_filters import apply_estimation_filters, moneyness_bucket, spot_over_strike  # noqa: E402

os_study.WINDOW_LABEL = "5 years"
os_study.ROLLING_MODES = ("monthly",)
os_study.N_PATHS = 2000
os_study.SEED = 42

WINDOW_LABEL = "5 years"
LOOKBACK_OFFSET = pd.DateOffset(years=5)
LOOKBACK_PHRASE = "5-year"
ROLLING_MODE = "monthly"
N_PATHS = 2000
SEED = 42
TICKERS = ("SPY", "AAPL", "MSFT")
MODELS = ("GBM", "GARCH", "Heston", "Merton", "GARCH–Merton", "Heston–Merton")
# Table row order requested in the experimental design
TABLE_MODELS = ("GBM", "GARCH", "Heston", "Merton", "GARCH–Merton", "Heston–Merton")
REGIME_ORDER = ["2008-2009", "2013-2014", "2018-2019", "2019-2020"]
REGIME_META = {
    "2008-2009": {
        "title": "Crisis",
        "window": "2008-08-01 → 2009-07-31",
        "note": "Global financial crisis / high realized volatility",
    },
    "2013-2014": {
        "title": "Normal",
        "window": "2014-01-01 → 2014-12-31",
        "note": "Low-volatility expansion year",
    },
    "2018-2019": {
        "title": "Late-cycle",
        "window": "2018-10-01 → 2019-09-30",
        "note": "Vol spike (Q4 2018) then recovery",
    },
    "2019-2020": {
        "title": "COVID",
        "window": "2019-09-01 → 2020-08-31",
        "note": "COVID crash and rebound (shares Sep 2019 with late-cycle file)",
    },
}
MODEL_FOLDERS = {
    "GBM": ("gbm notebook", "20*_gbm.ipynb"),
    "Modified GBM": ("modified gbm notebook", "20*_modified_gbm.ipynb"),
    "GARCH": ("garch notebook", "20*_garch.ipynb"),
    "Heston": ("heston notebook", "20*_heston.ipynb"),
    "Merton": ("merton notebook", "20*_merton.ipynb"),
    "GARCH–Merton": ("garch merton notebook", "20*_garch_merton.ipynb"),
    "Heston–Merton": ("heston merton notebook", "20*_heston_merton.ipynb"),
}

CACHE = ROOT / "results" / "empirical_study_5y_monthly"
SHORT = REPO / "Results_In_Short" / "V3 5-year monthly empirical study"
PDF_NAME = "V3_5y_monthly_empirical_study.pdf"
NB_NAME = "V3_5y_monthly_empirical_study.ipynb"
NOTEBOOK_IMPORT = "run_v3_5y_monthly_empirical_study"
ENGINE_SCRIPT = "run_v3_5y_monthly_empirical_study.py"
PAYLOAD_JSON = CACHE / "payload.json"
CONTRACTS_JSON = CACHE / "shared_contracts.json"
FILTER_JSON = CACHE / "filter_funnel.json"


def configure_lookback(
    *,
    window_label: str,
    lookback_offset: pd.DateOffset,
    lookback_phrase: str,
    cache_name: str,
    short_name: str,
    pdf_name: str,
    nb_name: str,
    notebook_import: str,
    engine_script: str,
    n_paths: int | None = None,
    tickers: tuple[str, ...] | None = None,
    rolling_mode: str | None = None,
) -> None:
    """Point this module at another lookback without rewriting the 5-year cache."""
    global WINDOW_LABEL, LOOKBACK_OFFSET, LOOKBACK_PHRASE, N_PATHS, TICKERS, ROLLING_MODE
    global CACHE, SHORT, PDF_NAME, NB_NAME, NOTEBOOK_IMPORT, ENGINE_SCRIPT
    global PAYLOAD_JSON, CONTRACTS_JSON, FILTER_JSON
    WINDOW_LABEL = window_label
    LOOKBACK_OFFSET = lookback_offset
    LOOKBACK_PHRASE = lookback_phrase
    os_study.WINDOW_LABEL = window_label
    if n_paths is not None:
        N_PATHS = int(n_paths)
        os_study.N_PATHS = int(n_paths)
    if tickers is not None:
        TICKERS = tuple(tickers)
    if rolling_mode is not None:
        ROLLING_MODE = rolling_mode
        os_study.ROLLING_MODES = (rolling_mode,)
    CACHE = ROOT / "results" / cache_name
    SHORT = REPO / "Results_In_Short" / short_name
    PDF_NAME = pdf_name
    NB_NAME = nb_name
    NOTEBOOK_IMPORT = notebook_import
    ENGINE_SCRIPT = engine_script
    PAYLOAD_JSON = CACHE / "payload.json"
    CONTRACTS_JSON = CACHE / "shared_contracts.json"
    FILTER_JSON = CACHE / "filter_funnel.json"


_orig_load_ns = os_study._load_ns


def _load_ns(nb_path):
    g = _orig_load_ns(nb_path)
    wo = g.get("WINDOW_OPTIONS")
    if isinstance(wo, dict) and WINDOW_LABEL not in wo:
        wo[WINDOW_LABEL] = LOOKBACK_OFFSET
    g["WINDOW_LABEL"] = WINDOW_LABEL
    prices = g.get("prices")
    if prices is not None:
        tickers = [t for t in TICKERS if t in prices.columns]
        if tickers:
            g["TICKERS"] = tickers
            g["period_prices"] = prices.loc[g["PERIOD_START"] : g["PERIOD_END"], tickers].copy()
            g["log_returns_all"] = np.log(prices[tickers]).diff()
    return g


os_study._load_ns = _load_ns

NAVY = "#1F3A5F"
MUTED = "#444444"
BEST_BG = "#E4EED8"
ALT_BG = "#F4F7FB"
MODEL_COLORS = {
    "GBM": "#4C72B0",
    "Modified GBM": "#C73E7B",
    "GARCH": "#C44E52",
    "Heston": "#DD8452",
    "Merton": "#55A868",
    "GARCH–Merton": "#8172B3",
    "Heston–Merton": "#937860",
}
MODEL_COVER_NAMES = {
    "GBM": "GBM",
    "Modified GBM": "Modified GBM (Markov direction)",
    "GARCH": "GARCH(1,1)",
    "Heston": "Heston (no jumps)",
    "Merton": "Merton jump-diffusion",
    "GARCH–Merton": "GARCH–Merton",
    "Heston–Merton": "Heston–Merton (Bates)",
}


def _n_cells() -> int:
    return len(TICKERS) * len(REGIME_ORDER)


def _underlyings_phrase() -> str:
    n = len(TICKERS)
    if n == 1:
        return TICKERS[0]
    if n == 3:
        return "three underlyings"
    if n == 4:
        return "four underlyings"
    return f"{n} underlyings"


def _ticker_grid_shape(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, max(n, 1)
    if n == 4:
        return 2, 2
    return (n + 2) // 3, 3


def _n_models() -> int:
    return len(TABLE_MODELS)


def _n_expected_cells() -> int:
    return len(TICKERS) * len(REGIME_ORDER) * len(TABLE_MODELS)


def _tickers_joined(sep: str = " / ") -> str:
    return sep.join(TICKERS)


def _tickers_listed() -> str:
    names = list(TICKERS)
    if names and names[0] == "SPY" and len(names) > 1:
        return "SPY (primary), " + ", ".join(names[1:])
    return ", ".join(names)


def _companies_phrase() -> str:
    n = len(TICKERS)
    words = {1: "one company", 2: "two companies", 3: "three companies", 4: "four companies"}
    return words.get(n, f"{n} companies")


def _subplots_tickers(figsize):
    n = len(TICKERS)
    nrows, ncols = _ticker_grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    axes_list = list(np.atleast_1d(axes).ravel())
    for ax in axes_list[n:]:
        ax.set_visible(False)
        ax.axis("off")
    return fig, axes_list[:n]


def _models_phrase(*, cap: bool = True) -> str:
    words = {
        1: "one model",
        2: "two models",
        3: "three models",
        4: "four models",
        5: "five models",
        6: "six models",
        7: "seven models",
    }
    phrase = words.get(_n_models(), f"{_n_models()} models")
    return phrase[:1].upper() + phrase[1:] if cap else phrase


def _models_listed() -> str:
    return ", ".join(MODEL_COVER_NAMES.get(m, m) for m in TABLE_MODELS)


def _rolling_phrase() -> str:
    if ROLLING_MODE == "none":
        return f"{LOOKBACK_PHRASE} fixed calibration"
    if ROLLING_MODE == "monthly":
        return f"{LOOKBACK_PHRASE} monthly calibration"
    return f"{LOOKBACK_PHRASE} {ROLLING_MODE} calibration"


def _rolling_detail() -> str:
    if ROLLING_MODE == "none":
        return (
            f"Calibration window: {LOOKBACK_PHRASE} ending at the first session of each 1-year evaluation window. "
            "Rolling: none (one estimate, held for the whole window). Parameters estimated at t0 are used only after t0."
        )
        return (
            f"Calibration window: {LOOKBACK_PHRASE} ending at each monthly update. Rolling: monthly (period start plus month-ends). "
            "Parameters estimated at t are used only after t (no look-ahead)."
        )


def _no_lookahead_note() -> str:
    if ROLLING_MODE == "none":
        return (
            "At the first session t0 of each 1-year evaluation window, parameters use only returns and listed quotes with "
            "`trading_date ≤ t0`. Those parameters are held for every Monday contract in the window. LSM prices each contract "
            "with that single calibration row. Option quotes are as-of the grid timestamp (same session, else the prior session)."
        )
    return (
        "At monthly update t, parameters use only returns and listed quotes with `trading_date ≤ t`. LSM prices a Monday "
        "contract with the latest calibration row dated on or before that Monday. Option quotes are as-of the grid timestamp "
        "(same session, else the prior session)."
    )


def _banner_line(payload: dict | None = None) -> str:
    extra = ""
    if payload:
        extra = str(payload.get("meta", {}).get("banner_extra") or "").strip()
    base = (
        f"{_models_phrase()}  ·  {_underlyings_phrase()}  ·  four volatility regimes  ·  "
        f"{_rolling_phrase()}"
    )
    return f"{base}  ·  {extra}" if extra else base


@contextmanager
def use_models(models):
    """Temporarily restrict TABLE_MODELS for grouped companion reports."""
    global TABLE_MODELS, MODELS
    old_table, old_models = TABLE_MODELS, MODELS
    TABLE_MODELS = tuple(models)
    MODELS = tuple(models)
    try:
        yield
    finally:
        TABLE_MODELS = old_table
        MODELS = old_models


def _jobs() -> list[tuple[str, Path]]:
    jobs = []
    for model in MODELS:
        folder, glob_pat = MODEL_FOLDERS[model]
        for nb in sorted((ROOT / folder).glob(glob_pat)):
            if "advanced" in nb.name:
                continue
            jobs.append((model, nb))
    return jobs


def _fingerprint(df: pd.DataFrame) -> list[dict]:
    cols = [c for c in ("trading_date", "expiration", "K", "S_t", "dte", "r", "option_price") if c in df.columns]
    out = []
    for rec in df[cols].to_dict(orient="records"):
        row = {}
        for k, v in rec.items():
            if hasattr(v, "isoformat"):
                row[k] = pd.Timestamp(v).isoformat()
            elif isinstance(v, (np.floating, float)):
                row[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                row[k] = int(v)
            else:
                row[k] = v
        out.append(row)
    return out


def _contracts_from_records(records: list[dict], ticker: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])
    df["underlying"] = ticker
    return df


def sample_shared_contracts() -> dict:
    """One contract list per (regime, ticker), reused by every model.

    Existing frozen samples are kept. Missing tickers (e.g. AMZN added later)
    are sampled and merged in without resampling names already on disk.
    """
    payload: dict = json.loads(CONTRACTS_JSON.read_text(encoding="utf-8")) if CONTRACTS_JSON.exists() else {}
    missing = [
        ticker
        for ticker in TICKERS
        if not payload
        or any(ticker not in block.get("tickers", {}) for block in payload.values())
    ]
    if payload and not missing:
        return payload

    os_study._install_notebook_stubs()
    panels: dict | None = None

    def _panels() -> dict:
        nonlocal panels
        if panels is None:
            print("  loading filtered call panels …", flush=True)
            panels = {t: load_calls(os_study.DATA_ROOT, t) for t in TICKERS}
        return panels

    for nb in sorted((ROOT / "gbm notebook").glob("20*_gbm.ipynb")):
        g = os_study._load_ns(nb)
        regime = os_study._regime_from_name(nb)
        payload.setdefault(
            regime,
            {
                "period_start": pd.Timestamp(g["PERIOD_START"]).date().isoformat(),
                "period_end": pd.Timestamp(g["PERIOD_END"]).date().isoformat(),
                "tickers": {},
            },
        )
        payload[regime].setdefault("tickers", {})
        for ticker in TICKERS:
            if ticker in payload[regime]["tickers"]:
                continue
            contracts = sample_calls(
                _panels()[ticker],
                g["PERIOD_START"],
                g["PERIOD_END"],
            )
            if contracts is None or len(contracts) == 0:
                raise RuntimeError(f"No shared contracts for {ticker} {regime}")
            contracts = contracts.copy()
            contracts["underlying"] = ticker
            payload[regime]["tickers"][ticker] = {
                "n": int(len(contracts)),
                "records": _fingerprint(contracts),
            }
            print(
                f"  sample {regime} {ticker}: n={len(contracts)} "
                f"{pd.Timestamp(contracts['trading_date'].min()).date()} → "
                f"{pd.Timestamp(contracts['trading_date'].max()).date()}",
                flush=True,
            )
    CONTRACTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTRACTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def filter_funnel() -> dict:
    """Observation counts after each V3 estimation filter, by ticker × regime."""
    if FILTER_JSON.exists():
        out = json.loads(FILTER_JSON.read_text(encoding="utf-8"))
        if all(t in out.get("tickers", {}) for t in TICKERS):
            return out

    contracts = sample_shared_contracts()
    out: dict = {"tickers": {}, "note": "Same filters for every model. No model-specific data selection."}
    for ticker in TICKERS:
        panel = pd.read_csv(
            os_study.DATA_ROOT / "options" / "processed" / f"{ticker}_calls_panel.csv",
            parse_dates=["trading_date", "expiration"],
        )
        t_block: dict = {}
        for regime in REGIME_ORDER:
            start = pd.Timestamp(contracts[regime]["period_start"])
            end = pd.Timestamp(contracts[regime]["period_end"])
            # Lookback-aware estimation sample: 5 calendar years before period end,
            # plus the evaluation window itself (quotes used at monthly updates).
            look_start = end - LOOKBACK_OFFSET
            sub = panel.loc[
                (panel["trading_date"] >= look_start) & (panel["trading_date"] <= end)
            ].copy()
            filtered, log = apply_estimation_filters(sub, audit=True)
            steps = [{"step": r["step"], "n": int(r["n"])} for r in log]
            raw_sk = spot_over_strike(sub) if len(sub) else pd.Series(dtype=float)
            keep_sk = spot_over_strike(filtered) if len(filtered) else pd.Series(dtype=float)
            t_block[regime] = {
                "lookback_start": look_start.date().isoformat(),
                "period_end": end.date().isoformat(),
                "steps": steps,
                "n_eval_contracts": int(contracts[regime]["tickers"][ticker]["n"]),
                "moneyness_before": moneyness_bucket(raw_sk).value_counts(dropna=False).astype(int).to_dict(),
                "moneyness_after": moneyness_bucket(keep_sk).value_counts(dropna=False).astype(int).to_dict(),
            }
        out["tickers"][ticker] = t_block
    FILTER_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _metrics_from_df(df: pd.DataFrame) -> dict:
    return {
        "rmse_pct": float(pct_rmse(df["model_price"], df["market"])),
        "mae": float(np.mean(np.abs(df["error"]))),
        "bias": float(np.mean(df["error"])),
        "bias_pct": float(pct_bias(df["model_price"], df["market"])),
        "early": float(df["early_ex_frac"].mean()),
        "n": int(len(df)),
    }


def enrich_bias_pct(payload: dict) -> dict:
    """Fill Bias% from saved contract CSVs so a frozen run still gets the new column."""
    for rec in payload.get("cells", {}).values():
        rel = rec.get("contracts_csv")
        if not rel:
            continue
        csv = CACHE / rel
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        rec["bias_pct"] = float(pct_bias(df["model_price"], df["market"]))
    for tab in payload.get("tables", {}).values():
        for row in tab.get("rows", []):
            key = f"{tab['ticker']}|{tab['regime']}|{row['model']}"
            if key in payload["cells"] and "bias_pct" in payload["cells"][key]:
                row["bias_pct"] = payload["cells"][key]["bias_pct"]
    payload.setdefault("meta", {})["bias_pct_definition"] = "100 × mean((model − market) / market)"
    return payload


def _run_one(model: str, nb_path: Path, shared: dict, tickers: tuple[str, ...] | list[str] | None = None) -> dict:
    os_study._install_notebook_stubs()
    g = os_study._load_ns(nb_path)
    regime = os_study._regime_from_name(nb_path)
    names = tuple(tickers) if tickers is not None else TICKERS
    out = {"model": model, "regime": regime, "stem": nb_path.stem, "tickers": {}}
    for ticker in names:
        recs = shared[regime]["tickers"][ticker]["records"]
        contracts = _contracts_from_records(recs, ticker)
        t0 = time.time()
        result = os_study._run_mode(g, ROLLING_MODE, contracts, ticker)
        mets = _metrics_from_df(result["df"])
        mets.update(
            {
                "n_updates": int(result["n_updates"]),
                "t_cal": float(result["t_cal"]),
                "t_lsm": float(result["t_lsm"]),
                "elapsed": float(time.time() - t0),
            }
        )
        fig_dir = CACHE / "contracts" / regime / ticker
        fig_dir.mkdir(parents=True, exist_ok=True)
        csv_path = fig_dir / f"{nb_path.stem}.csv"
        result["df"].to_csv(csv_path, index=False)
        mets["contracts_csv"] = str(csv_path.relative_to(CACHE))
        out["tickers"][ticker] = mets
        print(
            f"    {model:16s} {regime} {ticker}: RMSE%={mets['rmse_pct']:.2f} "
            f"MAE={mets['mae']:.4f} n={mets['n']} updates={mets['n_updates']} "
            f"cal={mets['t_cal']:.1f}s lsm={mets['t_lsm']:.1f}s",
            flush=True,
        )
    return out


def _partial_path(model: str, regime: str) -> Path:
    safe = model.replace("–", "-").replace(" ", "_")
    return CACHE / "partial" / f"{regime}_{safe}.json"


def run_study(only: list[str] | None = None) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "partial").mkdir(parents=True, exist_ok=True)
    print("Sampling shared contracts …", flush=True)
    shared = sample_shared_contracts()
    print("Filter funnel …", flush=True)
    funnel = filter_funnel()

    jobs = _jobs()
    if only:
        tokens = [t.lower().replace("–", "-").replace("_", "-") for t in only]
        model_alias = {
            "gbm": "GBM",
            "modified-gbm": "Modified GBM",
            "garch": "GARCH",
            "heston": "Heston",
            "merton": "Merton",
            "garch-merton": "GARCH–Merton",
            "heston-merton": "Heston–Merton",
        }
        keep = []
        for m, p in jobs:
            regime = os_study._regime_from_name(p)
            hit = False
            for tok in tokens:
                if tok in model_alias:
                    if m == model_alias[tok]:
                        hit = True
                elif tok == regime.lower() or tok == p.stem.lower():
                    hit = True
            if hit:
                keep.append((m, p))
        jobs = keep

    completed: list[dict] = []
    failures: list[str] = []
    t0 = time.time()
    for model, nb in jobs:
        regime = os_study._regime_from_name(nb)
        part = _partial_path(model, regime)
        if part.exists():
            row = json.loads(part.read_text(encoding="utf-8"))
            missing = [t for t in TICKERS if t not in row.get("tickers", {})]
            if missing:
                print(f"  fill {model} {regime}: {', '.join(missing)}", flush=True)
                try:
                    extra = _run_one(model, nb, shared, tickers=missing)
                    row.setdefault("tickers", {}).update(extra["tickers"])
                    part.write_text(json.dumps(row, indent=2), encoding="utf-8")
                except Exception:
                    failures.append(f"{model}/{nb.name}")
                    print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)
                    continue
            else:
                print(f"  resume {model} {regime}", flush=True)
            completed.append(row)
            continue
        print(f"\n=== {model} | {regime} ===", flush=True)
        try:
            row = _run_one(model, nb, shared)
            part.write_text(json.dumps(row, indent=2), encoding="utf-8")
            completed.append(row)
        except Exception:
            failures.append(f"{model}/{nb.name}")
            print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)

    payload = assemble_payload(shared, funnel, completed, failures, time.time() - t0)
    PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {PAYLOAD_JSON} in {(time.time()-t0)/60:.1f} min", flush=True)
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
    return payload


def assemble_payload(shared, funnel, completed, failures, elapsed) -> dict:
    cells: dict[str, dict] = {}
    for row in completed:
        for ticker, mets in row["tickers"].items():
            cells[f"{ticker}|{row['regime']}|{row['model']}"] = {
                "ticker": ticker,
                "regime": row["regime"],
                "model": row["model"],
                **mets,
            }
    tables = {}
    for ticker in TICKERS:
        for regime in REGIME_ORDER:
            rows = []
            for model in TABLE_MODELS:
                key = f"{ticker}|{regime}|{model}"
                if key not in cells:
                    continue
                rows.append(cells[key])
            if not rows:
                continue
            best = min(rows, key=lambda r: r["rmse_pct"])["model"]
            n_set = {int(r["n"]) for r in rows}
            tables[f"{ticker}|{regime}"] = {
                "ticker": ticker,
                "regime": regime,
                "best_model": best,
                "n_contracts": int(rows[0]["n"]) if rows else 0,
                "n_contracts_identical": len(n_set) == 1,
                "rows": rows,
            }

    # summaries
    by_regime = []
    for regime in REGIME_ORDER:
        for model in TABLE_MODELS:
            vals = [cells[f"{t}|{regime}|{model}"]["rmse_pct"] for t in TICKERS if f"{t}|{regime}|{model}" in cells]
            if not vals:
                continue
            by_regime.append(
                {
                    "regime": regime,
                    "model": model,
                    "mean_rmse_pct": float(np.mean(vals)),
                    "median_rmse_pct": float(np.median(vals)),
                    "n_tickers": len(vals),
                }
            )
    by_ticker = []
    for ticker in TICKERS:
        for model in TABLE_MODELS:
            vals = [cells[f"{ticker}|{r}|{model}"]["rmse_pct"] for r in REGIME_ORDER if f"{ticker}|{r}|{model}" in cells]
            if not vals:
                continue
            by_ticker.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "mean_rmse_pct": float(np.mean(vals)),
                    "median_rmse_pct": float(np.median(vals)),
                    "n_regimes": len(vals),
                    "n_best": sum(
                        1
                        for r in REGIME_ORDER
                        if tables.get(f"{ticker}|{r}", {}).get("best_model") == model
                    ),
                }
            )
    overall = []
    for model in TABLE_MODELS:
        vals = [
            cells[k]["rmse_pct"]
            for k, v in cells.items()
            if v["model"] == model
        ]
        if not vals:
            continue
        overall.append(
            {
                "model": model,
                "mean_rmse_pct": float(np.mean(vals)),
                "median_rmse_pct": float(np.median(vals)),
                "n_cells": len(vals),
                "n_best": sum(1 for t in tables.values() if t["best_model"] == model),
            }
        )
    overall = sorted(overall, key=lambda r: r["mean_rmse_pct"])
    ranking_grid = []
    for ticker in TICKERS:
        for regime in REGIME_ORDER:
            tab = tables.get(f"{ticker}|{regime}")
            if not tab:
                continue
            order = sorted(tab["rows"], key=lambda r: r["rmse_pct"])
            ranking_grid.append(
                {
                    "ticker": ticker,
                    "regime": regime,
                    "rank1": order[0]["model"],
                    "rank2": order[1]["model"] if len(order) > 1 else "",
                    "best_rmse_pct": order[0]["rmse_pct"],
                    "ranks": {r["model"]: i + 1 for i, r in enumerate(order)},
                }
            )

    return {
        "meta": {
            "window_label": WINDOW_LABEL,
            "rolling": ROLLING_MODE,
            "n_paths": N_PATHS,
            "seed": SEED,
            "models": list(TABLE_MODELS),
            "tickers": list(TICKERS),
            "regimes": REGIME_ORDER,
            "regime_meta": REGIME_META,
            "primary_metric": "percentage RMSE",
            "bias_definition": "model − market (dollars)",
            "bias_pct_definition": "100 × mean((model − market) / market)",
            "elapsed_sec": float(elapsed),
            "failures": failures,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "shared_contracts": {
            r: {
                "period_start": shared[r]["period_start"],
                "period_end": shared[r]["period_end"],
                "n": {t: shared[r]["tickers"][t]["n"] for t in TICKERS},
            }
            for r in REGIME_ORDER
            if r in shared
        },
        "filter_funnel": funnel,
        "cells": cells,
        "tables": tables,
        "summary_by_regime": by_regime,
        "summary_by_ticker": by_ticker,
        "summary_overall": overall,
        "ranking_grid": ranking_grid,
    }


def slice_payload_models(payload: dict, models, extra_meta: dict | None = None) -> dict:
    """Rebuild tables and rankings from existing cells. No new LSM."""
    models = tuple(models)
    cells = {k: dict(v) for k, v in payload.get("cells", {}).items() if v.get("model") in models}
    tables: dict[str, dict] = {}
    for ticker in TICKERS:
        for regime in REGIME_ORDER:
            rows = [cells[f"{ticker}|{regime}|{model}"] for model in models if f"{ticker}|{regime}|{model}" in cells]
            if not rows:
                continue
            tables[f"{ticker}|{regime}"] = {
                "ticker": ticker,
                "regime": regime,
                "best_model": min(rows, key=lambda r: r["rmse_pct"])["model"],
                "n_contracts": int(rows[0]["n"]) if rows else 0,
                "n_contracts_identical": len({int(r["n"]) for r in rows}) == 1,
                "rows": rows,
            }
    by_regime = []
    for regime in REGIME_ORDER:
        for model in models:
            vals = [cells[f"{t}|{regime}|{model}"]["rmse_pct"] for t in TICKERS if f"{t}|{regime}|{model}" in cells]
            if not vals:
                continue
            by_regime.append(
                {
                    "regime": regime,
                    "model": model,
                    "mean_rmse_pct": float(np.mean(vals)),
                    "median_rmse_pct": float(np.median(vals)),
                    "n_tickers": len(vals),
                }
            )
    by_ticker = []
    for ticker in TICKERS:
        for model in models:
            vals = [cells[f"{ticker}|{r}|{model}"]["rmse_pct"] for r in REGIME_ORDER if f"{ticker}|{r}|{model}" in cells]
            if not vals:
                continue
            by_ticker.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "mean_rmse_pct": float(np.mean(vals)),
                    "median_rmse_pct": float(np.median(vals)),
                    "n_regimes": len(vals),
                    "n_best": sum(
                        1
                        for r in REGIME_ORDER
                        if tables.get(f"{ticker}|{r}", {}).get("best_model") == model
                    ),
                }
            )
    overall = []
    for model in models:
        vals = [cells[k]["rmse_pct"] for k, v in cells.items() if v["model"] == model]
        if not vals:
            continue
        overall.append(
            {
                "model": model,
                "mean_rmse_pct": float(np.mean(vals)),
                "median_rmse_pct": float(np.median(vals)),
                "n_cells": len(vals),
                "n_best": sum(1 for t in tables.values() if t["best_model"] == model),
            }
        )
    overall = sorted(overall, key=lambda r: r["mean_rmse_pct"])
    ranking_grid = []
    for ticker in TICKERS:
        for regime in REGIME_ORDER:
            tab = tables.get(f"{ticker}|{regime}")
            if not tab:
                continue
            order = sorted(tab["rows"], key=lambda r: r["rmse_pct"])
            ranking_grid.append(
                {
                    "ticker": ticker,
                    "regime": regime,
                    "rank1": order[0]["model"],
                    "rank2": order[1]["model"] if len(order) > 1 else "",
                    "best_rmse_pct": order[0]["rmse_pct"],
                    "ranks": {r["model"]: i + 1 for i, r in enumerate(order)},
                }
            )
    meta = dict(payload.get("meta", {}))
    meta["models"] = list(models)
    meta["parent_models"] = list(payload.get("meta", {}).get("models", []))
    if extra_meta:
        meta.update(extra_meta)
    return {
        "meta": meta,
        "shared_contracts": payload.get("shared_contracts", {}),
        "filter_funnel": payload.get("filter_funnel", {}),
        "cells": cells,
        "tables": tables,
        "summary_by_regime": by_regime,
        "summary_by_ticker": by_ticker,
        "summary_overall": overall,
        "ranking_grid": ranking_grid,
    }


def load_payload() -> dict:
    if not PAYLOAD_JSON.exists():
        raise FileNotFoundError(PAYLOAD_JSON)
    return json.loads(PAYLOAD_JSON.read_text(encoding="utf-8"))


def run_or_load(*, recompute: bool = False, only: list[str] | None = None) -> dict:
    if recompute or not PAYLOAD_JSON.exists():
        return run_study(only=only)
    payload = load_payload()
    needed = {f"{t}|{r}|{m}" for t in TICKERS for r in REGIME_ORDER for m in TABLE_MODELS}
    if not needed.issubset(payload.get("cells", {})):
        return run_study(only=only)
    return payload


# ---------------------------------------------------------------------------
# Tables / figures / PDF
# ---------------------------------------------------------------------------

def _style_table(tbl, best_row: int | None) -> None:
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.6)
        cell.set_height(0.12)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif best_row is not None and r == best_row:
            cell.set_facecolor(BEST_BG)
            cell.set_text_props(weight="bold", color="#1A1A1A")
        elif r % 2 == 0:
            cell.set_facecolor(ALT_BG)
        else:
            cell.set_facecolor("white")


def _metric_rows(tab: dict) -> tuple[list[list[str]], list[str], int]:
    header = ["Model", "RMSE%", "MAE", "Bias $", "Bias%", "Early-ex. frac.", "Mark"]
    rows = []
    rmses = []
    for rec in tab["rows"]:
        rmses.append(rec["rmse_pct"])
        bp = rec.get("bias_pct", float("nan"))
        bp_s = "—" if not np.isfinite(bp) else f"{bp:+.2f}"
        rows.append(
            [
                rec["model"],
                f"{rec['rmse_pct']:.2f}",
                f"{rec['mae']:.4f}",
                f"{rec['bias']:+.4f}",
                bp_s,
                f"{100.0 * rec['early']:.1f}%",
                "",
            ]
        )
    order = np.argsort(np.asarray(rmses, dtype=float), kind="mergesort")
    ranks = np.empty(len(rmses), dtype=int)
    ranks[order] = np.arange(1, len(rmses) + 1)
    for i, row in enumerate(rows):
        row[-1] = str(int(ranks[i]))
    best_i = int(np.argmin(rmses))
    return rows, header, best_i + 1


def _equity_tickers() -> tuple[str, ...]:
    """Single-name companies shown on the post-SPY summary page."""
    return tuple(t for t in TICKERS if t != "SPY")


def _wants_company_summary(payload: dict) -> bool:
    return bool(payload.get("meta", {}).get("grouped_report")) and len(_equity_tickers()) >= 2


def _mean_or_nan(vals) -> float:
    x = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def _average_company_table(payload: dict, regime: str, names: tuple[str, ...]) -> dict:
    """One regime table: arithmetic mean of each metric across `names`."""
    rows = []
    for model in TABLE_MODELS:
        recs = []
        for ticker in names:
            tab = payload.get("tables", {}).get(f"{ticker}|{regime}")
            if not tab:
                continue
            rec = next((r for r in tab["rows"] if r["model"] == model), None)
            if rec is not None:
                recs.append(rec)
        if not recs:
            continue
        rows.append(
            {
                "model": model,
                "rmse_pct": _mean_or_nan(r["rmse_pct"] for r in recs),
                "mae": _mean_or_nan(r["mae"] for r in recs),
                "bias": _mean_or_nan(r["bias"] for r in recs),
                "bias_pct": _mean_or_nan(r.get("bias_pct") for r in recs),
                "early": _mean_or_nan(r["early"] for r in recs),
                "n": _mean_or_nan(r["n"] for r in recs),
            }
        )
    if not rows:
        return {
            "ticker": "Companies",
            "regime": regime,
            "best_model": "",
            "n_contracts": 0,
            "n_contracts_identical": True,
            "rows": [],
            "summary_of": list(names),
        }
    return {
        "ticker": "Companies",
        "regime": regime,
        "best_model": min(rows, key=lambda r: r["rmse_pct"])["model"],
        "n_contracts": int(round(_mean_or_nan(r["n"] for r in rows))),
        "n_contracts_identical": True,
        "rows": rows,
        "summary_of": list(names),
    }


def _cover(pdf: PdfPages, payload: dict) -> None:
    import textwrap

    def _instruction_page(title: str, subtitle: str, sections: list[tuple[str, list[str]]], footer: str | None) -> None:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.955, title, fontsize=18, weight="bold", color=NAVY, va="top")
        fig.text(0.08, 0.912, subtitle, fontsize=9.5, color=MUTED, va="top")
        fig.add_artist(plt.Line2D([0.08, 0.92], [0.892, 0.892], transform=fig.transFigure, color=NAVY, lw=1.4))
        y = 0.855

        def section(heading: str, bullets: list[str], y0: float) -> float:
            fig.text(0.08, y0, heading, fontsize=12, weight="bold", color=NAVY, va="top")
            y = y0 - 0.036
            for b in bullets:
                wrapped = textwrap.wrap(b, width=88) or [""]
                fig.text(0.10, y, "•", fontsize=10, color=NAVY, va="top")
                fig.text(0.13, y, wrapped[0], fontsize=9.4, color="#222222", va="top")
                y -= 0.0225
                for line in wrapped[1:]:
                    fig.text(0.13, y, line, fontsize=9.4, color="#222222", va="top")
                    y -= 0.0225
                y -= 0.010
            return y - 0.018

        for heading, bullets in sections:
            y = section(heading, bullets, y)
        if footer:
            fig.text(0.08, 0.035, footer, fontsize=7.8, color="#666666", va="bottom")
        pdf.savefig(fig)
        plt.close(fig)

    intro = [
        "Computational source of truth: the companion Jupyter notebook. This PDF is written from the same payload; every table number is identical.",
        f"Question: which of {_n_models()} American-call pricing models tracks listed market prices most closely, and does the ranking change across companies and volatility regimes?",
        "In every table the BEST row is the lowest percentage RMSE. That row is bold and shaded green, and the Mark column says BEST.",
    ]
    intro = list(payload.get("meta", {}).get("group_intro") or []) + intro
    _instruction_page(
        "V3 empirical study  ·  Instruction",
        _banner_line(payload),
        [
            (
                "What this report is",
                intro,
            ),
            (
                "Experimental design (held fixed for every model)",
                [
                    b
                    for b in [
                        f"Models: {_models_listed()}.",
                        (
                            "Modified GBM (Markov direction + split-normal magnitudes) is specified in V3_modified_gbm_model.pdf in this folder."
                            if "Modified GBM" in TABLE_MODELS
                            else None
                        ),
                        f"Companies: {_tickers_listed()}. Each name is calibrated and priced separately.",
                        "Regimes (V3 one-year evaluation windows): Crisis 2008-08-01→2009-07-31; Normal 2014-01-01→2014-12-31; Late-cycle 2018-10-01→2019-09-30; COVID 2019-09-01→2020-08-31.",
                        "The late-cycle and COVID files share September 2019 (V3 notebook dates). All other regime days are disjoint. Contracts are built independently in each window.",
                        _rolling_detail(),
                        "If history is shorter than the requested lookback, the estimator uses all available observations up to the update date (equity starts 2003-12-01; listed-option panels start 2008-01).",
                    ]
                    if b
                ],
            ),
        ],
        None,
    )
    _instruction_page(
        "V3 empirical study  ·  Instruction (continued)",
        _banner_line(payload),
        [
            (
                "Shared market sample and filters (not model-specific)",
                [
                    f"For each company × regime, one listed-call sample is drawn once and reused by all {_models_phrase(cap=False)} in this report: nearest-ATM call on each Monday (next session if Monday is closed), DTE 7–60, listed expiry, as-of that date.",
                    "Estimation filters, applied in order to every quote used for Heston/Heston–Merton calibration and to the LSM comparison set: (1) calls only, finite S, K, C; (2) no-arbitrage C ≥ max(0, S−K); (3) 7 ≤ DTE ≤ 60; (4) |S/K − 1| ≤ 10%; (5) premium ≥ 0.05, valid bid–ask with relative spread ≤ 50% when those columns exist, volume ≥ 1 when present.",
                    "Return-based models (GBM, Merton, GARCH, GARCH–Merton) never see option quotes in §4; they still price the same filtered LSM contracts in evaluation.",
                ],
            ),
            (
                "Model-correctness adjustments (V3)",
                [
                    "Heston / Heston–Merton: (κ, θ, ξ, ρ, v0) are option-implied Fourier NLS (Albrecher little-trap), not Method A realized-variance moments. μ is the P-measure lookback mean of stock returns; LSM replaces μ → r.",
                    "Euler ordering: the stock increment over [t, t+Δt] uses the current variance v_t; v_{t+Δt} is updated afterwards. No look-ahead in the variance.",
                    f"LSM: Longstaff–Schwartz American call on the full path cloud (n_paths = {N_PATHS}, seed 42, n_steps = DTE, Δt = 1/252). Paths are not averaged before stopping.",
                ],
            ),
            (
                "Metrics",
                [
                    "Primary: RMSE% = 100 × √ mean(((C_model − C_mkt) / C_mkt)²). Ranking uses this number only.",
                    "Secondary: MAE = mean |C_model − C_mkt| (dollars). Bias $ = mean(C_model − C_mkt) (dollars). Bias% = 100 × mean((C_model − C_mkt)/C_mkt). Positive bias = model expensive vs market. Early-exercise fraction = mean share of LSM paths that stop before expiry.",
                ],
            ),
        ],
        "Page map: p.1–2 method  ·  p.3 filters & sample  ·  then detailed tables (one company per page, four regimes; grouped reports add a company-summary page after SPY)  ·  then summaries, figures, ranking, conclusion.",
    )


def _filters_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "Page 3  ·  Filters, sample, and no-look-ahead", fontsize=14, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], transform=fig.transFigure, color=NAVY, lw=1.2))

    # shared n table
    ax = fig.add_axes([0.08, 0.62, 0.84, 0.26])
    ax.axis("off")
    ax.set_title(
        f"Shared evaluation sample  ·  n listed calls (identical for all {_models_phrase(cap=False)} in this report)",
        loc="left",
        fontsize=10.5,
        color=NAVY,
        weight="bold",
    )
    header = ["Regime", "Window"] + [f"{t} n" for t in TICKERS]
    rows = []
    for regime in REGIME_ORDER:
        info = payload["shared_contracts"][regime]
        rows.append(
            [
                f"{regime}  {REGIME_META[regime]['title']}",
                info["period_start"] + " → " + info["period_end"],
                *[str(info["n"][t]) for t in TICKERS],
            ]
        )
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    _style_table(tbl, None)

    # filter steps for SPY crisis as illustration + all tickers last-step n
    ax2 = fig.add_axes([0.08, 0.28, 0.84, 0.30])
    ax2.axis("off")
    ax2.set_title(f"Filter funnel  ·  last-step n in the {LOOKBACK_PHRASE} lookback ending at each regime (processed call panel)", loc="left", fontsize=9.5, color=NAVY, weight="bold")
    funnel = payload["filter_funnel"]["tickers"]
    header2 = ["Ticker", "Regime", "Input*", "No-arb", "DTE 7–60", "|S/K−1|≤10%", "Liquidity"]
    rows2 = []
    for ticker in TICKERS:
        for regime in REGIME_ORDER:
            steps = {s["step"]: s["n"] for s in funnel[ticker][regime]["steps"]}
            rows2.append(
                [
                    ticker,
                    regime,
                    str(steps.get("finite S, K, C", "")),
                    str(steps.get("1. no-arbitrage C ≥ max(0, S−K)", "")),
                    str(steps.get("2. maturity 7–60 DTE", "")),
                    str(steps.get("3a. |S/K − 1| ≤ 10%", "")),
                    str(steps.get("3b. liquidity / wide bid–ask", "")),
                ]
            )
    tbl2 = ax2.table(cellText=rows2, colLabels=header2, loc="center", cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.96])
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(6.6)
    for (r, c), cell in tbl2.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(ALT_BG)
        else:
            cell.set_facecolor("white")

    fig.text(
        0.08,
        0.22,
        f"*Input is the processed call panel restricted to (regime end − {LOOKBACK_PHRASE}, regime end]. "
        "Processed files already drop many raw quotes; the funnel above is the additional V3 research filter.",
        fontsize=8.0,
        color="#555555",
        va="top",
        wrap=True,
    )
    fig.text(
        0.08,
        0.16,
        "No look-ahead. Monthly parameters at update date t use only equity returns and listed quotes with trading_date ≤ t. "
        "LSM prices the Monday contract using the latest calibration row with date ≤ that Monday. "
        "Quotes are as-of the grid timestamp (same session, else the prior session).",
        fontsize=8.0,
        color="#222222",
        va="top",
        wrap=True,
    )
    fig.text(
        0.08,
        0.105,
        "SPY 2008–2009 uses bid–ask mid when the source mark is $0.01 (a GitHub field error, not missing trades). "
        f"After that repair the processed panel has weekly coverage, so n matches AAPL/MSFT (~50). All {_models_phrase(cap=False)} in this report still share that exact sample.",
        fontsize=8.0,
        color="#222222",
        va="top",
        wrap=True,
    )
    fig.text(
        0.08,
        0.055,
        "Heston / Heston–Merton calibration further subsamples the filtered lookback surface to ≤ 24 contracts (maturity × moneyness strata). "
        "GBM / Merton / GARCH / GARCH–Merton estimate from lookback log returns only (jumps: 3σ residual threshold where applicable).",
        fontsize=8.0,
        color="#222222",
        va="top",
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _company_tables_page(pdf: PdfPages, payload: dict, ticker: str, page_no: int) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Page {page_no}  ·  Table block  ·  {ticker}  ·  {_models_phrase(cap=False)} × four regimes",
        fontsize=13,
        color=NAVY,
        y=0.975,
        weight="bold",
    )
    fig.text(
        0.06,
        0.935,
        f"Same contracts and LSM settings for every model.  Bold green row = lowest RMSE%.  "
        f"{LOOKBACK_PHRASE} lookback, rolling={ROLLING_MODE}, n_paths={N_PATHS}, seed={SEED}.",
        fontsize=8.2,
        color=MUTED,
    )
    for i, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(2, 2, i + 1)
        tab = payload["tables"][f"{ticker}|{regime}"]
        rows, header, best = _metric_rows(tab)
        meta = REGIME_META[regime]
        title = (
            f"Table {ticker}-{i+1}   {ticker}  ·  {regime}  {meta['title']}\n"
            f"{meta['window']}   ·   n = {tab['n_contracts']} contracts"
        )
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9.3, color=NAVY, pad=8, weight="bold")
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.08, 0.96, 0.78])
        _style_table(tbl, best)
        tbl.auto_set_column_width(list(range(len(header))))
    fig.text(
        0.06,
        0.03,
        "RMSE% is 100×√mean(((model−market)/market)²).  MAE and Bias $ are in option-premium dollars.  "
        "Bias% is 100×mean((model−market)/market).  Early-ex. frac. is the mean LSM share of paths exercised before expiry.  "
        "Bias > 0 means the model is expensive vs the quote.",
        fontsize=7.6,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def _company_summary_page(pdf: PdfPages, payload: dict, page_no: int) -> None:
    names = _equity_tickers()
    names_txt = ", ".join(names)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Page {page_no}  ·  Table block  ·  Company summary  ·  {_models_phrase(cap=False)} × four regimes",
        fontsize=13,
        color=NAVY,
        y=0.975,
        weight="bold",
    )
    fig.text(
        0.06,
        0.935,
        f"Each number is the arithmetic mean of {names_txt}.  Same columns as the SPY page.  "
        f"Bold green row = lowest mean RMSE%.  {LOOKBACK_PHRASE} lookback, rolling={ROLLING_MODE}, "
        f"n_paths={N_PATHS}, seed={SEED}.",
        fontsize=8.2,
        color=MUTED,
    )
    for i, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(2, 2, i + 1)
        tab = _average_company_table(payload, regime, names)
        rows, header, best = _metric_rows(tab)
        meta = REGIME_META[regime]
        title = (
            f"Table Co-{i+1}   Company summary  ·  {regime}  {meta['title']}\n"
            f"{meta['window']}   ·   mean of {names_txt}   ·   n̄ = {tab['n_contracts']}"
        )
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9.3, color=NAVY, pad=8, weight="bold")
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.08, 0.96, 0.78])
        _style_table(tbl, best)
        tbl.auto_set_column_width(list(range(len(header))))
    fig.text(
        0.06,
        0.03,
        "RMSE%, MAE, Bias $, Bias%, and early-exercise fraction are each averaged across "
        f"{names_txt}.  This is not a pooled re-pricing of a combined contract sample.  Ranking uses the mean RMSE% only.",
        fontsize=7.6,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def _summary_pages(pdf: PdfPages, payload: dict) -> None:
    # overall + by regime
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 7  ·  Summary tables  ·  RMSE% across companies and regimes", fontsize=13, color=NAVY, y=0.97, weight="bold")

    ax = fig.add_axes([0.06, 0.55, 0.42, 0.35])
    ax.axis("off")
    ax.set_title(f"Table S1  ·  Overall ranking (mean RMSE% over {_n_cells()} cells)", loc="left", fontsize=10, color=NAVY, weight="bold")
    header = ["Rank", "Model", "Mean RMSE%", "Median RMSE%", f"# of {_n_cells()} cells best"]
    rows = []
    best_i = 1
    for i, rec in enumerate(payload["summary_overall"], start=1):
        rows.append(
            [
                str(i),
                rec["model"],
                f"{rec['mean_rmse_pct']:.2f}",
                f"{rec['median_rmse_pct']:.2f}",
                str(rec["n_best"]),
            ]
        )
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    _style_table(tbl, best_i)

    ax2 = fig.add_axes([0.52, 0.55, 0.42, 0.35])
    ax2.axis("off")
    ax2.set_title("Table S2  ·  Wins by company (lowest RMSE% in that company×regime)", loc="left", fontsize=10, color=NAVY, weight="bold")
    header2 = ["Model"] + [f"{t} wins" for t in TICKERS] + ["Total"]
    rows2 = []
    win_counts = {m: {t: 0 for t in TICKERS} for m in TABLE_MODELS}
    for rec in payload["ranking_grid"]:
        win_counts[rec["rank1"]][rec["ticker"]] += 1
    totals = [sum(win_counts[m].values()) for m in TABLE_MODELS]
    best_model = TABLE_MODELS[int(np.argmax(totals))] if totals else None
    best_row = None
    for i, m in enumerate(TABLE_MODELS):
        tot = sum(win_counts[m].values())
        rows2.append([m, *[str(win_counts[m][t]) for t in TICKERS], str(tot)])
        if m == best_model:
            best_row = i + 1
    tbl2 = ax2.table(cellText=rows2, colLabels=header2, loc="center", cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    _style_table(tbl2, best_row)

    ax3 = fig.add_axes([0.06, 0.08, 0.88, 0.42])
    ax3.axis("off")
    ax3.set_title(f"Table S3  ·  Mean RMSE% by regime (equal-weight mean of {_tickers_joined()})", loc="left", fontsize=10, color=NAVY, weight="bold")
    header3 = ["Model"] + [f"{r}\n{REGIME_META[r]['title']}" for r in REGIME_ORDER] + ["Mean of 4"]
    by = {(r["regime"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_regime"]}
    rows3 = []
    col_best = {r: min(TABLE_MODELS, key=lambda m: by.get((r, m), np.inf)) for r in REGIME_ORDER}
    means = []
    for m in TABLE_MODELS:
        vals = [by.get((r, m), np.nan) for r in REGIME_ORDER]
        mu = float(np.nanmean(vals))
        means.append(mu)
        rows3.append([m] + [f"{v:.2f}" for v in vals] + [f"{mu:.2f}"])
    best_row3 = int(np.nanargmin(means)) + 1
    tbl3 = ax3.table(cellText=rows3, colLabels=header3, loc="center", cellLoc="center", bbox=[0.0, 0.08, 1.0, 0.82])
    _style_table(tbl3, best_row3)
    fig.text(
        0.06,
        0.035,
        "Green row in S1 / S3 = lowest mean RMSE%.  Green row in S2 = most company×regime wins.  "
        "A model can win the most cells without having the lowest average error (and vice versa).",
        fontsize=8.0,
        color="#555555",
    )
    pdf.savefig(fig)
    plt.close(fig)

    # by ticker summary
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 8  ·  Summary by company  ·  mean RMSE% across four regimes", fontsize=13, color=NAVY, y=0.97, weight="bold")
    by_t = {(r["ticker"], r["model"]): r for r in payload["summary_by_ticker"]}
    nrows, ncols = _ticker_grid_shape(len(TICKERS))
    for i, ticker in enumerate(TICKERS):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.axis("off")
        ax.set_title(f"Table S4.{i+1}  ·  {ticker}", loc="left", fontsize=10.5, color=NAVY, weight="bold")
        header = ["Model", "Mean RMSE%", "Median", "# regimes best"]
        rows = []
        rmses = []
        for m in TABLE_MODELS:
            rec = by_t[(ticker, m)]
            rmses.append(rec["mean_rmse_pct"])
            rows.append(
                [
                    m,
                    f"{rec['mean_rmse_pct']:.2f}",
                    f"{rec['median_rmse_pct']:.2f}",
                    str(rec["n_best"]),
                ]
            )
        best = int(np.argmin(rmses)) + 1
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.15, 0.96, 0.7])
        _style_table(tbl, best)
    fig.text(
        0.06,
        0.06,
        "Each panel is one company. Green row = lowest mean RMSE% across the four V3 regimes for that name.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.97, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def _figure_pages(pdf: PdfPages, payload: dict) -> None:
    cells = payload["cells"]

    # RMSE bars 2x2 regimes, mean of tickers? Better: 3 pages or one 2x2 with grouped bars per ticker
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"Page 9  ·  Figure 1  ·  RMSE% by model and regime  ·  mean of {_tickers_joined()}", fontsize=12.5, color=NAVY, y=0.98, weight="bold")
    x = np.arange(len(TABLE_MODELS))
    for ax, regime in zip(axes.ravel(), REGIME_ORDER):
        vals = []
        for m in TABLE_MODELS:
            v = [cells[f"{t}|{regime}|{m}"]["rmse_pct"] for t in TICKERS]
            vals.append(float(np.mean(v)))
        bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in TABLE_MODELS], width=0.72, zorder=3)
        ymax = max(vals) if vals else 1.0
        best_i = int(np.argmin(vals))
        bars[best_i].set_edgecolor("#111111")
        bars[best_i].set_linewidth(1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * ymax, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x)
        ax.set_xticklabels(list(TABLE_MODELS), fontsize=7.2, rotation=18, ha="right")
        ax.set_ylabel("RMSE%", fontsize=8.5)
        ax.set_title(f"{regime}  ·  {REGIME_META[regime]['title']}", loc="left", fontsize=10, color=NAVY, weight="bold")
        ax.yaxis.grid(True, linestyle="-", alpha=0.28, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
    fig.text(0.06, 0.03, f"Outlined bar = lowest mean RMSE% in that regime. Equal-weight mean of the {_companies_phrase()}.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
    pdf.savefig(fig)
    plt.close(fig)

    # per-ticker RMSE
    fig, axes = _subplots_tickers((11, 8.5))
    fig.suptitle("Page 10  ·  Figure 2  ·  RMSE% by model  ·  each company (mean across four regimes)", fontsize=12.5, color=NAVY, y=0.98, weight="bold")
    by_t = {(r["ticker"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_ticker"]}
    for ax, ticker in zip(axes, TICKERS):
        vals = [by_t[(ticker, m)] for m in TABLE_MODELS]
        bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in TABLE_MODELS], width=0.72, zorder=3)
        best_i = int(np.argmin(vals))
        bars[best_i].set_edgecolor("#111111")
        bars[best_i].set_linewidth(1.5)
        ymax = max(vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * ymax, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x)
        ax.set_xticklabels(list(TABLE_MODELS), fontsize=7.0, rotation=22, ha="right")
        ax.set_title(ticker, loc="left", fontsize=11, color=NAVY, weight="bold")
        ax.set_ylabel("Mean RMSE%", fontsize=8.5)
        ax.yaxis.grid(True, linestyle="-", alpha=0.28, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, ymax * 1.22)
    fig.text(0.06, 0.03, "Outlined bar = lowest mean RMSE% for that company.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
    pdf.savefig(fig)
    plt.close(fig)

    # ranking heatmap
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 11  ·  Figure 3  ·  Model rank by company × regime  (1 = best RMSE%)", fontsize=12.5, color=NAVY, y=0.97, weight="bold")
    rank = np.zeros((len(TABLE_MODELS), len(TICKERS) * len(REGIME_ORDER)))
    col_labels = []
    j = 0
    for ticker in TICKERS:
        for regime in REGIME_ORDER:
            tab = payload["tables"][f"{ticker}|{regime}"]
            order = {r["model"]: i + 1 for i, r in enumerate(sorted(tab["rows"], key=lambda z: z["rmse_pct"]))}
            for i, m in enumerate(TABLE_MODELS):
                rank[i, j] = order[m]
            col_labels.append(f"{ticker}\n{regime[-7:]}")
            j += 1
    ax = fig.add_axes([0.16, 0.18, 0.78, 0.68])
    cmap = plt.cm.RdYlGn_r
    n_m = len(TABLE_MODELS)
    im = ax.imshow(rank, cmap=cmap, vmin=1, vmax=max(n_m, 2), aspect="auto")
    ax.set_xticks(range(rank.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=7.4)
    ax.set_yticks(range(len(TABLE_MODELS)))
    ax.set_yticklabels(list(TABLE_MODELS), fontsize=9)
    for i in range(rank.shape[0]):
        for j in range(rank.shape[1]):
            val = int(rank[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=8.5,
                color="black" if val not in (1, n_m) else ("#0B3D0B" if val == 1 else "white"),
                fontweight="bold" if val == 1 else "normal",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Rank (1 = best)", fontsize=8)
    fig.text(0.16, 0.06, "Green / 1 = lowest RMSE% in that company×regime cell. Rankings can and do change across names and volatility states.", fontsize=8.5, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)

    # bias figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"Page 12  ·  Figure 4  ·  Bias (model − market, $)  ·  mean of {_companies_phrase()}", fontsize=12.5, color=NAVY, y=0.98, weight="bold")
    for ax, regime in zip(axes.ravel(), REGIME_ORDER):
        vals = [float(np.mean([cells[f"{t}|{regime}|{m}"]["bias"] for t in TICKERS])) for m in TABLE_MODELS]
        colors = [MODEL_COLORS[m] for m in TABLE_MODELS]
        bars = ax.bar(x, vals, color=colors, width=0.72, zorder=3)
        ax.axhline(0.0, color="#333333", lw=0.8)
        near0 = int(np.argmin(np.abs(vals)))
        bars[near0].set_edgecolor("#111111")
        bars[near0].set_linewidth(1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(list(TABLE_MODELS), fontsize=7.0, rotation=18, ha="right")
        ax.set_ylabel("Bias ($)", fontsize=8.5)
        ax.set_title(f"{regime}  ·  {REGIME_META[regime]['title']}", loc="left", fontsize=10, color=NAVY, weight="bold")
        ax.yaxis.grid(True, linestyle="-", alpha=0.28, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.06, 0.03, "Outlined bar = closest to zero bias (not the RMSE ranking). Positive = model expensive vs listed quote.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def _analysis_text(payload: dict) -> list[str]:
    overall = payload["summary_overall"]
    winner = overall[0]["model"]
    runner = overall[1]["model"] if len(overall) > 1 else ""
    lines = [
        f"On mean percentage RMSE across the {_n_cells()} company×regime cells, {winner} is the best overall model "
        f"(mean RMSE% = {overall[0]['mean_rmse_pct']:.2f}; {overall[0]['n_best']} of {_n_cells()} cells).",
    ]
    if runner:
        lines.append(
            f"Second overall is {runner} (mean RMSE% = {overall[1]['mean_rmse_pct']:.2f}; {overall[1]['n_best']} cells)."
        )
    # does ranking change?
    rank1s = {(r["ticker"], r["regime"]): r["rank1"] for r in payload["ranking_grid"]}
    unique_winners = sorted(set(rank1s.values()))
    if len(unique_winners) == 1:
        lines.append(f"The ranking is stable: {unique_winners[0]} is best in every company and every regime.")
    else:
        lines.append(
            f"The ranking is not stable. Best-in-cell models across the {_n_cells()} tables: {', '.join(unique_winners)}."
        )
    # by company
    by_t = {(r["ticker"], r["model"]): r for r in payload["summary_by_ticker"]}
    for ticker in TICKERS:
        pick = min(TABLE_MODELS, key=lambda m: by_t[(ticker, m)]["mean_rmse_pct"])
        lines.append(
            f"{ticker}: lowest mean RMSE% is {pick} ({by_t[(ticker, pick)]['mean_rmse_pct']:.2f}); "
            f"wins {by_t[(ticker, pick)]['n_best']} of 4 regimes."
        )
    by_r = {(r["regime"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_regime"]}
    for regime in REGIME_ORDER:
        pick = min(TABLE_MODELS, key=lambda m: by_r[(regime, m)])
        lines.append(
            f"{regime} ({REGIME_META[regime]['title']}): best mean RMSE% is {pick} ({by_r[(regime, pick)]:.2f})."
        )
    # jump vs no jump (only models present in this report)
    jump = [m for m in ("Merton", "GARCH–Merton", "Heston–Merton") if m in TABLE_MODELS]
    nojump = [m for m in ("GBM", "Modified GBM", "GARCH", "Heston") if m in TABLE_MODELS]
    jump_vals = [r["mean_rmse_pct"] for r in overall if r["model"] in jump]
    nojump_vals = [r["mean_rmse_pct"] for r in overall if r["model"] in nojump]
    if jump_vals and nojump_vals:
        jump_mean = float(np.mean(jump_vals))
        nojump_mean = float(np.mean(nojump_vals))
        if jump_mean < nojump_mean:
            lines.append(
                f"Jump models as a group have lower mean RMSE% ({jump_mean:.2f}) than the no-jump models in this report ({nojump_mean:.2f})."
            )
        else:
            lines.append(
                f"No-jump models as a group have lower mean RMSE% ({nojump_mean:.2f}) than the jump models in this report ({jump_mean:.2f})."
            )
    # crisis vs normal
    crisis = by_r[("2008-2009", winner)]
    normal = by_r[("2013-2014", winner)]
    lines.append(
        f"For the overall winner ({winner}), crisis RMSE% is {crisis:.2f} vs {normal:.2f} in the 2014 normal year — "
        "absolute pricing error is regime-dependent even when the ranking is not."
    )
    lines.append(
        f"These conclusions are conditional on the V3 filters, {_rolling_phrase()}, Monday ATM sample, "
        f"LSM with {N_PATHS:,} paths, and American-call quotes only. They are not a statement about European options or other tenors."
    )
    return lines


def _conclusion_page(pdf: PdfPages, payload: dict) -> None:
    import textwrap

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "Page 13  ·  Final analysis", fontsize=16, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], transform=fig.transFigure, color=NAVY, lw=1.2))

    bullets = _analysis_text(payload)
    y = 0.88
    fig.text(0.08, y, "Which models perform best, and does the ranking change?", fontsize=11.5, weight="bold", color=NAVY, va="top")
    y -= 0.04
    for b in bullets:
        wrapped = textwrap.wrap(b, width=96) or [""]
        fig.text(0.10, y, "•", fontsize=10, color=NAVY, va="top")
        fig.text(0.13, y, wrapped[0], fontsize=9.4, color="#222222", va="top")
        y -= 0.022
        for line in wrapped[1:]:
            fig.text(0.13, y, line, fontsize=9.4, color="#222222", va="top")
            y -= 0.022
        y -= 0.010

    y -= 0.02
    fig.text(0.08, y, "How to read the 12 detailed tables", fontsize=11.5, weight="bold", color=NAVY, va="top")
    y -= 0.035
    notes = [
        f"Each of Tables SPY-1…MSFT-4 uses exactly the same option contracts for all {_models_phrase(cap=False)} in this report. Differences are the dynamics, not the sample.",
        "BEST is always lowest RMSE%. MAE, Bias $, Bias%, and early-exercise are reported, not used for the ranking.",
        "A model that is BEST on RMSE% can still be biased (systematically expensive or cheap).",
        "Early-exercise fractions are a model implication, not an accuracy score; American calls on these tenors often have modest early-exercise value.",
    ]
    for b in notes:
        wrapped = textwrap.wrap(b, width=96) or [""]
        fig.text(0.10, y, "•", fontsize=10, color=NAVY, va="top")
        fig.text(0.13, y, wrapped[0], fontsize=9.4, color="#222222", va="top")
        y -= 0.022
        for line in wrapped[1:]:
            fig.text(0.13, y, line, fontsize=9.4, color="#222222", va="top")
            y -= 0.022
        y -= 0.008

    nb_shown = payload.get("meta", {}).get("nb_name") or NB_NAME
    fig.text(
        0.08,
        0.05,
        f"Notebook: {SHORT.relative_to(REPO) / nb_shown}\n"
        f"Engine: V3-Models_result/scripts/{ENGINE_SCRIPT}  ·  payload.json is the shared number store.",
        fontsize=7.6,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    payload = enrich_bias_pct(payload)
    SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (SHORT / PDF_NAME)
    with PdfPages(path) as pdf:
        _cover(pdf, payload)
        _filters_page(pdf, payload)
        page_no = 4
        for ticker in TICKERS:
            _company_tables_page(pdf, payload, ticker, page_no=page_no)
            page_no += 1
            if ticker == "SPY" and _wants_company_summary(payload):
                _company_summary_page(pdf, payload, page_no=page_no)
                page_no += 1
        _summary_pages(pdf, payload)
        _figure_pages(pdf, payload)
        _conclusion_page(pdf, payload)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)
    return path


def _html_table(tab: dict) -> str:
    rows, header, best = _metric_rows(tab)
    parts = [
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;width:100%;'>",
        "<thead><tr>",
    ]
    for h in header:
        parts.append(f"<th style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{h}</th>")
    parts.append("</tr></thead><tbody>")
    for i, row in enumerate(rows, start=1):
        if i == best:
            bg, wt = BEST_BG, "700"
        elif i % 2 == 0:
            bg, wt = ALT_BG, "400"
        else:
            bg, wt = "white", "400"
        parts.append("<tr>")
        for j, val in enumerate(row):
            extra = "font-weight:700;" if i == best else ""
            parts.append(
                f"<td style='background:{bg};{extra}padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{val}</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_notebook(payload: dict, path: Path | None = None) -> Path:
    import base64
    import io
    import uuid

    payload = enrich_bias_pct(payload)
    SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (SHORT / NB_NAME)

    def md(text: str) -> dict:
        if not text.endswith("\n"):
            text += "\n"
        return {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]],
        }

    def code(src: str, html: str | None = None, png: bytes | None = None) -> dict:
        outputs = []
        if html:
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"text/html": [html], "text/plain": ["<IPython.core.display.HTML object>"]},
                    "metadata": {},
                }
            )
        if png:
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"image/png": base64.b64encode(png).decode("ascii"), "text/plain": ["<Figure>"]},
                    "metadata": {},
                }
            )
        return {
            "cell_type": "code",
            "execution_count": 1,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": outputs,
            "source": [ln + "\n" for ln in src.split("\n")[:-1]] + ([src.split("\n")[-1]] if src else [""]),
        }

    def fig_png(draw) -> bytes:
        fig = draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return buf.getvalue()

    cells = []
    grouped_id = payload.get("meta", {}).get("grouped_report")
    rerun_note = (
        "This notebook is a grouped companion. It loads stored cells and does **not** re-run LSM. "
        "To rebuild only this group's PDF/notebook, call `groups.write_group(...)`. "
        "Do not set RECOMPUTE on the six-model engine; that would overwrite the original study files."
        if grouped_id
        else "Set `RECOMPUTE = True` in the next cell and Run All. Default loads `payload.json` if the company×regime×model cells are already complete, then rewrites the PDF from that payload."
    )
    cells.append(
        md(
            f"""# V3 empirical study — {_rolling_phrase()}{payload.get("meta", {}).get("notebook_title_suffix", "")}

**This notebook is the computational source of truth.** The PDF in this folder is written from the same `payload` dict, so every table number matches.

| Item | Setting |
|------|---------|
| Models | {", ".join(TABLE_MODELS)} |
| Companies | {_tickers_listed()} |
| Regimes | Crisis 2008-08-01→2009-07-31 · Normal 2014-01-01→2014-12-31 · Late-cycle 2018-10-01→2019-09-30 · COVID 2019-09-01→2020-08-31 |
| Calibration | **{LOOKBACK_PHRASE}** lookback, **{ROLLING_MODE}** recalibration |
| Primary metric | Percentage RMSE |
| Secondary | MAE, Bias $ (model − market), Bias%, Early-exercise fraction |
| Paths | LSM American calls, `n_paths={N_PATHS}`, `seed=42` |
| Code | V3 notebooks + `option_filters.py` + Euler-corrected Heston step |

In every table the **BEST** model is the lowest RMSE%. That row is **bold and green**.
"""
        )
    )
    cells.append(
        md(
            f"""## 0. Methodology and conditions

### No look-ahead
{_no_lookahead_note()}

### Shared sample (no model-specific data selection)
For each company × regime the Monday nearest-ATM listed-call sample is drawn **once** and passed to all {_models_phrase(cap=False)} in this report. Filters, testing dates, simulation settings, and the market benchmark are therefore identical.

### Filters (applied in this order)
1. Calls only; finite S, K, C.
2. No-arbitrage: C ≥ max(0, S−K).
3. Maturity: 7 ≤ DTE ≤ 60.
4. Moneyness: |S/K − 1| ≤ 10%.
5. Liquidity: premium ≥ 0.05; valid bid–ask and relative spread ≤ 50% when those columns exist; volume ≥ 1 when present.

### Metrics
- Primary: RMSE%.
- Secondary: MAE (dollars), Bias $ = mean(model − market), **Bias% = 100 × mean((model − market)/market)**, early-exercise fraction.
- Ranking uses RMSE% only. The dollar bias column is kept; Bias% is the scale-free companion.

### Model-correctness adjustments
- Heston / Heston–Merton: option-implied Fourier NLS for (κ, θ, ξ, ρ, v0). Not Method A.
- Euler: stock return over [t, t+Δt] uses **current** v_t; then update v_{{t+Δt}}.
- LSM never averages paths before stopping.

### Data limits
Equity adj-close starts **2003-12-01**. Listed-option panels start **2008-01**. Estimators use all available history when the requested lookback is longer than the file.

### Re-run
{rerun_note}
"""
        )
    )
    grouped_id = payload.get("meta", {}).get("grouped_report")
    if grouped_id:
        bootstrap = (
            "import sys\n"
            "from pathlib import Path\n"
            "from IPython.display import display, HTML, Markdown, Image\n"
            "ROOT = Path.cwd()\n"
            "for cand in [ROOT, *ROOT.parents]:\n"
            "    scripts = cand / 'V3-Models_result' / 'scripts'\n"
            "    if scripts.exists():\n"
            "        sys.path.insert(0, str(scripts))\n"
            "        break\n"
            "import run_v3_1p5y_10k_monthly_empirical_study_groups as groups\n"
            f"payload = groups.load_group({grouped_id!r})\n"
            "print('cells', len(payload['cells']), 'models', payload['meta']['models'])\n"
            "# Rebuild this group's PDF/notebook only (does not overwrite the six-model study):\n"
            f"# groups.write_group({grouped_id!r})"
        )
        bootstrap_html = (
            f"<pre>cells {len(payload['cells'])} models {payload['meta']['models']}</pre>"
        )
    else:
        bootstrap = (
            "import sys\n"
            "from pathlib import Path\n"
            "from IPython.display import display, HTML, Markdown, Image\n"
            "ROOT = Path.cwd()\n"
            "for cand in [ROOT, *ROOT.parents]:\n"
            "    scripts = cand / 'V3-Models_result' / 'scripts'\n"
            "    if scripts.exists():\n"
            "        sys.path.insert(0, str(scripts))\n"
            "        break\n"
            f"import {NOTEBOOK_IMPORT} as study\n"
            "RECOMPUTE = False  # True to re-run all calibrations + LSM\n"
            "payload = study.run_or_load(recompute=RECOMPUTE)\n"
            "print('cells', len(payload['cells']), 'failures', payload['meta']['failures'])\n"
            "pdf_path = study.write_pdf(payload)\n"
            "print('PDF', pdf_path)"
        )
        bootstrap_html = f"<pre>cells {len(payload['cells'])} failures {payload['meta']['failures']}</pre>"
    cells.append(code(bootstrap, html=bootstrap_html))

    # filter / sample
    n_rows = []
    n_rows.append("<h2>1. Shared evaluation sample</h2>")
    n_rows.append(
        f"<p>One Monday ATM listed-call sample per company × regime. Copied to all {_models_phrase(cap=False)} in this report.</p>"
    )
    n_rows.append(
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;'>"
        f"<tr><th style='background:{NAVY};color:white;padding:6px 8px;'>Regime</th>"
        f"<th style='background:{NAVY};color:white;padding:6px 8px;'>Window</th>"
        + "".join(
            f"<th style='background:{NAVY};color:white;padding:6px 8px;'>{t}</th>"
            for t in TICKERS
        )
        + "</tr>"
    )
    for regime in REGIME_ORDER:
        info = payload["shared_contracts"][regime]
        n_cells = "".join(
            f"<td style='padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{info['n'][t]}</td>"
            for t in TICKERS
        )
        n_rows.append(
            f"<tr><td style='padding:5px 8px;border:1px solid #D0D5DD;'>{regime} {REGIME_META[regime]['title']}</td>"
            f"<td style='padding:5px 8px;border:1px solid #D0D5DD;'>{info['period_start']} → {info['period_end']}</td>"
            f"{n_cells}</tr>"
        )
    n_rows.append("</table>")
    cells.append(md("## 1. Shared evaluation sample"))
    cells.append(
        code(
            "from IPython.display import HTML, display\n"
            "rows = ['<h4>n contracts (identical across models)</h4>']\n"
            "display(HTML(''.join(study._html_table(payload['tables'][k]) if False else '')))\n"
            "meta = payload['shared_contracts']\n"
            "import pandas as pd\n"
            "df = pd.DataFrame([{ 'regime': r, **meta[r]['n'], 'start': meta[r]['period_start'], 'end': meta[r]['period_end']} for r in study.REGIME_ORDER])\n"
            "display(df)",
            html="".join(n_rows),
        )
    )

    k = 1
    sec = 1
    for ticker in TICKERS:
        cells.append(md(f"## 2.{sec} Detailed tables — `{ticker}`"))
        sec += 1
        for i, regime in enumerate(REGIME_ORDER, start=1):
            tab = payload["tables"][f"{ticker}|{regime}"]
            meta = REGIME_META[regime]
            title = (
                f"### Table {ticker}-{i}. {ticker} · {regime} · {meta['title']}\n\n"
                f"{meta['window']} · n = {tab['n_contracts']} contracts · "
                f"**BEST = {tab['best_model']}** (lowest RMSE%)."
            )
            cells.append(md(title))
            cells.append(
                code(
                    f"display(HTML(study._html_table(payload['tables']['{ticker}|{regime}'])))",
                    html=_html_table(tab),
                )
            )
            k += 1
        if ticker == "SPY" and _wants_company_summary(payload):
            names = _equity_tickers()
            names_txt = ", ".join(names)
            cells.append(md(f"## 2.{sec} Detailed tables — Company summary"))
            cells.append(
                md(
                    f"Same four-regime layout as SPY. Each value is the arithmetic mean of **{names_txt}**. "
                    "Ranking uses mean RMSE%."
                )
            )
            sec += 1
            for i, regime in enumerate(REGIME_ORDER, start=1):
                tab = _average_company_table(payload, regime, names)
                meta = REGIME_META[regime]
                title = (
                    f"### Table Co-{i}. Company summary · {regime} · {meta['title']}\n\n"
                    f"{meta['window']} · mean of {names_txt} · n̄ = {tab['n_contracts']} · "
                    f"**BEST = {tab['best_model']}** (lowest mean RMSE%)."
                )
                cells.append(md(title))
                cells.append(
                    code(
                        "display(HTML(study._html_table("
                        f"study._average_company_table(payload, '{regime}', study._equity_tickers())"
                        ")))",
                        html=_html_table(tab),
                    )
                )
                k += 1

    # summaries as markdown+html
    cells.append(md("## 3. Summary tables"))
    # S1
    s1 = payload["summary_overall"]
    html = [
        f"<h3>Table S1 · Overall ranking (mean RMSE% over {_n_cells()} cells)</h3>",
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;'>",
        f"<tr><th style='background:{NAVY};color:white;padding:6px 8px;'>Rank</th>"
        f"<th style='background:{NAVY};color:white;padding:6px 8px;'>Model</th>"
        f"<th style='background:{NAVY};color:white;padding:6px 8px;'>Mean RMSE%</th>"
        f"<th style='background:{NAVY};color:white;padding:6px 8px;'>Median RMSE%</th>"
        f"<th style='background:{NAVY};color:white;padding:6px 8px;'># cells best</th></tr>",
    ]
    for i, rec in enumerate(s1, start=1):
        bg = BEST_BG if i == 1 else ("white" if i % 2 else ALT_BG)
        wt = "700" if i == 1 else "400"
        mark = " BEST" if i == 1 else ""
        html.append(
            f"<tr><td style='background:{bg};font-weight:{wt};padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{i}</td>"
            f"<td style='background:{bg};font-weight:{wt};padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{rec['model']}{mark}</td>"
            f"<td style='background:{bg};font-weight:{wt};padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{rec['mean_rmse_pct']:.2f}</td>"
            f"<td style='background:{bg};font-weight:{wt};padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{rec['median_rmse_pct']:.2f}</td>"
            f"<td style='background:{bg};font-weight:{wt};padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{rec['n_best']}</td></tr>"
        )
    html.append("</table>")
    cells.append(md("### Table S1 — Overall ranking"))
    cells.append(code("display(HTML('overall ranking table is rendered from payload[\"summary_overall\"]'))", html="".join(html)))

    cells.append(md("## 4. Figures"))

    def draw_fig1():
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
        fig.suptitle(f"Figure 1 · RMSE% by model and regime · mean of {_tickers_joined()}", color=NAVY, fontsize=12, weight="bold")
        x = np.arange(len(TABLE_MODELS))
        cells_ = payload["cells"]
        for ax, regime in zip(axes.ravel(), REGIME_ORDER):
            vals = [float(np.mean([cells_[f"{t}|{regime}|{m}"]["rmse_pct"] for t in TICKERS])) for m in TABLE_MODELS]
            bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in TABLE_MODELS], width=0.72, zorder=3)
            bars[int(np.argmin(vals))].set_edgecolor("#111")
            bars[int(np.argmin(vals))].set_linewidth(1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(list(TABLE_MODELS), fontsize=7, rotation=18, ha="right")
            ax.set_title(f"{regime} · {REGIME_META[regime]['title']}", loc="left", color=NAVY, fontsize=10, weight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.28)
            ax.set_axisbelow(True)
            ax.set_ylabel("RMSE%")
        fig.tight_layout()
        return fig

    cells.append(md(f"### Figure 1 — RMSE% by regime (mean of {_companies_phrase()})"))
    cells.append(code("pass  # figure embedded from the same payload used for the PDF", png=fig_png(draw_fig1)))

    def draw_fig2():
        fig, axes = _subplots_tickers((10.5, 7.6 if len(TICKERS) > 3 else 4.2))
        fig.suptitle("Figure 2 · Mean RMSE% by company", color=NAVY, fontsize=12, weight="bold")
        x = np.arange(len(TABLE_MODELS))
        by_t = {(r["ticker"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_ticker"]}
        for ax, ticker in zip(axes, TICKERS):
            vals = [by_t[(ticker, m)] for m in TABLE_MODELS]
            bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in TABLE_MODELS], width=0.72, zorder=3)
            bars[int(np.argmin(vals))].set_edgecolor("#111")
            bars[int(np.argmin(vals))].set_linewidth(1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(list(TABLE_MODELS), fontsize=7, rotation=22, ha="right")
            ax.set_title(ticker, loc="left", color=NAVY, fontsize=11, weight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.28)
            ax.set_axisbelow(True)
        fig.tight_layout()
        return fig

    cells.append(md("### Figure 2 — RMSE% by company"))
    cells.append(code("pass", png=fig_png(draw_fig2)))

    def draw_fig3():
        fig = plt.figure(figsize=(10.5, 5.2))
        n_cells = len(TICKERS) * len(REGIME_ORDER)
        rank = np.zeros((len(TABLE_MODELS), n_cells))
        labels = []
        j = 0
        for ticker in TICKERS:
            for regime in REGIME_ORDER:
                tab = payload["tables"][f"{ticker}|{regime}"]
                order = {r["model"]: i + 1 for i, r in enumerate(sorted(tab["rows"], key=lambda z: z["rmse_pct"]))}
                for i, m in enumerate(TABLE_MODELS):
                    rank[i, j] = order[m]
                labels.append(f"{ticker}\n{regime[-7:]}")
                j += 1
        ax = fig.add_subplot(111)
        n_m = len(TABLE_MODELS)
        im = ax.imshow(rank, cmap="RdYlGn_r", vmin=1, vmax=max(n_m, 2), aspect="auto")
        ax.set_xticks(range(n_cells))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(list(TABLE_MODELS))
        for i in range(n_m):
            for j in range(n_cells):
                ax.text(j, i, str(int(rank[i, j])), ha="center", va="center", fontsize=8, fontweight="bold" if rank[i, j] == 1 else "normal")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Rank")
        fig.suptitle("Figure 3 · Rank heatmap (1 = best RMSE%)", color=NAVY, fontsize=12, weight="bold")
        fig.tight_layout()
        return fig

    cells.append(md("### Figure 3 — Rank heatmap"))
    cells.append(code("pass", png=fig_png(draw_fig3)))

    cells.append(md("## 5. Final analysis\n\n" + "\n\n".join(f"- {b}" for b in _analysis_text(payload))))
    if grouped_id:
        repro = f"""## 6. Reproducibility

This grouped report is assembled from the frozen 10,000-path cells. No new calibration or LSM.

```python
import run_v3_1p5y_10k_monthly_empirical_study_groups as groups
payload = groups.load_group({grouped_id!r})
groups.write_group({grouped_id!r})
```

The original six-model files `V3_1p5y_monthly_empirical_study.pdf` / `.ipynb` are not rewritten.
Partial results remain in `{CACHE.relative_to(REPO)}/partial/`.
Shared contracts are frozen in `shared_contracts.json`.
"""
    else:
        repro = f"""## 6. Reproducibility

```python
RECOMPUTE = True
payload = study.run_or_load(recompute=True)
study.write_pdf(payload)
study.build_notebook(payload)
```

Partial results resume from `{CACHE.relative_to(REPO)}/partial/`.
Shared contracts are frozen in `shared_contracts.json` so a resumed run cannot silently change the sample.
"""
    cells.append(md(repro))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


def write_outputs(payload: dict) -> tuple[Path, Path]:
    payload = enrich_bias_pct(payload)
    PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pdf = write_pdf(payload)
    nb = build_notebook(payload)
    return nb, pdf


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    recompute = "--recompute" in argv
    assemble_only = "--assemble" in argv
    only = [a for a in argv if not a.startswith("--")]
    if assemble_only:
        shared = sample_shared_contracts()
        funnel = filter_funnel()
        completed = []
        for part in sorted((CACHE / "partial").glob("*.json")):
            completed.append(json.loads(part.read_text(encoding="utf-8")))
        payload = enrich_bias_pct(assemble_payload(shared, funnel, completed, [], 0.0))
        PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"assembled {len(payload.get('cells', {}))}/{_n_expected_cells()} cells", flush=True)
        if len(payload.get("cells", {})) == _n_expected_cells():
            write_outputs(payload)
            return 0
        return 1
    payload = run_or_load(recompute=recompute, only=only or None)
    if len(payload.get("cells", {})) == _n_expected_cells() and not payload["meta"]["failures"]:
        write_outputs(payload)
        return 0
    print(
        f"Incomplete: {len(payload.get('cells', {}))}/{_n_expected_cells()} cells, "
        f"failures={payload['meta']['failures']}. PDF/notebook not finalized.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
