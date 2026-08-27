#!/usr/bin/env python3
"""V3 empirical study on 1-minute data: 12 monthly listed expiries.

Friday-before-expiry evaluation. Hourly stamps 09:59–15:59.
LSM / stock paths stop at the evaluation window’s Friday 15:59 (not listed expiry).

  7-day: lookback 7 RTH days, hourly rolling, 5-minute MC steps,
         five NYSE sessions ending the Friday before expiry.
  1-day: lookback 1 RTH day, hourly rolling, 1-minute MC steps,
         that Friday only.

Shared listed-call sample (DTE 7–60, nearest ATM as-of each hourly t).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPO = ROOT.parent
sys.path.insert(0, str(SCRIPTS))

import run_7d_1min_study as intra_load  # noqa: E402
import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
from american_lsm import (  # noqa: E402
    _pick_atm,
    _quotes_asof,
    lsm_american_call,
    load_calls,
    params_asof,
    pct_rmse,
    rth_grid,
)
from option_filters import apply_estimation_filters, moneyness_bucket, spot_over_strike  # noqa: E402

os_study._install_notebook_stubs()

EXPIRY_FRIDAYS = (
    "2022-10-21",
    "2022-11-18",
    "2022-12-16",
    "2023-01-20",
    "2023-02-17",
    "2023-03-17",
    "2023-04-21",
    "2023-05-19",
    "2023-06-16",
    "2023-07-21",
    "2023-08-18",
    "2023-09-15",
)
MONTH_TITLE = {
    "2022-10-21": "Oct 2022",
    "2022-11-18": "Nov 2022",
    "2022-12-16": "Dec 2022",
    "2023-01-20": "Jan 2023",
    "2023-02-17": "Feb 2023",
    "2023-03-17": "Mar 2023",
    "2023-04-21": "Apr 2023",
    "2023-05-19": "May 2023",
    "2023-06-16": "Jun 2023",
    "2023-07-21": "Jul 2023",
    "2023-08-18": "Aug 2023",
    "2023-09-15": "Sep 2023",
}
SHORT_LABEL = {
    "2022-10-21": "Oct22",
    "2022-11-18": "Nov22",
    "2022-12-16": "Dec22",
    "2023-01-20": "Jan23",
    "2023-02-17": "Feb23",
    "2023-03-17": "Mar23",
    "2023-04-21": "Apr23",
    "2023-05-19": "May23",
    "2023-06-16": "Jun23",
    "2023-07-21": "Jul23",
    "2023-08-18": "Aug23",
    "2023-09-15": "Sep23",
}
MODEL_NOTEBOOKS = {
    "7d": {
        "GBM": ("gbm notebook", "7d_1min_gbm.ipynb"),
        "Modified GBM": ("modified gbm notebook", "7d_1min_modified_gbm.ipynb"),
        "GARCH": ("garch notebook", "7d_1min_garch.ipynb"),
        "Heston": ("heston notebook", "7d_1min_heston.ipynb"),
        "Merton": ("merton notebook", "7d_1min_merton.ipynb"),
        "GARCH–Merton": ("garch merton notebook", "7d_1min_garch_merton.ipynb"),
        "Heston–Merton": ("heston merton notebook", "7d_1min_heston_merton.ipynb"),
    },
    "1d": {
        "GBM": ("gbm notebook", "1d_1min_gbm.ipynb"),
        "Modified GBM": ("modified gbm notebook", "1d_1min_modified_gbm.ipynb"),
        "GARCH": ("garch notebook", "1d_1min_garch.ipynb"),
        "Heston": ("heston notebook", "1d_1min_heston.ipynb"),
        "Merton": ("merton notebook", "1d_1min_merton.ipynb"),
        "GARCH–Merton": ("garch merton notebook", "1d_1min_garch_merton.ipynb"),
        "Heston–Merton": ("heston merton notebook", "1d_1min_heston_merton.ipynb"),
    },
}

STUDY_KIND = "7d"
WINDOW_LABEL = "7 days"
LOOKBACK_BARS = 2730
LOOKBACK_PHRASE = "7-day"
STEP_MINUTES = 5
ROLLING_MODE = "hourly"
N_PATHS = 10000
N_STEPS = 50000
SEED = 42
MODEL_STEM = {
    "GBM": "gbm",
    "Modified GBM": "modified_gbm",
    "GARCH": "garch",
    "Heston": "heston",
    "Merton": "merton",
    "GARCH–Merton": "garch_merton",
    "Heston–Merton": "heston_merton",
}
TICKERS = emp.TICKERS
TABLE_MODELS = emp.TABLE_MODELS
MODELS = emp.MODELS
REGIME_ORDER = list(EXPIRY_FRIDAYS)
NAVY = emp.NAVY
MUTED = emp.MUTED
BEST_BG = emp.BEST_BG
ALT_BG = emp.ALT_BG
_style_table = emp._style_table
pct_bias = emp.pct_bias
REPO = REPO
CACHE = ROOT / "results" / "empirical_study_7d_hourly"
SHORT = REPO / "Results_In_Short" / "V3 7-day hourly empirical study"
PDF_NAME = "V3_7d_hourly_empirical_study.pdf"
NB_NAME = "V3_7d_hourly_empirical_study.ipynb"
NOTEBOOK_IMPORT = "run_v3_7d_hourly_empirical_study"
ENGINE_SCRIPT = "run_v3_7d_hourly_empirical_study.py"
PAYLOAD_JSON = CACHE / "payload.json"
CONTRACTS_JSON = CACHE / "shared_contracts.json"
FILTER_JSON = CACHE / "filter_funnel.json"
REGIME_META: dict = {}
_NS_CACHE: dict = {}
_PX_INDEX: pd.DatetimeIndex | None = None


def configure_study(kind: str) -> None:
    global STUDY_KIND, WINDOW_LABEL, LOOKBACK_BARS, LOOKBACK_PHRASE, STEP_MINUTES
    global CACHE, SHORT, PDF_NAME, NB_NAME, NOTEBOOK_IMPORT, ENGINE_SCRIPT
    global PAYLOAD_JSON, CONTRACTS_JSON, FILTER_JSON, REGIME_META
    kind = str(kind).lower()
    if kind not in ("7d", "1d"):
        raise ValueError(kind)
    STUDY_KIND = kind
    if kind == "7d":
        WINDOW_LABEL = "7 days"
        LOOKBACK_BARS = 7 * 390
        LOOKBACK_PHRASE = "7-day"
        STEP_MINUTES = 5
        CACHE = ROOT / "results" / "empirical_study_7d_hourly"
        SHORT = REPO / "Results_In_Short" / "V3 7-day hourly empirical study"
        PDF_NAME = "V3_7d_hourly_empirical_study.pdf"
        NB_NAME = "V3_7d_hourly_empirical_study.ipynb"
        NOTEBOOK_IMPORT = "run_v3_7d_hourly_empirical_study"
        ENGINE_SCRIPT = "run_v3_7d_hourly_empirical_study.py"
    else:
        WINDOW_LABEL = "1 day"
        LOOKBACK_BARS = 390
        LOOKBACK_PHRASE = "1-day"
        STEP_MINUTES = 1
        CACHE = ROOT / "results" / "empirical_study_1d_hourly"
        SHORT = REPO / "Results_In_Short" / "V3 1-day hourly empirical study"
        PDF_NAME = "V3_1d_hourly_empirical_study.pdf"
        NB_NAME = "V3_1d_hourly_empirical_study.ipynb"
        NOTEBOOK_IMPORT = "run_v3_1d_hourly_empirical_study"
        ENGINE_SCRIPT = "run_v3_1d_hourly_empirical_study.py"
    PAYLOAD_JSON = CACHE / "payload.json"
    CONTRACTS_JSON = CACHE / "shared_contracts.json"
    FILTER_JSON = CACHE / "filter_funnel.json"
    os_study.WINDOW_LABEL = WINDOW_LABEL
    os_study.N_PATHS = N_PATHS
    os_study.SEED = SEED
    _NS_CACHE.clear()
    REGIME_META = _build_regime_meta()


def _px_index() -> pd.DatetimeIndex:
    global _PX_INDEX
    if _PX_INDEX is None:
        p = pd.read_csv(
            REPO / "research" / "data" / "equity" / "short_interval" / "prices_1min_rth" / "SPY.csv",
            parse_dates=["Datetime"],
        )
        _PX_INDEX = pd.DatetimeIndex(p["Datetime"]).sort_values().unique()
    return _PX_INDEX


def eval_friday(expiry: str) -> pd.Timestamp:
    """Friday immediately before the monthly third-Friday expiry."""
    return pd.Timestamp(expiry).normalize() - pd.Timedelta(days=7)


def window_bounds(expiry: str) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    idx = _px_index()
    days = pd.DatetimeIndex(idx.normalize().unique()).sort_values()
    fri = eval_friday(expiry)
    if fri not in set(days):
        prior = days[days <= fri]
        if len(prior) == 0:
            raise RuntimeError(f"No RTH session on/before Friday-before {expiry}")
        fri = pd.Timestamp(prior[-1])
    if STUDY_KIND == "1d":
        session_days = pd.DatetimeIndex([fri])
    else:
        prior = days[days <= fri]
        session_days = prior[-5:]
    if len(session_days) == 0:
        raise RuntimeError(f"No RTH session for Friday-before {expiry}")
    start = idx[idx.normalize() == session_days[0]].min()
    end = _friday_1559(session_days[-1])
    return pd.Timestamp(start), pd.Timestamp(end), [d.date().isoformat() for d in session_days]


def _friday_1559(day) -> pd.Timestamp:
    """Last 1-minute RTH bar of the 15:00 hour (15:59), else last bar with hour ≤ 15."""
    idx = _px_index()
    day = pd.Timestamp(day).normalize()
    session = idx[idx.normalize() == day]
    if session.empty:
        raise RuntimeError(f"No RTH bars on {day.date()}")
    hit = session[(session.hour == 15) & (session.minute == 59)]
    if len(hit):
        return pd.Timestamp(hit[-1])
    session = session[session.hour <= 15]
    return pd.Timestamp(session.max())


def hourly_stamps(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    idx = _px_index()
    session = idx[(idx >= start) & (idx <= end)]
    if session.empty:
        return session
    hours = session.floor("h")
    out = []
    for h in hours.unique():
        t = pd.Timestamp(session[hours == h].max())
        if 9 <= int(t.hour) <= 15:
            out.append(t)
    return pd.DatetimeIndex(out)


def remaining_n_steps(px: pd.Series, t, window_end) -> int:
    idx = pd.DatetimeIndex(px.dropna().index)
    n_min = int(((idx > pd.Timestamp(t)) & (idx <= pd.Timestamp(window_end))).sum())
    return int(n_min // max(int(STEP_MINUTES), 1))


def eval_clock(start, end) -> pd.DatetimeIndex:
    """RTH evaluation clock: every STEP_MINUTES bars from window open through Friday 15:59."""
    idx = _px_index()
    session = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    step = max(int(STEP_MINUTES), 1)
    if session.empty:
        return session
    take = session[::step]
    if take[-1] != session[-1]:
        take = take.append(pd.DatetimeIndex([session[-1]]))
    return pd.DatetimeIndex(take)


def _build_regime_meta() -> dict:
    out = {}
    for exp in EXPIRY_FRIDAYS:
        start, end, sessions = window_bounds(exp)
        out[exp] = {
            "title": MONTH_TITLE[exp],
            "window": f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}",
            "note": (
                f"Listed expiry {exp}; evaluate Friday-before {eval_friday(exp).date()}; "
                f"{len(sessions)} RTH session(s): {', '.join(sessions)}"
            ),
            "n_sessions": len(sessions),
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "eval_friday": eval_friday(exp).date().isoformat(),
        }
    return out


def _nb_path(model: str) -> Path:
    folder, name = MODEL_NOTEBOOKS[STUDY_KIND][model]
    return ROOT / folder / name


def _load_model_ns(model: str) -> dict:
    if model in _NS_CACHE:
        return _NS_CACHE[model]
    os_study._install_notebook_stubs()
    g = intra_load._load_ns(_nb_path(model))
    g["WINDOW_OPTIONS"] = {"1 hour": 60, "1 day": 390, "7 days": 7 * 390}
    g["WINDOW_LABEL"] = WINDOW_LABEL
    g["N_STEPS"] = N_STEPS
    g["N_DAYS"] = 252 * (390 // int(STEP_MINUTES))
    g["ROLLING_OPTIONS"] = ["minutely", "hourly"]
    g["_with_option_days"] = lambda fn, *a, **k: fn(*a, **k)
    _NS_CACHE[model] = g
    return g


def _set_period(g: dict, start: pd.Timestamp, end: pd.Timestamp) -> None:
    g["PERIOD_START"] = start
    g["PERIOD_END"] = end
    g["N_DAYS"] = 252 * (390 // int(STEP_MINUTES))
    g["N_STEPS"] = N_STEPS
    tickers = list(g["TICKERS"])
    g["period_prices"] = g["prices"].loc[start:end, tickers].copy()
    g["_with_option_days"] = lambda fn, *a, **k: fn(*a, **k)


def _lsm_n_steps(px: pd.Series, row) -> int:
    window_end = getattr(row, "window_end", None)
    if window_end is None:
        window_end = px.dropna().index.max()
    n = remaining_n_steps(px, row.trading_date, window_end)
    return int(n)


def _run_ticker(g: dict, ticker: str, contracts: pd.DataFrame) -> dict:
    t_cal0 = time.time()
    cal = g["calibrate_ticker"](ticker, WINDOW_LABEL, ROLLING_MODE)
    g["rolling"] = {ticker: cal}
    g["cal_meta"] = {"window_label": WINDOW_LABEL, "rolling_mode": ROLLING_MODE}
    t_cal = time.time() - t_cal0
    dt = 1.0 / float(g["N_DAYS"])
    px = g["prices"][ticker]
    rows = []
    t_lsm0 = time.time()
    for i, row in enumerate(contracts.itertuples(index=False)):
        n_steps = int(getattr(row, "remaining_horizon", 0) or _lsm_n_steps(px, row))
        if n_steps < 2:
            continue
        rec = row
        if hasattr(row, "_replace"):
            kwargs = {}
            if hasattr(row, "n_steps"):
                kwargs["n_steps"] = n_steps
            if kwargs:
                rec = row._replace(**kwargs)
        paths = g["_rn_paths_for_contract"](rec, N_PATHS, SEED + i)
        if paths.shape[1] - 1 > n_steps:
            paths = paths[:, : n_steps + 1]
        res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
        err = res.price - float(row.option_price)
        rows.append(
            {
                "ticker": ticker,
                "trading_date": row.trading_date,
                "evaluation_timestamp": getattr(row, "evaluation_timestamp", row.trading_date),
                "S_t": float(row.S_t),
                "K": float(row.K),
                "dte": int(row.dte),
                "DTE": int(row.dte),
                "expiration": str(getattr(row, "expiration", "")),
                "window_end": str(getattr(row, "window_end", "")),
                "remaining_horizon": int(n_steps),
                "n_steps": int(n_steps),
                "r": float(row.r),
                "market": float(row.option_price),
                "model_price": res.price,
                "error": err,
                "early_ex_frac": res.early_exercise_frac,
                "mean_ex_day": res.mean_exercise_step,
            }
        )
    t_lsm = time.time() - t_lsm0
    df = pd.DataFrame(rows)
    return {
        "df": df,
        "n_updates": int(len(cal)),
        "t_cal": float(t_cal),
        "t_lsm": float(t_lsm),
    }


def _fingerprint_intra(df: pd.DataFrame) -> list[dict]:
    cols = [
        c
        for c in (
            "trading_date",
            "evaluation_timestamp",
            "expiration",
            "K",
            "S_t",
            "dte",
            "r",
            "option_price",
            "window_end",
            "remaining_horizon",
            "n_steps",
        )
        if c in df.columns
    ]
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


def sample_shared_contracts() -> dict:
    if CONTRACTS_JSON.exists():
        return json.loads(CONTRACTS_JSON.read_text(encoding="utf-8"))
    print("  loading short_interval call panels …", flush=True)
    panels = {t: load_calls(os_study.DATA_ROOT, t, panel="short_interval") for t in TICKERS}
    g0 = _load_model_ns("GBM")
    payload: dict = {}
    for exp in EXPIRY_FRIDAYS:
        start, end, sessions = window_bounds(exp)
        stamps = hourly_stamps(start, end)
        payload[exp] = {
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "sessions": sessions,
            "n_stamps": int(len(stamps)),
            "tickers": {},
        }
        for ticker in TICKERS:
            px = g0["prices"][ticker]
            contracts = _sample_hourly_atm(panels[ticker], px, stamps, end)
            if contracts is None or len(contracts) == 0:
                raise RuntimeError(f"No shared contracts for {ticker} {exp}")
            contracts = contracts.copy()
            contracts["underlying"] = ticker
            payload[exp]["tickers"][ticker] = {
                "n": int(len(contracts)),
                "records": _fingerprint_intra(contracts),
            }
            print(
                f"  sample {exp} {ticker}: n={len(contracts)} "
                f"{pd.Timestamp(contracts['trading_date'].min())} → "
                f"{pd.Timestamp(contracts['trading_date'].max())}",
                flush=True,
            )
    CONTRACTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTRACTS_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _sample_hourly_atm(panel: pd.DataFrame, px: pd.Series, stamps: pd.DatetimeIndex, window_end) -> pd.DataFrame:
    px = pd.Series(px).dropna().sort_index()
    rows = []
    window_end = pd.Timestamp(window_end)
    for ts in stamps:
        ts = pd.Timestamp(ts)
        n_steps = remaining_n_steps(px, ts, window_end)
        if n_steps < 2:
            continue
        rec = _pick_atm(_quotes_asof(panel, ts))
        if rec is None:
            continue
        S = float(px.loc[ts]) if ts in px.index else float(px.asof(ts))
        if not np.isfinite(S) or S <= 0:
            continue
        row = rec.to_dict()
        row["trading_date"] = ts
        row["evaluation_timestamp"] = ts
        row["S_t"] = S
        row["window_end"] = window_end
        row["remaining_horizon"] = int(n_steps)
        row["n_steps"] = int(n_steps)
        row["moneyness"] = float(row["K"]) / S
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("trading_date").reset_index(drop=True)


def filter_funnel() -> dict:
    if FILTER_JSON.exists():
        return json.loads(FILTER_JSON.read_text(encoding="utf-8"))
    contracts = sample_shared_contracts()
    out: dict = {"tickers": {}, "note": "Same filters for every model. No model-specific data selection."}
    for ticker in TICKERS:
        panel = pd.read_csv(
            os_study.DATA_ROOT / "options" / "processed" / "short_interval" / f"{ticker}_calls_panel.csv",
            parse_dates=["trading_date", "expiration"],
        )
        t_block: dict = {}
        for exp in EXPIRY_FRIDAYS:
            end = pd.Timestamp(contracts[exp]["period_end"])
            idx = _px_index()
            past = idx[idx <= end]
            look_start = past[0] if len(past) <= LOOKBACK_BARS else past[-(LOOKBACK_BARS + 1)]
            sub = panel.loc[(panel["trading_date"] >= look_start) & (panel["trading_date"] <= end)].copy()
            filtered, log = apply_estimation_filters(sub, audit=True)
            steps = [{"step": r["step"], "n": int(r["n"])} for r in log]
            raw_sk = spot_over_strike(sub) if len(sub) else pd.Series(dtype=float)
            keep_sk = spot_over_strike(filtered) if len(filtered) else pd.Series(dtype=float)
            t_block[exp] = {
                "lookback_start": look_start.date().isoformat(),
                "period_end": end.date().isoformat(),
                "steps": steps,
                "n_eval_contracts": int(contracts[exp]["tickers"][ticker]["n"]),
                "moneyness_before": moneyness_bucket(raw_sk).value_counts(dropna=False).astype(int).to_dict(),
                "moneyness_after": moneyness_bucket(keep_sk).value_counts(dropna=False).astype(int).to_dict(),
            }
        out["tickers"][ticker] = t_block
    FILTER_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _partial_path(model: str, regime: str) -> Path:
    safe = model.replace("–", "-").replace(" ", "_")
    return CACHE / "partial" / f"{regime}_{safe}.json"


def _run_one(model: str, regime: str, shared: dict) -> dict:
    g = _load_model_ns(model)
    start, end, _ = window_bounds(regime)
    _set_period(g, start, end)
    out = {"model": model, "regime": regime, "stem": _nb_path(model).stem, "tickers": {}}
    for ticker in TICKERS:
        recs = shared[regime]["tickers"][ticker]["records"]
        contracts = emp._contracts_from_records(recs, ticker)
        for col in ("evaluation_timestamp", "window_end"):
            if col in contracts.columns:
                contracts[col] = pd.to_datetime(contracts[col])
        t0 = time.time()
        result = _run_ticker(g, ticker, contracts)
        mets = emp._metrics_from_df(result["df"])
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
        csv_path = fig_dir / f"{regime}_{MODEL_STEM[model]}.csv"
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
            best = min(rows, key=lambda r: r["rmse_pct"])
            n_set = {int(r["n"]) for r in rows}
            tables[f"{ticker}|{regime}"] = {
                "ticker": ticker,
                "regime": regime,
                "rows": rows,
                "best_model": best["model"],
                "n_contracts": int(rows[0]["n"]),
                "n_contracts_identical": len(n_set) == 1,
            }
    overall = []
    for model in TABLE_MODELS:
        vals = [c["rmse_pct"] for c in cells.values() if c["model"] == model]
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
                    "n_best": sum(1 for r in REGIME_ORDER if tables.get(f"{ticker}|{r}", {}).get("best_model") == model),
                }
            )
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
    n_by = {exp: {t: shared[exp]["tickers"][t]["n"] for t in TICKERS} for exp in EXPIRY_FRIDAYS}
    return {
        "meta": {
            "window_label": WINDOW_LABEL,
            "rolling": ROLLING_MODE,
            "n_paths": N_PATHS,
            "n_steps_cap": N_STEPS,
            "step_minutes": STEP_MINUTES,
            "seed": SEED,
            "models": list(TABLE_MODELS),
            "tickers": list(TICKERS),
            "regimes": REGIME_ORDER,
            "regime_meta": REGIME_META,
            "primary_metric": "percentage RMSE",
            "bias_definition": "model − market (dollars)",
            "bias_pct_definition": "100 × mean((model − market) / market)",
            "elapsed_sec": float(elapsed),
            "failures": list(failures),
            "study_kind": STUDY_KIND,
        },
        "shared_contracts": {
            exp: {
                "period_start": shared[exp]["period_start"][:10],
                "period_end": shared[exp]["period_end"][:10],
                "n": n_by[exp],
            }
            for exp in EXPIRY_FRIDAYS
        },
        "filter_funnel": funnel,
        "cells": cells,
        "tables": tables,
        "summary_overall": overall,
        "summary_by_regime": by_regime,
        "summary_by_ticker": by_ticker,
        "ranking_grid": ranking_grid,
    }


def run_study(only: list[str] | None = None) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "partial").mkdir(parents=True, exist_ok=True)
    configure_study(STUDY_KIND)
    print("Sampling shared contracts …", flush=True)
    shared = sample_shared_contracts()
    print("Filter funnel …", flush=True)
    funnel = filter_funnel()
    jobs = [(m, exp) for m in MODELS for exp in EXPIRY_FRIDAYS]
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
        for m, exp in jobs:
            hit = False
            for tok in tokens:
                if tok in model_alias and m == model_alias[tok]:
                    hit = True
                elif tok == exp.lower() or tok in exp.lower():
                    hit = True
            if hit:
                keep.append((m, exp))
        jobs = keep
    completed: list[dict] = []
    failures: list[str] = []
    t0 = time.time()
    for model, exp in jobs:
        part = _partial_path(model, exp)
        if part.exists():
            print(f"  resume {model} {exp}", flush=True)
            completed.append(json.loads(part.read_text(encoding="utf-8")))
            continue
        print(f"\n=== {model} | {exp} ===", flush=True)
        try:
            row = _run_one(model, exp, shared)
            part.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            completed.append(row)
        except Exception:
            failures.append(f"{model}/{exp}")
            print(f"FAILED {model} {exp}:\n{traceback.format_exc()}", flush=True)
    payload = assemble_payload(shared, funnel, completed, failures, time.time() - t0)
    PAYLOAD_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {PAYLOAD_JSON} in {(time.time()-t0)/60:.1f} min", flush=True)
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
    return payload


def load_payload() -> dict:
    return json.loads(PAYLOAD_JSON.read_text(encoding="utf-8"))


def run_or_load(*, recompute: bool = False, only: list[str] | None = None) -> dict:
    needed = {f"{t}|{r}|{m}" for t in TICKERS for r in REGIME_ORDER for m in TABLE_MODELS}
    if recompute or not PAYLOAD_JSON.exists():
        return run_study(only=only)
    payload = load_payload()
    if not needed.issubset(payload.get("cells", {})):
        return run_study(only=only)
    return payload


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _company_tables_page(pdf, payload: dict, ticker: str, regimes: list[str], page_no: int) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Page {page_no}  ·  Table block  ·  {ticker}  ·  six models × four expiry windows",
        fontsize=13,
        color=NAVY,
        y=0.975,
        weight="bold",
    )
    fig.text(
        0.06,
        0.935,
        f"Same contracts and LSM settings for every model.  Bold green row = lowest RMSE%.  "
        f"{LOOKBACK_PHRASE} lookback, hourly rolling, Δt = {STEP_MINUTES} min, n_paths={N_PATHS}, seed={SEED}.  "
        f"LSM horizon = remaining time to Friday 15:59 (not listed expiry).",
        fontsize=7.8,
        color=MUTED,
    )
    for i, regime in enumerate(regimes):
        ax = fig.add_subplot(2, 2, i + 1)
        tab = payload["tables"][f"{ticker}|{regime}"]
        rows, header, best = emp._metric_rows(tab)
        meta = REGIME_META[regime]
        k = REGIME_ORDER.index(regime) + 1
        title = (
            f"Table {ticker}-{k}   {ticker}  ·  {meta['title']}  listed expiry {regime}\n"
            f"{meta['window']}   ·   n = {tab['n_contracts']} contracts"
        )
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.4, color=NAVY, pad=8, weight="bold")
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.08, 0.96, 0.78])
        emp._style_table(tbl, best)
        tbl.auto_set_column_width(list(range(len(header))))
    fig.text(
        0.06,
        0.03,
        "RMSE% is 100×√mean(((model−market)/market)²).  MAE and Bias $ are in option-premium dollars.  "
        "Bias% is 100×mean((model−market)/market).  Early-ex. frac. is the mean LSM share of paths exercised before the window end.  "
        "Bias > 0 means the model is expensive vs the quote.",
        fontsize=7.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    import run_v3_intraday_hourly_reports as reports
    return reports.write_pdf(payload, path)


def build_notebook(payload: dict, path: Path | None = None) -> Path:
    import run_v3_intraday_hourly_reports as reports
    return reports.build_notebook(payload, path)


def write_outputs(payload: dict):
    payload = emp.enrich_bias_pct(payload)
    PAYLOAD_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pdf = write_pdf(payload)
    nb = build_notebook(payload)
    return nb, pdf


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    recompute = "--recompute" in argv
    assemble_only = "--assemble" in argv
    only = [a for a in argv if not a.startswith("--")]
    n_needed = len(TICKERS) * len(REGIME_ORDER) * len(TABLE_MODELS)
    if assemble_only:
        shared = sample_shared_contracts()
        funnel = filter_funnel()
        completed = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((CACHE / "partial").glob("*.json"))]
        payload = emp.enrich_bias_pct(assemble_payload(shared, funnel, completed, [], 0.0))
        PAYLOAD_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"assembled {len(payload.get('cells', {}))}/{n_needed} cells", flush=True)
        if len(payload.get("cells", {})) == n_needed:
            write_outputs(payload)
            return 0
        return 1
    payload = run_or_load(recompute=recompute, only=only or None)
    if len(payload.get("cells", {})) == n_needed and not payload["meta"]["failures"]:
        write_outputs(payload)
        return 0
    print(
        f"Incomplete: {len(payload.get('cells', {}))}/{n_needed} cells, "
        f"failures={payload['meta']['failures']}. PDF/notebook not finalized.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    configure_study("7d")
    raise SystemExit(main())
