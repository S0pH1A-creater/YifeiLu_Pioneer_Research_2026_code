#!/usr/bin/env python3
"""Headless 7-day 1-minute study (V2) for a dated window.

Default window: 2022-10-17 → 2022-10-21 (5 RTH sessions ending monthly expiry Friday; lookback 2022-10-14).
Rolling: hourly / minutely. Lookback: 1 day.

Writes:
  V3-Models_result/results/7d_1min/<window_id>/
  Results_In_Short/7 days regimes/<window_id>/   (parameters, stopping, ALL_TABLES, RMSE bars)

Listed 2022–2023 American-call panels live in
research/data/options/processed/short_interval/. §6 uses the same 24-contract /
seed-42 / listed-expiry / LSM-vs-market workflow as the 2-year notebooks.
Quote times are unique RTH minutes. Same contracts/seeds across models. Prices
are stitched to continuous RTH (overnight/weekend straight lines removed).
§5 notebooks also report RMSE(S_t).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA_ROOT = REPO / "research" / "data"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_parameter_estimation_study as pest  # noqa: E402
from american_lsm import (  # noqa: E402
    lsm_american_call,
    load_calls,
    params_asof,
    pct_rmse,
    sample_listed_minute_calls,
    stitch_continuous,
    session_gap,
)

WINDOW_ID = "2022-10-17_to_2022-10-21"
WINDOW_LABEL = "1 day"  # 390 trading minutes after notebook patch
ROLLING_MODES = ("hourly", "minutely")
EXPIRY = pd.Timestamp("2022-10-21 15:59:00")
REGIME = WINDOW_ID
N_PATHS = 2000
SEED = 42
DTE_DAYS = 2
TICKERS = ("SPY", "AAPL", "MSFT")
COLORS = {"AAPL": "#1f77b4", "MSFT": "#ff7f0e", "SPY": "#2ca02c"}
ROLL_COLORS = {"hourly": "#4C72B0", "minutely": "#C44E52"}
MODEL_ORDER = {"GBM": 0, "Merton": 1, "Heston": 2, "Heston–Merton": 3, "GARCH": 4, "GARCH–Merton": 5}
STUDIES = [
    ("GBM", "gbm notebook", "7d_1min_gbm.ipynb", "gbm"),
    ("Merton", "merton notebook", "7d_1min_merton.ipynb", "merton"),
    ("Heston", "heston notebook", "7d_1min_heston.ipynb", "heston"),
    ("Heston–Merton", "heston merton notebook", "7d_1min_heston_merton.ipynb", "heston_merton"),
    ("GARCH", "garch notebook", "7d_1min_garch.ipynb", "garch"),
    ("GARCH–Merton", "garch merton notebook", "7d_1min_garch_merton.ipynb", "garch_merton"),
]

RESULTS = ROOT / "results" / "7d_1min" / WINDOW_ID
SHORT = REPO / "Results_In_Short" / "V3" / "7 days regimes" / WINDOW_ID
RATES_PATH = DATA_ROOT / "rates" / "risk_free_dgs3mo_short_interval.csv"
INTRADAY_DIR = DATA_ROOT / "equity" / "short_interval" / "prices_1min_rth"


def _md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    source = [ln + "\n" for ln in lines[:-1]]
    if lines[-1] != "":
        source.append(lines[-1] + "\n")
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}


def _write_nb(path: Path, cells: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _load_ns(nb_path: Path) -> dict:
    """Load 7d notebook namespace; keep the 1-minute prices from the setup cell."""
    nb_dir = nb_path.parent
    os.chdir(nb_dir)
    nb = json.loads(nb_path.read_text())
    code_like = []
    for c in nb["cells"]:
        src = pest._cell_source(c)
        if c["cell_type"] == "code" or "def calibrate_ticker" in src or "def _rn_paths_for_contract" in src:
            code_like.append(src)

    setup = next(s for s in code_like if "DATA = Path" in s and "PERIOD_START" in s)
    cal = next(s for s in code_like if "def calibrate_ticker" in s)
    sim = next((s for s in code_like if "def simulate_" in s and "def calibrate_ticker" not in s), "")
    stop = next((s for s in code_like if "def _rn_paths_for_contract" in s), "")
    if not stop:
        raise RuntimeError(f"Missing _rn_paths_for_contract in {nb_path.name}")

    g: dict = {"__name__": "__main__", "Path": Path, "np": np, "pd": pd, "plt": plt}
    exec("from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n", g)
    exec(pest._extract_defs(setup, keep_assigns=False), g)
    if "_with_option_days" not in g:
        def _with_option_days(fn, *args, **kwargs):
            saved = g["N_DAYS"]
            g["N_DAYS"] = 252
            try:
                return fn(*args, **kwargs)
            finally:
                g["N_DAYS"] = saved

        g["_with_option_days"] = _with_option_days
    g.update(
        {
            "DATA": DATA_ROOT,
            "TICKERS": ["AAPL", "MSFT", "SPY"],
            "BARS_PER_DAY": 390,
            "N_DAYS": 252 * 390,
            "N_STEPS": 5000,
            "COLORS": dict(COLORS),
            "WINDOW_OPTIONS": {"1 hour": 60, "1 day": 390, "7 days": 7 * 390},
            "ROLLING_OPTIONS": ["minutely", "hourly"],
            "WINDOW_ID": WINDOW_ID,
            "JUMP_THRESH": 3.0,
            "MIN_WINDOW": 60,
            "GAP_MIN": pd.Timedelta(minutes=2),
        }
    )
    intra = INTRADAY_DIR
    frames = []
    for t in g["TICKERS"]:
        p = pd.read_csv(intra / f"{t}.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
        frames.append(p["Close"].rename(t))
    prices_raw = pd.concat(frames, axis=1).sort_index()
    stitch = g.get("stitch_continuous", stitch_continuous)
    prices = pd.concat(
        [stitch(prices_raw[t]).rename(t) for t in g["TICKERS"]],
        axis=1,
    ).sort_index()
    g["prices"] = prices
    g["PERIOD_START"] = pd.Timestamp("2022-10-17 09:30:00")
    g["PERIOD_END"] = pd.Timestamp("2022-10-21 15:59:00")
    g["period_prices"] = prices.loc[g["PERIOD_START"] : g["PERIOD_END"], list(g["TICKERS"])].copy()
    log_returns_all = np.log(prices[list(g["TICKERS"])]).diff()
    gap_fn = g.get("_session_gap", session_gap)
    for t in g["TICKERS"]:
        gg = gap_fn(prices[t].dropna().index)
        log_returns_all.loc[gg.reindex(log_returns_all.index, fill_value=False), t] = np.nan
    g["log_returns_all"] = log_returns_all
    g["rolling"] = {}
    g["cal_meta"] = {}
    exec(pest._extract_defs(cal, keep_assigns=True), g)
    if sim:
        exec(pest._extract_defs(sim, keep_assigns=False), g)
    g.update(
        {
            "lsm_american_call": lsm_american_call,
            "params_asof": params_asof,
            "sys": sys,
        }
    )
    exec(pest._extract_defs(stop, keep_assigns=False), g)
    if "calibrate_ticker" not in g or "_rn_paths_for_contract" not in g:
        raise RuntimeError(f"Missing calibrate/_rn_paths in {nb_path.name}")
    return g


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))


def _load_rates() -> pd.Series:
    r = pd.read_csv(RATES_PATH, parse_dates=["observation_date"])
    s = r.set_index("observation_date")["DGS3MO"].astype(float) / 100.0
    return s.sort_index()


def make_listed_minute_calls(g: dict, ticker: str) -> pd.DataFrame:
    """2-year sample_calls + unique 1-minute quote times; listed/natural expiry."""
    df = sample_listed_minute_calls(
        load_calls(DATA_ROOT, ticker, panel="short_interval"),
        g["prices"][ticker],
        g["PERIOD_START"],
        g["PERIOD_END"],
    )
    if df.empty:
        raise RuntimeError(
            f"No listed contracts for {ticker} in {g['PERIOD_START']} → {g['PERIOD_END']}"
        )
    if not df["trading_date"].is_unique:
        raise RuntimeError(f"{ticker}: duplicate trading minutes in §6 sample")
    minute = df["trading_date"].dt.floor("min")
    if not (minute == df["trading_date"]).all():
        raise RuntimeError(f"{ticker}: §6 quote times are not on the 1-minute grid")
    return df.reset_index(drop=True)


def _fmt_table_md(df: pd.DataFrame, model_key: str) -> str:
    show = df.copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d %H:%M")
    param_cols = pest._param_cols(show, model_key)
    for c in param_cols:
        show[c] = show[c].map(lambda x: f"{float(x):.4g}")
    headers = ["date", "ticker"] + param_cols
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["------" if h in ("date", "ticker") else "------:" for h in headers]) + "|",
    ]
    for r in show.itertuples(index=False):
        row = {h: getattr(r, h) for h in headers}
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _short_table(full: pd.DataFrame, rolling_mode: str, model_key: str) -> pd.DataFrame:
    cols = ["date", "ticker"] + pest._param_cols(full, model_key)
    df = full[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    if rolling_mode == "hourly":
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)
    keys = pest._pick_daily_key_dates(df["date"], n_keep=10)
    return df[df["date"].isin(keys)].sort_values(["date", "ticker"]).reset_index(drop=True)


def run_params(nb_path: Path, model: str, model_key: str, g: dict) -> list[dict]:
    stem = nb_path.stem
    metas = []
    for mode in ROLLING_MODES:
        t0 = time.time()
        rolling = {t: g["calibrate_ticker"](t, WINDOW_LABEL, mode) for t in TICKERS}
        out_dir = RESULTS / "parameters" / mode / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for t in TICKERS:
            d = rolling[t].copy()
            d.insert(0, "ticker", t)
            parts.append(d)
        full = pd.concat(parts, ignore_index=True)
        full.to_csv(out_dir / "params.csv", index=False)
        short = _short_table(full, mode, model_key)
        short.to_csv(out_dir / "params_short.csv", index=False)
        pest.WINDOW_LABEL = WINDOW_LABEL
        pest._save_panel(rolling, model_key, model, REGIME, mode, out_dir / "panel.png")

        short_dir = SHORT / "parameters" / mode / stem
        short_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / "panel.png", short_dir / "panel.png")
        shutil.copy2(out_dir / "params_short.csv", short_dir / "params_short.csv")

        n_updates = {t: int(len(rolling[t])) for t in TICKERS}
        print(
            f"  params {mode}: updates={n_updates} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        metas.append(
            {
                "model": model,
                "model_key": model_key,
                "regime": REGIME,
                "stem": stem,
                "rolling": mode,
                "n_updates": n_updates,
                "panel_rel": f"parameters/{mode}/{stem}/panel.png",
                "table_md": _fmt_table_md(short, model_key),
            }
        )
    return metas


def _save_stop_figs(model: str, mode: str, ticker: str, out: dict, fig_dir: Path) -> tuple[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = out["df"]
    color = COLORS.get(ticker, "#2ca02c")
    fig = Figure(figsize=(13.5, 4.0))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    ax = axes[0]
    ax.scatter(df["market"], df["model_price"], alpha=0.75, color=color)
    lo = min(df["market"].min(), df["model_price"].min())
    hi = max(df["market"].max(), df["model_price"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("market option_price")
    ax.set_ylabel("model LSM price")
    ax.set_title("Price: model vs market")
    axes[1].bar(["model", "market"], [df["model_price"].mean(), df["market"].mean()], color=[color, "#7f7f7f"])
    axes[1].set_title("Mean option value")
    axes[1].set_ylabel("price")
    axes[2].hist(df["mean_ex_day"], bins=12, color=color, alpha=0.85, edgecolor="white")
    axes[2].set_xlabel("mean exercise step (by contract)")
    axes[2].set_title("Optimal exercise timing")
    fig.suptitle(
        f"{model} | {ticker} | {REGIME} | rolling={mode} | lookback={WINDOW_LABEL}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    panel_name = f"{mode}_panel.png"
    fig.savefig(fig_dir / panel_name, dpi=110, bbox_inches="tight")
    plt.close(fig)

    row, paths, res = out["example"]
    j = int(np.argmin(np.abs(res.exercise_steps - res.mean_exercise_step)))
    t_ex = int(res.exercise_steps[j])
    fig2 = Figure(figsize=(10, 3.8))
    FigureCanvasAgg(fig2)
    ax = fig2.subplots()
    ax.plot(paths[j], color=color, lw=1.5, label="one RN path")
    ax.axhline(float(row.K), color="gray", ls="--", lw=1, label=f"K={row.K:g}")
    ax.scatter([t_ex], [paths[j, t_ex]], color="crimson", zorder=5, s=50, label=f"exercise step {t_ex}")
    ax.set_xlabel("step (trading day)")
    ax.set_ylabel("S")
    ax.set_title(
        f"{ticker} example path | {pd.Timestamp(row.trading_date)} | "
        f"dte={int(row.dte)}d | model={res.price:.3f} vs mkt={float(row.option_price):.3f}"
    )
    ax.legend(frameon=False, loc="best")
    fig2.tight_layout()
    line_name = f"{mode}_path.png"
    fig2.savefig(fig_dir / line_name, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    return panel_name, line_name


def run_stopping(nb_path: Path, model: str, g: dict, contracts_by: dict) -> dict[str, tuple]:
    stem = nb_path.stem
    out: dict[str, tuple] = {}
    dt = 1.0 / 252
    for ticker in TICKERS:
        contracts = contracts_by[ticker]
        fig_dir = RESULTS / ticker / "figures" / stem
        modes: dict[str, dict] = {}
        for mode in ROLLING_MODES:
            t0 = time.time()
            cal = g["calibrate_ticker"](ticker, WINDOW_LABEL, mode)
            g["rolling"] = {ticker: cal}
            g["cal_meta"] = {"window_label": WINDOW_LABEL, "rolling_mode": mode}
            rows = []
            example = None
            for i, row in enumerate(contracts.itertuples(index=False)):
                paths = g["_rn_paths_for_contract"](row, N_PATHS, SEED + i)
                res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
                err = res.price - float(row.option_price)
                rows.append(
                    {
                        "ticker": ticker,
                        "trading_date": row.trading_date,
                        "S_t": float(row.S_t),
                        "K": float(row.K),
                        "dte": int(row.dte),
                        "r": float(row.r),
                        "market": float(row.option_price),
                        "model_price": res.price,
                        "error": err,
                        "early_ex_frac": res.early_exercise_frac,
                        "mean_ex_day": res.mean_exercise_step,
                    }
                )
                if example is None:
                    example = (row, paths, res)
            df = pd.DataFrame(rows)
            packed = {
                "df": df,
                "rmse": pct_rmse(df["model_price"], df["market"]),
                "mae": float(np.mean(np.abs(df["error"]))),
                "bias": float(np.mean(df["error"])),
                "early": float(df["early_ex_frac"].mean()),
                "n_updates": int(len(cal)),
                "example": example,
            }
            panel, line = _save_stop_figs(model, mode, ticker, packed, fig_dir)
            packed["panel"], packed["line"] = panel, line
            df.to_csv(fig_dir / f"{mode}_contracts.csv", index=False)
            short_fig = SHORT / "results" / ticker / "figures" / stem
            short_fig.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig_dir / panel, short_fig / panel)
            shutil.copy2(fig_dir / line, short_fig / line)
            modes[mode] = packed
            print(
                f"    stop {ticker}/{mode}: RMSE%={packed['rmse']:.2f}% "
                f"updates={packed['n_updates']} ({time.time()-t0:.1f}s)",
                flush=True,
            )
        slim = {
            m: {k: modes[m][k] for k in ("rmse", "mae", "bias", "early", "n_updates")}
            for m in ROLLING_MODES
        }
        out[ticker] = (model, REGIME, stem, slim)
    return out


def build_param_notebooks(metas: list[dict]) -> None:
    for mode in ROLLING_MODES:
        items = [m for m in metas if m["rolling"] == mode]
        items = sorted(items, key=lambda m: MODEL_ORDER.get(m["model"], 9))
        note = (
            "every hourly update × all tickers"
            if mode == "hourly"
            else "key timestamps only (~10 evenly spaced, incl. first & last)"
        )
        cells = [
            _md_cell(
                f"""# V2 Parameter estimation — 7-day 1-minute — rolling = `{mode}`

**Window:** `{WINDOW_ID}` (calendar 2022-10-17 → 2022-10-21; 5 RTH sessions).  
**Source:** `V3-Models_result/results/7d_1min/{WINDOW_ID}/parameters/{mode}/`  
Lookback fixed at **1 day**. Parameters estimated **separately** for AAPL / MSFT / SPY (same graph).

| Fixed | Value |
|-------|-------|
| Window | `{WINDOW_ID}` |
| Lookback | 1 day |
| Rolling | `{mode}` only |
| Grid | 1-minute RTH bars |
| Underlyings | AAPL, MSFT, SPY |

## Layout
6 studies: GBM → Merton → Heston → Heston–Merton → GARCH → GARCH–Merton.  
Each block: parameter graph + short value table ({note}).
"""
            )
        ]
        n = len(items)
        for i, m in enumerate(items, start=1):
            n_up = m["n_updates"]
            cells.append(
                _md_cell(
                    f"""---

## Study {i}/{n} — {m['model']} · {m['regime']}

> **model=`{m['model']}` · regime=`{m['regime']}` · rolling=`{mode}` · lookback=1 day**

Updates: AAPL={n_up['AAPL']}, MSFT={n_up['MSFT']}, SPY={n_up['SPY']}

### Parameter graph

![params {m['stem']} {mode}]({m['panel_rel']})

### Parameter values

{m['table_md']}
"""
                )
            )
        out = _write_nb(SHORT / f"parameters_rolling_{mode}.ipynb", cells)
        print(f"  wrote {out.relative_to(REPO)}", flush=True)


def build_compare_notebooks(by_ticker: dict) -> None:
    for mode in ROLLING_MODES:
        cells = [
            _md_cell(
                f"""# V2 Optimal stopping comparison — 7-day 1-minute — rolling = `{mode}`

**Window:** `{WINDOW_ID}`.  
**Source:** `V3-Models_result/results/7d_1min/{WINDOW_ID}/{{TICKER}}/`  
Same systematic 15-minute sample / LSM harness for **SPY**, **AAPL**, and **MSFT** (n_paths={N_PATHS}; DTE 7–60 from the short_interval panel).

**Percentage RMSE is model LSM vs market `option_price`**, same decision metric as the 2-year notebooks.

| Fixed | Value |
|-------|-------|
| Window | `{WINDOW_ID}` |
| Lookback | 1 day |
| Rolling | `{mode}` only |
| Underlyings | SPY, AAPL, MSFT |

## Layout
For each ticker: GBM → Merton → Heston → Heston–Merton → GARCH → GARCH–Merton.  
Each block: label + RMSE + three-panel graph.
"""
            )
        ]
        for ticker in TICKERS:
            completed = sorted(by_ticker.get(ticker, []), key=lambda r: MODEL_ORDER.get(r[0], 9))
            cells.append(_md_cell(f"# Underlying: `{ticker}`\n"))
            table = [
                f"## Quick RMSE table — `{ticker}` · `{mode}`",
                "",
                "| # | Regime | Model | RMSE | MAE | Bias |",
                "|---|--------|-------|-----:|----:|-----:|",
            ]
            for i, (model, regime, stem, modes) in enumerate(completed, start=1):
                m = modes[mode]
                table.append(
                    f"| {i} | {regime} | {model} | **{m['rmse']:.2f}%** | {m['mae']:.4f} | {m['bias']:.4f} |"
                )
            cells.append(_md_cell("\n".join(table)))
            n = len(completed)
            for i, (model, regime, stem, modes) in enumerate(completed, start=1):
                m = modes[mode]
                fig_rel = f"results/{ticker}/figures/{stem}/{mode}_panel.png"
                cells.append(
                    _md_cell(
                        f"""---

## Study {i}/{n} — {model} · {regime} · {ticker}

> **underlying=`{ticker}` · model=`{model}` · regime=`{regime}` · rolling=`{mode}` · lookback=1 day**

| Metric | Value |
|--------|------:|
| **RMSE%** | **{m['rmse']:.2f}%** |
| MAE | {m['mae']:.4f} |
| Bias (model − market) | {m['bias']:.4f} |
| Mean early-exercise frac | {m['early']:.3f} |

### Three-panel graph — {model} · {regime} · {ticker} · `{mode}`

![Study {i} {ticker}]({fig_rel})
"""
                    )
                )
        out = _write_nb(SHORT / f"compare_rolling_{mode}.ipynb", cells)
        print(f"  wrote {out.relative_to(REPO)}", flush=True)


def build_all_tables(by_ticker: dict) -> dict[str, list[dict]]:
    """Return parsed rows and write ALL_TABLES.ipynb + PDF."""
    parsed: dict[str, list[dict]] = {}
    cells = []
    for ticker in TICKERS:
        completed = sorted(by_ticker.get(ticker, []), key=lambda r: MODEL_ORDER.get(r[0], 9))
        rows = []
        lines = [
            f"### {ticker} — RMSE index (hourly / minutely)",
            "",
            "| Regime | Model | hourly | minutely | best rolling | file |",
            "|--------|-------|-------:|---------:|--------------|------|",
        ]
        for model, regime, stem, modes in completed:
            best = min(ROLLING_MODES, key=lambda k: modes[k]["rmse"])
            lines.append(
                f"| {regime} | {model} | {modes['hourly']['rmse']:.2f}% | "
                f"{modes['minutely']['rmse']:.2f}% | `{best}` | [{stem}.md]({stem}.md) |"
            )
            rows.append(
                {
                    "Regime": regime,
                    "Model": model,
                    "hourly": modes["hourly"]["rmse"],
                    "minutely": modes["minutely"]["rmse"],
                    "best rolling": best,
                    "file": f"{stem}.md",
                }
            )
        parsed[ticker] = rows
        cells.append(_md_cell("\n".join(lines)))

    intro = _md_cell(
        f"""# RMSE index — 7-day 1-minute `{WINDOW_ID}`

Calendar week **2022-10-17 → 2022-10-21** (RTH: 17–21 Oct; monthly expiry Friday).  
Lookback **1 day**. Rolling **hourly** / **minutely**.

RMSE = LSM American price vs listed market `option_price` (short_interval panel, same filters as the 2-year files).
"""
    )
    _write_nb(SHORT / "ALL_TABLES.ipynb", [intro] + cells)

    fig = Figure(figsize=(11, 9.5))
    FigureCanvasAgg(fig)
    fig.suptitle(f"RMSE index — 7-day 1-minute  {WINDOW_ID}", fontsize=13, y=0.98)
    for i, ticker in enumerate(TICKERS):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.axis("off")
        ax.set_title(ticker, loc="left", fontsize=11, pad=8)
        rows = parsed[ticker]
        col_labels = ["Regime", "Model", "hourly", "minutely", "best rolling"]
        cell = [[r[c] if c in ("Regime", "Model", "best rolling") else f"{r[c]:.4f}" for c in col_labels] for r in rows]
        tbl = ax.table(cellText=cell, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(SHORT / "ALL_TABLES.pdf", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote { (SHORT / 'ALL_TABLES.pdf').relative_to(REPO) }", flush=True)
    return parsed


def build_rmse_bars(parsed: dict[str, list[dict]]) -> None:
    models = ["GBM", "Merton", "Heston", "Heston–Merton", "GARCH", "GARCH–Merton"]
    x = np.arange(len(models))
    width = 0.35

    def plot_one(ax, ticker: str) -> None:
        by_model = {r["Model"]: r for r in parsed[ticker]}
        h = [by_model[m]["hourly"] for m in models if m in by_model]
        u = [by_model[m]["minutely"] for m in models if m in by_model]
        labels = [m for m in models if m in by_model]
        xloc = np.arange(len(labels))
        ax.bar(xloc - width / 2, h, width, label="hourly", color=ROLL_COLORS["hourly"])
        ax.bar(xloc + width / 2, u, width, label="minutely", color=ROLL_COLORS["minutely"])
        ax.set_xticks(xloc)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE (%)")
        ax.set_title(f"{ticker} — percentage RMSE by model · {WINDOW_ID}")
        ax.legend(frameon=False)
        ax.set_xlabel("Model")

    # notebook
    cells = [
        _md_cell(
            f"""# RMSE bar charts — 7-day 1-minute `{WINDOW_ID}`

Data source: the three company summary tables in `ALL_TABLES.ipynb` (SPY, AAPL, MSFT) — one window × models × rolling methods (hourly / minutely).

Three charts: SPY, then AAPL, then MSFT. Each chart: models on the x-axis; two colored bars per model for the rolling methods.
"""
        )
    ]
    _write_nb(SHORT / "RMSE_BAR_CHARTS.ipynb", cells)

    pdf_path = SHORT / "RMSE_BAR_CHARTS.pdf"
    with PdfPages(pdf_path) as pdf:
        for ticker in TICKERS:
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            plot_one(ax, ticker)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(SHORT / f"RMSE_BAR_{ticker}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    print(f"  wrote {pdf_path.relative_to(REPO)}", flush=True)


def write_readme() -> None:
    (SHORT / "README.md").write_text(
        f"""# 7-day regime — `{WINDOW_ID}`

Calendar week **17–21 Oct 2022** (monthly expiry Friday 21 Oct). Regular sessions: **17, 18, 19, 20, 21 Oct**. Lookback buffer **14 Oct**.

Same Results_In_Short layout as the 2-year regimes, with rolling methods that match the 1-minute notebooks:

| Daily-regime file | This folder |
|-------------------|-------------|
| `parameters_rolling_{{none,monthly,daily}}.ipynb` | `parameters_rolling_{{hourly,minutely}}.ipynb` |
| `compare_rolling_{{none,monthly,daily}}.ipynb` | `compare_rolling_{{hourly,minutely}}.ipynb` |
| `ALL_TABLES.pdf` / `.ipynb` | same names |
| `RMSE_BAR_CHARTS.pdf` / `.ipynb` | same names |

1-minute bars: `research/data/equity/short_interval/prices_1min_rth/`.  
Listed calls: `research/data/options/processed/short_interval/`.  
Canonical numbers/figures: `V3-Models_result/results/7d_1min/{WINDOW_ID}/`.

Stopping RMSE uses listed market `option_price` (same as the 2-year notebooks).
""",
        encoding="utf-8",
    )
    parent = SHORT.parent / "README.md"
    if not parent.exists():
        parent.write_text(
            """# 7 days regimes

Dated 7-day 1-minute windows cut from `research/data/equity/intraday/source_1min/` (Sep 2022 – Sep 2023).

Each subfolder is named `YYYY-MM-DD_to_YYYY-MM-DD` so later weeks can be added beside this one.
""",
            encoding="utf-8",
        )


def main() -> int:
    pest._install_notebook_stubs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    SHORT.mkdir(parents=True, exist_ok=True)
    write_readme()

    all_metas: list[dict] = []
    by_ticker: dict[str, list] = {t: [] for t in TICKERS}
    failures: list[str] = []
    t0 = time.time()

    contracts_cache: dict[str, pd.DataFrame] | None = None
    for model, folder, nb_name, key in STUDIES:
        nb = ROOT / folder / nb_name
        print(f"\n=== {model} | {REGIME} ({nb.name}) ===", flush=True)
        try:
            g = _load_ns(nb)
            if contracts_cache is None:
                contracts_cache = {t: make_listed_minute_calls(g, t) for t in TICKERS}
                for t, df in contracts_cache.items():
                    exps = sorted({pd.Timestamp(x).normalize().date() for x in df["expiration"]})
                    print(
                        f"  {t}: {len(df)} contracts | unique minutes={df['trading_date'].nunique()} "
                        f"| listed expiries={exps}",
                        flush=True,
                    )
                    df.to_csv(RESULTS / f"synthetic_calls_{t}.csv", index=False)
            all_metas.extend(run_params(nb, model, key, g))
            per = run_stopping(nb, model, g, contracts_cache)
            for t, row in per.items():
                by_ticker[t].append(row)
        except Exception:
            failures.append(nb.name)
            print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)

    if all_metas:
        build_param_notebooks(all_metas)
        (RESULTS / "parameters" / "summary.json").write_text(json.dumps(all_metas, indent=2), encoding="utf-8")
    if any(by_ticker.values()):
        for t in TICKERS:
            (RESULTS / t).mkdir(parents=True, exist_ok=True)
            payload = [
                {"model": m, "regime": r, "stem": s, "modes": modes}
                for m, r, s, modes in by_ticker[t]
            ]
            (RESULTS / t / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        build_compare_notebooks(by_ticker)
        parsed = build_all_tables(by_ticker)
        if all(len(by_ticker[t]) >= 4 for t in TICKERS):
            build_rmse_bars(parsed)
        else:
            print("RMSE bars skipped (not all 4 models finished)", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min → {SHORT}", flush=True)
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
