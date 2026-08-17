#!/usr/bin/env python3
"""P-measure stock-path accuracy vs realized S_t for the hourly 7d/1d studies.

One rolling cloud per (ticker × window × model) on the evaluation clock,
Friday 15:59 close. Drift μ is kept (not replaced by r). Reported path = p50.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
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

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
import run_v3_intraday_hourly_empirical_study as intra  # noqa: E402
from american_lsm import pct_rmse  # noqa: E402

NAVY = emp.NAVY
MUTED = emp.MUTED
MODEL_COLORS = emp.MODEL_COLORS


def _stock_paths() -> tuple[Path, Path, Path]:
    cache = intra.CACHE / "stock"
    short = intra.SHORT
    if intra.STUDY_KIND == "7d":
        pdf = short / "V3_7d_hourly_stock_price.pdf"
        nb = short / "V3_7d_hourly_stock_price.ipynb"
    else:
        pdf = short / "V3_1d_hourly_stock_price.pdf"
        nb = short / "V3_1d_hourly_stock_price.ipynb"
    return cache, pdf, nb


def _simulate_p_measure(g: dict, ticker: str, cal: pd.DataFrame):
    sched = g["param_schedule_for_steps"](ticker, cal)
    n_paths = intra.N_PATHS
    seed = intra.SEED
    if "simulate_gbm_rolling" in g:
        _dates, mu, sig, S0, hist = sched
        paths = g["simulate_gbm_rolling"](mu, sig, S0, n_paths, seed)
    elif "simulate_garch_merton_rolling" in g:
        _dates, steps, S0, hist = sched
        paths = g["simulate_garch_merton_rolling"](steps, S0, n_paths, seed)
    elif "simulate_garch_rolling" in g:
        _dates, steps, S0, hist = sched
        paths = g["simulate_garch_rolling"](steps, S0, n_paths, seed)
    elif "simulate_heston_merton_rolling" in g:
        (
            _dates,
            mu,
            kappa,
            theta,
            xi,
            rho,
            v0,
            lam,
            muj,
            sj,
            kapj,
            S0,
            hist,
        ) = sched
        paths = g["simulate_heston_merton_rolling"](
            mu, kappa, theta, xi, rho, v0, lam, muj, sj, kapj, S0, n_paths, seed
        )
    elif "simulate_heston_rolling" in g:
        _dates, mu, kappa, theta, xi, rho, v0, S0, hist = sched
        paths = g["simulate_heston_rolling"](mu, kappa, theta, xi, rho, v0, S0, n_paths, seed)
    else:
        _dates, mu, sig, lam, muj, sj, kap, S0, hist = sched
        paths = g["simulate_merton_rolling"](mu, sig, lam, muj, sj, kap, S0, n_paths, seed)
    return paths, hist


def _stock_metrics(paths, hist) -> dict:
    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    hist_v = np.asarray(hist.values if hasattr(hist, "values") else hist, dtype=float)
    n = min(len(p50), len(hist_v))
    p10, p25, p50, p75, p90, hist_v = p10[:n], p25[:n], p50[:n], p75[:n], p90[:n], hist_v[:n]
    rel = (p50 - hist_v) / np.maximum(np.abs(hist_v), 1e-12)
    return {
        "rmse_pct": float(pct_rmse(p50, hist_v)),
        "mae": float(np.mean(np.abs(p50 - hist_v))),
        "bias": float(np.mean(p50 - hist_v)),
        "bias_pct": float(100.0 * np.mean(rel)),
        "icp": float(np.mean((hist_v >= p25) & (hist_v <= p75))),
        "abw": float(np.mean(p75 - p25)),
        "icp_10_90": float(np.mean((hist_v >= p10) & (hist_v <= p90))),
        "n": int(n),
        "early": float("nan"),
    }


def _metric_rows(tab: dict) -> tuple[list[list[str]], list[str], int]:
    header = ["Model", "RMSE%", "MAE", "Bias $", "Bias%", "ICP 25–75", "ABW", "Mark"]
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
                f"{100.0 * rec['icp']:.1f}%",
                f"{rec['abw']:.4f}",
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


def _partial_path(model: str, regime: str) -> Path:
    cache, _, _ = _stock_paths()
    safe = model.replace("–", "-").replace(" ", "_")
    return cache / "partial" / f"{regime}_{safe}.json"


def _run_one(model: str, regime: str) -> dict:
    g = intra._load_model_ns(model)
    start, end, _ = intra.window_bounds(regime)
    intra._set_period(g, start, end)
    clock = intra.eval_clock(start, end)
    px = g["prices"].reindex(clock)
    g["period_prices"] = px
    g["N_STEPS"] = max(int(len(clock) - 1), 2)
    out = {"model": model, "regime": regime, "tickers": {}}
    cache, _, _ = _stock_paths()
    for ticker in intra.TICKERS:
        t0 = time.time()
        cal = g["calibrate_ticker"](ticker, intra.WINDOW_LABEL, intra.ROLLING_MODE)
        g["rolling"] = {ticker: cal}
        paths, hist = _simulate_p_measure(g, ticker, cal)
        mets = _stock_metrics(paths, hist)
        mets["n_updates"] = int(len(cal))
        mets["elapsed"] = float(time.time() - t0)
        fig_dir = cache / "series" / regime / ticker
        fig_dir.mkdir(parents=True, exist_ok=True)
        csv_path = fig_dir / f"{regime}_{intra.MODEL_STEM[model]}.csv"
        hist_v = np.asarray(hist.values if hasattr(hist, "values") else hist, dtype=float)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        n = min(len(p50), len(hist_v), len(clock))
        pd.DataFrame(
            {
                "timestamp": list(clock[:n]),
                "S_t": hist_v[:n],
                "p25": p25[:n],
                "p50": p50[:n],
                "p75": p75[:n],
            }
        ).to_csv(csv_path, index=False)
        mets["series_csv"] = str(csv_path.relative_to(cache))
        out["tickers"][ticker] = mets
        print(
            f"    {model:16s} {regime} {ticker}: RMSE%={mets['rmse_pct']:.2f} "
            f"MAE={mets['mae']:.4f} ICP={100*mets['icp']:.1f}% n={mets['n']}",
            flush=True,
        )
    return out


def assemble_payload(completed, failures, elapsed) -> dict:
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
    for ticker in intra.TICKERS:
        for regime in intra.REGIME_ORDER:
            rows = []
            for model in intra.TABLE_MODELS:
                key = f"{ticker}|{regime}|{model}"
                if key not in cells:
                    continue
                rows.append(cells[key])
            if not rows:
                continue
            best = min(rows, key=lambda r: r["rmse_pct"])
            tables[f"{ticker}|{regime}"] = {
                "ticker": ticker,
                "regime": regime,
                "rows": rows,
                "best_model": best["model"],
                "n_contracts": int(rows[0]["n"]),
                "n_contracts_identical": True,
            }
    overall = []
    for model in intra.TABLE_MODELS:
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
    for regime in intra.REGIME_ORDER:
        for model in intra.TABLE_MODELS:
            vals = [cells[f"{t}|{regime}|{model}"]["rmse_pct"] for t in intra.TICKERS if f"{t}|{regime}|{model}" in cells]
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
    for ticker in intra.TICKERS:
        for model in intra.TABLE_MODELS:
            vals = [cells[f"{ticker}|{r}|{model}"]["rmse_pct"] for r in intra.REGIME_ORDER if f"{ticker}|{r}|{model}" in cells]
            if not vals:
                continue
            by_ticker.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "mean_rmse_pct": float(np.mean(vals)),
                    "median_rmse_pct": float(np.median(vals)),
                    "n_regimes": len(vals),
                    "n_best": sum(1 for r in intra.REGIME_ORDER if tables.get(f"{ticker}|{r}", {}).get("best_model") == model),
                }
            )
    ranking_grid = []
    for ticker in intra.TICKERS:
        for regime in intra.REGIME_ORDER:
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
            "window_label": intra.WINDOW_LABEL,
            "rolling": intra.ROLLING_MODE,
            "n_paths": intra.N_PATHS,
            "step_minutes": intra.STEP_MINUTES,
            "seed": intra.SEED,
            "models": list(intra.TABLE_MODELS),
            "tickers": list(intra.TICKERS),
            "regimes": intra.REGIME_ORDER,
            "regime_meta": intra.REGIME_META,
            "primary_metric": "percentage RMSE of p50 vs realized S_t",
            "measure": "P-measure (μ kept, not replaced by r)",
            "elapsed_sec": float(elapsed),
            "failures": list(failures),
            "study_kind": intra.STUDY_KIND,
        },
        "cells": cells,
        "tables": tables,
        "summary_overall": overall,
        "summary_by_regime": by_regime,
        "summary_by_ticker": by_ticker,
        "ranking_grid": ranking_grid,
    }


def run_study(only: list[str] | None = None) -> dict:
    cache, _, _ = _stock_paths()
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "partial").mkdir(parents=True, exist_ok=True)
    intra.configure_study(intra.STUDY_KIND)
    jobs = [(m, exp) for m in intra.MODELS for exp in intra.EXPIRY_FRIDAYS]
    if only:
        tokens = [t.lower().replace("–", "-").replace("_", "-") for t in only]
        model_alias = {
            "gbm": "GBM",
            "garch": "GARCH",
            "heston": "Heston",
            "merton": "Merton",
            "garch-merton": "GARCH–Merton",
            "heston-merton": "Heston–Merton",
        }
        jobs = [
            (m, exp)
            for m, exp in jobs
            if any(
                (tok in model_alias and m == model_alias[tok]) or tok == exp.lower() or tok in exp.lower()
                for tok in tokens
            )
        ]
    completed: list[dict] = []
    failures: list[str] = []
    t0 = time.time()
    for model, exp in jobs:
        part = _partial_path(model, exp)
        if part.exists():
            print(f"  resume {model} {exp}", flush=True)
            completed.append(json.loads(part.read_text(encoding="utf-8")))
            continue
        print(f"\n=== stock {model} | {exp} ===", flush=True)
        try:
            row = _run_one(model, exp)
            part.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            completed.append(row)
        except Exception:
            failures.append(f"{model}/{exp}")
            print(f"FAILED {model} {exp}:\n{traceback.format_exc()}", flush=True)
    payload = assemble_payload(completed, failures, time.time() - t0)
    dest = cache / "payload.json"
    dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {dest} in {(time.time()-t0)/60:.1f} min", flush=True)
    return payload


def load_payload() -> dict:
    cache, _, _ = _stock_paths()
    return json.loads((cache / "payload.json").read_text(encoding="utf-8"))


def run_or_load(*, recompute: bool = False, only: list[str] | None = None) -> dict:
    cache, _, _ = _stock_paths()
    needed = {f"{t}|{r}|{m}" for t in intra.TICKERS for r in intra.REGIME_ORDER for m in intra.TABLE_MODELS}
    dest = cache / "payload.json"
    if recompute or not dest.exists():
        return run_study(only=only)
    payload = load_payload()
    if not needed.issubset(payload.get("cells", {})):
        return run_study(only=only)
    return payload


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
        f"P-measure rolling cloud, reported path = p50.  Bold green row = lowest RMSE% vs realized S_t.  "
        f"{intra.LOOKBACK_PHRASE} lookback, hourly rolling, Δt = {intra.STEP_MINUTES} min, n_paths={intra.N_PATHS}, seed={intra.SEED}.",
        fontsize=7.6,
        color=MUTED,
    )
    for i, regime in enumerate(regimes):
        ax = fig.add_subplot(2, 2, i + 1)
        tab = payload["tables"][f"{ticker}|{regime}"]
        rows, header, best = _metric_rows(tab)
        meta = intra.REGIME_META[regime]
        k = intra.REGIME_ORDER.index(regime) + 1
        title = (
            f"Table {ticker}-{k}   {ticker}  ·  {meta['title']}  listed expiry {regime}\n"
            f"{meta['window']}   ·   n = {tab['n_contracts']} bars"
        )
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.2, color=NAVY, pad=8, weight="bold")
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.08, 0.96, 0.78])
        emp._style_table(tbl, best)
        tbl.auto_set_column_width(list(range(len(header))))
    fig.text(
        0.06,
        0.03,
        "RMSE% is 100×√mean(((p50−S_t)/S_t)²).  MAE and Bias $ are in stock-price dollars.  "
        "ICP 25–75 is the share of historical S_t inside [p25, p75].  ABW = mean(p75−p25).  "
        "Ranking uses RMSE% only.",
        fontsize=7.0,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def _cover(pdf: PdfPages, payload: dict) -> None:
    import textwrap

    def page(title, sections, footer=None):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.955, title, fontsize=18, weight="bold", color=NAVY, va="top")
        fig.text(
            0.08,
            0.912,
            f"Six models  ·  three underlyings  ·  twelve expiry windows  ·  {intra.LOOKBACK_PHRASE} hourly calibration",
            fontsize=9.5,
            color=MUTED,
            va="top",
        )
        fig.add_artist(plt.Line2D([0.08, 0.92], [0.892, 0.892], transform=fig.transFigure, color=NAVY, lw=1.4))
        y = 0.855

        def section(heading, bullets, y0):
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

    page(
        "V3 empirical study  ·  Instruction",
        [
            (
                "What this report is",
                [
                    "Computational source of truth: the companion Jupyter notebook. This PDF is written from the same payload; every table number is identical.",
                    "Question: which of six models tracks realized stock price S_t most closely on the evaluation window, and does the ranking change across companies and the twelve windows?",
                    "This report does not price options. Option RMSE lives in the companion decision PDF.",
                    "In every table the BEST row is the lowest percentage RMSE of the p50 path versus realized S_t. That row is bold and shaded green; Mark ranks 1–6.",
                ],
            ),
            (
                "Experimental design (held fixed for every model)",
                [
                    "Models: GBM, GARCH(1,1), Heston (no jumps), Merton jump-diffusion, GARCH–Merton, Heston–Merton (Bates).",
                    "Companies: SPY, AAPL, MSFT. Each name is calibrated and simulated separately.",
                    f"Windows: twelve Friday-before-expiry evaluation windows (Oct 2022–Sep 2023). This is the {intra.LOOKBACK_PHRASE} study.",
                    f"Calibration: {intra.LOOKBACK_PHRASE} of 1-minute RTH returns (and listed quotes for Heston) ending at each hourly stamp 09:59–15:59. No look-ahead.",
                    "P-measure: keep the lookback drift μ. Do not replace μ → r (that replacement is only for LSM option pricing).",
                ],
            ),
        ],
        None,
    )
    page(
        "V3 empirical study  ·  Instruction (continued)",
        [
            (
                "Path construction",
                [
                    f"One rolling path cloud per ticker × window × model, n_paths = {intra.N_PATHS}, seed 42.",
                    f"Clock: first RTH bar of the window through Friday 15:59 in {intra.STEP_MINUTES}-minute steps. Not to listed option expiry.",
                    "Start at the observed open S_0. Refresh parameters at each hourly stamp with data ≤ that stamp.",
                    "At each clock time, summarize the cloud by percentiles p10, p25, p50, p75, p90. The reported path is p50, not the Monte Carlo mean.",
                ],
            ),
            (
                "Metrics",
                [
                    "Primary: RMSE% = 100 × √ mean(((p50 − S_t)/S_t)²). Ranking uses this only (lowest is Mark 1).",
                    "MAE = mean |p50 − S_t| (dollars). Bias $ = mean(p50 − S_t). Bias% = 100 × mean((p50 − S_t)/S_t).",
                    "ICP 25–75 = share of historical S_t inside [p25, p75]. ABW = mean(p75 − p25). ICP 10–90 = share inside [p10, p90] (method/appendix; not a table column).",
                    "ICP is higher-better and is not used for the green row.",
                ],
            ),
        ],
        "Page map: p.1–2 method  ·  then company tables  ·  S1–S3, figures, ranking, conclusion.",
    )


def _stock_conclusion(pdf: PdfPages, payload: dict) -> None:
    import textwrap

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "Final analysis", fontsize=16, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], transform=fig.transFigure, color=NAVY, lw=1.2))
    overall = payload["summary_overall"]
    winner = overall[0]["model"]
    bullets = [
        f"On mean RMSE% of the p50 path versus realized S_t across 36 company×window cells, {winner} is best "
        f"(mean RMSE% = {overall[0]['mean_rmse_pct']:.2f}; {overall[0]['n_best']} of 36 cells).",
    ]
    unique = sorted({r["rank1"] for r in payload["ranking_grid"]})
    if len(unique) == 1:
        bullets.append(f"The ranking is stable: {unique[0]} is best in every company and every window.")
    else:
        bullets.append(f"The ranking is not stable. Best-in-cell models: {', '.join(unique)}.")
    bullets.append(
        f"These conclusions are conditional on {intra.LOOKBACK_PHRASE} hourly P-measure calibration, "
        f"{intra.N_PATHS} paths, {intra.STEP_MINUTES}-minute steps to Friday 15:59, and the p50 scoring convention."
    )
    y = 0.88
    fig.text(0.08, y, "Which models track realized S_t most closely?", fontsize=11.5, weight="bold", color=NAVY, va="top")
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
    fig.text(
        0.08,
        0.05,
        "Do not mix these stock-path RMSE% numbers with the option-pricing RMSE% in the decision PDF.",
        fontsize=8.0,
        color="#555555",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    import run_v3_intraday_hourly_reports as reports

    _, pdf_path, _ = _stock_paths()
    path = path or pdf_path
    intra.SHORT.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _cover(pdf, payload)
        page = 3
        for ticker in intra.TICKERS:
            for chunk in intra._chunks(intra.REGIME_ORDER, 4):
                _company_tables_page(pdf, payload, ticker, chunk, page)
                page += 1
        reports._summary_pages(pdf, payload)
        reports._figure_pages(pdf, payload)
        _stock_conclusion(pdf, payload)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
    return path


def build_notebook(payload: dict, path: Path | None = None) -> Path:
    import uuid

    _, _, nb_path = _stock_paths()
    path = path or nb_path
    intra.SHORT.mkdir(parents=True, exist_ok=True)
    nb_import = (
        "run_v3_7d_hourly_stock_study" if intra.STUDY_KIND == "7d" else "run_v3_1d_hourly_stock_study"
    )

    def md(text: str) -> dict:
        if not text.endswith("\n"):
            text += "\n"
        lines = text.split("\n")
        source = [ln + "\n" for ln in lines[:-1]]
        if lines[-1] != "":
            source.append(lines[-1] + "\n")
        return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}

    def code(src: str, html: str | None = None) -> dict:
        outputs = []
        if html:
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"text/html": [html], "text/plain": ["<IPython.core.display.HTML object>"]},
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

    def html_table(tab):
        rows, header, best = _metric_rows(tab)
        parts = [
            "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;width:100%;'>",
            "<thead><tr>",
        ]
        for h in header:
            parts.append(
                f"<th style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{h}</th>"
            )
        parts.append("</tr></thead><tbody>")
        for i, row in enumerate(rows, start=1):
            if i == best:
                bg, extra = emp.BEST_BG, "font-weight:700;"
            elif i % 2 == 0:
                bg, extra = emp.ALT_BG, ""
            else:
                bg, extra = "white", ""
            parts.append("<tr>")
            for val in row:
                parts.append(
                    f"<td style='background:{bg};{extra}padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{val}</td>"
                )
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    cells = [
        md(
            f"""# V3 stock-price study — {intra.LOOKBACK_PHRASE} hourly calibration

**This notebook is the computational source of truth.** The PDF is written from the same payload.

P-measure rolling clouds vs realized S_t. Reported path = **p50**. `n_paths = {intra.N_PATHS}`, Δt = {intra.STEP_MINUTES} min, seed {intra.SEED}. Horizon = window Friday 15:59.
"""
        ),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "from IPython.display import display, HTML\n"
            "ROOT = Path.cwd()\n"
            "for cand in [ROOT, *ROOT.parents]:\n"
            "    scripts = cand / 'V3-Models_result' / 'scripts'\n"
            "    if scripts.exists():\n"
            "        sys.path.insert(0, str(scripts))\n"
            "        break\n"
            f"import {nb_import} as study\n"
            "payload = study.run_or_load(recompute=False)\n"
            "print('cells', len(payload['cells']), 'failures', payload['meta']['failures'])",
            html=f"<pre>cells {len(payload['cells'])} failures {payload['meta']['failures']}</pre>",
        ),
    ]
    for ticker in intra.TICKERS:
        cells.append(md(f"## {ticker}"))
        for exp in intra.REGIME_ORDER:
            tab = payload["tables"][f"{ticker}|{exp}"]
            k = intra.REGIME_ORDER.index(exp) + 1
            cells.append(
                md(
                    f"### Table {ticker}-{k}  ·  {intra.REGIME_META[exp]['title']}  ·  "
                    f"n = {tab['n_contracts']}  ·  BEST = {tab['best_model']}"
                )
            )
            cells.append(code("display(HTML('tab'))", html=html_table(tab)))
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


def write_outputs(payload: dict):
    cache, _, _ = _stock_paths()
    (cache / "payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pdf = write_pdf(payload)
    nb = build_notebook(payload)
    return nb, pdf


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    recompute = "--recompute" in argv
    only = [a for a in argv if not a.startswith("--")]
    n_needed = len(intra.TICKERS) * len(intra.REGIME_ORDER) * len(intra.TABLE_MODELS)
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
    raise SystemExit(main())
