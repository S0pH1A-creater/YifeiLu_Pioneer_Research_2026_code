#!/usr/bin/env python3
"""P-measure stock-path accuracy vs realized S_t for the 1.5-year monthly 10k study.

One daily rolling cloud per (ticker × regime × model). Drift μ is kept.
Reported path = p50. n_paths = 10000.
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

import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402
from american_lsm import pct_rmse  # noqa: E402

NAVY = emp.NAVY
MUTED = emp.MUTED
MODEL_COLORS = emp.MODEL_COLORS
MODEL_STEM = {
    "GBM": "gbm",
    "GARCH": "garch",
    "Heston": "heston",
    "Merton": "merton",
    "GARCH–Merton": "garch_merton",
    "Heston–Merton": "heston_merton",
}


def _paths() -> tuple[Path, Path, Path]:
    cache = emp.CACHE / "stock"
    short = emp.SHORT
    return cache, short / "V3_1p5y_monthly_stock_price.pdf", short / "V3_1p5y_monthly_stock_price.ipynb"


def _simulate_p_measure(g: dict, ticker: str, cal: pd.DataFrame):
    sched = g["param_schedule_for_steps"](ticker, cal)
    n_paths = emp.N_PATHS
    seed = emp.SEED
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
            _dates, mu, kappa, theta, xi, rho, v0, lam, muj, sj, kapj, S0, hist
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


def _metric_rows(tab: dict):
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
    return rows, header, int(np.argmin(rmses)) + 1


def _partial_path(model: str, regime: str) -> Path:
    cache, _, _ = _paths()
    safe = model.replace("–", "-").replace(" ", "_")
    return cache / "partial" / f"{regime}_{safe}.json"


def _run_one(model: str, nb_path: Path) -> dict:
    import run_optimal_stopping_study as os_study

    g = emp._load_ns(nb_path)
    regime = os_study._regime_from_name(nb_path)
    out = {"model": model, "regime": regime, "tickers": {}}
    cache, _, _ = _paths()
    for ticker in emp.TICKERS:
        t0 = time.time()
        cal = g["calibrate_ticker"](ticker, emp.WINDOW_LABEL, emp.ROLLING_MODE)
        g["rolling"] = {ticker: cal}
        paths, hist = _simulate_p_measure(g, ticker, cal)
        mets = _stock_metrics(paths, hist)
        mets["n_updates"] = int(len(cal))
        mets["elapsed"] = float(time.time() - t0)
        fig_dir = cache / "series" / regime / ticker
        fig_dir.mkdir(parents=True, exist_ok=True)
        csv_path = fig_dir / f"{regime}_{MODEL_STEM[model]}.csv"
        hist_v = np.asarray(hist.values if hasattr(hist, "values") else hist, dtype=float)
        idx = list(hist.index) if hasattr(hist, "index") else list(range(len(hist_v)))
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        n = min(len(p50), len(hist_v), len(idx))
        pd.DataFrame(
            {"timestamp": list(idx[:n]), "S_t": hist_v[:n], "p25": p25[:n], "p50": p50[:n], "p75": p75[:n]}
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
    cells = {}
    for row in completed:
        for ticker, mets in row["tickers"].items():
            cells[f"{ticker}|{row['regime']}|{row['model']}"] = {
                "ticker": ticker,
                "regime": row["regime"],
                "model": row["model"],
                **mets,
            }
    tables = {}
    for ticker in emp.TICKERS:
        for regime in emp.REGIME_ORDER:
            rows = [cells[f"{ticker}|{regime}|{m}"] for m in emp.TABLE_MODELS if f"{ticker}|{regime}|{m}" in cells]
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
    for model in emp.TABLE_MODELS:
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
    for regime in emp.REGIME_ORDER:
        for model in emp.TABLE_MODELS:
            vals = [cells[f"{t}|{regime}|{model}"]["rmse_pct"] for t in emp.TICKERS if f"{t}|{regime}|{model}" in cells]
            if vals:
                by_regime.append({"regime": regime, "model": model, "mean_rmse_pct": float(np.mean(vals)), "median_rmse_pct": float(np.median(vals)), "n_tickers": len(vals)})
    by_ticker = []
    for ticker in emp.TICKERS:
        for model in emp.TABLE_MODELS:
            vals = [cells[f"{ticker}|{r}|{model}"]["rmse_pct"] for r in emp.REGIME_ORDER if f"{ticker}|{r}|{model}" in cells]
            if vals:
                by_ticker.append(
                    {
                        "ticker": ticker,
                        "model": model,
                        "mean_rmse_pct": float(np.mean(vals)),
                        "median_rmse_pct": float(np.median(vals)),
                        "n_regimes": len(vals),
                        "n_best": sum(1 for r in emp.REGIME_ORDER if tables.get(f"{ticker}|{r}", {}).get("best_model") == model),
                    }
                )
    ranking_grid = []
    for ticker in emp.TICKERS:
        for regime in emp.REGIME_ORDER:
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
            "window_label": emp.WINDOW_LABEL,
            "rolling": emp.ROLLING_MODE,
            "n_paths": emp.N_PATHS,
            "seed": emp.SEED,
            "models": list(emp.TABLE_MODELS),
            "tickers": list(emp.TICKERS),
            "regimes": emp.REGIME_ORDER,
            "regime_meta": emp.REGIME_META,
            "primary_metric": "percentage RMSE of p50 vs realized S_t",
            "measure": "P-measure (μ kept)",
            "elapsed_sec": float(elapsed),
            "failures": list(failures),
        },
        "cells": cells,
        "tables": tables,
        "summary_overall": overall,
        "summary_by_regime": by_regime,
        "summary_by_ticker": by_ticker,
        "ranking_grid": ranking_grid,
    }


def run_study(only=None) -> dict:
    wrap.apply_1p5y_10k_config()
    cache, _, _ = _paths()
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "partial").mkdir(parents=True, exist_ok=True)
    jobs = emp._jobs()
    if only:
        tokens = [t.lower().replace("–", "-").replace("_", "-") for t in only]
        jobs = [(m, p) for m, p in jobs if any(tok in m.lower().replace("–", "-") or tok in p.stem.lower() for tok in tokens)]
    completed, failures = [], []
    t0 = time.time()
    import run_optimal_stopping_study as os_study

    for model, nb in jobs:
        regime = os_study._regime_from_name(nb)
        part = _partial_path(model, regime)
        if part.exists():
            print(f"  resume {model} {regime}", flush=True)
            completed.append(json.loads(part.read_text(encoding="utf-8")))
            continue
        print(f"\n=== stock {model} | {regime} ===", flush=True)
        try:
            row = _run_one(model, nb)
            part.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            completed.append(row)
        except Exception:
            failures.append(f"{model}/{regime}")
            print(f"FAILED {model} {regime}:\n{traceback.format_exc()}", flush=True)
    payload = assemble_payload(completed, failures, time.time() - t0)
    (cache / "payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {cache / 'payload.json'} in {(time.time()-t0)/60:.1f} min", flush=True)
    return payload


def run_or_load(*, recompute: bool = False, only=None) -> dict:
    wrap.apply_1p5y_10k_config()
    cache, _, _ = _paths()
    dest = cache / "payload.json"
    needed = {f"{t}|{r}|{m}" for t in emp.TICKERS for r in emp.REGIME_ORDER for m in emp.TABLE_MODELS}
    if recompute or not dest.exists():
        return run_study(only=only)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    if not needed.issubset(payload.get("cells", {})):
        return run_study(only=only)
    return payload


def _company_tables_page(pdf, payload, ticker, page_no):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Page {page_no}  ·  Table block  ·  {ticker}  ·  six models × four regimes", fontsize=13, color=NAVY, y=0.975, weight="bold")
    fig.text(
        0.06,
        0.935,
        f"P-measure daily cloud, reported path = p50.  Bold green = lowest RMSE% vs realized S_t.  "
        f"{emp.LOOKBACK_PHRASE} lookback, monthly rolling, n_paths={emp.N_PATHS}, seed={emp.SEED}.",
        fontsize=7.8,
        color=MUTED,
    )
    for i, regime in enumerate(emp.REGIME_ORDER):
        ax = fig.add_subplot(2, 2, i + 1)
        tab = payload["tables"][f"{ticker}|{regime}"]
        rows, header, best = _metric_rows(tab)
        meta = emp.REGIME_META[regime]
        title = f"Table {ticker}-{i+1}   {ticker}  ·  {regime}  {meta['title']}\n{meta['window']}   ·   n = {tab['n_contracts']} days"
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9.0, color=NAVY, pad=8, weight="bold")
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.08, 0.96, 0.78])
        emp._style_table(tbl, best)
        tbl.auto_set_column_width(list(range(len(header))))
    fig.text(
        0.06,
        0.03,
        "RMSE% is 100×√mean(((p50−S_t)/S_t)²).  MAE and Bias $ are in stock-price dollars.  "
        "ICP 25–75 = share of S_t inside [p25, p75].  ABW = mean(p75−p25).  Ranking uses RMSE% only.",
        fontsize=7.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def _cover(pdf, payload):
    import textwrap

    def page(title, sections, footer=None):
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.955, title, fontsize=18, weight="bold", color=NAVY, va="top")
        fig.text(
            0.08,
            0.912,
            f"Six models  ·  three underlyings  ·  four volatility regimes  ·  {emp.LOOKBACK_PHRASE} monthly calibration",
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
                    "Question: which of six models tracks realized stock price S_t most closely, and does the ranking change across companies and the four volatility regimes?",
                    "This report does not price options. Option RMSE lives in the companion decision PDF.",
                    "In every table the BEST row is the lowest percentage RMSE of the p50 path versus realized S_t. That row is bold and shaded green; Mark ranks 1–6.",
                ],
            ),
            (
                "Experimental design (held fixed for every model)",
                [
                    "Models: GBM, GARCH(1,1), Heston (no jumps), Merton jump-diffusion, GARCH–Merton, Heston–Merton (Bates).",
                    "Companies: SPY, AAPL, MSFT. Each name is calibrated and simulated separately.",
                    "Regimes: Crisis 2008-08-01→2009-07-31; Normal 2014-01-01→2014-12-31; Late-cycle 2018-10-01→2019-09-30; COVID 2019-09-01→2020-08-31.",
                    f"Calibration window: {emp.LOOKBACK_PHRASE} ending at each monthly update. Rolling: monthly. No look-ahead.",
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
                    f"One rolling path cloud per ticker × regime × model, n_paths = {emp.N_PATHS}, seed 42.",
                    "Clock: daily trading days from the first bar of the regime through the regime end. Δt = 1/252.",
                    "Start at the observed open S_0. Refresh parameters at month-ends with data ≤ that date.",
                    "At each day, summarize the cloud by p10, p25, p50, p75, p90. The reported path is p50, not the Monte Carlo mean.",
                ],
            ),
            (
                "Metrics",
                [
                    "Primary: RMSE% = 100 × √ mean(((p50 − S_t)/S_t)²). Ranking uses this only.",
                    "MAE = mean |p50 − S_t|. Bias $ = mean(p50 − S_t). Bias% = 100 × mean((p50 − S_t)/S_t).",
                    "ICP 25–75 = share of S_t inside [p25, p75]. ABW = mean(p75 − p25). ICP 10–90 is recorded but not a table column.",
                    "ICP is higher-better and is not used for the green row.",
                ],
            ),
        ],
        "Page map: p.1–2 method  ·  then company tables  ·  S1–S3, figures, ranking, conclusion.",
    )


def _stock_conclusion(pdf, payload):
    import textwrap

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "Final analysis", fontsize=16, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], transform=fig.transFigure, color=NAVY, lw=1.2))
    overall = payload["summary_overall"]
    winner = overall[0]["model"]
    unique = sorted({r["rank1"] for r in payload["ranking_grid"]})
    bullets = [
        f"On mean RMSE% of the p50 path versus realized S_t across 12 company×regime cells, {winner} is best "
        f"(mean RMSE% = {overall[0]['mean_rmse_pct']:.2f}; {overall[0]['n_best']} of 12 cells).",
        f"The ranking is {'stable: ' + unique[0] + ' is best in every cell' if len(unique)==1 else 'not stable. Best-in-cell models: ' + ', '.join(unique)}.",
        f"These conclusions are conditional on {emp.LOOKBACK_PHRASE} monthly P-measure calibration, {emp.N_PATHS} paths, daily steps, and the p50 scoring convention.",
    ]
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
    fig.text(0.08, 0.05, "Do not mix these stock-path RMSE% numbers with the option-pricing RMSE% in the decision PDF.", fontsize=8.0, color="#555555", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload, path=None):
    _, pdf_path, _ = _paths()
    path = path or pdf_path
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _cover(pdf, payload)
        for i, ticker in enumerate(emp.TICKERS):
            _company_tables_page(pdf, payload, ticker, page_no=3 + i)
        emp._summary_pages(pdf, payload)
        emp._figure_pages(pdf, payload)
        _stock_conclusion(pdf, payload)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
    return path


def build_notebook(payload, path=None):
    import uuid

    _, _, nb_path = _paths()
    path = path or nb_path
    emp.SHORT.mkdir(parents=True, exist_ok=True)

    def md(text):
        if not text.endswith("\n"):
            text += "\n"
        lines = text.split("\n")
        source = [ln + "\n" for ln in lines[:-1]]
        if lines[-1] != "":
            source.append(lines[-1] + "\n")
        return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}

    def html_table(tab):
        rows, header, best = _metric_rows(tab)
        parts = ["<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;width:100%;'><thead><tr>"]
        for h in header:
            parts.append(f"<th style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{h}</th>")
        parts.append("</tr></thead><tbody>")
        for i, row in enumerate(rows, start=1):
            bg = emp.BEST_BG if i == best else (emp.ALT_BG if i % 2 == 0 else "white")
            extra = "font-weight:700;" if i == best else ""
            parts.append("<tr>")
            for val in row:
                parts.append(f"<td style='background:{bg};{extra}padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>{val}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    cells = [
        md(
            f"""# V3 stock-price study — {emp.LOOKBACK_PHRASE} monthly calibration

P-measure daily clouds vs realized S_t. Reported path = **p50**. `n_paths = {emp.N_PATHS}`, seed {emp.SEED}.
"""
        )
    ]
    for ticker in emp.TICKERS:
        cells.append(md(f"## {ticker}"))
        for i, exp in enumerate(emp.REGIME_ORDER, start=1):
            tab = payload["tables"][f"{ticker}|{exp}"]
            cells.append(md(f"### Table {ticker}-{i}  ·  {emp.REGIME_META[exp]['title']}  ·  BEST = {tab['best_model']}"))
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "id": uuid.uuid4().hex[:8],
                    "metadata": {},
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {"text/html": [html_table(tab)], "text/plain": ["<IPython.core.display.HTML object>"]},
                            "metadata": {},
                        }
                    ],
                    "source": ["from IPython.display import display, HTML\ndisplay(HTML('tab'))\n"],
                }
            )
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


def write_outputs(payload):
    cache, _, _ = _paths()
    (cache / "payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pdf = write_pdf(payload)
    nb = build_notebook(payload)
    return nb, pdf


def main(argv=None) -> int:
    wrap.apply_1p5y_10k_config()
    argv = list(sys.argv[1:] if argv is None else argv)
    recompute = "--recompute" in argv
    only = [a for a in argv if not a.startswith("--")]
    n_needed = len(emp.TICKERS) * len(emp.REGIME_ORDER) * len(emp.TABLE_MODELS)
    payload = run_or_load(recompute=recompute, only=only or None)
    if len(payload.get("cells", {})) == n_needed and not payload["meta"]["failures"]:
        write_outputs(payload)
        return 0
    print(f"Incomplete: {len(payload.get('cells', {}))}/{n_needed} cells, failures={payload['meta']['failures']}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
