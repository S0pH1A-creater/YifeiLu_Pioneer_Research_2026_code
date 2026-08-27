"""Yearly-style PDF / notebook drawing for the 7-day and 1-day hourly studies."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import run_v3_5y_monthly_empirical_study as emp
import run_v3_intraday_hourly_empirical_study as intra

NAVY = emp.NAVY
MUTED = emp.MUTED
MODEL_COLORS = emp.MODEL_COLORS


def _subtitle() -> str:
    return (
        f"Six models  ·  three underlyings  ·  twelve expiry windows  ·  "
        f"{intra.LOOKBACK_PHRASE} hourly calibration"
    )


def _cover(pdf: PdfPages, payload: dict) -> None:
    eval_txt = (
        "that Friday only (seven hourly stamps 09:59–15:59)"
        if intra.STUDY_KIND == "1d"
        else "the five NYSE sessions ending that Friday (35 hourly stamps; Labor Day week = 1, 5, 6, 7, 8 Sep)"
    )

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

    _instruction_page(
        "V3 empirical study  ·  Instruction",
        _subtitle(),
        [
            (
                "What this report is",
                [
                    "Computational source of truth: the companion Jupyter notebook. This PDF is written from the same payload; every table number is identical.",
                    "Question: which of six American-call pricing models tracks listed market prices most closely, and does the ranking change across companies and the twelve expiry windows?",
                    "In every table the BEST row is the lowest percentage RMSE. That row is bold and shaded green, and the Mark column ranks 1–6 (1 = lowest RMSE%).",
                ],
            ),
            (
                "Experimental design (held fixed for every model)",
                [
                    "Models: GBM, GARCH(1,1), Heston (no jumps), Merton jump-diffusion, GARCH–Merton, Heston–Merton (Bates).",
                    "Companies: SPY (primary), AAPL, MSFT. Each name is calibrated and priced separately.",
                    "Windows: twelve monthly third-Friday expiries (Oct 2022–Sep 2023). Evaluation is the Friday immediately before expiry, not the expiry Friday itself.",
                    f"1-day study = that Friday. 7-day study = five NYSE sessions ending that Friday. This report is the {intra.LOOKBACK_PHRASE} study: {eval_txt}.",
                    f"Calibration window: {intra.LOOKBACK_PHRASE} of 1-minute RTH equity returns and listed quotes ending at each hourly update (09:59–15:59). Parameters estimated at t use only data with timestamp ≤ t (no look-ahead).",
                    "If history is shorter than the requested lookback (first hours of 10 Oct 2022), the estimator uses all available 1-minute observations up to t (equity file starts 2022-09-30).",
                ],
            ),
        ],
        None,
    )
    _instruction_page(
        "V3 empirical study  ·  Instruction (continued)",
        _subtitle(),
        [
            (
                "Shared market sample and filters (not model-specific)",
                [
                    "For each company × window, one listed-call sample is drawn once and reused by all six models: nearest-ATM call as-of each hourly evaluation timestamp (09:59, 10:59, 11:59, 12:59, 13:59, 14:59, 15:59), DTE 7–60, listed expiry.",
                    "LSM does not run on a 15-minute or 5-minute quote grid. Friday 15:59 is listed as a scheduled stamp but remaining_horizon = 0, so LSM is skipped (at least two MC steps required). RMSE% uses only timestamps with a valid remaining horizon.",
                    "Estimation filters, applied in order to every quote used for Heston/Heston–Merton calibration and to the LSM comparison set: (1) calls only, finite S, K, C; (2) no-arbitrage C ≥ max(0, S−K); (3) 7 ≤ DTE ≤ 60; (4) |S/K − 1| ≤ 10%; (5) premium ≥ 0.05, valid bid–ask with relative spread ≤ 50% when those columns exist, volume ≥ 1 when present.",
                    "Return-based models (GBM, Merton, GARCH, GARCH–Merton) never see option quotes in §4; they still price the same filtered LSM contracts in evaluation.",
                ],
            ),
            (
                "Model-correctness adjustments (V3)",
                [
                    "Heston / Heston–Merton: (κ, θ, ξ, ρ, v0) are option-implied Fourier NLS (Albrecher little-trap), not Method A realized-variance moments. μ is the P-measure lookback mean of stock returns; LSM replaces μ → r.",
                    "Euler ordering: the stock increment over [t, t+Δt] uses the current variance v_t; v_{t+Δt} is updated afterwards. No look-ahead in the variance.",
                    f"LSM: Longstaff–Schwartz American call on the full path cloud (n_paths = {intra.N_PATHS}, seed 42). n_steps = remaining MC steps from t to window Friday 15:59, not listed DTE. Δt = {intra.STEP_MINUTES} minute(s). Paths are not averaged before stopping. Do not extend paths to listed expiry.",
                ],
            ),
            (
                "Metrics",
                [
                    "Primary: RMSE% = 100 × √ mean(((C_model − C_mkt) / C_mkt)²). Ranking uses this number only.",
                    "Secondary: MAE = mean |C_model − C_mkt| (dollars). Bias $ = mean(C_model − C_mkt) (dollars). Bias% = 100 × mean((C_model − C_mkt)/C_mkt). Positive bias = model expensive vs market. Early-exercise fraction = mean share of LSM paths that stop before the window end.",
                ],
            ),
        ],
        "Page map: p.1–2 method  ·  p.3 filters & sample  ·  then company tables (three landscape pages per name)  ·  S1–S3, figures, ranking, conclusion.",
    )


def _filters_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.955, "Page 3  ·  Filters, sample, and no-look-ahead", fontsize=14, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.925, 0.925], transform=fig.transFigure, color=NAVY, lw=1.2))

    ax = fig.add_axes([0.06, 0.48, 0.88, 0.42])
    ax.axis("off")
    ax.set_title(
        "Shared evaluation sample  ·  n listed calls (identical for all six models)",
        loc="left",
        fontsize=10.5,
        color=NAVY,
        weight="bold",
    )
    header = ["Window", "Eval Friday", "Sessions", "SPY n", "AAPL n", "MSFT n"]
    rows = []
    for regime in intra.REGIME_ORDER:
        info = payload["shared_contracts"][regime]
        meta = intra.REGIME_META[regime]
        rows.append(
            [
                f"{meta['title']}  {regime}",
                meta.get("eval_friday", info["period_end"]),
                str(meta["n_sessions"]),
                str(info["n"]["SPY"]),
                str(info["n"]["AAPL"]),
                str(info["n"]["MSFT"]),
            ]
        )
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.92])
    emp._style_table(tbl, None)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)

    ax2 = fig.add_axes([0.06, 0.08, 0.88, 0.36])
    ax2.axis("off")
    ax2.set_title(
        f"Filter funnel  ·  last-step n in the {intra.LOOKBACK_PHRASE} lookback ending at each window (SPY processed calls)",
        loc="left",
        fontsize=9.2,
        color=NAVY,
        weight="bold",
    )
    funnel = payload["filter_funnel"]["tickers"]["SPY"]
    header2 = ["Window", "Input*", "No-arb", "DTE 7–60", "|S/K−1|≤10%", "Liquidity", "Eval n"]
    rows2 = []
    for regime in intra.REGIME_ORDER:
        steps = {s["step"]: s["n"] for s in funnel[regime]["steps"]}
        rows2.append(
            [
                intra.SHORT_LABEL[regime],
                str(steps.get("finite S, K, C", "")),
                str(steps.get("1. no-arbitrage C ≥ max(0, S−K)", "")),
                str(steps.get("2. maturity 7–60 DTE", "")),
                str(steps.get("3a. |S/K − 1| ≤ 10%", "")),
                str(steps.get("3b. liquidity / wide bid–ask", "")),
                str(funnel[regime].get("n_eval_contracts", "")),
            ]
        )
    tbl2 = ax2.table(cellText=rows2, colLabels=header2, loc="center", cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.92])
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(6.6)
    emp._style_table(tbl2, None)
    fig.text(
        0.06,
        0.025,
        f"*Input is the processed call panel restricted to the {intra.LOOKBACK_PHRASE} 1-minute lookback ending at window Friday 15:59. "
        "Same filters for AAPL/MSFT. No look-ahead: hourly parameters at t use only data ≤ t. "
        "LSM horizon is remaining time to Friday 15:59, not listed expiry.",
        fontsize=7.2,
        color="#555555",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _summary_pages(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Summary tables  ·  RMSE% across companies and twelve windows", fontsize=13, color=NAVY, y=0.97, weight="bold")

    ax = fig.add_axes([0.06, 0.55, 0.42, 0.35])
    ax.axis("off")
    ax.set_title("Table S1  ·  Overall ranking (mean RMSE% over 36 cells)", loc="left", fontsize=10, color=NAVY, weight="bold")
    header = ["Rank", "Model", "Mean RMSE%", "Median RMSE%", "# of 36 cells best"]
    rows = []
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
    emp._style_table(tbl, 1)

    ax2 = fig.add_axes([0.52, 0.55, 0.42, 0.35])
    ax2.axis("off")
    ax2.set_title("Table S2  ·  Wins by company (lowest RMSE% in that company×window)", loc="left", fontsize=9.6, color=NAVY, weight="bold")
    header2 = ["Model", "SPY wins", "AAPL wins", "MSFT wins", "Total"]
    rows2 = []
    win_counts = {m: {t: 0 for t in intra.TICKERS} for m in intra.TABLE_MODELS}
    for rec in payload["ranking_grid"]:
        win_counts[rec["rank1"]][rec["ticker"]] += 1
    totals = [sum(win_counts[m].values()) for m in intra.TABLE_MODELS]
    best_model = intra.TABLE_MODELS[int(np.argmax(totals))] if totals else None
    best_row = None
    for i, m in enumerate(intra.TABLE_MODELS):
        tot = sum(win_counts[m].values())
        rows2.append([m, str(win_counts[m]["SPY"]), str(win_counts[m]["AAPL"]), str(win_counts[m]["MSFT"]), str(tot)])
        if m == best_model:
            best_row = i + 1
    tbl2 = ax2.table(cellText=rows2, colLabels=header2, loc="center", cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    emp._style_table(tbl2, best_row)

    ax3 = fig.add_axes([0.06, 0.08, 0.88, 0.42])
    ax3.axis("off")
    ax3.set_title("Table S3  ·  Mean RMSE% by window (equal-weight mean of SPY / AAPL / MSFT)", loc="left", fontsize=10, color=NAVY, weight="bold")
    header3 = ["Model"] + [intra.SHORT_LABEL[r] for r in intra.REGIME_ORDER] + ["Mean of 12"]
    by = {(r["regime"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_regime"]}
    rows3 = []
    means = []
    for m in intra.TABLE_MODELS:
        vals = [by.get((r, m), np.nan) for r in intra.REGIME_ORDER]
        mu = float(np.nanmean(vals))
        means.append(mu)
        rows3.append([m] + [f"{v:.2f}" for v in vals] + [f"{mu:.2f}"])
    best_row3 = int(np.nanargmin(means)) + 1
    tbl3 = ax3.table(cellText=rows3, colLabels=header3, loc="center", cellLoc="center", bbox=[0.0, 0.08, 1.0, 0.82])
    emp._style_table(tbl3, best_row3)
    tbl3.auto_set_font_size(False)
    tbl3.set_fontsize(7.0)
    fig.text(
        0.06,
        0.035,
        "Green row in S1 / S3 = lowest mean RMSE%.  Green row in S2 = most company×window wins.  "
        "A model can win the most cells without having the lowest average error (and vice versa).",
        fontsize=8.0,
        color="#555555",
    )
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Summary by company  ·  mean RMSE% across twelve windows", fontsize=13, color=NAVY, y=0.97, weight="bold")
    by_t = {(r["ticker"], r["model"]): r for r in payload["summary_by_ticker"]}
    for i, ticker in enumerate(intra.TICKERS):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(f"Table S4.{i+1}  ·  {ticker}", loc="left", fontsize=10.5, color=NAVY, weight="bold")
        header = ["Model", "Mean RMSE%", "Median", "# windows best"]
        rows = []
        rmses = []
        for m in intra.TABLE_MODELS:
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
        emp._style_table(tbl, best)
    fig.text(
        0.06,
        0.06,
        "Each column is one company. Green row = lowest mean RMSE% across the twelve Friday-before-expiry windows for that name.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.97, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def _figure_pages(pdf: PdfPages, payload: dict) -> None:
    cells = payload["cells"]
    x = np.arange(len(intra.TABLE_MODELS))
    for chunk in intra._chunks(intra.REGIME_ORDER, 4):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(
            "Figure 1  ·  RMSE% by model and window  ·  mean of SPY / AAPL / MSFT",
            fontsize=12.5,
            color=NAVY,
            y=0.98,
            weight="bold",
        )
        for ax, regime in zip(axes.ravel(), chunk):
            vals = []
            for m in intra.TABLE_MODELS:
                v = [cells[f"{t}|{regime}|{m}"]["rmse_pct"] for t in intra.TICKERS]
                vals.append(float(np.mean(v)))
            bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in intra.TABLE_MODELS], width=0.72, zorder=3)
            ymax = max(vals) if vals else 1.0
            best_i = int(np.argmin(vals))
            bars[best_i].set_edgecolor("#111111")
            bars[best_i].set_linewidth(1.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * ymax, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
            ax.set_xticks(x)
            ax.set_xticklabels(list(intra.TABLE_MODELS), fontsize=7.2, rotation=18, ha="right")
            ax.set_ylabel("RMSE%", fontsize=8.5)
            ax.set_title(f"{intra.SHORT_LABEL[regime]}  ·  {intra.REGIME_META[regime]['title']}", loc="left", fontsize=10, color=NAVY, weight="bold")
            ax.yaxis.grid(True, linestyle="-", alpha=0.28, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
        fig.text(0.06, 0.03, "Outlined bar = lowest mean RMSE% in that window. Equal-weight mean of the three companies.", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
        pdf.savefig(fig)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 8.5), sharey=False)
    fig.suptitle("Figure 2  ·  RMSE% by model  ·  each company (mean across twelve windows)", fontsize=12.5, color=NAVY, y=0.98, weight="bold")
    by_t = {(r["ticker"], r["model"]): r["mean_rmse_pct"] for r in payload["summary_by_ticker"]}
    for ax, ticker in zip(axes, intra.TICKERS):
        vals = [by_t[(ticker, m)] for m in intra.TABLE_MODELS]
        bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in intra.TABLE_MODELS], width=0.72, zorder=3)
        best_i = int(np.argmin(vals))
        bars[best_i].set_edgecolor("#111111")
        bars[best_i].set_linewidth(1.5)
        ymax = max(vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 * ymax, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x)
        ax.set_xticklabels(list(intra.TABLE_MODELS), fontsize=7.0, rotation=22, ha="right")
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

    fig, axes = plt.subplots(1, 3, figsize=(11, 8.5))
    fig.suptitle("Figure 3  ·  Model rank by company × window  (1 = best RMSE%)", fontsize=12.5, color=NAVY, y=0.97, weight="bold")
    for ax, ticker in zip(axes, intra.TICKERS):
        rank = np.zeros((len(intra.TABLE_MODELS), len(intra.REGIME_ORDER)))
        for j, regime in enumerate(intra.REGIME_ORDER):
            tab = payload["tables"][f"{ticker}|{regime}"]
            order = {r["model"]: i + 1 for i, r in enumerate(sorted(tab["rows"], key=lambda z: z["rmse_pct"]))}
            for i, m in enumerate(intra.TABLE_MODELS):
                rank[i, j] = order[m]
        im = ax.imshow(rank, cmap=plt.cm.RdYlGn_r, vmin=1, vmax=6, aspect="auto")
        ax.set_xticks(range(len(intra.REGIME_ORDER)))
        ax.set_xticklabels([intra.SHORT_LABEL[r] for r in intra.REGIME_ORDER], fontsize=6.2, rotation=45, ha="right")
        ax.set_yticks(range(len(intra.TABLE_MODELS)))
        ax.set_yticklabels(list(intra.TABLE_MODELS), fontsize=7.4)
        ax.set_title(ticker, loc="left", fontsize=11, color=NAVY, weight="bold")
        for i in range(rank.shape[0]):
            for j in range(rank.shape[1]):
                val = int(rank[i, j])
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="black" if val not in (1, 6) else ("#0B3D0B" if val == 1 else "white"),
                    fontweight="bold" if val == 1 else "normal",
                )
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02).set_label("Rank (1 = best)", fontsize=8)
    fig.text(0.06, 0.03, "Green / 1 = lowest RMSE% in that company×window cell. Rankings can and do change across names and windows.", fontsize=8.5, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def _analysis_text(payload: dict) -> list[str]:
    overall = payload["summary_overall"]
    winner = overall[0]["model"]
    runner = overall[1]["model"] if len(overall) > 1 else ""
    lines = [
        f"On mean percentage RMSE across the 36 company×window cells, {winner} is the best overall model "
        f"(mean RMSE% = {overall[0]['mean_rmse_pct']:.2f}; {overall[0]['n_best']} of 36 cells).",
    ]
    if runner:
        lines.append(
            f"Second overall is {runner} (mean RMSE% = {overall[1]['mean_rmse_pct']:.2f}; {overall[1]['n_best']} cells)."
        )
    rank1s = {(r["ticker"], r["regime"]): r["rank1"] for r in payload["ranking_grid"]}
    unique_winners = sorted(set(rank1s.values()))
    if len(unique_winners) == 1:
        lines.append(f"The ranking is stable: {unique_winners[0]} is best in every company and every window.")
    else:
        lines.append(
            f"The ranking is not stable. Best-in-cell models across the 36 tables: {', '.join(unique_winners)}."
        )
    by_t = {(r["ticker"], r["model"]): r for r in payload["summary_by_ticker"]}
    for ticker in intra.TICKERS:
        pick = min(intra.TABLE_MODELS, key=lambda m: by_t[(ticker, m)]["mean_rmse_pct"])
        lines.append(
            f"{ticker}: lowest mean RMSE% is {pick} ({by_t[(ticker, pick)]['mean_rmse_pct']:.2f}); "
            f"wins {by_t[(ticker, pick)]['n_best']} of 12 windows."
        )
    jump = ["Merton", "GARCH–Merton", "Heston–Merton"]
    nojump = ["GBM", "GARCH", "Heston"]
    jump_mean = float(np.mean([r["mean_rmse_pct"] for r in overall if r["model"] in jump]))
    nojump_mean = float(np.mean([r["mean_rmse_pct"] for r in overall if r["model"] in nojump]))
    if jump_mean < nojump_mean:
        lines.append(
            f"Jump models as a group have lower mean RMSE% ({jump_mean:.2f}) than the three no-jump models ({nojump_mean:.2f})."
        )
    else:
        lines.append(
            f"No-jump models as a group have lower mean RMSE% ({nojump_mean:.2f}) than the three jump models ({jump_mean:.2f})."
        )
    lines.append(
        f"These conclusions are conditional on the V3 filters, {intra.LOOKBACK_PHRASE} hourly calibration, "
        f"nearest-ATM hourly sample, LSM with {intra.N_PATHS} paths and {intra.STEP_MINUTES}-minute steps to Friday 15:59, "
        "and American-call quotes only. They are not a statement about European options or other tenors."
    )
    return lines


def _conclusion_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "Final analysis", fontsize=16, weight="bold", color=NAVY, va="top")
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
    fig.text(0.08, y, "How to read the 36 detailed tables", fontsize=11.5, weight="bold", color=NAVY, va="top")
    y -= 0.035
    notes = [
        "Each of Tables SPY-1…MSFT-12 uses exactly the same option contracts for all six models. Differences are the dynamics, not the sample.",
        "Mark 1 is always lowest RMSE%. MAE, Bias $, Bias%, and early-exercise are reported, not used for the ranking.",
        "A model that is Mark 1 on RMSE% can still be biased (systematically expensive or cheap).",
        "Early-exercise fractions are a model implication, not an accuracy score; LSM stops at window Friday close, not listed expiry.",
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

    fig.text(
        0.08,
        0.05,
        f"Notebook: {intra.SHORT.relative_to(intra.REPO) / intra.NB_NAME}\n"
        f"Engine: V4-Models_result/scripts/{intra.ENGINE_SCRIPT}  ·  payload.json is the shared number store.",
        fontsize=7.6,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    payload = emp.enrich_bias_pct(payload)
    intra.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (intra.SHORT / intra.PDF_NAME)
    with PdfPages(path) as pdf:
        _cover(pdf, payload)
        _filters_page(pdf, payload)
        page = 4
        for ticker in intra.TICKERS:
            for chunk in intra._chunks(intra.REGIME_ORDER, 4):
                intra._company_tables_page(pdf, payload, ticker, chunk, page)
                page += 1
        _summary_pages(pdf, payload)
        _figure_pages(pdf, payload)
        _conclusion_page(pdf, payload)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
    return path


def build_notebook(payload: dict, path: Path | None = None) -> Path:
    import uuid

    payload = emp.enrich_bias_pct(payload)
    intra.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (intra.SHORT / intra.NB_NAME)
    eval_line = (
        "Friday immediately before expiry only (hourly 09:59–15:59)"
        if intra.STUDY_KIND == "1d"
        else "five NYSE sessions ending the Friday before expiry (hourly 09:59–15:59)"
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

    cells = [
        md(
            f"""# V3 empirical study — {intra.LOOKBACK_PHRASE} hourly calibration

**This notebook is the computational source of truth.** The PDF in this folder is written from the same `payload` dict.

| Item | Setting |
|------|---------|
| Models | GBM, GARCH, Heston, Merton, GARCH–Merton, Heston–Merton |
| Companies | SPY, AAPL, MSFT |
| Windows | 12 third-Friday monthly expiries, 21 Oct 2022 – 15 Sep 2023 |
| Evaluation | **{eval_line}** |
| Calibration | **{intra.LOOKBACK_PHRASE}** lookback, **hourly** rolling, 1-minute RTH data |
| LSM horizon | remaining time **to window Friday 15:59**, not listed expiry |
| LSM grid | Δt = **{intra.STEP_MINUTES} minute(s)**, n_steps = remaining RTH / Δt |
| Paths | `{intra.N_PATHS}`, seed `{intra.SEED}` |
| Sample | one nearest-ATM listed call per hourly timestamp, DTE 7–60 |

Green / Mark 1 = lowest RMSE%. Friday 15:59 is scheduled but LSM is skipped (`remaining_horizon = 0`).
"""
        ),
        md(
            """## Methodology

- Shared sample: nearest-ATM, DTE 7–60, listed expiry, as-of each hourly *t*. All six models price the same contracts.
- No look-ahead: parameters at *t* use only equity returns and listed quotes with timestamp ≤ *t*.
- V3 Heston: option-implied NLS; Euler uses *v_t* then updates variance.
- LSM on the full path cloud. Start every path at the observed *S_t*.
"""
        ),
        code(
            "import sys\n"
            "from pathlib import Path\n"
            "from IPython.display import display, HTML\n"
            "ROOT = Path.cwd()\n"
            "for cand in [ROOT, *ROOT.parents]:\n"
            "    scripts = cand / 'V4-Models_result' / 'scripts'\n"
            "    if scripts.exists():\n"
            "        sys.path.insert(0, str(scripts))\n"
            "        break\n"
            f"import {intra.NOTEBOOK_IMPORT} as study\n"
            "RECOMPUTE = False\n"
            "payload = study.run_or_load(recompute=RECOMPUTE)\n"
            "print('cells', len(payload['cells']), 'failures', payload['meta']['failures'])",
            html=f"<pre>cells {len(payload['cells'])} failures {payload['meta']['failures']}</pre>",
        ),
    ]
    header = ["Window", "Eval Friday", "SPY n", "AAPL n", "MSFT n"]
    rows = []
    for exp in intra.REGIME_ORDER:
        info = payload["shared_contracts"][exp]
        meta = intra.REGIME_META[exp]
        rows.append(
            [
                f"{meta['title']} {exp}",
                meta.get("eval_friday", ""),
                str(info["n"]["SPY"]),
                str(info["n"]["AAPL"]),
                str(info["n"]["MSFT"]),
            ]
        )
    cells.append(md("## Shared sample *n*"))
    cells.append(
        code(
            "display(HTML('shared n'))",
            html=emp._html_table(
                {
                    "rows": [
                        {
                            "model": r[0],
                            "rmse_pct": 0,
                            "mae": 0,
                            "bias": 0,
                            "bias_pct": 0,
                            "early": 0,
                        }
                        for r in rows
                    ]
                }
            )
            if False
            else _html_simple(header, rows),
        )
    )
    for ticker in intra.TICKERS:
        cells.append(md(f"## {ticker}"))
        for exp in intra.REGIME_ORDER:
            tab = payload["tables"][f"{ticker}|{exp}"]
            k = intra.REGIME_ORDER.index(exp) + 1
            cells.append(
                md(
                    f"### Table {ticker}-{k}  ·  {intra.REGIME_META[exp]['title']}  {exp}  ·  "
                    f"n = {tab['n_contracts']}  ·  BEST = {tab['best_model']}"
                )
            )
            cells.append(
                code(
                    f"display(HTML(study.emp._html_table(payload['tables']['{ticker}|{exp}'])))",
                    html=emp._html_table(tab),
                )
            )
    cells.append(md("## Summary S1"))
    s1_header = ["Rank", "Model", "Mean RMSE%", "Median RMSE%", "# of 36 cells best"]
    s1_rows = [
        [
            str(i),
            r["model"],
            f"{r['mean_rmse_pct']:.2f}",
            f"{r['median_rmse_pct']:.2f}",
            str(r["n_best"]),
        ]
        for i, r in enumerate(payload["summary_overall"], start=1)
    ]
    cells.append(code("display(HTML('s1'))", html=_html_simple(s1_header, s1_rows, best=1)))
    cells.append(md("## Findings\n\n" + "\n\n".join(f"- {b}" for b in _analysis_text(payload))))
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


def _html_simple(header: list[str], rows: list[list[str]], best: int | None = None) -> str:
    parts = [
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;width:100%;'>",
        "<thead><tr>",
    ]
    for h in header:
        parts.append(f"<th style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{h}</th>")
    parts.append("</tr></thead><tbody>")
    for i, row in enumerate(rows, start=1):
        if best is not None and i == best:
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
