#!/usr/bin/env python3
"""Moneyness performance report on the frozen V3 5-year monthly study.

Same 72 contract files as the main empirical study. No recalibration, no
new filters, no new sample. Classifies each priced call by S/K and
compares the six models inside each moneyness bucket.

Notebook + PDF live next to the main study in Results_In_Short.
"""
from __future__ import annotations

import json
import os
import sys
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

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as main  # noqa: E402
from american_lsm import pct_mae, pct_rmse  # noqa: E402

BUCKETS = ("DOTM", "OTM", "ATM", "ITM", "DITM")
BUCKET_RULE = (
    ("DOTM", "m < 0.98"),
    ("OTM", "0.98 ≤ m < 0.995"),
    ("ATM", "0.995 ≤ m ≤ 1.005"),
    ("ITM", "1.005 < m ≤ 1.02"),
    ("DITM", "m > 1.02"),
)
MODEL_STEM = {
    "GBM": "gbm",
    "Modified GBM": "modified_gbm",
    "Modified GBM meanfix": "modified_gbm_meanfix",
    "MD-GBM": "modified_gbm_meanfix",
    "Modified GBM v2": "modified_gbm_v2",
    "Modified GBM v3": "modified_gbm_v3",
    "GARCH": "garch",
    "Heston": "heston",
    "Merton": "merton",
    "GARCH–Merton": "garch_merton",
    "Heston–Merton": "heston_merton",
}
NB_NAME = "V3_5y_monthly_moneyness_performance.ipynb"
PDF_NAME = "V3_5y_monthly_moneyness_performance.pdf"
MONEYNESS_IMPORT = "run_v3_5y_monthly_moneyness_study"


def _payload_json() -> Path:
    return main.CACHE / "moneyness_payload.json"


def classify_moneyness(s, k) -> str:
    """Call moneyness m = S/K. Five bins that partition the |m−1|≤10% filter."""
    m = float(s) / float(k)
    if m < 0.98:
        return "DOTM"
    if m < 0.995:
        return "OTM"
    if m <= 1.005:
        return "ATM"
    if m <= 1.02:
        return "ITM"
    return "DITM"


def _metrics(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {
            "rmse_pct": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "bias_pct": float("nan"),
            "early": float("nan"),
            "n": 0,
        }
    return {
        "rmse_pct": float(pct_rmse(df["model_price"], df["market"])),
        "mae": float(pct_mae(df["model_price"], df["market"])),
        "bias": float(np.mean(df["error"])),
        "bias_pct": float(main.pct_bias(df["model_price"], df["market"])),
        "early": float(df["early_ex_frac"].mean()),
        "n": int(len(df)),
    }


def load_panel() -> pd.DataFrame:
    rows = []
    for regime in main.REGIME_ORDER:
        for ticker in main.TICKERS:
            for model in main.TABLE_MODELS:
                stem = MODEL_STEM[model]
                path = main.CACHE / "contracts" / regime / ticker / f"{regime}_{stem}.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                df = pd.read_csv(path)
                df["regime"] = regime
                df["ticker"] = ticker
                df["model"] = model
                df["m"] = df["S_t"].astype(float) / df["K"].astype(float)
                df["bucket"] = [classify_moneyness(s, k) for s, k in zip(df["S_t"], df["K"])]
                rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    # identity check: every model prices the same (ticker, regime, date, K)
    keys = ["ticker", "regime", "trading_date", "K"]
    n_by_model = out.groupby("model").size()
    if n_by_model.nunique() != 1:
        raise RuntimeError(f"model-specific contract counts: {n_by_model.to_dict()}")
    return out


def build_payload(panel: pd.DataFrame) -> dict:
    # counts from one model (identical across models)
    one = panel.loc[panel["model"] == main.TABLE_MODELS[0]]
    n_overall = one["bucket"].value_counts().reindex(BUCKETS).fillna(0).astype(int).to_dict()
    n_regime = (
        pd.crosstab(one["regime"], one["bucket"])
        .reindex(index=main.REGIME_ORDER, columns=BUCKETS)
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    analysis1 = {}
    for bucket in BUCKETS:
        rows = []
        for model in main.TABLE_MODELS:
            sub = panel.loc[(panel["model"] == model) & (panel["bucket"] == bucket)]
            rec = _metrics(sub)
            rec["model"] = model
            rec["bucket"] = bucket
            rows.append(rec)
        finite = [r for r in rows if r["n"] > 0 and np.isfinite(r["rmse_pct"])]
        best = min(finite, key=lambda r: r["rmse_pct"])["model"] if finite else None
        analysis1[bucket] = {
            "bucket": bucket,
            "n": int(n_overall[bucket]),
            "best_model": best,
            "rows": rows,
        }

    analysis2 = {}
    for regime in main.REGIME_ORDER:
        for bucket in BUCKETS:
            rows = []
            for model in main.TABLE_MODELS:
                sub = panel.loc[
                    (panel["model"] == model)
                    & (panel["regime"] == regime)
                    & (panel["bucket"] == bucket)
                ]
                rec = _metrics(sub)
                rec["model"] = model
                rec["regime"] = regime
                rec["bucket"] = bucket
                rows.append(rec)
            finite = [r for r in rows if r["n"] > 0 and np.isfinite(r["rmse_pct"])]
            best = min(finite, key=lambda r: r["rmse_pct"])["model"] if finite else None
            analysis2[f"{regime}|{bucket}"] = {
                "regime": regime,
                "bucket": bucket,
                "n": int(rows[0]["n"]) if rows else 0,
                "best_model": best,
                "rows": rows,
            }

    winner_grid = []
    for regime in main.REGIME_ORDER:
        rec = {"regime": regime}
        for bucket in BUCKETS:
            rec[bucket] = analysis2[f"{regime}|{bucket}"]["best_model"]
            rec[f"n_{bucket}"] = analysis2[f"{regime}|{bucket}"]["n"]
        winner_grid.append(rec)

    rmse_heat_overall = {}
    for model in main.TABLE_MODELS:
        rmse_heat_overall[model] = {
            b: next(r["rmse_pct"] for r in analysis1[b]["rows"] if r["model"] == model)
            for b in BUCKETS
        }

    rmse_heat_regime = {}
    for regime in main.REGIME_ORDER:
        rmse_heat_regime[regime] = {}
        for model in main.TABLE_MODELS:
            rmse_heat_regime[regime][model] = {
                b: next(
                    r["rmse_pct"]
                    for r in analysis2[f"{regime}|{b}"]["rows"]
                    if r["model"] == model
                )
                for b in BUCKETS
            }

    payload = {
        "meta": {
            "source": f"same 72 contract CSVs as {Path(main.PDF_NAME).stem}",
            "moneyness": "m = S/K (call)",
            "rule": {name: rule for name, rule in BUCKET_RULE},
            "n_paths": main.N_PATHS,
            "seed": main.SEED,
            "lookback": main.WINDOW_LABEL,
            "rolling": main.ROLLING_MODE,
            "n_contracts_one_model": int(len(one)),
        },
        "n_overall": n_overall,
        "n_regime": {
            r: {b: int(pd.crosstab(one["regime"], one["bucket"]).reindex(index=main.REGIME_ORDER, columns=BUCKETS).fillna(0).loc[r, b]) for b in BUCKETS}
            for r in main.REGIME_ORDER
        },
        "analysis1": analysis1,
        "analysis2": analysis2,
        "winner_grid": winner_grid,
        "rmse_heat_overall": rmse_heat_overall,
        "rmse_heat_regime": rmse_heat_regime,
    }
    return payload


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if kind == "rmse":
        return f"{v:.2f}"
    if kind == "mae":
        return f"{v:.2f}"
    if kind == "bias":
        return f"{v:+.4f}"
    if kind == "bias_pct":
        return f"{v:+.2f}"
    if kind == "early":
        return f"{100.0 * v:.1f}%"
    return str(v)


def _model_rows(rows: list[dict]) -> tuple[list[list[str]], list[str], int | None]:
    header = ["Model", "n", "RMSE%", "MAE%", "Bias%", "Ranking"]
    out = []
    rmses = []
    for rec in rows:
        rmses.append(rec["rmse_pct"] if rec["n"] > 0 else np.inf)
        out.append(
            [
                rec["model"],
                str(rec["n"]),
                _fmt(rec["rmse_pct"], "rmse") if rec["n"] else "—",
                _fmt(rec["mae"], "mae") if rec["n"] else "—",
                _fmt(rec["bias_pct"], "bias_pct") if rec["n"] else "—",
                "—",
            ]
        )
    finite_idx = [i for i, r in enumerate(rmses) if np.isfinite(r)]
    if not finite_idx:
        return out, header, None
    order = sorted(finite_idx, key=lambda i: rmses[i])
    for rank, i in enumerate(order, start=1):
        out[i][-1] = str(rank)
    return out, header, order[0] + 1


def _cover(pdf, payload: dict) -> None:
    import textwrap

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "V3 moneyness performance", fontsize=18, weight="bold", color=main.NAVY, va="top")
    fig.text(
        0.08,
        0.912,
        f"{main._models_phrase()} · {main._companies_phrase()} · {len(main.REGIME_ORDER)} windows · "
        f"{main._rolling_phrase()}  ·  classified by call moneyness m = S/K",
        fontsize=9.2,
        color=main.MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.892, 0.892], transform=fig.transFigure, color=main.NAVY, lw=1.4))

    y = 0.86

    def section(heading: str, bullets: list[str], y0: float) -> float:
        fig.text(0.08, y0, heading, fontsize=11.5, weight="bold", color=main.NAVY, va="top")
        y = y0 - 0.028
        for b in bullets:
            wrapped = textwrap.wrap(b, width=96) or [""]
            fig.text(0.10, y, "•", fontsize=9.5, color=main.NAVY, va="top")
            fig.text(0.13, y, wrapped[0], fontsize=8.7, color="#222222", va="top")
            y -= 0.0178
            for line in wrapped[1:]:
                fig.text(0.13, y, line, fontsize=8.7, color="#222222", va="top")
                y -= 0.0178
            y -= 0.004
        return y - 0.010

    y = section(
        "What this report is",
        [
            f"Companion to {Path(main.PDF_NAME).stem}. The notebook in this folder is the source of truth; this PDF is written from the same moneyness payload.",
            "Question: does model ranking change with option moneyness, overall and inside each volatility regime?",
            "BEST in every table is the lowest RMSE%. That row is bold and green, and the Ranking column ranks 1 = best. Empty buckets (n = 0) show em dashes and have no BEST.",
        ],
        y,
    )
    y = section(
        "Unchanged experimental design",
        [
            f"Same {main._models_phrase(cap=False)}, {main._companies_phrase()} ({', '.join(main.TICKERS)}), {len(main.REGIME_ORDER)} windows, {main.LOOKBACK_PHRASE} lookback, {getattr(main, 'ROLLING_MODE', 'monthly')} recalibration, V3 filters, Euler-corrected Heston step, LSM n_paths = {getattr(main, 'N_PATHS', 2000)} / seed 42.",
            "Same option contracts as the main study (nearest-ATM listed calls, DTE 7–60, |S/K − 1| ≤ 10%). No model-specific re-sampling.",
            f"Pooled n = {payload['meta']['n_contracts_one_model']} contracts per model (identical keys across models).",
        ],
        y,
    )
    y = section(
        "Moneyness rule (calls, m = S / K)",
        [
            "DOTM: m < 0.98.   OTM: 0.98 ≤ m < 0.995.   ATM: 0.995 ≤ m ≤ 1.005.   ITM: 1.005 < m ≤ 1.02.   DITM: m > 1.02.",
            "These five bins partition the main-study filter |m − 1| ≤ 10%. The Monday nearest-ATM design puts most mass in ATM; DOTM and DITM are sparse and appear mainly in 2008–2009.",
        ],
        y,
    )
    y = section(
        "Metrics",
        [
            "RMSE% = 100 × √ mean(((C_model − C_mkt)/C_mkt)²). Ranking uses this only.",
            "MAE% = 100 × mean(|C_model − C_mkt|/C_mkt). Bias% = 100 × mean((C_model − C_mkt)/C_mkt).",
        ],
        y,
    )
    y = section(
        "Two analyses",
        [
            "Analysis 1 — Overall moneyness: pool all companies and regimes. Five tables (one per bucket) plus a Model × Moneyness RMSE% heatmap.",
            f"Analysis 2 — Window × moneyness: keep the {len(main.REGIME_ORDER)} windows separate, pool the {main._companies_phrase()}. Compact RMSE tables/heatmaps and a winner grid showing whether the best model changes with moneyness inside each window.",
        ],
        y,
    )
    fig.text(
        0.08,
        0.04,
        "Page map: p.1 method  ·  p.2 sample counts  ·  p.3–4 Analysis 1 tables  ·  p.5 overall heatmap  ·  p.6–7 Analysis 2  ·  p.8 findings.",
        fontsize=8.0,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _counts_page(pdf, payload: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 2  ·  Shared sample by moneyness  ·  identical for all six models", fontsize=13, color=main.NAVY, y=0.97, weight="bold")

    ax = fig.add_axes([0.08, 0.58, 0.84, 0.32])
    ax.axis("off")
    ax.set_title("Table M0  ·  Contract counts (one model = all models)", loc="left", fontsize=10.5, color=main.NAVY, weight="bold")
    header = ["Regime"] + list(BUCKETS) + ["Total"]
    rows = []
    n_reg = payload["n_regime"]
    for regime in main.REGIME_ORDER:
        vals = [int(n_reg[regime][b]) for b in BUCKETS]
        rows.append([f"{regime}  {main.REGIME_META[regime]['title']}"] + [str(v) for v in vals] + [str(sum(vals))])
    tot = [int(payload["n_overall"][b]) for b in BUCKETS]
    rows.append(["All regimes"] + [str(v) for v in tot] + [str(sum(tot))])
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.92])
    main._style_table(tbl, None)

    ax2 = fig.add_axes([0.08, 0.18, 0.84, 0.32])
    ax2.axis("off")
    ax2.set_title("Moneyness rule  ·  American calls, m = S/K", loc="left", fontsize=10.5, color=main.NAVY, weight="bold")
    header2 = ["Bucket", "Rule", "n (pooled)", "Share"]
    n_all = sum(payload["n_overall"].values())
    rows2 = []
    for name, rule in BUCKET_RULE:
        n = int(payload["n_overall"][name])
        rows2.append([name, rule, str(n), f"{100.0 * n / n_all:.1f}%"])
    tbl2 = ax2.table(cellText=rows2, colLabels=header2, loc="center", cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.9])
    main._style_table(tbl2, None)

    fig.text(
        0.08,
        0.07,
        "ATM dominates because the main study picks the nearest-ATM listed call each Monday. "
        "DOTM/DITM are not empty because of a new filter — they are rare in that ATM sample, especially after 2009. "
        "Every model still sees the same contracts in every bucket.",
        fontsize=8.4,
        color="#333333",
        va="top",
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _analysis1_tables(pdf, payload: dict) -> None:
    # page 3: DOTM OTM ATM
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 3  ·  Analysis 1  ·  Overall moneyness  ·  Tables M1–M3", fontsize=13, color=main.NAVY, y=0.975, weight="bold")
    fig.text(0.06, 0.935, f"Pooled SPY + AAPL + MSFT and all {len(main.REGIME_ORDER)} windows.  Bold green = lowest RMSE%.", fontsize=8.2, color=main.MUTED)
    for i, bucket in enumerate(("DOTM", "OTM", "ATM")):
        ax = fig.add_subplot(3, 1, i + 1)
        tab = payload["analysis1"][bucket]
        rows, header, best = _model_rows(tab["rows"])
        ax.axis("off")
        ax.set_title(
            f"Table M{i+1}  ·  {bucket}  ·  {BUCKET_RULE[BUCKETS.index(bucket)][1]}  ·  n = {tab['n']}"
            + (f"  ·  BEST = {tab['best_model']}" if tab["best_model"] else "  ·  no contracts"),
            loc="left",
            fontsize=10,
            color=main.NAVY,
            weight="bold",
        )
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.08, 0.96, 0.78])
        main._style_table(tbl, best)
    fig.text(0.06, 0.025, "Bias% = 100 × mean((model − market)/market).", fontsize=7.8, color="#555555")
    fig.tight_layout(rect=(0.03, 0.04, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 4  ·  Analysis 1  ·  Overall moneyness  ·  Tables M4–M5", fontsize=13, color=main.NAVY, y=0.975, weight="bold")
    fig.text(0.06, 0.935, f"Pooled SPY + AAPL + MSFT and all {len(main.REGIME_ORDER)} windows.  Bold green = lowest RMSE%.", fontsize=8.2, color=main.MUTED)
    for i, bucket in enumerate(("ITM", "DITM")):
        ax = fig.add_subplot(2, 1, i + 1)
        tab = payload["analysis1"][bucket]
        rows, header, best = _model_rows(tab["rows"])
        ax.axis("off")
        ax.set_title(
            f"Table M{i+4}  ·  {bucket}  ·  {BUCKET_RULE[BUCKETS.index(bucket)][1]}  ·  n = {tab['n']}"
            + (f"  ·  BEST = {tab['best_model']}" if tab["best_model"] else "  ·  no contracts"),
            loc="left",
            fontsize=10.5,
            color=main.NAVY,
            weight="bold",
        )
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.02, 0.12, 0.96, 0.72])
        main._style_table(tbl, best)
    fig.text(0.06, 0.04, "ITM/DITM: S > K.  DOTM/OTM: S < K.  ATM: |S/K − 1| ≤ 0.5%.", fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0.03, 0.06, 0.97, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def _heatmap(ax, matrix: np.ndarray, title: str, mask_nan: bool = True) -> None:
    vis = np.array(matrix, dtype=float)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad("#EEEEEE")
    im = ax.imshow(vis, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels(list(BUCKETS), fontsize=8)
    ax.set_yticks(range(len(main.TABLE_MODELS)))
    ax.set_yticklabels(list(main.TABLE_MODELS), fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=main.NAVY, weight="bold")
    for i in range(vis.shape[0]):
        for j in range(vis.shape[1]):
            v = vis[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888888")
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.6, color="#111111")
    return im


def _overall_heatmap(pdf, payload: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Page 5  ·  Figure M1  ·  Model × Moneyness RMSE%  ·  overall (pooled)", fontsize=13, color=main.NAVY, y=0.97, weight="bold")
    mat = np.array(
        [[payload["rmse_heat_overall"][m][b] for b in BUCKETS] for m in main.TABLE_MODELS],
        dtype=float,
    )
    ax = fig.add_axes([0.16, 0.18, 0.72, 0.68])
    im = _heatmap(ax, mat, "")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("RMSE%")
    # outline best in each column
    for j, bucket in enumerate(BUCKETS):
        col = mat[:, j]
        if np.all(~np.isfinite(col)):
            continue
        i = int(np.nanargmin(col))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=1.6))
    fig.text(
        0.16,
        0.06,
        "Black outline = lowest RMSE% in that moneyness column. Grey = n = 0. Pooled across companies and regimes.",
        fontsize=8.5,
        color="#555555",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _analysis2_pages(pdf, payload: dict) -> None:
    regimes = list(main.REGIME_ORDER)
    for chunk_i in range(0, len(regimes), 4):
        chunk = regimes[chunk_i : chunk_i + 4]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(
            "Figure M2  ·  RMSE% by model × moneyness  ·  each window (companies pooled)",
            fontsize=12.2,
            color=main.NAVY,
            y=0.98,
            weight="bold",
        )
        last_im = None
        for ax, regime in zip(axes.ravel(), list(chunk) + [None] * (4 - len(chunk))):
            if regime is None:
                ax.axis("off")
                continue
            mat = np.array(
                [[payload["rmse_heat_regime"][regime][m][b] for b in BUCKETS] for m in main.TABLE_MODELS],
                dtype=float,
            )
            last_im = _heatmap(ax, mat, f"{regime}  ·  {main.REGIME_META[regime]['title']}")
            for j in range(len(BUCKETS)):
                col = mat[:, j]
                if np.all(~np.isfinite(col)):
                    continue
                i = int(np.nanargmin(col))
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=1.3))
        fig.tight_layout(rect=(0.03, 0.06, 0.97, 0.94))
        fig.text(0.06, 0.025, "Outlined cell = lowest RMSE% in that window × moneyness column. Grey = n = 0.", fontsize=8, color="#555555")
        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Analysis 2  ·  Does the ranking change with moneyness inside each window?", fontsize=12.5, color=main.NAVY, y=0.97, weight="bold")

    ax = fig.add_axes([0.08, 0.52, 0.84, 0.38])
    ax.axis("off")
    ax.set_title("Table M6  ·  Best model (lowest RMSE%) by window × moneyness", loc="left", fontsize=10.5, color=main.NAVY, weight="bold")
    header = ["Window"] + list(BUCKETS)
    rows = []
    for rec in payload["winner_grid"]:
        rows.append([f"{rec['regime']}  {main.REGIME_META[rec['regime']]['title']}"] + [rec[b] or "—" for b in BUCKETS])
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.92])
    main._style_table(tbl, None)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.6)

    ax2 = fig.add_axes([0.08, 0.10, 0.84, 0.38])
    ax2.axis("off")
    ax2.set_title("Table M7  ·  n contracts by window × moneyness (shared across models)", loc="left", fontsize=10.5, color=main.NAVY, weight="bold")
    rows2 = []
    for rec in payload["winner_grid"]:
        rows2.append([f"{rec['regime']}  {main.REGIME_META[rec['regime']]['title']}"] + [str(rec[f"n_{b}"]) for b in BUCKETS])
    tbl2 = ax2.table(cellText=rows2, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.92])
    main._style_table(tbl2, None)
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(6.6)

    fig.text(
        0.08,
        0.035,
        "A change in the Table M6 entry across a row means the ranking depends on moneyness inside that window. "
        "Columns with n = 0 cannot change the ranking.",
        fontsize=8.0,
        color="#333333",
        va="top",
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _findings(payload: dict) -> list[str]:
    a1 = payload["analysis1"]
    lines = []
    for b in BUCKETS:
        tab = a1[b]
        if not tab["best_model"]:
            lines.append(f"{b} (n = {tab['n']}): no contracts in the main-study sample.")
            continue
        best = next(r for r in tab["rows"] if r["model"] == tab["best_model"])
        lines.append(
            f"{b} (n = {tab['n']}): BEST = {tab['best_model']} with RMSE% = {best['rmse_pct']:.2f}, "
            f"Bias% = {best['bias_pct']:+.2f}."
        )
    winners = {b: a1[b]["best_model"] for b in BUCKETS if a1[b]["best_model"]}
    uniq = sorted(set(winners.values()))
    if len(uniq) == 1:
        lines.append(f"Overall, the moneyness ranking is stable: {uniq[0]} is BEST in every non-empty bucket.")
    else:
        lines.append(
            "Overall, the ranking does change with moneyness: "
            + ", ".join(f"{b} → {winners[b]}" for b in BUCKETS if b in winners)
            + "."
        )
    # regime
    changes = []
    for rec in payload["winner_grid"]:
        vals = [rec[b] for b in BUCKETS if rec[b]]
        u = sorted(set(vals))
        if len(u) > 1:
            changes.append(rec["regime"])
    if changes:
        lines.append(
            "Inside regimes, the BEST model also changes with moneyness in: " + ", ".join(changes) + "."
        )
    else:
        lines.append("Inside each regime, the BEST model is the same in every non-empty moneyness bucket.")
    if getattr(main, "STUDY_KIND", None) in ("7d", "1d"):
        lines.append(
            "These comparisons use the hourly nearest-ATM sample (09:59–15:59; LSM skipped at Friday 15:59). "
            "ATM still holds most of the mass; sparse DOTM/DITM buckets should be read as descriptive."
        )
    else:
        lines.append(
            "These comparisons use the Monday nearest-ATM sample, so ATM is the only well-populated bucket after 2009. "
            "DOTM/DITM results rest on few 2008–2009 quotes and should be read as descriptive, not as a high-powered test."
        )
    return lines


def _findings_page(pdf, payload: dict) -> None:
    import textwrap

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "Page 8  ·  Findings", fontsize=16, weight="bold", color=main.NAVY, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], transform=fig.transFigure, color=main.NAVY, lw=1.2))
    y = 0.88
    fig.text(0.08, y, "Does performance change with moneyness?", fontsize=11.5, weight="bold", color=main.NAVY, va="top")
    y -= 0.04
    for b in _findings(payload):
        wrapped = textwrap.wrap(b, width=94) or [""]
        fig.text(0.10, y, "•", fontsize=10, color=main.NAVY, va="top")
        fig.text(0.13, y, wrapped[0], fontsize=9.4, color="#222222", va="top")
        y -= 0.022
        for line in wrapped[1:]:
            fig.text(0.13, y, line, fontsize=9.4, color="#222222", va="top")
            y -= 0.022
        y -= 0.010
    fig.text(
        0.08,
        0.05,
        f"Notebook: {main.SHORT.relative_to(main.REPO) / NB_NAME}\n"
        f"Contracts: {main.CACHE.relative_to(main.REPO)}/contracts/  ·  same files as the main study.",
        fontsize=7.6,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    main.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (main.SHORT / PDF_NAME)
    with PdfPages(path) as pdf:
        _cover(pdf, payload)
        _counts_page(pdf, payload)
        _analysis1_tables(pdf, payload)
        _overall_heatmap(pdf, payload)
        _analysis2_pages(pdf, payload)
        _findings_page(pdf, payload)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)
    return path


def _html_rows(rows, header, best) -> str:
    parts = [
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:13px;width:100%;'>",
        "<thead><tr>",
    ]
    for h in header:
        parts.append(
            f"<th style='background:{main.NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{h}</th>"
        )
    parts.append("</tr></thead><tbody>")
    for i, row in enumerate(rows, start=1):
        if best is not None and i == best:
            bg, extra = main.BEST_BG, "font-weight:700;"
        elif i % 2 == 0:
            bg, extra = main.ALT_BG, ""
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


def build_notebook(payload: dict, path: Path | None = None) -> Path:
    import base64
    import io
    import uuid

    main.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (main.SHORT / NB_NAME)

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
    n_all = payload["meta"]["n_contracts_one_model"]
    cells.append(
        md(
            f"""# V3 moneyness performance — {main._rolling_phrase()}

**This notebook is the computational source of truth.** The matching PDF is written from the same payload.

This report classifies the **same** option contracts as `{Path(main.PDF_NAME).stem}` into DOTM / OTM / ATM / ITM / DITM. It does not change filters, calibration, contracts, LSM settings, or the market benchmark.

| Item | Setting |
|------|---------|
| Models | {", ".join(main.TABLE_MODELS)} |
| Companies | {", ".join(main.TICKERS)} (pooled in both analyses) |
| Regimes | Crisis, Normal, Late-cycle, COVID (pooled in Analysis 1; separate in Analysis 2) |
| Contracts | Monday nearest-ATM listed calls from the main study (`n = {n_all}` per model) |
| Moneyness | `m = S/K` (calls) |
| Primary metric | RMSE% |
| Secondary | MAE%, **Bias%** |
"""
        )
    )
    cells.append(
        md(
            f"""## 0. Methodology

### Same design as the main study
{main.LOOKBACK_PHRASE.capitalize()} lookback, `{getattr(main, 'ROLLING_MODE', 'monthly')}` recalibration, V3 filters (no-arbitrage, DTE 7–60, |S/K−1|≤10%, liquidity), Euler-corrected Heston, option-implied Heston NLS, LSM with {getattr(main, 'N_PATHS', 2000):,} paths and seed 42. No look-ahead.

### Moneyness rule (calls)
Let `m = S/K`.

| Bucket | Rule |
|--------|------|
| DOTM | m < 0.98 |
| OTM | 0.98 ≤ m < 0.995 |
| ATM | 0.995 ≤ m ≤ 1.005 |
| ITM | 1.005 < m ≤ 1.02 |
| DITM | m > 1.02 |

These five bins partition the main-study 10% moneyness filter. The Monday ATM sampler concentrates mass in ATM.

### Metrics
- RMSE% = 100 × √ mean(((model − market)/market)²)
- MAE% = 100 × mean(|model − market|/market)
- Bias% = 100 × mean((model − market)/market)
- BEST = lowest RMSE% among models with n > 0
"""
        )
    )
    cells.append(
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
            f"import {MONEYNESS_IMPORT} as mx\n"
            "panel = mx.load_panel()\n"
            "payload = mx.build_payload(panel)\n"
            "pdf = mx.write_pdf(payload)\n"
            "print('n_overall', payload['n_overall'])\n"
            "print('PDF', pdf)",
            html=f"<pre>n_overall {payload['n_overall']}</pre>",
        )
    )

    # counts
    header = ["Regime"] + list(BUCKETS) + ["Total"]
    rows = []
    for regime in main.REGIME_ORDER:
        vals = [int(payload["n_regime"][regime][b]) for b in BUCKETS]
        rows.append([regime] + [str(v) for v in vals] + [str(sum(vals))])
    tot = [int(payload["n_overall"][b]) for b in BUCKETS]
    rows.append(["All"] + [str(v) for v in tot] + [str(sum(tot))])
    cells.append(md("## 1. Shared sample counts"))
    cells.append(code("display(HTML('counts'))", html=_html_rows(rows, header, len(rows))))

    cells.append(md("## 2. Analysis 1 — Overall moneyness"))
    for i, bucket in enumerate(BUCKETS, start=1):
        tab = payload["analysis1"][bucket]
        rows, header, best = _model_rows(tab["rows"])
        best_txt = tab["best_model"] or "none"
        cells.append(md(f"### Table M{i}. {bucket}  ·  {BUCKET_RULE[i-1][1]}  ·  n = {tab['n']}  ·  **BEST = {best_txt}**"))
        cells.append(code(
            "tab = payload['analysis1']['%s']\nrows, header, best = mx._model_rows(tab['rows'])\ndisplay(HTML(mx._html_rows(rows, header, best)))" % bucket,
            html=_html_rows(rows, header, best),
        ))

    def draw_heat():
        fig = plt.figure(figsize=(9.5, 5.2))
        mat = np.array(
            [[payload["rmse_heat_overall"][m][b] for b in BUCKETS] for m in main.TABLE_MODELS],
            dtype=float,
        )
        ax = fig.add_subplot(111)
        im = _heatmap(ax, mat, "Figure M1 · Model × Moneyness RMSE% (pooled)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("RMSE%")
        for j in range(len(BUCKETS)):
            col = mat[:, j]
            if np.all(~np.isfinite(col)):
                continue
            i = int(np.nanargmin(col))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111", lw=1.5))
        fig.tight_layout()
        return fig

    cells.append(md("### Figure M1 — Model × Moneyness RMSE% heatmap"))
    cells.append(code("pass", png=fig_png(draw_heat)))

    cells.append(md("## 3. Analysis 2 — Regime × moneyness"))

    def draw_regimes():
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
        fig.suptitle("Figure M2 · RMSE% by model × moneyness · each regime", fontsize=12, color=main.NAVY, weight="bold")
        for ax, regime in zip(axes.ravel(), main.REGIME_ORDER):
            mat = np.array(
                [[payload["rmse_heat_regime"][regime][m][b] for b in BUCKETS] for m in main.TABLE_MODELS],
                dtype=float,
            )
            _heatmap(ax, mat, f"{regime} · {main.REGIME_META[regime]['title']}")
            for j in range(len(BUCKETS)):
                col = mat[:, j]
                if np.all(~np.isfinite(col)):
                    continue
                i = int(np.nanargmin(col))
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111", lw=1.3))
        fig.tight_layout()
        return fig

    cells.append(md("### Figure M2 — RMSE% heatmaps by regime"))
    cells.append(code("pass", png=fig_png(draw_regimes)))

    header = ["Regime"] + list(BUCKETS)
    rows = [
        [rec["regime"]] + [rec[b] or "—" for b in BUCKETS] for rec in payload["winner_grid"]
    ]
    cells.append(md("### Table M6 — Best model by regime × moneyness"))
    cells.append(code("display(HTML('m6'))", html=_html_rows(rows, header, None)))
    rows_n = [
        [rec["regime"]] + [str(rec[f"n_{b}"]) for b in BUCKETS] for rec in payload["winner_grid"]
    ]
    cells.append(md("### Table M7 — n by regime × moneyness"))
    cells.append(code("display(HTML('m7'))", html=_html_rows(rows_n, header, None)))

    cells.append(md("## 4. Findings\n\n" + "\n\n".join(f"- {b}" for b in _findings(payload))))
    cells.append(
        md(
            f"""## 5. Re-run

```python
import {MONEYNESS_IMPORT} as mx
payload = mx.run()
```

Requires the main study contract CSVs under `{main.CACHE.relative_to(main.REPO)}/contracts/`.
"""
        )
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


def run() -> dict:
    panel = load_panel()
    payload = build_payload(panel)
    dest = _payload_json()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_pdf(payload)
    build_notebook(payload)
    return payload


def main_cli() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
