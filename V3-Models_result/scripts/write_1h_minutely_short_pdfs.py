#!/usr/bin/env python3
"""Write Results_In_Short PDFs for the V2 1-hour / minutely RMSE report.

Reads V3-Models_result/results/rmse_report_1h_minutely.json and writes:

  Results_In_Short/1 hour minutely/ALL_TABLES.pdf
  Results_In_Short/1 hour minutely/RMSE_BAR_CHARTS.pdf

Layout matches the 7-day ALL_TABLES / RMSE_BAR_CHARTS form:
  tables — one block per ticker, 7-day | 1-day columns
  bars   — models on x, grouped bars (blue 7-day / red 1-day)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v2_rmse_mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
JSON_PATH = ROOT / "results" / "rmse_report_1h_minutely.json"
SHORT = REPO / "Results_In_Short" / "1 hour minutely"

TICKERS = ("SPY", "AAPL", "MSFT")
MODELS = ("GBM", "Merton", "Heston–Merton", "GARCH", "GARCH–Merton")
WIN_COLORS = {"7-day": "#4C72B0", "1-day": "#C44E52"}  # same blue/red as hourly/minutely
WIDTH = 0.35


def _load() -> dict[tuple[str, str, str], dict]:
    rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {(r["window"], r["model"], r["ticker"]): r for r in rows}


def _val(idx, window, model, ticker, key) -> float:
    return float(idx[(window, model, ticker)][key])


def _best_window(a: float, b: float) -> str:
    return "7-day" if a <= b else "1-day"


def _style_table(tbl) -> None:
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FB")


def _ticker_table(ax, idx, ticker: str, metric: str, col_title: str) -> None:
    ax.axis("off")
    ax.set_title(f"{ticker}  —  {col_title}", loc="left", fontsize=11, pad=10)
    header = ["Model", "7-day", "1-day", "best window"]
    cells = []
    for model in MODELS:
        a = _val(idx, "7-day", model, ticker, metric)
        b = _val(idx, "1-day", model, ticker, metric)
        cells.append([model, f"{a:.4f}", f"{b:.4f}", _best_window(a, b)])
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    _style_table(tbl)
    tbl.auto_set_column_width(col=list(range(len(header))))


def write_tables(idx) -> Path:
    SHORT.mkdir(parents=True, exist_ok=True)
    path = SHORT / "ALL_TABLES.pdf"
    pages = [
        {
            "metric": "rmse_st",
            "title": "V2  ·  RMSE(Sₜ)  ·  lookback 1 hour  ·  rolling minutely",
            "subtitle": (
                "Stock-path RMSE: Monte Carlo mean path (1,000 paths) vs historical close.  "
                "Continuous RTH (overnight / weekend gaps removed).\n"
                "7-day = 9–15 Mar 2023 (1,945 minutely calibrations).    "
                "1-day = 15 Mar 2023 (389 minutely calibrations)."
            ),
            "col": "RMSE(Sₜ)",
        },
        {
            "metric": "rmse_stop",
            "title": "V2  ·  RMSE(stopping)  ·  lookback 1 hour  ·  rolling minutely",
            "subtitle": (
                "Optimal-stopping RMSE: LSM American call vs Black–Scholes European.  "
                "12 synthetic ATM/OTM/ITM contracts per name, expiry 2023-03-15 15:59, 500 LSM paths.\n"
                "No 2023 listed quotes in the project options panel.    "
                "Same contracts and seeds across models."
            ),
            "col": "RMSE(LSM vs BS)",
        },
    ]
    with PdfPages(path) as pdf:
        for page in pages:
            fig = plt.figure(figsize=(11, 9.5))
            fig.suptitle(page["title"], fontsize=13, y=0.98)
            fig.text(0.5, 0.935, page["subtitle"], ha="center", va="top", fontsize=8.2, color="#333333")
            for i, ticker in enumerate(TICKERS):
                ax = fig.add_subplot(3, 1, i + 1)
                _ticker_table(ax, idx, ticker, page["metric"], page["col"])
            fig.tight_layout(rect=(0.03, 0.02, 0.97, 0.91))
            pdf.savefig(fig)
            plt.close(fig)
    return path


def _annotate_bars(ax, xloc, values, offset) -> None:
    ymax = max(values) if values else 1.0
    for x, v in zip(xloc, values):
        ax.text(
            x + offset,
            v + 0.012 * ymax,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222222",
        )


def _plot_grouped(ax, idx, ticker: str, metric: str, ylabel: str, title: str) -> None:
    seven = [_val(idx, "7-day", m, ticker, metric) for m in MODELS]
    one = [_val(idx, "1-day", m, ticker, metric) for m in MODELS]
    xloc = np.arange(len(MODELS))
    ax.bar(xloc - WIDTH / 2, seven, WIDTH, label="7-day", color=WIN_COLORS["7-day"])
    ax.bar(xloc + WIDTH / 2, one, WIDTH, label="1-day", color=WIN_COLORS["1-day"])
    _annotate_bars(ax, xloc, seven, -WIDTH / 2)
    _annotate_bars(ax, xloc, one, WIDTH / 2)
    ax.set_xticks(xloc)
    ax.set_xticklabels(list(MODELS))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, linestyle="-", alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ymax = max(seven + one)
    ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1.0)


def write_bars(idx) -> Path:
    SHORT.mkdir(parents=True, exist_ok=True)
    path = SHORT / "RMSE_BAR_CHARTS.pdf"
    charts = []
    for ticker in TICKERS:
        charts.append(
            {
                "ticker": ticker,
                "metric": "rmse_st",
                "ylabel": "RMSE(Sₜ)",
                "title": (
                    f"{ticker}  —  stock-path RMSE(Sₜ) by model  ·  "
                    "lookback 1 hour  ·  rolling minutely"
                ),
            }
        )
    for ticker in TICKERS:
        charts.append(
            {
                "ticker": ticker,
                "metric": "rmse_stop",
                "ylabel": "RMSE(LSM vs BS)",
                "title": (
                    f"{ticker}  —  stopping RMSE (LSM American vs BS European) by model  ·  "
                    "lookback 1 hour  ·  rolling minutely"
                ),
            }
        )
    with PdfPages(path) as pdf:
        for spec in charts:
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            _plot_grouped(ax, idx, spec["ticker"], spec["metric"], spec["ylabel"], spec["title"])
            fig.text(
                0.08,
                0.02,
                "V2  ·  lookback = 1 hour (60 RTH minutes)  ·  rolling = minutely  ·  "
                "7-day: 9–15 Mar 2023    1-day: 15 Mar 2023",
                fontsize=7.5,
                color="#555555",
                ha="left",
                va="bottom",
            )
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            pdf.savefig(fig)
            plt.close(fig)
    return path


def main() -> int:
    idx = _load()
    t = write_tables(idx)
    b = write_bars(idx)
    print(f"wrote {t.relative_to(REPO)}")
    print(f"wrote {b.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
