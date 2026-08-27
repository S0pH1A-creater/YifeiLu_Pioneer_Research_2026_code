#!/usr/bin/env python3
"""Four Results_In_Short PDFs for the Friday 17 Mar 2023 / 1-hour lookback study."""
from __future__ import annotations

import json
import os
import textwrap
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
JSON_PATH = ROOT / "results" / "rmse_report_20230317_1h.json"
SHORT = REPO / "Results_In_Short" / "2023-03-17 expiry 1 hour lookback"

TICKERS = ("SPY", "AAPL", "MSFT")
MODELS_ALL = ("GBM", "Merton", "Heston", "Heston–Merton", "GARCH", "GARCH–Merton")
MODELS_BARS = ("GBM", "Merton", "GARCH", "GARCH–Merton")
BAR_COLORS = {
    "GBM": "#4C72B0",
    "Merton": "#55A868",
    "GARCH": "#C44E52",
    "GARCH–Merton": "#8172B3",
}
BLOCKS = (
    ("1-day", "hourly", "1-day  ·  hourly rolling"),
    ("1-day", "minutely", "1-day  ·  minutely rolling"),
    ("7-day", "hourly", "5-day  ·  hourly rolling"),
    ("7-day", "minutely", "5-day  ·  minutely rolling"),
)

NAVY = "#1F3A5F"
MUTED = "#444444"


def _load() -> dict[tuple[str, str, str, str], dict]:
    rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {(r["window"], r["rolling"], r["model"], r["ticker"]): r for r in rows}


def _mean(idx, window: str, rolling: str, model: str, key: str, tickers=TICKERS) -> float:
    return float(np.mean([idx[(window, rolling, model, t)][key] for t in tickers]))


def _val(idx, window: str, rolling: str, model: str, ticker: str, key: str) -> float:
    return float(idx[(window, rolling, model, ticker)][key])


def _cover(pdf: PdfPages, title: str, purpose: list[str], method: list[str], extra: str | None = None) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=17, weight="bold", color=NAVY, va="top", ha="left")
    fig.text(
        0.08,
        0.905,
        "Friday 17 March 2023 monthly expiry   ·   1-hour lookback   ·   SPY / AAPL / MSFT",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.885, 0.885], transform=fig.transFigure, color=NAVY, lw=1.4))

    y = 0.85

    def section(heading: str, bullets: list[str], y0: float) -> float:
        fig.text(0.08, y0, heading, fontsize=12.5, weight="bold", color=NAVY, va="top")
        y = y0 - 0.035
        for b in bullets:
            wrapped = textwrap.wrap(b, width=94) or [""]
            fig.text(0.10, y, "•", fontsize=10.2, color=NAVY, va="top")
            fig.text(0.13, y, wrapped[0], fontsize=10.0, color="#222222", va="top")
            y -= 0.0205
            for line in wrapped[1:]:
                fig.text(0.13, y, line, fontsize=10.0, color="#222222", va="top")
                y -= 0.0205
            y -= 0.006
        return y - 0.014

    y = section("What this PDF is for", purpose, y)
    y = section("Methodology", method, y)
    if extra:
        fig.text(0.08, 0.07, extra, fontsize=8.2, color="#666666", va="bottom", wrap=True)
    fig.text(
        0.08,
        0.04,
        "Windows: 1-day = Fri 17 Mar 2023 09:30–15:59 ET (RTH).   "
        "5-day = Mon 13 Mar – Fri 17 Mar 2023 (five RTH sessions).",
        fontsize=8.0,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _style_table(tbl, best_row: int | None) -> None:
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif best_row is not None and r == best_row:
            cell.set_facecolor("#E4EED8")
            cell.set_text_props(weight="bold", color="#1A1A1A")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FB")
        else:
            cell.set_facecolor("white")


def _metric_table(ax, rows: list[list[str]], header: list[str], title: str, best_row: int) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10.5, color=NAVY, pad=8, weight="bold")
    tbl = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.06, 0.96, 0.82],
    )
    _style_table(tbl, best_row)


def _stock_rows(getter) -> tuple[list[list[str]], int]:
    header_skip = 1
    rows = []
    rmses = []
    for model in MODELS_ALL:
        rmse = getter(model, "rmse_st")
        mae = getter(model, "mae_st")
        icp = getter(model, "icp_st")
        abw = getter(model, "abw_st")
        rmses.append(rmse)
        rows.append([model, f"{rmse:.3f}", f"{mae:.3f}", f"{100 * icp:.1f}%", f"{abw:.3f}"])
    best = int(np.argmin(rmses)) + header_skip
    return rows, best


def _option_rows(getter) -> tuple[list[list[str]], int]:
    header_skip = 1
    rows = []
    rmses = []
    for model in MODELS_ALL:
        rmse = getter(model, "rmse_stop")
        mae = getter(model, "mae_stop")
        rmses.append(rmse)
        rows.append([model, f"{rmse:.3f}", f"{mae:.3f}"])
    best = int(np.argmin(rmses)) + header_skip
    return rows, best


def _four_tables(pdf: PdfPages, page_title: str, footnote: str, make_rows) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(page_title, fontsize=13, color=NAVY, y=0.97, weight="bold")
    for i, (window, rolling, label) in enumerate(BLOCKS):
        ax = fig.add_subplot(2, 2, i + 1)
        rows, best = make_rows(window, rolling)
        header = ["Model", "RMSE", "MAE", "ICP", "Avg band width"] if len(rows[0]) == 5 else ["Model", "RMSE", "MAE"]
        _metric_table(ax, rows, header, label, best)
    fig.text(0.08, 0.035, footnote, fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def _bars_2x2(
    pdf: PdfPages,
    page_title: str,
    ylabel: str,
    values_fn,
    higher_better: bool = False,
    outline_best: bool = True,
    footnote: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(page_title, fontsize=13, color=NAVY, y=0.98, weight="bold")
    x = np.arange(len(MODELS_BARS))
    for ax, (window, rolling, label) in zip(axes.ravel(), BLOCKS):
        vals = [values_fn(window, rolling, m) for m in MODELS_BARS]
        colors = [BAR_COLORS[m] for m in MODELS_BARS]
        bars = ax.bar(x, vals, color=colors, width=0.72, zorder=3)
        ymax = max(vals) if vals else 1.0
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02 * ymax,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )
        if outline_best:
            best_i = int(np.argmax(vals) if higher_better else np.argmin(vals))
            bars[best_i].set_edgecolor("#111111")
            bars[best_i].set_linewidth(1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(list(MODELS_BARS), fontsize=8.0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(label, loc="left", fontsize=10.5, color=NAVY, weight="bold")
        ax.yaxis.grid(True, linestyle="-", alpha=0.28, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(MODELS_BARS) - 0.4)
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if footnote is None:
        footnote = "Lowest bar outlined  ·  Heston and Heston–Merton omitted (see tables PDF)."
        if higher_better:
            footnote = "Highest bar outlined  ·  Heston and Heston–Merton omitted (see tables PDF)."
    fig.text(0.08, 0.03, footnote, fontsize=8.0, color="#555555")
    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


STOCK_PURPOSE = [
    "This PDF records stock-path simulation results for the Friday 17 March 2023 monthly-expiry study.",
    "Each table compares six models on how closely the simulated path tracks the historical 1-minute price.",
    "Heston (no jumps) and Heston–Merton are in the tables. The matching bar-chart PDF drops both so the other four models stay readable.",
    "The first results page is the equal-weight mean of SPY, AAPL, and MSFT. Later pages repeat the layout for each ticker.",
    "In every block the best model is bold and shaded: lowest RMSE of the median (50th-percentile) path versus actual S_t.",
]

STOCK_METHOD = [
    "Lookback is 1 hour (60 consecutive regular-trading-hours minutes) at every calibration time.",
    "Rolling is either hourly (parameters held for one hour) or minutely (parameters updated every minute).",
    "Models: GBM, Merton jump-diffusion, Heston (no jumps), Heston–Merton, GARCH(1,1), and GARCH–Merton. Same 1-minute notebooks as the rest of V2.",
    "Each name is simulated with 1,000 Monte Carlo paths and a common seed of 42.",
    "The reported path is the 50th percentile across paths at each minute — not the Monte Carlo mean.",
    "RMSE and MAE compare that median path with historical S_t on the continuous RTH clock (overnight and weekend gaps removed).",
    "ICP is the share of historical prices that fall inside the 25th–75th percentile band.",
    "Average band width is the mean of (75th − 25th percentile) over the window.",
]

STOCK_BAR_PURPOSE = [
    "This PDF is the bar-chart companion to the stock-path simulation tables for 17 March 2023.",
    "Each page is a 2×2 of the four study cells: 1-day vs 5-day, and hourly vs minutely rolling.",
    "Four bars per panel: GBM, Merton, GARCH, GARCH–Merton.",
    "Heston and Heston–Merton are excluded from every chart. Their errors are larger on this 1-hour / 1-minute setup and would squash the other bars; the values remain in the tables PDF.",
    "The outlined bar is the best model on that panel (lowest RMSE or MAE; highest ICP).",
]

OPTION_PURPOSE = [
    "This PDF records option-exercise (optimal stopping) results: Longstaff–Schwartz American call prices versus listed market quotes.",
    "The same 12 contracts per ticker are priced under every model, so differences come from the dynamics, not from the sample.",
    "Six models are in the tables, including Heston (no jumps) and Heston–Merton. On this 1-hour / 1-minute calibration both Heston and Heston–Merton risk-neutral paths often explode, so their LSM RMSE is typically ~90–100.",
    "The first results page is the equal-weight mean of SPY, AAPL, and MSFT. Later pages are per ticker.",
    "In every block the best model is bold and shaded: lowest RMSE of LSM price versus market option_price.",
]

OPTION_METHOD = [
    "Calibration and rolling are the same as the stock-path study (1-hour lookback; hourly and minutely; 1-day and 5-day windows).",
    "Contracts are 12 listed American calls per name from the short-interval options panel, sampled on the study window (seed 42).",
    "The sample is OTM / ATM / ITM × DTE 7–60 trading days, with unique 1-minute RTH quotes.",
    "Each contract is priced with Longstaff–Schwartz LSM: 500 risk-neutral paths, n_steps = DTE, Δt = 1/252.",
    "RMSE and MAE are computed on (model LSM price − listed market price) across the 12 contracts.",
    "The same seeds and the same 12 contracts are reused across models, including Heston.",
    "Models priced here: GBM, Merton, Heston, Heston–Merton, GARCH(1,1), and GARCH–Merton.",
]

OPTION_BAR_PURPOSE = [
    "This PDF is the bar-chart companion to the option-decision tables for 17 March 2023.",
    "Each page is a 2×2 of 1-day vs 5-day and hourly vs minutely rolling.",
    "Four bars per panel: GBM, Merton, GARCH, GARCH–Merton.",
    "Heston and Heston–Merton are excluded from every chart. Their LSM RMSE is an order of magnitude larger and would flatten the other bars; see the tables PDF for those numbers.",
    "The outlined bar is the lowest RMSE or MAE on that panel.",
]


def write_stock_tables(idx) -> Path:
    path = SHORT / "STOCK_SIMULATION_TABLES.pdf"
    footnote = (
        "Bold shaded row = lowest RMSE (median path vs S_t).   "
        "ICP = % of actual prices inside the 25th–75th band.   "
        "Heston–Merton is included."
    )
    with PdfPages(path) as pdf:
        _cover(pdf, "Stock-path simulation  ·  tables", STOCK_PURPOSE, STOCK_METHOD)
        _four_tables(
            pdf,
            "Stock-path metrics  ·  mean of SPY / AAPL / MSFT",
            footnote + "   Mean is equal-weight across the three names.",
            lambda w, r: _stock_rows(lambda m, k: _mean(idx, w, r, m, k)),
        )
        for ticker in TICKERS:
            _four_tables(
                pdf,
                f"Stock-path metrics  ·  {ticker}",
                footnote,
                lambda w, r, t=ticker: _stock_rows(lambda m, k: _val(idx, w, r, m, t, k)),
            )
    return path


def write_stock_bars(idx) -> Path:
    path = SHORT / "STOCK_SIMULATION_BAR_CHARTS.pdf"
    with PdfPages(path) as pdf:
        _cover(
            pdf,
            "Stock-path simulation  ·  bar charts",
            STOCK_BAR_PURPOSE,
            STOCK_METHOD,
            extra="Heston and Heston–Merton are in the tables PDF only.",
        )
        _bars_2x2(
            pdf,
            "RMSE of median path vs S_t  ·  mean of SPY / AAPL / MSFT",
            "RMSE",
            lambda w, r, m: _mean(idx, w, r, m, "rmse_st"),
        )
        _bars_2x2(
            pdf,
            "MAE of median path vs S_t  ·  mean of SPY / AAPL / MSFT",
            "MAE",
            lambda w, r, m: _mean(idx, w, r, m, "mae_st"),
        )
        _bars_2x2(
            pdf,
            "ICP (25th–75th band)  ·  mean of SPY / AAPL / MSFT",
            "ICP (%)",
            lambda w, r, m: 100 * _mean(idx, w, r, m, "icp_st"),
            higher_better=True,
        )
        _bars_2x2(
            pdf,
            "Average 25th–75th band width  ·  mean of SPY / AAPL / MSFT",
            "Width",
            lambda w, r, m: _mean(idx, w, r, m, "abw_st"),
            outline_best=False,
            footnote="Narrower is not a ranking  ·  Heston and Heston–Merton omitted (see tables PDF).",
        )
        for ticker in TICKERS:
            _bars_2x2(
                pdf,
                f"RMSE of median path vs S_t  ·  {ticker}",
                "RMSE",
                lambda w, r, m, t=ticker: _val(idx, w, r, m, t, "rmse_st"),
            )
    return path


def write_option_tables(idx) -> Path:
    path = SHORT / "OPTION_DECISION_TABLES.pdf"
    footnote = (
        "Bold shaded row = lowest RMSE (LSM American call vs listed market).   "
        "12 contracts per ticker, 500 RN paths, seed 42.   "
        "Heston–Merton is included."
    )
    with PdfPages(path) as pdf:
        _cover(pdf, "Option decision (LSM vs market)  ·  tables", OPTION_PURPOSE, OPTION_METHOD)
        _four_tables(
            pdf,
            "Option LSM vs market  ·  mean of SPY / AAPL / MSFT",
            footnote + "   Mean is equal-weight across the three names.",
            lambda w, r: _option_rows(lambda m, k: _mean(idx, w, r, m, k)),
        )
        for ticker in TICKERS:
            _four_tables(
                pdf,
                f"Option LSM vs market  ·  {ticker}",
                footnote,
                lambda w, r, t=ticker: _option_rows(lambda m, k: _val(idx, w, r, m, t, k)),
            )
    return path


def write_option_bars(idx) -> Path:
    path = SHORT / "OPTION_DECISION_BAR_CHARTS.pdf"
    with PdfPages(path) as pdf:
        _cover(
            pdf,
            "Option decision (LSM vs market)  ·  bar charts",
            OPTION_BAR_PURPOSE,
            OPTION_METHOD,
            extra="Heston and Heston–Merton are in the tables PDF only.",
        )
        _bars_2x2(
            pdf,
            "Option RMSE (LSM vs market)  ·  mean of SPY / AAPL / MSFT",
            "RMSE",
            lambda w, r, m: _mean(idx, w, r, m, "rmse_stop"),
        )
        _bars_2x2(
            pdf,
            "Option MAE (LSM vs market)  ·  mean of SPY / AAPL / MSFT",
            "MAE",
            lambda w, r, m: _mean(idx, w, r, m, "mae_stop"),
        )
        for ticker in TICKERS:
            _bars_2x2(
                pdf,
                f"Option RMSE (LSM vs market)  ·  {ticker}",
                "RMSE",
                lambda w, r, m, t=ticker: _val(idx, w, r, m, t, "rmse_stop"),
            )
    return path


def main() -> int:
    idx = _load()
    SHORT.mkdir(parents=True, exist_ok=True)
    paths = [
        write_stock_tables(idx),
        write_stock_bars(idx),
        write_option_tables(idx),
        write_option_bars(idx),
    ]
    for p in paths:
        print(f"wrote {p.relative_to(REPO)}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
