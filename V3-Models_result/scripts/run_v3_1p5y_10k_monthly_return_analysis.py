#!/usr/bin/env python3
"""Realized return statistics and time-series for the 1.5-year 10k study windows.

One PDF + notebook: four periods × (statistics table + three return charts).
R_t = ln(S_t / S_{t-1}) from prices_clean.csv. Does not rerun LSM.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402

PDF_NAME = "V3_1p5y_monthly_return_analysis.pdf"
NB_NAME = "V3_1p5y_monthly_return_analysis.ipynb"
NAVY = emp.NAVY
MUTED = emp.MUTED
ALT_BG = emp.ALT_BG
TICKER_COLORS = {"SPY": "#1F3A5F", "AAPL": "#C44E52", "MSFT": "#4C72B0"}
ROW_LABELS = (
    "Mean",
    "Std dev.",
    "Skewness statistic",
    "Ex. kurt. statistic",
    "Normality statistics (JB)",
)


def _paths() -> tuple[Path, Path, Path]:
    cache = emp.CACHE / "return_analysis"
    return cache, emp.SHORT / PDF_NAME, emp.SHORT / NB_NAME


def _prices() -> pd.DataFrame:
    path = emp.REPO / "research" / "data" / "equity" / "prices_clean.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df.loc[:, list(emp.TICKERS)].astype(float)


def _window_bounds(regime: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    left, right = emp.REGIME_META[regime]["window"].split("→")
    return pd.Timestamp(left.strip()), pd.Timestamp(right.strip())


def _long_date(value) -> str:
    d = pd.Timestamp(value)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _period_returns(prices: pd.DataFrame, regime: str) -> pd.DataFrame:
    start, end = _window_bounds(regime)
    log_all = np.log(prices).diff()
    return log_all.loc[start:end].dropna(how="any")


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _t_jarque_bera(r: np.ndarray) -> tuple[float, float]:
    """JB_t = t_skew² + t_kurt², p-value from χ²(2)."""
    t_s, _ = stats.skewtest(r)
    t_k, _ = stats.kurtosistest(r)
    stat = float(t_s**2 + t_k**2)
    p = float(stats.chi2.sf(stat, 2))
    return stat, p


def _ticker_stats(r: pd.Series) -> dict:
    x = np.asarray(r.to_numpy(dtype=float), dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    skew = float(stats.skew(x, bias=False))
    exkurt = float(stats.kurtosis(x, fisher=True, bias=False))
    _, skew_p = stats.skewtest(x)
    _, kurt_p = stats.kurtosistest(x)
    jb_stat, jb_p = _t_jarque_bera(x)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skewness": skew,
        "excess_kurtosis": exkurt,
        "skewness_p": float(skew_p),
        "excess_kurtosis_p": float(kurt_p),
        "jb_stat": jb_stat,
        "jb_p": jb_p,
        "mean_pct": 100.0 * mean,
        "std_pct": 100.0 * std,
        "ann_mean_pct": 100.0 * 252.0 * mean,
        "ann_std_pct": 100.0 * float(np.sqrt(252.0) * std),
    }


def _cell(stat: str, rec: dict) -> str:
    if stat == "Mean":
        return f"{rec['mean_pct']:.4f}"
    if stat == "Std dev.":
        return f"{rec['std_pct']:.4f}"
    if stat == "Skewness statistic":
        return f"{rec['skewness']:.3f}  [{_fmt_p(rec['skewness_p'])}]"
    if stat == "Ex. kurt. statistic":
        return f"{rec['excess_kurtosis']:.3f}  [{_fmt_p(rec['excess_kurtosis_p'])}]"
    return f"{rec['jb_stat']:.2f}  [{_fmt_p(rec['jb_p'])}]"


def _table_rows(period: dict) -> tuple[list[list[str]], list[str]]:
    header = ["Statistic", *emp.TICKERS]
    rows = []
    for label in ROW_LABELS:
        rows.append([label] + [_cell(label, period["tickers"][t]) for t in emp.TICKERS])
    return rows, header


def _table_note(period: dict) -> str:
    start = _long_date(period["start"])
    end = _long_date(period["end"])
    n = period["n"]
    return (
        f"Notes: This table shows sample statistics for the continuously compounded returns, "
        f"R_t = ln(S_t / S_{{t-1}}), for SPY, AAPL, and MSFT. The sample period is {start} to {end}, "
        f"for a total of {n} daily observations. Mean return and standard deviation are in percent per "
        f"trading day. For the skewness and excess kurtosis statistics, the brackets next to the "
        f"statistics report the p-values from testing the significance of the difference between the "
        f"empirical values and the theoretical values from the Normal distribution using a t-test. For "
        f"the normality statistic the p-value of a t-version of the well known Jarque–Bera test for "
        f"normality is reported in brackets next to the statistics."
    )


def _figure_note(period: dict) -> str:
    start = _long_date(period["start"])
    end = _long_date(period["end"])
    n = period["n"]
    return (
        f"Notes: This figure plots the annualized continuously compounded returns 252 × R_t for SPY, "
        f"AAPL, and MSFT over {start} to {end} ({n} trading days). Date is on the x-axis and the "
        f"annualized return, in percent, is on the y-axis. Each panel is one name. The series show the "
        f"evolution and fluctuations of returns throughout the period; large spikes are one-day moves "
        f"scaled by 252, not cumulative performance."
    )


def build_payload() -> dict:
    wrap.apply_1p5y_10k_config()
    prices = _prices()
    periods = []
    for regime in emp.REGIME_ORDER:
        rets = _period_returns(prices, regime)
        start, end = _window_bounds(regime)
        meta = emp.REGIME_META[regime]
        tickers = {t: _ticker_stats(rets[t]) for t in emp.TICKERS}
        n = int(len(rets))
        dates = [d.date().isoformat() for d in rets.index]
        series = {t: (252.0 * 100.0 * rets[t]).astype(float).tolist() for t in emp.TICKERS}
        periods.append(
            {
                "regime": regime,
                "title": meta["title"],
                "window": meta["window"],
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "n": n,
                "tickers": tickers,
                "dates": dates,
                "ann_return_pct": series,
            }
        )
    return {
        "meta": {
            "study": "V3 1.5-year monthly empirical study 10000 paths",
            "measure": "R_t = ln(S_t / S_{t-1})",
            "mean_std_units": "percent per trading day",
            "figure": "annualized continuously compounded return 252 × R_t, in percent",
            "prices": "research/data/equity/prices_clean.csv",
            "tickers": list(emp.TICKERS),
            "regimes": list(emp.REGIME_ORDER),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "periods": periods,
    }


def _style_table(tbl) -> None:
    emp._style_table(tbl, None)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(0.145)
        if c == 0 and r > 0:
            cell.set_text_props(ha="left", weight="medium")


def _wrapped_note(fig, text: str, y: float, *, width: int = 118, fontsize: float = 8.0) -> None:
    lines = textwrap.wrap(text, width=width) or [""]
    for i, line in enumerate(lines):
        fig.text(0.08, y - 0.018 * i, line, fontsize=fontsize, color="#555555", va="top")


def _cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "V3 return analysis", fontsize=18, weight="bold", color=NAVY, va="top")
    fig.text(
        0.08,
        0.912,
        f"SPY, AAPL, MSFT  ·  four study windows  ·  {emp.LOOKBACK_PHRASE} monthly study",
        fontsize=9.2,
        color=MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.892, 0.892], transform=fig.transFigure, color=NAVY, lw=1.4))

    y = 0.86

    def section(heading: str, bullets: list[str], y0: float) -> float:
        fig.text(0.08, y0, heading, fontsize=11.5, weight="bold", color=NAVY, va="top")
        y = y0 - 0.030
        for b in bullets:
            wrapped = textwrap.wrap(b, width=94) or [""]
            fig.text(0.10, y, "•", fontsize=9.5, color=NAVY, va="top")
            fig.text(0.13, y, wrapped[0], fontsize=8.8, color="#222222", va="top")
            y -= 0.0185
            for line in wrapped[1:]:
                fig.text(0.13, y, line, fontsize=8.8, color="#222222", va="top")
                y -= 0.0185
            y -= 0.006
        return y - 0.012

    y = section(
        "What this report is",
        [
            "Descriptive statistics of realized equity returns in the same four windows as the 1.5-year monthly 10,000-path study. It does not price options or resimulate paths.",
            "One table and one three-panel time-series figure per window, always SPY, AAPL, and MSFT, always the same five rows and the same return definition.",
            "The companion notebook is the computational source of truth. This PDF is written from the same payload.",
        ],
        y,
    )
    y = section(
        "Return definition",
        [
            "Daily continuously compounded return R_t = ln(S_t / S_{t-1}) from adjusted closes in prices_clean.csv.",
            "Tables: Mean, Std dev. (percent per trading day), Skewness statistic, Ex. kurt. statistic, and Normality statistics (JB), all computed on daily R_t.",
            "Figures: annualized continuously compounded return 252 × R_t, reported in percent.",
        ],
        y,
    )
    y = section(
        "Tests in the table",
        [
            "Skewness and excess kurtosis: brackets are p-values from t-tests against the Normal values 0 and 0.",
            "Jarque–Bera: t-version JB_t = t_skew² + t_kurtosis² with p-value from χ²(2), shown in brackets.",
        ],
        y,
    )
    y = section(
        "Windows (inclusive)",
        [
            "Crisis 2008-08-01 → 2009-07-31.  Normal 2014-01-01 → 2014-12-31.",
            "Late-cycle 2018-10-01 → 2019-09-30.  COVID 2019-09-01 → 2020-08-31.",
        ],
        y,
    )
    fig.text(
        0.08,
        0.045,
        "Page map: cover  ·  Table R1–R4 each followed by Figure R1–R4  ·  notebook has the same tables, figures, and notes.",
        fontsize=7.6,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf: PdfPages, period: dict, idx: int) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(
        0.08,
        0.955,
        f"Table R{idx}  ·  {period['regime']}  {period['title']}  ·  return statistics",
        fontsize=13.5,
        weight="bold",
        color=NAVY,
        va="top",
    )
    fig.text(
        0.08,
        0.918,
        f"{period['window']}   ·   n = {period['n']} daily observations   ·   R_t in percent per trading day",
        fontsize=9.0,
        color=MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.898, 0.898], transform=fig.transFigure, color=NAVY, lw=1.2))

    ax = fig.add_axes([0.14, 0.38, 0.72, 0.46])
    ax.axis("off")
    rows, header = _table_rows(period)
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.95])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    _style_table(tbl)
    tbl.auto_set_column_width(list(range(len(header))))

    _wrapped_note(fig, _table_note(period), 0.30, width=118, fontsize=8.1)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_return_figure(period: dict, idx: int, *, for_pdf: bool = True):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    fig.patch.set_facecolor("white")
    dates = pd.to_datetime(period["dates"])
    fig.suptitle(
        f"Figure R{idx}  ·  {period['regime']}  {period['title']}  ·  annualized continuously compounded returns",
        fontsize=12.5,
        color=NAVY,
        weight="bold",
        y=0.98,
        x=0.08,
        ha="left",
    )
    for ax, ticker in zip(axes, emp.TICKERS):
        y = np.asarray(period["ann_return_pct"][ticker], dtype=float)
        ax.plot(dates, y, color=TICKER_COLORS[ticker], lw=0.85, solid_capstyle="round")
        ax.axhline(0.0, color="#B0B7C3", lw=0.7, zorder=0)
        ax.set_ylabel("Annualized return (%)", fontsize=8.2, color="#333333")
        ax.set_title(ticker, loc="left", fontsize=10.5, color=NAVY, weight="bold", pad=4)
        ax.yaxis.grid(True, linestyle="-", alpha=0.28)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8.0, colors="#333333")
        pad = 0.08 * (np.nanmax(np.abs(y)) if np.isfinite(y).any() else 1.0)
        ymax = float(np.nanmax(np.abs(y))) + pad
        ax.set_ylim(-ymax, ymax)
    axes[-1].set_xlabel("Date", fontsize=8.5, color="#333333")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    if for_pdf:
        fig.tight_layout(rect=(0.04, 0.13, 0.98, 0.94))
        _wrapped_note(fig, _figure_note(period), 0.105, width=124, fontsize=7.8)
    else:
        fig.tight_layout()
    return fig


def _figure_page(pdf: PdfPages, period: dict, idx: int) -> None:
    fig = _draw_return_figure(period, idx, for_pdf=True)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    wrap.apply_1p5y_10k_config()
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (emp.SHORT / PDF_NAME)
    with PdfPages(path) as pdf:
        _cover(pdf)
        for i, period in enumerate(payload["periods"], start=1):
            _table_page(pdf, period, i)
            _figure_page(pdf, period, i)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)
    return path


def _html_table(period: dict) -> str:
    rows, header = _table_rows(period)
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
        bg = ALT_BG if i % 2 == 0 else "white"
        parts.append("<tr>")
        for j, val in enumerate(row):
            align = "left" if j == 0 else "center"
            weight = "600" if j == 0 else "400"
            parts.append(
                f"<td style='background:{bg};padding:6px 8px;border:1px solid #D0D5DD;"
                f"text-align:{align};font-weight:{weight};'>{val}</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_notebook(payload: dict, path: Path | None = None) -> Path:
    import base64
    import io
    import uuid

    wrap.apply_1p5y_10k_config()
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (emp.SHORT / NB_NAME)

    def md(text: str) -> dict:
        if not text.endswith("\n"):
            text += "\n"
        lines = text.split("\n")
        return {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]],
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

    def fig_png(period: dict, idx: int) -> bytes:
        fig = _draw_return_figure(period, idx, for_pdf=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return buf.getvalue()

    cells = [
        md(
            f"""# V3 return analysis — {emp.LOOKBACK_PHRASE} monthly study

**This notebook is the computational source of truth.** The matching PDF is written from the same payload.

Realized equity returns in the four windows of `V3 1.5-year monthly empirical study 10000 paths`. This report does not price options and does not resimulate LSM paths.

| Item | Setting |
|------|---------|
| Names | SPY, AAPL, MSFT |
| Windows | Crisis, Normal, Late-cycle, COVID (same bounds as the option study) |
| Return | $R_t = \\ln(S_t / S_{{t-1}})$ from `research/data/equity/prices_clean.csv` |
| Table | Mean, Std dev., Skewness statistic, Ex. kurt. statistic, Normality statistics (JB) |
| Brackets | t-test p-values vs Normal for skewness and excess kurtosis; t-version Jarque–Bera p-value |
| Figures | Annualized continuously compounded returns $252 \\times R_t$, in percent |
"""
        )
    ]
    for i, period in enumerate(payload["periods"], start=1):
        cells.append(
            md(
                f"## {period['regime']}  ·  {period['title']}\n\n"
                f"{period['window']}  ·  n = {period['n']} daily observations."
            )
        )
        cells.append(md(f"### Table R{i}. Return statistics"))
        cells.append(
            code(
                "from IPython.display import display, HTML\n"
                f"display(HTML(ra._html_table(payload['periods'][{i-1}])))",
                html=_html_table(period),
            )
        )
        cells.append(md(_table_note(period)))
        cells.append(md(f"### Figure R{i}. Annualized continuously compounded returns"))
        cells.append(
            code(
                f"fig = ra._draw_return_figure(payload['periods'][{i-1}], {i}, for_pdf=False)\nfig",
                png=fig_png(period, i),
            )
        )
        cells.append(md(_figure_note(period)))

    cells.append(
        md(
            """## Re-run

```python
import run_v3_1p5y_10k_monthly_return_analysis as ra
payload = ra.run()
```
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
    wrap.apply_1p5y_10k_config()
    cache, _, _ = _paths()
    cache.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (cache / "payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_pdf(payload)
    build_notebook(payload)
    return payload


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
