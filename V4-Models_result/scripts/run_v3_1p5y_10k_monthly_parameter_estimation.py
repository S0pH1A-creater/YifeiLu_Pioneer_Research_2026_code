#!/usr/bin/env python3
"""Monthly 1.5-year parameter estimates for the 1.5-year empirical study.

Calibration only: 18-month lookback, monthly rolling, four regimes, three
companies, six models. No LSM and no path simulation.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402

PDF_NAME = "V3_1p5y_monthly_parameter_estimation.pdf"
NB_NAME = "V3_1p5y_monthly_parameter_estimation.ipynb"
NAVY = emp.NAVY
MUTED = emp.MUTED
ALT_BG = emp.ALT_BG
STATS = ("Mean", "SD", "Min", "Max")
REGIME_TAG = {
    "2008-2009": "R1",
    "2013-2014": "R2",
    "2018-2019": "R3",
    "2019-2020": "R4",
}
PARAM_SPEC = {
    "GBM": [("mu", "μ"), ("sigma", "σ")],
    "Modified GBM": [
        ("p_uu", "P(U|U)"),
        ("p_dd", "P(D|D)"),
        ("p_ud", "P(U|D)"),
        ("p_du", "P(D|U)"),
        ("mu_u", "μ_U"),
        ("sig_u", "σ_U"),
        ("mu_d", "μ_D"),
        ("sig_d", "σ_D"),
    ],
    "Modified GBM v2": [
        ("p_uu", "P(U|U)"),
        ("p_dd", "P(D|D)"),
        ("p_ud", "P(U|D)"),
        ("p_du", "P(D|U)"),
        ("p_hh", "P(H|H)"),
        ("p_ll", "P(L|L)"),
        ("mu_u_l", "μ_U,L"),
        ("sig_u_l", "σ_U,L"),
        ("mu_u_h", "μ_U,H"),
        ("sig_u_h", "σ_U,H"),
        ("mu_d_l", "μ_D,L"),
        ("sig_d_l", "σ_D,L"),
        ("mu_d_h", "μ_D,H"),
        ("sig_d_h", "σ_D,H"),
    ],
    "Modified GBM v3": [
        ("p_u_uuu", "P(U|UUU)"),
        ("p_u_uud", "P(U|UUD)"),
        ("p_u_udu", "P(U|UDU)"),
        ("p_u_udd", "P(U|UDD)"),
        ("p_u_duu", "P(U|DUU)"),
        ("p_u_dud", "P(U|DUD)"),
        ("p_u_ddu", "P(U|DDU)"),
        ("p_u_ddd", "P(U|DDD)"),
        ("mu_u", "μ_U"),
        ("sig_u", "σ_U"),
        ("mu_d", "μ_D"),
        ("sig_d", "σ_D"),
    ],
    "GARCH": [("lambda", "λ"), ("omega", "ω"), ("alpha", "α"), ("beta", "β"), ("sigma0", "σ₀")],
    "Heston": [("mu", "μ"), ("kappa", "κ"), ("theta", "θ"), ("xi", "ξ"), ("rho", "ρ"), ("v0", "v₀")],
    "Merton": [
        ("mu", "μ"),
        ("sigma", "σ"),
        ("lam", "λ"),
        ("mu_j", "μ_J"),
        ("sigma_j", "σ_J"),
        ("kappa", "κ"),
    ],
    "GARCH–Merton": [
        ("mu", "μ"),
        ("omega", "ω"),
        ("alpha", "α"),
        ("beta", "β"),
        ("sigma0", "σ₀"),
        ("lam", "λ"),
        ("mu_j", "μ_J"),
        ("sigma_j", "σ_J"),
        ("kappa", "κ"),
    ],
    "Heston–Merton": [
        ("mu", "μ"),
        ("kappa", "κ"),
        ("theta", "θ"),
        ("xi", "ξ"),
        ("rho", "ρ"),
        ("v0", "v₀"),
        ("lam", "λ"),
        ("mu_j", "μ_J"),
        ("sigma_j", "σ_J"),
        ("kappa_j", "κ_J"),
    ],
}


def _paths() -> tuple[Path, Path, Path]:
    cache = emp.CACHE / "parameter_estimation"
    return cache, emp.SHORT / PDF_NAME, emp.SHORT / NB_NAME


def _raw_csv(cache: Path, model: str, regime: str, ticker: str) -> Path:
    safe = emp.file_stem(model)
    folder = cache / "raw" / regime / ticker
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe}.csv"


def _fmt_num(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    av = abs(float(v))
    if av == 0:
        return "0.0000"
    if av < 1e-3 or av >= 100:
        return f"{float(v):.3e}"
    return f"{float(v):.4f}"


def _summarize(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "Mean": np.nan, "SD": np.nan, "Min": np.nan, "Max": np.nan}
    return {
        "n": int(x.size),
        "Mean": float(np.mean(x)),
        "SD": float(np.std(x, ddof=1)) if x.size >= 2 else np.nan,
        "Min": float(np.min(x)),
        "Max": float(np.max(x)),
    }


def _calibrate_one(model: str, nb_path: Path, ticker: str) -> pd.DataFrame:
    os_study._install_notebook_stubs()
    g = os_study._load_ns(nb_path)
    cal = g["calibrate_ticker"](ticker, emp.WINDOW_LABEL, emp.ROLLING_MODE)
    if cal is None or len(cal) == 0:
        return pd.DataFrame()
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def run_estimation() -> dict:
    wrap.apply_1p5y_10k_config()
    cache, _, _ = _paths()
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "raw").mkdir(parents=True, exist_ok=True)
    tables = {m: {} for m in emp.TABLE_MODELS}
    n_updates = {m: {} for m in emp.TABLE_MODELS}
    jobs = emp._jobs()
    for model, nb in jobs:
        regime = os_study._regime_from_name(nb)
        print(f"\n=== calibrate {model} | {regime} ===", flush=True)
        for ticker in emp.TICKERS:
            dest = _raw_csv(cache, model, regime, ticker)
            if dest.exists():
                cal = pd.read_csv(dest, parse_dates=["date"])
                print(f"  resume {ticker} n={len(cal)}", flush=True)
            else:
                cal = _calibrate_one(model, nb, ticker)
                cal.to_csv(dest, index=False)
                print(f"  {ticker} n={len(cal)}", flush=True)
            n_updates[model][f"{ticker}|{regime}"] = int(len(cal))
            rec = {}
            for key, _label in PARAM_SPEC[model]:
                rec[key] = _summarize(cal[key] if key in cal.columns else pd.Series(dtype=float))
            tables[model][f"{ticker}|{regime}"] = rec
    payload = {
        "meta": {
            "study": emp.SHORT.name,
            "window_label": emp.WINDOW_LABEL,
            "lookback": emp.LOOKBACK_PHRASE,
            "rolling": emp.ROLLING_MODE,
            "purpose": "parameter estimation only; no LSM",
            "tickers": list(emp.TICKERS),
            "models": list(emp.TABLE_MODELS),
            "regimes": list(emp.REGIME_ORDER),
            "regime_meta": emp.REGIME_META,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "tables": tables,
        "n_updates": n_updates,
        "param_spec": {m: PARAM_SPEC[m] for m in emp.TABLE_MODELS},
    }
    (cache / "payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _table_rows(model: str, payload: dict) -> tuple[list[list[str]], list[str], list[str], set[int]]:
    top = ["Regime / param"]
    for ticker in emp.TICKERS:
        top.extend([ticker, "", "", ""])
    header = [""] + list(STATS) * len(emp.TICKERS)
    header[0] = "Regime / param"
    rows = []
    banner_idx = set()
    r0 = 2
    for regime in emp.REGIME_ORDER:
        tag = REGIME_TAG[regime]
        title = emp.REGIME_META[regime]["title"]
        ns = [
            payload["n_updates"][model].get(f"{t}|{regime}", 0) for t in emp.TICKERS
        ]
        n_txt = ns[0] if len(set(ns)) == 1 else "/".join(str(n) for n in ns)
        banner = [f"{tag}  {title}  (n = {n_txt})"] + [""] * (len(STATS) * len(emp.TICKERS))
        rows.append(banner)
        banner_idx.add(r0)
        r0 += 1
        for key, label in PARAM_SPEC[model]:
            row = [f"    {label}"]
            for ticker in emp.TICKERS:
                rec = payload["tables"][model][f"{ticker}|{regime}"][key]
                for stat in STATS:
                    row.append(_fmt_num(rec[stat]))
            rows.append(row)
            r0 += 1
    return rows, top, header, banner_idx


def _table_note(model: str, payload: dict) -> str:
    params = ", ".join(lab for _k, lab in PARAM_SPEC[model])
    if emp.ROLLING_MODE == "none":
        how = (
            "Each company×regime cell is one estimate: 18-month lookback ending on the first session of the window, held for the year. "
            "n is 1. SD is undefined with a single update (shown as —)."
        )
    else:
        how = (
            "Each month-end update uses an 18-month lookback ending on that date, with no look-ahead. "
            "R1–R4 are Crisis, Normal, Late-cycle, and COVID. n is the number of monthly updates "
            "in that regime. SD is the sample standard deviation across those monthly estimates. "
        )
    return (
        f"Notes: This table reports Mean, SD, Min, and Max of the {emp.ROLLING_MODE} "
        f"{model} estimates for {', '.join(emp.TICKERS)}. Parameters are {params}. "
        f"{how} "
        f"This is parameter estimation only; it is not an LSM pricing table."
    )


def _style_table(tbl, banner_idx: set[int], n_rows: int) -> None:
    tbl.auto_set_font_size(False)
    fs = 6.2 if n_rows > 28 else 7.0
    h = 0.055 if n_rows > 28 else 0.085
    tbl.set_fontsize(fs)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.5)
        cell.set_height(h)
        if r <= 1:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold", fontsize=fs)
        elif r in banner_idx:
            cell.set_facecolor("#D9E2EC")
            cell.set_text_props(weight="bold", color=NAVY, fontsize=fs, ha="left" if c == 0 else "center")
        elif c == 0:
            cell.set_text_props(ha="left", color="#1A1A1A", fontsize=fs)
            cell.set_facecolor(ALT_BG if r % 2 == 0 else "white")
        elif r % 2 == 0:
            cell.set_facecolor(ALT_BG)
            cell.set_text_props(fontsize=fs - 0.2)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(fontsize=fs - 0.2)


def _wrapped_note(fig, text: str, y: float, *, width: int = 132, fontsize: float = 7.4) -> None:
    lines = textwrap.wrap(text, width=width) or [""]
    for i, line in enumerate(lines):
        fig.text(0.055, y - 0.016 * i, line, fontsize=fontsize, color="#555555", va="top")


def _cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "V4 parameter estimation", fontsize=17, weight="bold", color=NAVY, va="top")
    fig.text(
        0.08,
        0.912,
        f"{emp._models_phrase()}  ·  {', '.join(emp.TICKERS)}  ·  four windows  ·  {emp._rolling_phrase()}",
        fontsize=9.0,
        color=MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.892, 0.892], transform=fig.transFigure, color=NAVY, lw=1.4))
    y = 0.86

    def section(heading: str, bullets: list[str], y0: float) -> float:
        fig.text(0.08, y0, heading, fontsize=11.5, weight="bold", color=NAVY, va="top")
        y = y0 - 0.028
        for b in bullets:
            wrapped = textwrap.wrap(b, width=94) or [""]
            fig.text(0.10, y, "•", fontsize=9.5, color=NAVY, va="top")
            fig.text(0.13, y, wrapped[0], fontsize=8.6, color="#222222", va="top")
            y -= 0.018
            for line in wrapped[1:]:
                fig.text(0.13, y, line, fontsize=8.6, color="#222222", va="top")
                y -= 0.018
            y -= 0.005
        return y - 0.012

    y = section(
        "What this report is",
        [
            (
                "The rolling parameter paths used in this study, summarized by regime. One table per model."
                if emp.ROLLING_MODE != "none"
                else "The single t0 parameter vector used in this study, summarized by regime. One table per model. n = 1 so SD is —."
            ),
            "No LSM, no option prices, and no simulated paths. Only calibrate_ticker with the 18-month lookback.",
            "The companion notebook is the computational source of truth. This PDF is written from the same payload.",
        ],
        y,
    )
    y = section(
        "How estimates are formed",
        [
            emp._no_lookahead_note(),
            emp._rolling_detail(),
            (
                "Cells are Mean, SD, Min, and Max of those monthly estimates inside the regime."
                if emp.ROLLING_MODE != "none"
                else "Cells are Mean, SD, Min, and Max. With n = 1, Mean = Min = Max and SD is —."
            ),
        ],
        y,
    )
    y = section(
        "Tables",
        [
            "  ".join(f"P{i} {m}." for i, m in enumerate(emp.TABLE_MODELS, start=1)),
            f"R1 Crisis, R2 Normal, R3 Late-cycle, R4 COVID. Columns are {', '.join(emp.TICKERS)}.",
        ],
        y,
    )
    fig.text(
        0.08,
        0.045,
        "Page map: cover  ·  Tables P1–P{len(emp.TABLE_MODELS)}. Pricing results live in the companion empirical-study PDF.",
        fontsize=7.6,
        color="#666666",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _model_page(pdf: PdfPages, payload: dict, model: str, idx: int) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(
        0.055,
        0.955,
        f"Table P{idx}  ·  {model}  ·  {emp.ROLLING_MODE} parameter estimates",
        fontsize=13.0,
        weight="bold",
        color=NAVY,
        va="top",
    )
    fig.text(
        0.055,
        0.920,
        f"{emp.LOOKBACK_PHRASE} lookback  ·  rolling={emp.ROLLING_MODE}  ·  Mean, SD, Min, Max within each regime",
        fontsize=8.8,
        color=MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.900, 0.900], transform=fig.transFigure, color=NAVY, lw=1.2))
    rows, top, header, banner_idx = _table_rows(model, payload)
    ax = fig.add_axes([0.04, 0.30 if len(rows) > 28 else 0.34, 0.92, 0.52 if len(rows) > 28 else 0.50])
    ax.axis("off")
    cell_text = [top, header, *rows]
    tbl = ax.table(cellText=cell_text, loc="center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    _style_table(tbl, banner_idx, len(cell_text))
    _wrapped_note(fig, _table_note(model, payload), 0.265 if len(rows) > 28 else 0.30)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(payload: dict, path: Path | None = None) -> Path:
    wrap.apply_1p5y_10k_config()
    emp.SHORT.mkdir(parents=True, exist_ok=True)
    path = path or (emp.SHORT / PDF_NAME)
    with PdfPages(path) as pdf:
        _cover(pdf)
        for i, model in enumerate(emp.TABLE_MODELS, start=1):
            _model_page(pdf, payload, model, i)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)
    return path


def _html_table(model: str, payload: dict) -> str:
    parts = [
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:12px;width:100%;'>",
        "<thead><tr>",
        f"<th rowspan='2' style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;text-align:left;'>Regime / param</th>",
    ]
    for ticker in emp.TICKERS:
        parts.append(
            f"<th colspan='4' style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{ticker}</th>"
        )
    parts.append("</tr><tr>")
    for _ in emp.TICKERS:
        for stat in STATS:
            parts.append(
                f"<th style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{stat}</th>"
            )
    parts.append("</tr></thead><tbody>")
    for regime in emp.REGIME_ORDER:
        tag = REGIME_TAG[regime]
        title = emp.REGIME_META[regime]["title"]
        ns = [payload["n_updates"][model].get(f"{t}|{regime}", 0) for t in emp.TICKERS]
        n_txt = ns[0] if len(set(ns)) == 1 else "/".join(str(n) for n in ns)
        parts.append("<tr>")
        ncols = 1 + len(STATS) * len(emp.TICKERS)
        parts.append(
            f"<td colspan='{ncols}' style='background:#D9E2EC;color:{NAVY};font-weight:700;padding:6px 8px;"
            f"border:1px solid #D0D5DD;'>{tag}  {title}  (n = {n_txt})</td>"
        )
        parts.append("</tr>")
        for i, (key, label) in enumerate(PARAM_SPEC[model], start=1):
            bg = ALT_BG if i % 2 == 0 else "white"
            parts.append("<tr>")
            parts.append(
                f"<td style='background:{bg};padding:5px 8px;border:1px solid #D0D5DD;text-align:left;'>{label}</td>"
            )
            for ticker in emp.TICKERS:
                rec = payload["tables"][model][f"{ticker}|{regime}"][key]
                for stat in STATS:
                    parts.append(
                        f"<td style='background:{bg};padding:5px 8px;border:1px solid #D0D5DD;text-align:center;'>"
                        f"{_fmt_num(rec[stat])}</td>"
                    )
            parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_notebook(payload: dict, path: Path | None = None) -> Path:
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
            f"""# V4 parameter estimation — {emp._rolling_phrase()}

**This notebook is the computational source of truth.** The matching PDF is written from the same payload.

`calibrate_ticker` paths for `{emp.SHORT.name}`. **No LSM.**

| Item | Setting |
|------|---------|
| Lookback | 18 months ending at each update |
| Rolling | `{emp.ROLLING_MODE}` |
| Names | {", ".join(emp.TICKERS)} |
| Windows | R1 Crisis, R2 Normal, R3 Late-cycle, R4 COVID |
| Cell | Mean, SD, Min, Max of the estimates inside that regime |
"""
        )
    ]
    for i, model in enumerate(emp.TABLE_MODELS, start=1):
        cells.append(md(f"## Table P{i}. {model}"))
        cells.append(
            code(
                "from IPython.display import display, HTML\n"
                f"display(HTML(pe._html_table('{model}', payload)))",
                html=_html_table(model, payload),
            )
        )
        cells.append(md(_table_note(model, payload)))
    cells.append(
        md(
            """## Re-run

```python
import run_v3_1p5y_10k_monthly_parameter_estimation as pe
payload = pe.run()
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


def run(*, recompute: bool = False) -> dict:
    wrap.apply_1p5y_10k_config()
    cache, _, _ = _paths()
    dest = cache / "payload.json"
    if recompute and cache.exists():
        import shutil

        shutil.rmtree(cache / "raw", ignore_errors=True)
        if dest.exists():
            dest.unlink()
    payload = run_estimation()
    write_pdf(payload)
    build_notebook(payload)
    return payload


def main() -> int:
    recompute = "--recompute" in sys.argv
    run(recompute=recompute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
