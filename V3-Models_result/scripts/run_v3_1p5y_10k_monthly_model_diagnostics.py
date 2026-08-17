#!/usr/bin/env python3
"""Model diagnostic verification for the 1.5-year monthly 10k study.

Four regime tables: 6 models × 3 companies × JB, Q(20), Q²(20), ARCH-LM
on standardized residuals from that regime's daily returns only.
Does not rerun LSM and is not a pricing-accuracy report.
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
from scipy import stats

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_garch_parameter_recovery as garch_rec  # noqa: E402
import run_v3_1p5y_10k_monthly_empirical_study as wrap  # noqa: E402
import run_v3_1p5y_10k_monthly_return_analysis as ra  # noqa: E402
import run_v3_5y_monthly_empirical_study as emp  # noqa: E402

PDF_NAME = "V3_1p5y_monthly_model_diagnostics.pdf"
NB_NAME = "V3_1p5y_monthly_model_diagnostics.ipynb"
NAVY = emp.NAVY
MUTED = emp.MUTED
ALT_BG = emp.ALT_BG
N_DAYS = 252
DT = 1.0 / N_DAYS
JUMP_THRESH = 3.0
LB_LAGS = 20
ARCH_LAGS = 5
TESTS = ("JB", "Q(20)", "Q²(20)", "ARCH-LM")
NA = "N/A"


def _paths() -> tuple[Path, Path, Path]:
    cache = emp.CACHE / "model_diagnostics"
    return cache, emp.SHORT / PDF_NAME, emp.SHORT / NB_NAME


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_pair(stat, p) -> str:
    if stat is None or p is None:
        return NA
    if not np.isfinite(stat) or not np.isfinite(p):
        return NA
    return f"{stat:.2f}  [{_fmt_p(p)}]"


def _ljung_box(z: np.ndarray, lags: int = LB_LAGS) -> tuple[float, float] | tuple[None, None]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = int(z.size)
    if n <= lags + 2:
        return None, None
    z = z - z.mean()
    acf = []
    for k in range(1, lags + 1):
        a = float(np.dot(z[k:], z[:-k]) / np.dot(z, z))
        acf.append(a)
    acf = np.asarray(acf, dtype=float)
    q = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1))))
    p = float(stats.chi2.sf(q, lags))
    return q, p


def _arch_lm(z: np.ndarray, lags: int = ARCH_LAGS) -> tuple[float, float] | tuple[None, None]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = int(z.size)
    if n <= lags + 5:
        return None, None
    y = z[lags:] ** 2
    cols = [np.ones(y.size)]
    for i in range(1, lags + 1):
        cols.append(z[lags - i : n - i] ** 2)
    x = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    if ss_tot <= 0:
        return None, None
    r2 = 1.0 - ss_res / ss_tot
    lm = float(y.size * max(r2, 0.0))
    p = float(stats.chi2.sf(lm, lags))
    return lm, p


def _jb(z: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 8:
        return None, None
    stat, p = stats.jarque_bera(z)
    return float(stat), float(p)


def _diagnostics(z) -> dict:
    if z is None:
        return {t: {"stat": None, "p": None} for t in TESTS}
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 8:
        return {t: {"stat": None, "p": None} for t in TESTS}
    jb_s, jb_p = _jb(z)
    q_s, q_p = _ljung_box(z, LB_LAGS)
    q2_s, q2_p = _ljung_box(z**2, LB_LAGS)
    a_s, a_p = _arch_lm(z, ARCH_LAGS)
    return {
        "JB": {"stat": jb_s, "p": jb_p},
        "Q(20)": {"stat": q_s, "p": q_p},
        "Q²(20)": {"stat": q2_s, "p": q2_p},
        "ARCH-LM": {"stat": a_s, "p": a_p},
    }


def _heston_filter(r: np.ndarray, p: dict) -> np.ndarray:
    n = len(r)
    v = np.empty(n, dtype=float)
    v[0] = float(max(p["v0"], 1e-8))
    k, th, xi = float(p["kappa"]), float(p["theta"]), float(p["xi"])
    w = float(np.clip(xi / (xi + 1.0), 0.05, 0.35))
    for t in range(1, n):
        pred = v[t - 1] + k * (th - v[t - 1]) * DT
        rv = (r[t] ** 2) / DT
        v[t] = max((1.0 - w) * pred + w * rv, 1e-8)
    return v


def _heston_moments(r: np.ndarray) -> dict | None:
    x = pd.Series(np.asarray(r, dtype=float)).dropna()
    n = int(x.shape[0])
    if n < 60:
        return None
    sigma_day = float(x.std(ddof=1))
    if not np.isfinite(sigma_day) or sigma_day <= 0:
        return None
    mu = float(x.mean() * N_DAYS)
    rv = x**2
    theta = float(rv.mean() * N_DAYS)
    v0 = float(rv.iloc[-min(21, n) :].mean() * N_DAYS)
    if not np.isfinite(theta) or theta <= 0:
        return None
    if not np.isfinite(v0) or v0 <= 0:
        v0 = theta
    rho1 = rv.autocorr(lag=1)
    if rho1 is None or not np.isfinite(rho1) or rho1 <= 1e-6:
        kappa = 2.0
    elif rho1 >= 0.999:
        kappa = 0.05
    else:
        kappa = float(np.clip(-np.log(float(rho1)) / DT, 0.05, 20.0))
    v_ann = (rv * N_DAYS).astype(float)
    dv = v_ann.diff().dropna()
    v_lag = v_ann.loc[dv.index]
    resid = dv.values - kappa * (theta - v_lag.values) * DT
    mean_v = float(np.mean(v_lag.values))
    var_resid = float(np.var(resid, ddof=1)) if resid.size >= 2 else np.nan
    if mean_v > 0 and np.isfinite(var_resid) and var_resid > 0:
        xi = float(np.clip(np.sqrt(var_resid / (mean_v * DT)), 0.05, 3.0))
    else:
        xi = 0.5
    return {"mu": mu, "kappa": kappa, "theta": theta, "xi": xi, "v0": v0}


def _merton_params(r: np.ndarray) -> dict | None:
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None
    sigma_day = float(np.std(x, ddof=1))
    if not np.isfinite(sigma_day) or sigma_day <= 0:
        return None
    jump_mask = np.abs(x) > JUMP_THRESH * sigma_day
    normal = x[~jump_mask]
    base = normal if normal.size >= 2 else x
    mu = float(base.mean() * N_DAYS)
    years = x.size / float(N_DAYS)
    n_j = int(jump_mask.sum())
    lam = float(n_j / years) if years > 0 else 0.0
    jumps = x[jump_mask]
    if n_j >= 2:
        mu_j, sj = float(jumps.mean()), float(jumps.std(ddof=1))
    elif n_j == 1:
        mu_j, sj = float(jumps[0]), 0.0
    else:
        mu_j, sj = 0.0, 0.0
    if not np.isfinite(sj) or sj < 0:
        sj = 0.0
    var_tot = float(np.var(x, ddof=1) * N_DAYS)
    jump_var = lam * (mu_j**2 + sj**2)
    sig = float(np.sqrt(max(var_tot - jump_var, 1e-12)))
    return {"mu": mu, "sigma": sig}


def _residuals(model: str, r: np.ndarray):
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None
    if model == "GBM":
        mu = float(np.mean(x))
        sig = float(np.std(x, ddof=1))
        if not np.isfinite(sig) or sig <= 0:
            return None
        return (x - mu) / sig
    if model == "Modified GBM":
        up = x > 0
        mag = np.abs(x)
        if int(up.sum()) < 2 or int((~up).sum()) < 2:
            return None
        mu_u, sig_u = float(mag[up].mean()), float(mag[up].std(ddof=1))
        mu_d, sig_d = float(mag[~up].mean()), float(mag[~up].std(ddof=1))
        if min(sig_u, sig_d) <= 0 or not np.isfinite(sig_u) or not np.isfinite(sig_d):
            return None
        z = np.empty_like(x)
        z[up] = (x[up] - mu_u) / sig_u
        z[~up] = (x[~up] + mu_d) / sig_d
        return z
    if model == "GARCH":
        fit = garch_rec.fit_garch11(x)
        if fit is None:
            return None
        mu, _omega, _a, _b, sigma, _ok = fit
        return (x - mu) / np.maximum(sigma, 1e-12)
    if model == "Merton":
        p = _merton_params(x)
        if p is None:
            return None
        mu_d = p["mu"] / N_DAYS
        sig_d = p["sigma"] / np.sqrt(N_DAYS)
        if not np.isfinite(sig_d) or sig_d <= 0:
            return None
        return (x - mu_d) / sig_d
    if model == "GARCH–Merton":
        fit = garch_rec.fit_garch11(x)
        if fit is None:
            return None
        mu, _omega, _a, _b, sigma, _ok = fit
        z = (x - mu) / np.maximum(sigma, 1e-12)
        jump = np.abs(z) > JUMP_THRESH
        z = z.copy()
        z[jump] = 0.0
        return z
    if model == "Heston":
        p = _heston_moments(x)
        if p is None:
            return None
        v = _heston_filter(x, p)
        mu_d = p["mu"] / N_DAYS
        vol = np.sqrt(np.maximum(v, 1e-12) * DT)
        return (x - mu_d) / vol
    if model == "Heston–Merton":
        sigma_day = float(np.std(x, ddof=1))
        if not np.isfinite(sigma_day) or sigma_day <= 0:
            return None
        jump = np.abs(x) > JUMP_THRESH * sigma_day
        x_c = x.copy()
        base = x[~jump]
        x_c[jump] = float(base.mean()) if base.size else 0.0
        p = _heston_moments(x_c)
        if p is None:
            return None
        v = _heston_filter(x_c, p)
        mu_d = p["mu"] / N_DAYS
        vol = np.sqrt(np.maximum(v, 1e-12) * DT)
        return (x_c - mu_d) / vol
    return None


def build_payload() -> dict:
    wrap.apply_1p5y_10k_config()
    prices = ra._prices()
    periods = []
    cells = {}
    for regime in emp.REGIME_ORDER:
        rets = ra._period_returns(prices, regime)
        start, end = ra._window_bounds(regime)
        meta = emp.REGIME_META[regime]
        models = {}
        for model in emp.TABLE_MODELS:
            tickers = {}
            for ticker in emp.TICKERS:
                diag = _diagnostics(_residuals(model, rets[ticker].to_numpy()))
                tickers[ticker] = diag
                cells[f"{ticker}|{regime}|{model}"] = diag
            models[model] = tickers
        periods.append(
            {
                "regime": regime,
                "title": meta["title"],
                "window": meta["window"],
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "n": int(len(rets)),
                "models": models,
            }
        )
    return {
        "meta": {
            "study": "V3 1.5-year monthly empirical study 10000 paths",
            "purpose": "model adequacy diagnostics on regime returns; not pricing RMSE/MAE",
            "return": "R_t = ln(S_t / S_{t-1})",
            "lb_lags": LB_LAGS,
            "arch_lags": ARCH_LAGS,
            "jump_thresh": JUMP_THRESH,
            "tickers": list(emp.TICKERS),
            "models": list(emp.TABLE_MODELS),
            "regimes": list(emp.REGIME_ORDER),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "periods": periods,
        "cells": cells,
    }


def _table_rows(period: dict) -> tuple[list[list[str]], list[str], list[str]]:
    top = ["Model"]
    for ticker in emp.TICKERS:
        top.extend([ticker, "", "", ""])
    header = [""] + list(TESTS) * len(emp.TICKERS)
    header[0] = "Model"
    rows = []
    for model in emp.TABLE_MODELS:
        row = [model]
        for ticker in emp.TICKERS:
            rec = period["models"][model][ticker]
            for test in TESTS:
                row.append(_fmt_pair(rec[test]["stat"], rec[test]["p"]))
        rows.append(row)
    return rows, top, header


def _table_note(period: dict) -> str:
    start = ra._long_date(period["start"])
    end = ra._long_date(period["end"])
    n = period["n"]
    return (
        f"Notes: This table reports residual diagnostics for {', '.join(emp.TABLE_MODELS)} "
        f"fitted to daily continuously compounded returns "
        f"R_t = ln(S_t / S_{{t-1}}) of {', '.join(emp.TICKERS)}. The sample is {start} to {end} "
        f"({n} trading days). Each model is estimated from that regime and company only. "
        f"Jump models use standardized residuals after the 3σ jump filter. "
        f"JB is the usual Jarque–Bera normality test for the standardized residuals. "
        f"Q(20) is the Ljung–Box portmanteau test for serial correlation in the standardized "
        f"residuals up to 20 lags; Q²(20) is the same test on the squared standardized residuals. "
        f"ARCH-LM is Engle's (1982) test for remaining ARCH effects with {ARCH_LAGS} lags. "
        f"The test statistic is reported with the p-value in brackets. N/A means the diagnostic "
        f"is not applicable or the residual series is not well-defined. These tests check model "
        f"adequacy on returns; they are not option pricing RMSE or MAE."
    )


def _style_diag_table(tbl) -> None:
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.5)
        cell.set_height(0.105)
        if r <= 1:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold", fontsize=6.6)
        elif c == 0:
            cell.set_text_props(ha="left", color="#1A1A1A", fontsize=7.0)
            cell.set_facecolor(ALT_BG if r % 2 == 0 else "white")
        elif r % 2 == 0:
            cell.set_facecolor(ALT_BG)
            cell.set_text_props(fontsize=6.3)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(fontsize=6.3)


def _wrapped_note(fig, text: str, y: float, *, width: int = 128, fontsize: float = 7.6) -> None:
    lines = textwrap.wrap(text, width=width) or [""]
    for i, line in enumerate(lines):
        fig.text(0.06, y - 0.0165 * i, line, fontsize=fontsize, color="#555555", va="top")


def _cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, "V3 model diagnostic verification", fontsize=17, weight="bold", color=NAVY, va="top")
    fig.text(
        0.08,
        0.912,
        f"Six models  ·  SPY, AAPL, MSFT  ·  four windows  ·  {emp.LOOKBACK_PHRASE} monthly study",
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
            "Return-based specification tests for each model × company × regime cell. It does not price options and does not use LSM RMSE or MAE.",
            f"One table per regime. Rows are {', '.join(emp.TABLE_MODELS)}. Columns are {', '.join(emp.TICKERS)}, each with JB, Q(20), Q²(20), and ARCH-LM.",
            "The companion notebook is the computational source of truth. This PDF is written from the same payload.",
        ],
        y,
    )
    y = section(
        "Sample and residuals",
        [
            "Daily R_t = ln(S_t / S_{t-1}) inside that regime window only. No lookback from outside the window.",
            "GBM: (R_t − μ̂) / σ̂.  Modified GBM: split-normal magnitude residual, up vs down.  GARCH: GARCH(1,1) z_t.  Merton: jump-adjusted diffusion σ.  GARCH–Merton / Heston–Merton: residuals after the 3σ jump filter.  Heston: filtered variance path from returns.",
            "A diagnostic is N/A if the residual series is not well-defined or the sample is too short for that test.",
        ],
        y,
    )
    y = section(
        "Tests",
        [
            "JB: usual Jarque–Bera test of normality of the standardized residuals.",
            "Q(20): Ljung–Box test of residual autocorrelation through lag 20.",
            "Q²(20): Ljung–Box test of autocorrelation in squared standardized residuals through lag 20.",
            f"ARCH-LM: Engle (1982) test for remaining ARCH effects, {ARCH_LAGS} lags. Statistic with p-value in brackets.",
        ],
        y,
    )
    fig.text(
        0.08,
        0.045,
        "Page map: cover  ·  Tables D1–D4 (one regime each). Pricing tables live in the companion empirical-study PDF.",
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
        0.06,
        0.955,
        f"Table D{idx}  ·  {period['regime']}  {period['title']}  ·  model diagnostics",
        fontsize=13.0,
        weight="bold",
        color=NAVY,
        va="top",
    )
    fig.text(
        0.06,
        0.920,
        f"{period['window']}   ·   n = {period['n']} daily observations   ·   statistic [p-value]",
        fontsize=8.8,
        color=MUTED,
        va="top",
    )
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.900, 0.900], transform=fig.transFigure, color=NAVY, lw=1.2))

    ax = fig.add_axes([0.04, 0.34, 0.92, 0.52])
    ax.axis("off")
    rows, top, header = _table_rows(period)
    cell_text = [top, header, *rows]
    tbl = ax.table(cellText=cell_text, loc="center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    _style_diag_table(tbl)
    # Widen the model column slightly by leaving auto widths.
    _wrapped_note(fig, _table_note(period), 0.30, width=132, fontsize=7.5)
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
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)", flush=True)
    return path


def _html_table(period: dict) -> str:
    parts = [
        "<table style='border-collapse:collapse;font-family:Helvetica,Arial,sans-serif;font-size:12px;width:100%;'>",
        "<thead>",
        "<tr>",
        f"<th rowspan='2' style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>Model</th>",
    ]
    for ticker in emp.TICKERS:
        parts.append(
            f"<th colspan='4' style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{ticker}</th>"
        )
    parts.append("</tr><tr>")
    for _ in emp.TICKERS:
        for test in TESTS:
            parts.append(
                f"<th style='background:{NAVY};color:white;padding:6px 8px;border:1px solid #D0D5DD;'>{test}</th>"
            )
    parts.append("</tr></thead><tbody>")
    for i, model in enumerate(emp.TABLE_MODELS, start=1):
        bg = ALT_BG if i % 2 == 0 else "white"
        parts.append("<tr>")
        parts.append(
            f"<td style='background:{bg};padding:6px 8px;border:1px solid #D0D5DD;font-weight:600;text-align:left;'>{model}</td>"
        )
        for ticker in emp.TICKERS:
            rec = period["models"][model][ticker]
            for test in TESTS:
                val = _fmt_pair(rec[test]["stat"], rec[test]["p"])
                parts.append(
                    f"<td style='background:{bg};padding:6px 8px;border:1px solid #D0D5DD;text-align:center;'>{val}</td>"
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
            f"""# V3 model diagnostic verification — {emp.LOOKBACK_PHRASE} monthly study

**This notebook is the computational source of truth.** The matching PDF is written from the same payload.

Specification tests on realized returns in the four windows of `V3 1.5-year monthly empirical study 10000 paths`. This report does **not** price options and does **not** use LSM RMSE or MAE.

| Item | Setting |
|------|---------|
| Names | {", ".join(emp.TICKERS)} |
| Models | {", ".join(emp.TABLE_MODELS)} |
| Sample | Daily $R_t=\\ln(S_t/S_{{t-1}})$ inside each regime window only |
| Tests | JB; Ljung–Box Q(20); Q²(20); ARCH-LM ({ARCH_LAGS} lags) |
| Cell | Test statistic with p-value in brackets; N/A if not applicable |
| Combinations | {len(emp.REGIME_ORDER)} regimes × {len(emp.TICKERS)} companies × {len(emp.TABLE_MODELS)} models = {len(emp.REGIME_ORDER) * len(emp.TICKERS) * len(emp.TABLE_MODELS)} |
"""
        )
    ]
    for i, period in enumerate(payload["periods"], start=1):
        cells.append(
            md(
                f"## Table D{i}. {period['regime']}  ·  {period['title']}\n\n"
                f"{period['window']}  ·  n = {period['n']} daily observations."
            )
        )
        cells.append(
            code(
                "from IPython.display import display, HTML\n"
                f"display(HTML(dx._html_table(payload['periods'][{i-1}])))",
                html=_html_table(period),
            )
        )
        cells.append(md(_table_note(period)))

    cells.append(
        md(
            """## Re-run

```python
import run_v3_1p5y_10k_monthly_model_diagnostics as dx
payload = dx.run()
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
