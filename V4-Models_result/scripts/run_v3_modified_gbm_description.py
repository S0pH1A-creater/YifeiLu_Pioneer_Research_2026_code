#!/usr/bin/env python3
"""Write the Modified GBM model-specification PDF (and a matching notebook).

This is documentation only: no calibration, LSM, or new Monte Carlo.
Copies the same files into every 1.5-year Results_In_Short folder that uses
the model.
"""
from __future__ import annotations

import json
import os
import shutil
import textwrap
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
from matplotlib.backends.backend_pdf import PdfPages

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPO = ROOT.parent

NAVY = "#1F3A5F"
MUTED = "#444444"
ALT = "#F4F7FB"
PDF_NAME = "V3_modified_gbm_model.pdf"
NB_NAME = "V3_modified_gbm_model.ipynb"

DEST_FOLDERS = [
    REPO / "Results_In_Short" / "V3 1.5-year monthly Modified GBM",
    REPO / "Results_In_Short" / "V3 1.5-year monthly empirical study 10000 paths",
    REPO / "Results_In_Short" / "V3 1.5-year monthly Modified GBM AMZN",
    REPO / "Results_In_Short" / "V3 1.5-year fixed 10000 paths",
]


def _page(title: str, subtitle: str, sections: list[tuple[str, list[str]]], footer: str | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=16, weight="bold", color=NAVY, va="top")
    fig.text(0.08, 0.918, subtitle, fontsize=9.2, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.898, 0.898], transform=fig.transFigure, color=NAVY, lw=1.3))
    y = 0.868

    def section(heading: str, bullets: list[str], y0: float) -> float:
        fig.text(0.08, y0, heading, fontsize=11.5, weight="bold", color=NAVY, va="top")
        y = y0 - 0.028
        for b in bullets:
            wrapped = textwrap.wrap(b, width=92) or [""]
            fig.text(0.10, y, "•", fontsize=9.4, color=NAVY, va="top")
            fig.text(0.13, y, wrapped[0], fontsize=8.7, color="#222222", va="top")
            y -= 0.0178
            for line in wrapped[1:]:
                fig.text(0.13, y, line, fontsize=8.7, color="#222222", va="top")
                y -= 0.0178
            y -= 0.006
        return y - 0.010

    for heading, bullets in sections:
        y = section(heading, bullets, y)
    if footer:
        fig.text(0.08, 0.035, footer, fontsize=7.6, color="#666666", va="bottom")
    return fig


def _eq_page(title: str, subtitle: str, blocks: list[tuple[str, str, list[str]]]) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=16, weight="bold", color=NAVY, va="top")
    fig.text(0.08, 0.918, subtitle, fontsize=9.2, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.898, 0.898], transform=fig.transFigure, color=NAVY, lw=1.3))
    y = 0.868
    for heading, eq, notes in blocks:
        fig.text(0.08, y, heading, fontsize=11.5, weight="bold", color=NAVY, va="top")
        y -= 0.045
        ax = fig.add_axes([0.10, y - 0.055, 0.80, 0.058])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(ALT)
        ax.text(0.5, 0.5, eq, ha="center", va="center", fontsize=12.5, color=NAVY)
        y -= 0.072
        for note in notes:
            wrapped = textwrap.wrap(note, width=92) or [""]
            for i, line in enumerate(wrapped):
                prefix = "• " if i == 0 else "  "
                fig.text(0.13, y, prefix + line, fontsize=8.6, color="#222222", va="top")
                y -= 0.0175
            y -= 0.006
        y -= 0.012
    return fig


def write_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = _page(
            "Modified GBM  ·  model specification",
            "Markov direction  ·  split-normal magnitudes  ·  discrete log-price recursion",
            [
                (
                    "What this model is",
                    [
                        "Modified GBM is a discrete-time replacement for geometric Brownian motion. Standard GBM draws each log-return from one normal N(μΔt, σ²Δt). That forces a single drift, a single volatility, and independent signs from one bar to the next.",
                        "The replacement keeps the GBM price map S_{t+1} = S_t exp(r_t), but splits r_t into a direction and a magnitude. Direction is a two-state Markov chain (up / down). Magnitude is drawn from a different truncated-normal law on up bars than on down bars.",
                        "The model is return-based. Parameters are estimated from lookback log-returns only. Listed option quotes are not used in calibration. American calls are priced afterwards by Longstaff–Schwartz LSM on risk-neutral paths.",
                    ],
                ),
                (
                    "Why it exists",
                    [
                        "Equity returns cluster in sign: up days tend to follow up days, and down days follow down days, more often than an i.i.d. sign would imply.",
                        "Up and down moves need not have the same typical size. A single σ cannot represent that split.",
                        "GBM, GARCH, Heston, and Merton already sit in the V3 comparison. Modified GBM is the extra return-based specification that carries those two empirical facts without adding stochastic variance or Poisson jumps.",
                    ],
                ),
                (
                    "What it is not",
                    [
                        "Not Black–Scholes GBM with a patched drift. The sign process is Markov, not independent.",
                        "Not a regime-switching GBM with hidden states. The state is the previous bar’s observed sign, which is known at t.",
                        "Not Heston: variance is not a mean-reverting diffusion. Not Merton: there is no jump intensity. Not GARCH: there is no ω, α, β recursion on σ²_t.",
                    ],
                ),
            ],
            "Companion to the 1.5-year monthly 10,000-path study. Documentation only; numbers in the empirical PDFs are from stored LSM cells.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _eq_page(
            "Modified GBM  ·  three stages",
            "Every simulated bar uses the same three-stage map.",
            [
                (
                    "Stage 1 — direction",
                    r"$U=\{r_t>0\},\quad D=\{r_t<0\}$",
                    [
                        "The previous non-zero return’s sign selects the next-bar probability. After an up bar the next bar is up with P(U|U) and down with P(D|U)=1−P(U|U). After a down bar the pair is P(U|D), P(D|D).",
                        "High P(U|U) or P(D|D) produces runs of the same sign. High P(U|D) or P(D|U) produces reversals.",
                        "Zero returns are dropped from the transition count. The last observed sign in the lookback, last_up, starts the Monte Carlo chain.",
                    ],
                ),
                (
                    "Stage 2 — magnitude",
                    r"$m_t|U\sim|N(\mu_U,\sigma_U^2)|,\quad m_t|D\sim|N(\mu_D,\sigma_D^2)|$",
                    [
                        "μ_U and σ_U are the sample mean and sample SD of |r| on up bars in the lookback. μ_D and σ_D are the same statistics on down bars. Units are the bar’s log-return (daily in the 1.5-year study).",
                        "The absolute value keeps magnitudes strictly positive, so the sign comes only from Stage 1.",
                    ],
                ),
                (
                    "Stage 3 — price",
                    r"$r_t=+m_t$ on $U$,   $r_t=-m_t$ on $D$,   $S_{t+1}=S_t e^{r_t}$",
                    [
                        "This is the GBM exponential map with a signed, state-dependent increment instead of N(μΔt, σ²Δt).",
                        "One bar = one trading day in the 1.5-year monthly study (Δt = 1/252 when a risk-free shift is applied).",
                    ],
                ),
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _eq_page(
            "Modified GBM  ·  estimation",
            "Closed form on the lookback window. No numerical optimizer.",
            [
                (
                    "Laplace-smoothed transitions",
                    r"$\hat P(U|U)=(n_{UU}+0.5)/(n_{fromU}+1)$",
                    [
                        "n_UU is the number of up→up pairs among consecutive non-zero returns. n_from_U is the number of pairs that start with an up. P(D|D) is the same formula on down→down counts.",
                        "The +1/2 and +1 are Laplace (add-one-half) smoothing so a window with no reversals does not pin a probability at 0 or 1.",
                        "Then P(D|U)=1−P(U|U) and P(U|D)=1−P(D|D). The four probabilities are a two-parameter object.",
                    ],
                ),
                (
                    "Split-normal magnitudes",
                    r"$\hat\mu_U=\mathrm{mean}(|r|_{up}),\ \hat\sigma_U=\mathrm{sd}(|r|_{up})$",
                    [
                        "Down-bar analogues use r<0. If a side has one observation, σ is replaced by the median absolute return in the window. If a side is empty, both μ and σ fall back to that median (or 10^{-6}).",
                        "σ is floored at 10^{-6} so the simulator never draws a degenerate normal.",
                    ],
                ),
                (
                    "Rolling window",
                    r"$\hat\theta_t=\hat\theta(\{R_s: t-L<s\leq t\})$",
                    [
                        "L is the lookback (18 months in the 1.5-year 10k study). Update dates are the first trading day of the evaluation window plus each month-end.",
                        "Parameters dated t are used only on steps after t (no look-ahead). If the file is shorter than L, the estimator uses all available history up to t. Equity starts 2003-12-01.",
                    ],
                ),
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _page(
            "Modified GBM  ·  simulation and pricing",
            "P-measure stock paths  ·  Q-measure LSM calls",
            [
                (
                    "P-measure (stock-path report)",
                    [
                        "Start every path at S_0. Initialize the Markov state from last_up in the current parameter row.",
                        "For each step i: read (P(U|U), P(U|D), μ_U, σ_U, μ_D, σ_D) from the rolling schedule at that date. Draw the next sign from the state-dependent Bernoulli. Draw |N(μ,σ)| on that side. Set r = ±m and S ← S exp(r). Do not shift the mean.",
                        "Reported path = 50th percentile across n_paths = 10,000. Stock RMSE% is that p50 path versus realized adj-close.",
                    ],
                ),
                (
                    "Q-measure (American-call LSM)",
                    [
                        "Keep the calibrated transitions and magnitudes. After drawing the P-measure increment r on a step, shift the whole cross-section of paths so that the mean of exp(r) equals exp(r_f Δt), with Δt = 1/252 and r_f the 3-month Treasury as of that date.",
                        "The shift is r ← r + (r_f Δt − log mean exp(r)). It is a multiplicative drift correction, not a change of the Markov law.",
                        "LSM is Longstaff–Schwartz on the full path cloud (n_paths = 10,000, seed 42, n_steps = DTE). Paths are not averaged before the stopping decision. The contract sample is the same Monday ATM listed-call set used by every other model.",
                    ],
                ),
                (
                    "Parameter vector written in the estimation PDF",
                    [
                        "P(U|U), P(D|D), P(U|D), P(D|U), μ_U, σ_U, μ_D, σ_D. last_up is stored for simulation but is not a free parameter; it is the last lookback sign.",
                        "Tables in V3_1p5y_monthly_parameter_estimation.pdf report Mean, SD, Min, Max of the monthly estimates inside each regime and company.",
                    ],
                ),
            ],
            None,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _page(
            "Modified GBM  ·  where the numbers live",
            "Same Monday ATM sample, 18-month lookback, monthly rolling, 10,000 LSM paths.",
            [
                (
                    "Empirical ranking (decision)",
                    [
                        "V3 1.5-year monthly empirical study 10000 paths / V3_1p5y_monthly_empirical_study.pdf — seven models, four companies.",
                        "Return-based companion V3_1p5y_monthly_empirical_study_return_based.pdf ranks Modified GBM with GBM, GARCH, Merton, and GARCH–Merton.",
                        "Option-implied companion does not include Modified GBM (Heston family only).",
                    ],
                ),
                (
                    "Parameters, stock paths, moneyness",
                    [
                        "Parameter estimation: V3_1p5y_monthly_parameter_estimation.pdf (Modified GBM table P*).",
                        "Stock paths: V3_1p5y_monthly_stock_price.pdf (P-measure p50 vs S_t).",
                        "Moneyness: V3_1p5y_monthly_moneyness_performance.pdf (same LSM contracts, split by S/K).",
                        "Standalone copies of those four reports also sit in V3 1.5-year monthly Modified GBM/.",
                    ],
                ),
                (
                    "Code",
                    [
                        "Notebooks: V4-Models_result/modified gbm notebook/20*_modified_gbm.ipynb. Functions: estimate_modified_gbm, calibrate_ticker, simulate_modified_gbm_rolling.",
                        "Engine: run_v3_1p5y_10k_monthly_modified_gbm.py (standalone cache) and run_v3_1p5y_10k_monthly_empirical_study.py (seven-model 10k cache).",
                    ],
                ),
            ],
            "This PDF does not replace the ranking tables. It only states the model that those tables evaluate.",
        )
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {path}", flush=True)
    return path


def write_notebook(path: Path) -> Path:
    md = r"""# Modified GBM — model specification

**This notebook is the documentation source of truth.** The PDF `V3_modified_gbm_model.pdf` in this folder is the same text.

Modified GBM is a discrete-time replacement for geometric Brownian motion used in the V3 1.5-year monthly 10,000-path study.

## 1. Idea

Standard GBM draws each log-return from one normal \(N(\mu\Delta t,\sigma^2\Delta t)\). Signs are independent and up/down sizes share one volatility.

Modified GBM keeps the price map \(S_{t+1}=S_t e^{r_t}\) but splits \(r_t\) into:

1. **Direction** — two-state Markov chain on \(\{U,D\}\).
2. **Magnitude** — \(|N(\mu_U,\sigma_U^2)|\) on up bars and \(|N(\mu_D,\sigma_D^2)|\) on down bars.
3. **Price** — \(r_t=+m_t\) or \(r_t=-m_t\), then \(S\leftarrow S e^{r_t}\).

It is return-based. Option quotes are not used in calibration. American calls are priced by LSM on risk-neutral paths.

It is **not** Heston (no variance diffusion), **not** Merton (no jumps), **not** GARCH (no \(\omega,\alpha,\beta\)), and **not** hidden-state regime-switching (the state is the previous observed sign).

## 2. Estimation

On each lookback window of log-returns \(R_s=\ln(S_s/S_{s-1})\):

- Drop zeros. Count consecutive sign pairs. Laplace-smoothed
  \(\hat P(U\mid U)=(n_{UU}+1/2)/(n_{\mathrm{from\ }U}+1)\), and the analogue for \(\hat P(D\mid D)\).
- \(\mu_U,\sigma_U\) (resp. \(\mu_D,\sigma_D\)) = mean and sample SD of \(|R|\) on up (resp. down) bars.
- `last_up` = sign of the last non-zero lookback return (starts the simulator).

Rolling: 18-month window ending at each month-end (plus period start). Parameters dated \(t\) are used only after \(t\).

## 3. Simulation

**P-measure** (stock PDF): no drift shift. Path cloud \(n=10000\), seed 42. Reported path = p50 vs realized adj-close.

**Q-measure** (decision / moneyness PDFs): after drawing \(r\), shift
\(r\leftarrow r+(r_f\Delta t-\log\mathbb{E}e^{r})\) so \(\mathbb{E}e^{r}=e^{r_f\Delta t}\), \(\Delta t=1/252\). Then Longstaff–Schwartz on the same Monday ATM listed-call sample as every other model.

## 4. Parameters in the estimation PDF

\(P(U\mid U),\ P(D\mid D),\ P(U\mid D),\ P(D\mid U),\ \mu_U,\ \sigma_U,\ \mu_D,\ \sigma_D\).

## 5. Where results are

- Seven-model ranking: `V3_1p5y_monthly_empirical_study.pdf`
- Return-based group (includes Modified GBM): `V3_1p5y_monthly_empirical_study_return_based.pdf`
- Parameters / stock / moneyness: the matching `V3_1p5y_monthly_*.pdf` files in this folder
- Code: `V4-Models_result/modified gbm notebook/20*_modified_gbm.ipynb`
"""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": uuid.uuid4().hex[:8],
                "metadata": {},
                "source": [ln + "\n" for ln in md.strip().split("\n")],
            }
        ],
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


def main() -> int:
    primary = DEST_FOLDERS[0]
    pdf = write_pdf(primary / PDF_NAME)
    nb = write_notebook(primary / NB_NAME)
    for folder in DEST_FOLDERS[1:]:
        folder.mkdir(parents=True, exist_ok=True)
        if folder.resolve() == primary.resolve():
            continue
        shutil.copy2(pdf, folder / PDF_NAME)
        shutil.copy2(nb, folder / NB_NAME)
        print(f"copied to {folder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
