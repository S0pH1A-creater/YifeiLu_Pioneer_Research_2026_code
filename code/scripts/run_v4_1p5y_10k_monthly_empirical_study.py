#!/usr/bin/env python3
"""1.5-year monthly empirical study, 10,000 LSM paths.

Return-based models only: GBM, MD-GBM, GARCH, Merton, GARCH–Merton.
Same Monday ATM sample, 18-month lookback, monthly rolling, four companies,
four 1-year regimes.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_optimal_stopping_study as os_study  # noqa: E402
import run_v3_5y_monthly_empirical_study as study  # noqa: E402
from garch_duan_lrnvr import report_p_and_q  # noqa: E402
from pq_risk_premium import report_merton_pq  # noqa: E402

STUDY_TICKERS = ("SPY", "AAPL", "MSFT", "AMZN")
STUDY_MODELS = (
    "GBM",
    "MD-GBM",
    "GARCH",
    "Merton",
    "GARCH–Merton",
)
MEANFIX_CACHE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000_mgbm_meanfix"
V3_CONTRACTS = (
    study.ROOT
    / "results"
    / "empirical_study_1p5y_monthly_10000"
    / "shared_contracts.json"
)
GARCH_CACHE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000_garch"


def _report_garch(row):
    p_tbl, q_tbl = report_p_and_q(row)
    return p_tbl, {}, q_tbl


PQ_REPORTERS = {
    "GARCH": _report_garch,
    "Merton": report_merton_pq,
}


def apply_v4_10k_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name="empirical_study_1p5y_monthly_10000",
        short_name="results/1p5y_monthly_return_based",
        pdf_name="V4_1p5y_monthly_empirical_study.pdf",
        nb_name="V4_1p5y_monthly_empirical_study.ipynb",
        notebook_import="run_v4_1p5y_10k_monthly_empirical_study",
        engine_script="run_v4_1p5y_10k_monthly_empirical_study.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}


def patch_v3_wrap() -> None:
    """Point companion scripts at this 10k cache and results folder."""
    import run_v3_1p5y_10k_monthly_empirical_study as wrap

    wrap.apply_1p5y_10k_config = apply_v4_10k_config
    apply_v4_10k_config()


def layout_short_outputs() -> None:
    """Place reports into the labeled results subfolders."""
    apply_v4_10k_config()
    short = study.SHORT
    mapping = {
        "V4_1p5y_monthly_empirical_study_return_based.pdf": "01_optimal_stopping",
        "V4_1p5y_monthly_empirical_study_return_based.ipynb": "01_optimal_stopping",
        "V4_1p5y_monthly_empirical_study_return_based_8tables.xlsx": "01_optimal_stopping",
        "V4_1p5y_monthly_stock_price_return_based.pdf": "02_stock_price",
        "V4_1p5y_monthly_stock_price_return_based.ipynb": "02_stock_price",
        "V4_1p5y_monthly_stock_price_return_based_8tables.xlsx": "02_stock_price",
        "V4_MD-GBM_model.pdf": "03_md_gbm_spec",
        "V4_MD-GBM_model.ipynb": "03_md_gbm_spec",
        "V4_1p5y_monthly_return_analysis.pdf": "04_other_reports",
        "V4_1p5y_monthly_return_analysis.ipynb": "04_other_reports",
        "pq_parameters.csv": "04_other_reports",
    }
    for name, folder in mapping.items():
        dest_dir = short / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = short / name
        if not src.exists():
            nested = short / "Notebooks" / name
            src = nested if nested.exists() else src
        if src.exists() and src.parent.resolve() != dest_dir.resolve():
            dest = dest_dir / name
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
            print(f"moved {name} → {folder}/", flush=True)


MODIFIED_GBM_NB_MD = """# MD-GBM — model specification

**MD-GBM means Markov Directional Geometric Brownian Motion.**

**This notebook is the documentation source of truth.** The PDF `V4_MD-GBM_model.pdf` in the parent folder is the same text.

MD-GBM is the return-based model in the V4 1.5-year monthly 10,000-path study. It is original Modified GBM with one calibration fix: folded-normal means matched to sample |R|.

Same as original Modified GBM: 1-lag up/down Markov chain; sizes $|N(\\mu,\\sigma^{2})|$ on up and down bars; additive $Q$ shift so $E[e^{R}]=e^{r_f\\Delta t}$.

Only change: $\\mu,\\sigma$ are inverted so $E[|N(\\mu,\\sigma)|]$ equals the sample mean of $|R|$ in that up/down bucket. Original set $\\mu=$ mean$(|R|)$ and then drew $|N(\\mu,\\sigma)|$, which makes simulated sizes too large. If the half-normal at the sample SD already overshoots that mean, $\\mu=0$ and $\\sigma$ is shrunk to match.

## Symbols

| Symbol | Meaning |
|--------|---------|
| $R_t=\\ln(S_t/S_{t-1})$ | log-return |
| $U,D$ | up / down |
| $P(U\\mid U),\\ P(D\\mid D),\\ P(U\\mid D),\\ P(D\\mid U)$ | sign coins |
| $\\mu_U,\\sigma_U$ | folded-normal parameters for up sizes |
| $\\mu_D,\\sigma_D$ | folded-normal parameters for down sizes |

## Equations

On an up bar the size is $|N(\\mu_U,\\sigma_U^{2})|$; on a down bar it is $|N(\\mu_D,\\sigma_D^{2})|$. Then $R_t=+m_t$ or $-m_t$ and $S_{t+1}=S_t e^{R_t}$.

$\\mu,\\sigma$ in each bucket solve $E[|N(\\mu,\\sigma)|]=\\overline{|R|}$ (and keep the sample SD of $|R|$ when that is feasible).

Under $Q$, after drawing $R$ add $r_f\\Delta t-\\log E[e^{R}]$ so $E[e^{R}]=e^{r_f\\Delta t}$. Then Longstaff–Schwartz on the same Monday ATM listed-call sample as every other model.

It is return-based. Option quotes are not used in $P$ calibration.

It is **not** Merton and **not** GARCH.

## Estimation

On each 18-month lookback: drop zeros; Laplace-smoothed sign transitions; invert folded-normal $\\mu,\\sigma$ on up and down $|R|$; start from the last non-zero sign. Parameters dated $t$ are used only after $t$.

## Where results are

- LSM ranking: `results/1p5y_monthly_return_based/01_optimal_stopping/`
- Stock paths: `results/1p5y_monthly_return_based/02_stock_price/`
- Code: `code/md_gbm/20*_modified_gbm_meanfix.ipynb`
"""


def write_modified_gbm_notebook(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "id": "md-gbm-spec",
                "metadata": {},
                "source": [ln + "\n" for ln in MODIFIED_GBM_NB_MD.strip().split("\n")],
            }
        ],
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


def write_modified_gbm_v2_pdf(path: Path) -> Path:
    import os
    import textwrap

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(study.REPO / ".mplconfig"),
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    navy, muted = study.NAVY, study.MUTED
    path.parent.mkdir(parents=True, exist_ok=True)

    def _page(title: str, sections: list[tuple[str, list[str]]], footer: str) -> None:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.955, title, fontsize=15, weight="bold", color=navy, va="top")
        fig.text(
            0.08,
            0.918,
            "V4 1.5-year monthly 10,000-path study  ·  documentation source of truth is the companion notebook",
            fontsize=8.8,
            color=muted,
            va="top",
        )
        fig.add_artist(plt.Line2D([0.08, 0.92], [0.898, 0.898], transform=fig.transFigure, color=navy, lw=1.2))
        y = 0.868

        def section(heading: str, bullets: list[str], y0: float) -> float:
            fig.text(0.08, y0, heading, fontsize=11.2, weight="bold", color=navy, va="top")
            y = y0 - 0.028
            for b in bullets:
                wrapped = textwrap.wrap(b, width=94) or [""]
                fig.text(0.10, y, "•", fontsize=9.2, color=navy, va="top")
                fig.text(0.13, y, wrapped[0], fontsize=8.5, color="#222222", va="top")
                y -= 0.0176
                for line in wrapped[1:]:
                    fig.text(0.13, y, line, fontsize=8.5, color="#222222", va="top")
                    y -= 0.0176
                y -= 0.005
            return y - 0.010

        for heading, bullets in sections:
            y = section(heading, bullets, y)
        fig.text(0.08, 0.035, footer, fontsize=7.6, color="#666666", va="bottom")
        return fig

    with PdfPages(path) as pdf:
        fig = _page(
            "MD-GBM  ·  model specification",
            [
                (
                    "Name",
                    [
                        "MD-GBM means Markov Directional Geometric Brownian Motion.",
                    ],
                ),
                (
                    "Idea",
                    [
                        "MD-GBM is original Modified GBM with one calibration fix. It is the return-based model in the V4 monthly 10,000-path study. Price map S_{t+1}=S_t e^{R_t}. Direction under P is the 1-lag up/down Markov chain. Sizes are |N(μ, σ²)| on up and down bars. Q is the usual additive shift so E[e^R]=e^{r_f Δt}.",
                        "Only change: μ and σ are inverted so E[|N(μ, σ)|] equals the sample mean of |R| in that bucket. Original set μ = mean(|R|) and then drew |N(μ, σ)|, so simulated sizes were too large. If the half-normal at the sample SD already overshoots, μ=0 and σ is shrunk to match the mean.",
                    ],
                ),
                (
                    "What it is not",
                    [
                        "Not Merton and not GARCH.",
                    ],
                ),
            ],
            "Page 1 of 2  ·  notebook: Notebooks/V4_MD-GBM_model.ipynb",
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig = _page(
            "MD-GBM  ·  estimation, Q, and files",
            [
                (
                    "Estimation (each 18-month lookback)",
                    [
                        "Drop zero log-returns. Laplace-smoothed P(U|U) and P(D|D) from consecutive sign pairs.",
                        "On up (down) bars, invert folded-normal μ, σ so E[|N|] equals the sample mean of |R|.",
                        "Simulator start: last observed non-zero sign. Parameters dated t are used only after t.",
                    ],
                ),
                (
                    "Simulation",
                    [
                        "P-measure (stock PDF): no drift shift. Path cloud n=10,000, seed 42. Reported path = p50 vs realized adj-close.",
                        "Q-measure (LSM): after drawing R, add r_f Δt − log E[e^R]. Then Longstaff–Schwartz on the same Monday ATM listed-call sample as every other model.",
                    ],
                ),
                (
                    "Parameters in the estimation PDF",
                    [
                        "P(U|U), P(D|D), P(U|D), P(D|U), μ_U, σ_U, μ_D, σ_D.",
                    ],
                ),
                (
                    "Where results are",
                    [
                        "LSM ranking: results/1p5y_monthly_return_based/01_optimal_stopping/.",
                        "Code: code/md_gbm/20*_modified_gbm_meanfix.ipynb.",
                    ],
                ),
            ],
            "Page 2 of 2  ·  this PDF matches the companion notebook",
        )
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {path}", flush=True)
    return path


def _is_legacy_modified_gbm_file(path: Path) -> bool:
    name = path.name.lower()
    if any(tag in name for tag in ("_v2", "v2.", "v3", "meanfix", "_ai", " ai")):
        return False
    return "modified_gbm" in name


def _is_v2_artifact(path: Path) -> bool:
    name = path.name.lower()
    return "modified_gbm_v2" in name


def _is_meanfix_artifact(path: Path) -> bool:
    name = path.name.lower()
    return "modified_gbm_meanfix" in name


def _relabel_meanfix_json(path: Path) -> None:
    if path.suffix.lower() != ".json":
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    def walk(obj):
        changed = False
        if isinstance(obj, dict):
            if obj.get("model") == "Modified GBM meanfix":
                obj["model"] = "MD-GBM"
                changed = True
            for val in obj.values():
                if walk(val):
                    changed = True
        elif isinstance(obj, list):
            for val in obj:
                if walk(val):
                    changed = True
        return changed

    if walk(data):
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def swap_modified_gbm_meanfix_cache() -> None:
    """Replace official-cache v2 / original Modified GBM artifacts with meanfix cells."""
    apply_v4_10k_config()
    src, dst = MEANFIX_CACHE, study.CACHE
    if not src.exists():
        return
    n_copy = 0
    for path in src.rglob("*"):
        if not path.is_file() or not _is_meanfix_artifact(path):
            continue
        dest = dst / path.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        n_copy += 1
        _relabel_meanfix_json(dest)
    n_del = 0
    for path in list(dst.rglob("*")):
        if not path.is_file():
            continue
        if _is_v2_artifact(path) or _is_legacy_modified_gbm_file(path):
            path.unlink()
            n_del += 1
    for stale in (
        dst / "payload_option_implied.json",
        dst / "moneyness_payload.json",
    ):
        if stale.exists():
            stale.unlink()
    for path in dst.rglob("*.json"):
        _relabel_meanfix_json(path)
    print(
        f"swapped v2 → MD-GBM in {dst.name}: copied {n_copy} meanfix files (labelled MD-GBM), removed {n_del} v2/original files",
        flush=True,
    )


def swap_modified_gbm_v2_cache() -> None:
    swap_modified_gbm_meanfix_cache()


def copy_modified_gbm_spec() -> None:
    apply_v4_10k_config()
    dest = study.SHORT / "03_md_gbm_spec"
    dest.mkdir(parents=True, exist_ok=True)
    write_modified_gbm_v2_pdf(dest / "V4_MD-GBM_model.pdf")
    write_modified_gbm_notebook(dest / "V4_MD-GBM_model.ipynb")


apply_v4_10k_config()


def _copy_missing(src: Path, dst: Path, pattern: str) -> int:
    if not src.exists():
        return 0
    n = 0
    for path in src.rglob(pattern):
        if not path.is_file():
            continue
        dest = dst / path.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
            n += 1
    return n


def merge_garch_cache() -> None:
    """Reuse the finished GARCH-only 10k cells. Do not re-price GARCH."""
    src, dst = GARCH_CACHE, study.CACHE
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "partial").mkdir(parents=True, exist_ok=True)
    n = 0
    n += _copy_missing(src / "partial", dst / "partial", "*.json")
    n += _copy_missing(src / "contracts", dst / "contracts", "*garch.csv")
    n += _copy_missing(
        src / "parameter_estimation" / "raw",
        dst / "parameter_estimation" / "raw",
        "GARCH*",
    )
    if n:
        print(f"merged {n} GARCH artifacts from {src.name}", flush=True)


def _seed_shared_contracts() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if not study.CONTRACTS_JSON.exists():
        src = GARCH_CACHE / "shared_contracts.json"
        if not src.exists():
            src = V3_CONTRACTS
        if not src.exists():
            raise FileNotFoundError(
                f"Need the frozen V3 Monday sample at {V3_CONTRACTS}"
            )
        shutil.copy2(src, study.CONTRACTS_JSON)
        print(f"copied shared contracts from {src}", flush=True)
    if not study.FILTER_JSON.exists() and (GARCH_CACHE / "filter_funnel.json").exists():
        shutil.copy2(GARCH_CACHE / "filter_funnel.json", study.FILTER_JSON)
    merge_garch_cache()


def _safe_model(model: str) -> str:
    return model.replace("–", "-").replace(" ", "_")


def _save_calibration(model: str, regime: str, ticker: str, cal: pd.DataFrame) -> None:
    if cal is None or len(cal) == 0:
        return
    dest = study.CACHE / "parameter_estimation" / "raw" / regime / ticker
    dest.mkdir(parents=True, exist_ok=True)
    cal.to_csv(dest / f"{_safe_model(model)}.csv", index=False)
    reporter = PQ_REPORTERS.get(model)
    if reporter is None:
        return
    last = cal.iloc[-1]
    p_tbl, prem, q_tbl = reporter(last)
    dest.joinpath(f"{_safe_model(model)}_P_Q.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "regime": regime,
                "model": model,
                "n_updates": int(len(cal)),
                "asof": str(last.get("date")),
                "P": {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in p_tbl.items()},
                "premium": {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in prem.items()},
                "Q": {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in q_tbl.items()},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


_study_load_ns = os_study._load_ns
_orig_run_mode = os_study._run_mode


def _load_ns(nb_path):
    g = _study_load_ns(nb_path)
    g["_v4_regime"] = os_study._regime_from_name(Path(nb_path))
    g["_v4_model"] = None
    return g


def _run_mode(g, rolling_mode, contracts, ticker):
    result = _orig_run_mode(g, rolling_mode, contracts, ticker)
    cal = g.get("rolling", {}).get(ticker)
    regime = g.get("_v4_regime")
    model = g.get("_v4_model")
    if cal is None or regime is None or model is None:
        return result
    _save_calibration(model, regime, ticker, cal)
    return result


os_study._load_ns = _load_ns
os_study._run_mode = _run_mode

_orig_run_one = study._run_one


def _run_one(model, nb_path, shared, tickers=None):
    prev = os_study._load_ns

    def _load_with_model(path):
        ns = prev(path)
        ns["_v4_model"] = model
        ns["_v4_regime"] = os_study._regime_from_name(Path(path))
        return ns

    os_study._load_ns = _load_with_model
    try:
        return _orig_run_one(model, nb_path, shared, tickers=tickers)
    finally:
        os_study._load_ns = prev


study._run_one = _run_one


def _flatten_measure(prefix: str, block: dict) -> dict:
    out = {}
    for key, val in (block or {}).items():
        if key == "measure":
            continue
        out[f"{prefix}{key}"] = val
    return out


def collect_pq_tables() -> pd.DataFrame:
    rows = []
    raw = study.CACHE / "parameter_estimation" / "raw"
    if not raw.exists():
        return pd.DataFrame()
    for path in sorted(raw.glob("*/*/*_P_Q.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        model = rec.get("model")
        if not model:
            stem = path.name.replace("_P_Q.json", "")
            model = stem.replace("Heston-Merton", "Heston–Merton").replace("_", " ")
        row = {
            "ticker": rec.get("ticker"),
            "regime": rec.get("regime"),
            "model": model,
            "n_updates": rec.get("n_updates"),
            "asof": rec.get("asof"),
        }
        row.update(_flatten_measure("P_", rec.get("P") or {}))
        row.update(_flatten_measure("prem_", rec.get("premium") or {}))
        row.update(_flatten_measure("Q_", rec.get("Q") or {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        dest = study.CACHE / "pq_parameters.csv"
        df.to_csv(dest, index=False)
        out_dir = study.SHORT / "04_other_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, out_dir / "pq_parameters.csv")
    return df


def main(argv: list[str] | None = None) -> int:
    apply_v4_10k_config()
    swap_modified_gbm_meanfix_cache()
    _seed_shared_contracts()
    rc = study.main(argv)
    df = collect_pq_tables()
    if len(df):
        print("\n=== P / premium / Q (last monthly update) ===", flush=True)
        cols = [c for c in df.columns if c in ("ticker", "regime", "model", "n_updates", "asof")]
        print(df[cols].to_string(index=False), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
