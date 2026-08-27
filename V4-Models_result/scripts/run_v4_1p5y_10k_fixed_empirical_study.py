#!/usr/bin/env python3
"""V4 1.5-year fixed empirical study, 10,000 LSM paths.

Same seven models, grouping, Monday ATM sample, 18-month lookback, four
companies, and four 1-year regimes as the V4 monthly 10k study.

Calibration is fixed: one 1.5-year window ending at the first session of
each regime, held for every Monday contract in that year.

GARCH is re-estimated under this rule (Duan LRNVR). The monthly GARCH
cache is not reused.
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
from pq_risk_premium import report_bates_pq, report_heston_pq, report_merton_pq  # noqa: E402

STUDY_TICKERS = ("SPY", "AAPL", "MSFT", "AMZN")
STUDY_MODELS = (
    "GBM",
    "Modified GBM",
    "GARCH",
    "Heston",
    "Merton",
    "GARCH–Merton",
    "Heston–Merton",
)
MONTHLY_CACHE = study.ROOT / "results" / "empirical_study_1p5y_monthly_10000"
V3_CONTRACTS = (
    study.REPO
    / "V3-Models_result"
    / "results"
    / "empirical_study_1p5y_monthly_10000"
    / "shared_contracts.json"
)
SHORT_NAME = "V4/V4 1.5-year fixed empirical study 10000 paths"
CACHE_NAME = "empirical_study_1p5y_fixed_10000"


def _report_garch(row):
    p_tbl, q_tbl = report_p_and_q(row)
    return p_tbl, {}, q_tbl


PQ_REPORTERS = {
    "GARCH": _report_garch,
    "Heston": report_heston_pq,
    "Merton": report_merton_pq,
    "Heston–Merton": report_bates_pq,
}


def apply_v4_fixed_10k_config() -> None:
    study.configure_lookback(
        window_label="1.5 years",
        lookback_offset=pd.DateOffset(months=18),
        lookback_phrase="1.5-year",
        cache_name=CACHE_NAME,
        short_name=SHORT_NAME,
        pdf_name="V4_1p5y_fixed_empirical_study.pdf",
        nb_name="V4_1p5y_fixed_empirical_study.ipynb",
        notebook_import="run_v4_1p5y_10k_fixed_empirical_study",
        engine_script="run_v4_1p5y_10k_fixed_empirical_study.py",
        n_paths=10000,
        tickers=STUDY_TICKERS,
        rolling_mode="none",
    )
    study.TABLE_MODELS = STUDY_MODELS
    study.MODELS = STUDY_MODELS
    os_study.COLORS = {**os_study.COLORS, "AMZN": "#C73E7B"}


def patch_v3_wrap() -> None:
    """Point V3 companion scripts at this V4 10k fixed cache and Results_In_Short folder."""
    import run_v3_1p5y_10k_monthly_empirical_study as wrap

    wrap.apply_1p5y_10k_config = apply_v4_fixed_10k_config
    apply_v4_fixed_10k_config()


def layout_short_outputs() -> None:
    """PDFs at the folder root, notebooks in Notebooks/, matching the monthly V4 study."""
    apply_v4_fixed_10k_config()
    short = study.SHORT
    short.mkdir(parents=True, exist_ok=True)
    nb_dir = short / "Notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "V4_1p5y_fixed_empirical_study_pan_q.pdf",
        "V4_1p5y_fixed_empirical_study_pan_q.ipynb",
    ):
        path = short / stale
        if path.exists():
            path.unlink()
        nested = nb_dir / stale
        if nested.exists():
            nested.unlink()
    for path in list(short.glob("*.ipynb")):
        dest = nb_dir / path.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(path), str(dest))
        print(f"moved {path.name} → Notebooks/", flush=True)


MODIFIED_GBM_NB_MD = """# Modified GBM — model specification

**This notebook is the documentation source of truth.** The PDF `V4_modified_gbm_model.pdf` in the parent folder is the same text.

Modified GBM is a discrete-time replacement for geometric Brownian motion used in the V4 1.5-year fixed 10,000-path study.

## 1. Idea

Standard GBM draws each log-return from one normal $N(\\mu\\Delta t,\\sigma^{2}\\Delta t)$. Signs are independent and up/down sizes share one volatility.

Modified GBM keeps the price map $S_{t+1}=S_t e^{r_t}$ but splits $r_t$ using the eight parameters in §4: $P(U\\mid U)$, $P(D\\mid D)$, $P(U\\mid D)$, $P(D\\mid U)$, $\\mu_U$, $\\sigma_U$, $\\mu_D$, $\\sigma_D$.

1. **Direction.** Given the last non-zero sign, draw the next sign from:
   - $P(U\\mid U)$ — probability the next bar is up if the last bar was up. Use after an up bar.
   - $P(D\\mid D)$ — probability the next bar is down if the last bar was down. Use after a down bar.
   - $P(U\\mid D)$ — probability the next bar is up if the last bar was down. Use after a down bar.
   - $P(D\\mid U)$ — probability the next bar is down if the last bar was up. Use after an up bar.
2. **Magnitude.** On an up bar the size is $|N(\\mu_U,\\sigma_U^{2})|$; on a down bar it is $|N(\\mu_D,\\sigma_D^{2})|$. So $\\mu_U,\\sigma_U$ are the typical up-move size and spread, and $\\mu_D,\\sigma_D$ are the down-move analogues.
3. **Price.** $r_t=+m_t$ on $U$ and $r_t=-m_t$ on $D$, then $S\\leftarrow S e^{r_t}$.

It is return-based. Option quotes are not used in calibration. American calls are priced by LSM on risk-neutral paths.

It is **not** Heston (no variance diffusion), **not** Merton (no jumps), **not** GARCH (no $\\omega,\\alpha,\\beta$), and **not** hidden-state regime-switching (the state is the previous observed sign).

## 2. Estimation

On each lookback window of log-returns $R_s=\\ln(S_s/S_{s-1})$:

- Drop zeros. Count consecutive sign pairs. Laplace-smoothed
  $\\hat{P}(U\\mid U)=(n_{UU}+1/2)/(n_{\\mathrm{from\\ }U}+1)$, and the analogue for $\\hat{P}(D\\mid D)$.
- $\\mu_U,\\sigma_U$ (resp. $\\mu_D,\\sigma_D$) = mean and sample SD of $|R|$ on up (resp. down) bars.
- `last_up` = sign of the last non-zero lookback return (starts the simulator).

Fixed: one 18-month window ending at the first session of each 1-year evaluation window. Parameters dated $t_0$ are held for every Monday contract in that window and are used only after $t_0$.

## 3. Simulation

**P-measure** (stock PDF): no drift shift. Path cloud $n=10000$, seed 42. Reported path = p50 vs realized adj-close.

**Q-measure** (decision / moneyness PDFs): after drawing $r$, shift
$r\\leftarrow r+(r_f\\Delta t-\\log\\mathbb{E}e^{r})$ so $\\mathbb{E}e^{r}=e^{r_f\\Delta t}$, $\\Delta t=1/252$. Then Longstaff–Schwartz on the same Monday ATM listed-call sample as every other model.

## 4. Parameters in the estimation PDF

The quantities used in §1, written once per regime (the single fixed lookback):

$P(U\\mid U),\\ P(D\\mid D),\\ P(U\\mid D),\\ P(D\\mid U),\\ \\mu_U,\\ \\sigma_U,\\ \\mu_D,\\ \\sigma_D$.

## 5. Where results are

- Seven-model ranking: `V4_1p5y_fixed_empirical_study.pdf`
- Return-based group (includes Modified GBM): `V4_1p5y_fixed_empirical_study_return_based.pdf`
- Parameters / stock / moneyness: the matching `V4_1p5y_fixed_*.pdf` files in the parent folder
- Code: `V4-Models_result/modified gbm notebook/20*_modified_gbm.ipynb`
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
                "id": "9a2e2d09",
                "metadata": {},
                "source": [ln + "\n" for ln in MODIFIED_GBM_NB_MD.strip().split("\n")],
            }
        ],
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return path


def copy_modified_gbm_spec() -> None:
    apply_v4_fixed_10k_config()
    study.SHORT.mkdir(parents=True, exist_ok=True)
    candidates = [
        study.REPO / "Results_In_Short" / "V4" / "V4 1.5-year monthly empirical study 10000 paths",
        study.REPO / "Results_In_Short" / "V3" / "V3 1.5-year monthly empirical study",
        study.REPO / "Results_In_Short" / "Other" / "V3 1.5-year monthly Modified GBM",
    ]
    dest_pdf = study.SHORT / "V4_modified_gbm_model.pdf"
    for src_dir in candidates:
        src = src_dir / "V4_modified_gbm_model.pdf"
        if not src.exists():
            src = src_dir / "V3_modified_gbm_model.pdf"
        if not src.exists() and (src_dir / "Notebooks" / "V3_modified_gbm_model.pdf").exists():
            src = src_dir / "Notebooks" / "V3_modified_gbm_model.pdf"
        if src.exists():
            shutil.copy2(src, dest_pdf)
            print(f"copied V4_modified_gbm_model.pdf from {src_dir.name}", flush=True)
            break
    else:
        print("missing V4_modified_gbm_model.pdf", flush=True)
    nb_dir = study.SHORT / "Notebooks"
    if nb_dir.is_dir():
        write_modified_gbm_notebook(nb_dir / "V4_modified_gbm_model.ipynb")
    else:
        write_modified_gbm_notebook(study.SHORT / "V4_modified_gbm_model.ipynb")


apply_v4_fixed_10k_config()


def _seed_shared_contracts() -> None:
    study.CACHE.mkdir(parents=True, exist_ok=True)
    (study.CACHE / "partial").mkdir(parents=True, exist_ok=True)
    if not study.CONTRACTS_JSON.exists():
        src = MONTHLY_CACHE / "shared_contracts.json"
        if not src.exists():
            src = V3_CONTRACTS
        if not src.exists():
            raise FileNotFoundError(
                f"Need the frozen Monday sample at {MONTHLY_CACHE / 'shared_contracts.json'}"
            )
        shutil.copy2(src, study.CONTRACTS_JSON)
        print(f"copied shared contracts from {src}", flush=True)
    if not study.FILTER_JSON.exists():
        src = MONTHLY_CACHE / "filter_funnel.json"
        if src.exists():
            shutil.copy2(src, study.FILTER_JSON)


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
        study.SHORT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, study.SHORT / "pq_parameters.csv")
    return df


def main(argv: list[str] | None = None) -> int:
    apply_v4_fixed_10k_config()
    _seed_shared_contracts()
    rc = study.main(argv)
    df = collect_pq_tables()
    if len(df):
        print("\n=== P / premium / Q (fixed calibration as-of) ===", flush=True)
        cols = [c for c in df.columns if c in ("ticker", "regime", "model", "n_updates", "asof")]
        print(df[cols].to_string(index=False), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
