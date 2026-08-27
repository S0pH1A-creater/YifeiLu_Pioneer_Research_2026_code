#!/usr/bin/env python3
"""Split the V4 1.5-year fixed 10k LSM study into V3's two calibration-group reports.

The split is how P is estimated, not the P→Q map:

- return-based: GBM, Modified GBM, GARCH, Merton, GARCH–Merton
  (dynamics from lookback returns)
- option-implied: Heston, Heston–Merton
  (CIR / Bates parameters from listed calls)

No new Monte Carlo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as study  # noqa: E402
import run_v4_1p5y_10k_fixed_empirical_study as emp  # noqa: E402

GROUPS = {
    "option_implied": {
        "models": ("Heston", "Heston–Merton"),
        "pdf_name": "V4_1p5y_fixed_empirical_study_option_implied.pdf",
        "nb_name": "V4_1p5y_fixed_empirical_study_option_implied.ipynb",
        "payload_name": "payload_option_implied.json",
        "banner_extra": "option-implied Heston family",
        "notebook_title_suffix": " — option-implied Heston family",
        "group_intro": [
            "Companion to the seven-model 10,000-path V4 fixed-calibration study. This report keeps only Heston and Heston–Merton. In V4 their P dynamics still come from returns (Method A), and listed calls identify the Pan (2002) volatility / jump-size premia used for LSM. Rankings are within this pair; they are not compared here with return-based models.",
            "Companies are SPY, AAPL, MSFT, and AMZN. No new calibration or LSM.",
            "Return-based companion: V4_1p5y_fixed_empirical_study_return_based.pdf.",
        ],
    },
    "return_based": {
        "models": ("GBM", "Modified GBM", "GARCH", "Merton", "GARCH–Merton"),
        "pdf_name": "V4_1p5y_fixed_empirical_study_return_based.pdf",
        "nb_name": "V4_1p5y_fixed_empirical_study_return_based.ipynb",
        "payload_name": "payload_return_based.json",
        "banner_extra": "return-based calibration",
        "notebook_title_suffix": " — return-based models",
        "group_intro": [
            "Companion to the seven-model 10,000-path V4 fixed-calibration study. This report keeps GBM, Modified GBM, GARCH, Merton, and GARCH–Merton, whose P dynamics are estimated from lookback returns. That is the same V3 split: return-based vs option-implied Heston, not a split by the P→Q map. GBM still uses μ→r_f; GARCH uses Duan LRNVR; Merton uses Pan jump-size premium μ_J* from listed calls on top of return-based P jumps; GARCH–Merton is unchanged from V3.",
            "Companies are SPY, AAPL, MSFT, and AMZN. No new calibration or LSM.",
            "Option-implied companion: V4_1p5y_fixed_empirical_study_option_implied.pdf.",
        ],
    },
}

ORIGINAL_PDF = "V4_1p5y_fixed_empirical_study.pdf"
ORIGINAL_NB = "V4_1p5y_fixed_empirical_study.ipynb"


def _original_paths() -> tuple[Path, Path]:
    emp.apply_v4_fixed_10k_config()
    return study.SHORT / ORIGINAL_PDF, study.SHORT / ORIGINAL_NB


def load_full_payload() -> dict:
    emp.apply_v4_fixed_10k_config()
    payload = study.load_payload()
    n = len(payload.get("cells", {}))
    n_needed = len(study.TICKERS) * len(study.REGIME_ORDER) * len(study.TABLE_MODELS)
    if n != n_needed:
        raise RuntimeError(
            f"Need the frozen {n_needed}-cell payload "
            f"({len(study.TICKERS)} companies × 4 regimes × {len(study.TABLE_MODELS)} models), "
            f"found {n} cells in {study.PAYLOAD_JSON}"
        )
    return payload


def _slice_group(full: dict, group_id: str) -> dict:
    spec = GROUPS[group_id]
    extra = {
        "grouped_report": group_id,
        "banner_extra": spec["banner_extra"],
        "notebook_title_suffix": spec["notebook_title_suffix"],
        "group_intro": spec["group_intro"],
        "pdf_name": spec["pdf_name"],
        "nb_name": spec["nb_name"],
    }
    return study.slice_payload_models(full, spec["models"], extra_meta=extra)


def write_group(group_id: str, *, full: dict | None = None) -> tuple[Path, Path]:
    emp.apply_v4_fixed_10k_config()
    orig_pdf, orig_nb = _original_paths()
    spec = GROUPS[group_id]
    pdf_path = study.SHORT / spec["pdf_name"]
    nb_path = study.SHORT / spec["nb_name"]
    if pdf_path.resolve() == orig_pdf.resolve() or nb_path.resolve() == orig_nb.resolve():
        raise RuntimeError("Refusing to overwrite the original full-study files")

    full = full if full is not None else load_full_payload()
    payload = _slice_group(full, group_id)
    sliced_json = study.CACHE / spec["payload_name"]
    sliced_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with study.use_models(spec["models"]):
        pdf = study.write_pdf(payload, path=pdf_path)
        nb = study.build_notebook(payload, path=nb_path)
    return nb, pdf


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wanted = [a for a in argv if not a.startswith("--")]
    ids = wanted or list(GROUPS)
    orig_pdf, orig_nb = _original_paths()
    before = {
        orig_pdf: orig_pdf.stat().st_mtime if orig_pdf.exists() else None,
        orig_nb: orig_nb.stat().st_mtime if orig_nb.exists() else None,
    }
    full = load_full_payload()
    for gid in ids:
        nb, pdf = write_group(gid, full=full)
        print(f"{gid}: {pdf.name}  {nb.name}", flush=True)
    after_pdf = orig_pdf.stat().st_mtime if orig_pdf.exists() else None
    after_nb = orig_nb.stat().st_mtime if orig_nb.exists() else None
    if before[orig_pdf] != after_pdf or before[orig_nb] != after_nb:
        raise RuntimeError("Original full-study PDF/notebook timestamps changed")
    print(f"original files unchanged: {orig_pdf.name}, {orig_nb.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
