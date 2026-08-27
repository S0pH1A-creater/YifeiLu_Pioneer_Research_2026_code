#!/usr/bin/env python3
"""Split the 1.5-year 10k LSM study into two calibration-group reports.

No new Monte Carlo. Reads the frozen payload, ranks within each group, and
writes companion PDF/notebooks beside the original full-study files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_1p5y_10k_monthly_empirical_study as emp  # noqa: E402
import run_v3_5y_monthly_empirical_study as study  # noqa: E402

GROUPS = {
    "option_implied": {
        "models": ("Heston", "Heston–Merton"),
        "pdf_name": "V3_1p5y_monthly_empirical_study_option_implied.pdf",
        "nb_name": "V3_1p5y_monthly_empirical_study_option_implied.ipynb",
        "payload_name": "payload_option_implied.json",
        "banner_extra": "option-implied Heston family",
        "notebook_title_suffix": " — option-implied Heston family",
        "group_intro": [
            "Companion to the seven-model 10,000-path study. This report keeps only Heston and Heston–Merton, which are calibrated to listed calls (Fourier NLS). Rankings are within this pair; they are not compared here with return-based models.",
            "Companies are SPY, AAPL, MSFT, and AMZN. No new calibration or LSM. Every number is the stored cell from V3_1p5y_monthly_empirical_study.",
            "Return-based companion: V3_1p5y_monthly_empirical_study_return_based.pdf.",
        ],
    },
    "return_based": {
        "models": ("GBM", "Modified GBM", "GARCH", "Merton", "GARCH–Merton"),
        "pdf_name": "V3_1p5y_monthly_empirical_study_return_based.pdf",
        "nb_name": "V3_1p5y_monthly_empirical_study_return_based.ipynb",
        "payload_name": "payload_return_based.json",
        "banner_extra": "return-based calibration",
        "notebook_title_suffix": " — return-based models",
        "group_intro": [
            "Companion to the seven-model 10,000-path study. This report keeps GBM, Modified GBM, GARCH, Merton, and GARCH–Merton, which are calibrated from lookback returns. Rankings are within this group; they are not compared here with option-implied Heston models.",
            "Companies are SPY, AAPL, MSFT, and AMZN. No new calibration or LSM. Every number is the stored cell from V3_1p5y_monthly_empirical_study.",
            "Option-implied companion: V3_1p5y_monthly_empirical_study_option_implied.pdf.",
        ],
    },
}

ORIGINAL_PDF = "V3_1p5y_monthly_empirical_study.pdf"
ORIGINAL_NB = "V3_1p5y_monthly_empirical_study.ipynb"


def _original_paths() -> tuple[Path, Path]:
    emp.apply_1p5y_10k_config()
    return study.SHORT / ORIGINAL_PDF, study.SHORT / ORIGINAL_NB


def load_full_payload() -> dict:
    emp.apply_1p5y_10k_config()
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


def load_group(group_id: str) -> dict:
    emp.apply_1p5y_10k_config()
    spec = GROUPS[group_id]
    sliced_path = study.CACHE / spec["payload_name"]
    if sliced_path.exists():
        return json.loads(sliced_path.read_text(encoding="utf-8"))
    return _slice_group(load_full_payload(), group_id)


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
    emp.apply_1p5y_10k_config()
    orig_pdf, orig_nb = _original_paths()
    if group_id not in GROUPS:
        raise KeyError(group_id)
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
