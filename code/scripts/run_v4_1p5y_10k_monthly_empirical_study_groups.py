#!/usr/bin/env python3
"""Write the return-based 1.5-year monthly 10k LSM report.

Models: MD-GBM, GBM, GARCH, Merton, GARCH–Merton.
No new Monte Carlo; slices stored cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import run_v3_5y_monthly_empirical_study as study  # noqa: E402
import run_v4_1p5y_10k_monthly_empirical_study as emp  # noqa: E402

GROUPS = {
    "return_based": {
        "models": ("MD-GBM", "GBM", "GARCH", "Merton", "GARCH–Merton"),
        "pdf_name": "V4_1p5y_monthly_empirical_study_return_based.pdf",
        "nb_name": "V4_1p5y_monthly_empirical_study_return_based.ipynb",
        "payload_name": "payload_return_based.json",
        "banner_extra": "return-based models",
        "notebook_title_suffix": " — return-based models",
        "group_intro": [
            "1.5-year lookback, monthly rolling, 10,000 LSM paths. Models: MD-GBM, GBM, GARCH, Merton, and GARCH–Merton. P dynamics come from lookback returns. GBM / MD-GBM use an additive Q shift; GARCH uses Duan LRNVR; Merton uses Pan jump-size premium μ_J* from listed calls; GARCH–Merton uses Duan LRNVR plus Pan μ_J*. MD-GBM is Markov Directional Geometric Brownian Motion: 1-lag Modified GBM with folded-normal means matched to sample |R|.",
            "Companies are SPY, AAPL, MSFT, and AMZN.",
        ],
    },
}


def load_full_payload() -> dict:
    emp.apply_v4_10k_config()
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
    emp.apply_v4_10k_config()
    spec = GROUPS[group_id]
    out_dir = study.SHORT / "01_optimal_stopping"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / spec["pdf_name"]
    nb_path = out_dir / spec["nb_name"]

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
    full = load_full_payload()
    for gid in ids:
        nb, pdf = write_group(gid, full=full)
        print(f"{gid}: {pdf.name}  {nb.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
