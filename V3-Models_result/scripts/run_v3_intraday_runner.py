#!/usr/bin/env python3
"""Runner: load approved expiry windows, run intraday study, and write notebook + PDF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONFIG = REPO / "V3-Models_result" / "config" / "expiry_eval_windows.json"

if __name__ == "__main__":
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expiries = [e["expiry"] for e in cfg["expiry_plan"]]
    month_title = {e["expiry"]: Path(e["expiry"]).name for e in cfg["expiry_plan"]}

    # import study module
    sys.path.insert(0, str(HERE))
    import run_v3_intraday_hourly_empirical_study as study  # type: ignore

    # override expiry list and titles
    study.EXPIRY_FRIDAYS = tuple(expiries)
    study.MONTH_TITLE = {e["expiry"]: e["expiry"][:7].replace("-", " ") for e in cfg["expiry_plan"]}

    # set STUDY_KIND default to 7d (keeps existing behavior); callers can modify if needed
    study.configure_study("7d")

    print("Running intraday study (will sample shared contracts, compute filters, and run models)...")
    payload = study.run_or_load(recompute=True)

    print("Writing notebook and PDF...")
    nb = study.build_notebook(payload)
    pdf = study.write_pdf(payload)
    print(f"Notebook written: {nb}")
    print(f"PDF written: {pdf}")
