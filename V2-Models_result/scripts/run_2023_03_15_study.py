#!/usr/bin/env python3
"""V2 1-day 1-minute study for Friday 2022-10-21 (monthly expiry).

Cloned notebooks already live beside the 7d files. Evaluation is one RTH
session on the expiry date. Prices are continuous RTH (overnight/weekend removed).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_7d_1min_study as seven  # noqa: E402
import run_parameter_estimation_study as pest  # noqa: E402

PERIOD_START = pd.Timestamp("2022-10-21 09:30:00")
PERIOD_END = pd.Timestamp("2022-10-21 15:59:00")

seven.WINDOW_LABEL = "1 day"
seven.REGIME = "2022-10-21"
seven.WINDOW_ID = "2022-10-21"
seven.RESULTS = ROOT / "results" / "1d_1min" / "2022-10-21"
seven.SHORT = seven.REPO / "Results_In_Short" / "1 day regimes" / "2022-10-21"
seven.STUDIES = [
    ("GBM", "gbm notebook", "1d_1min_gbm.ipynb", "gbm"),
    ("Merton", "merton notebook", "1d_1min_merton.ipynb", "merton"),
    ("Heston", "heston notebook", "1d_1min_heston.ipynb", "heston"),
    ("Heston–Merton", "heston merton notebook", "1d_1min_heston_merton.ipynb", "heston_merton"),
    ("GARCH", "garch notebook", "1d_1min_garch.ipynb", "garch"),
    ("GARCH–Merton", "garch merton notebook", "1d_1min_garch_merton.ipynb", "garch_merton"),
]

_orig_load = seven._load_ns


def _load_ns_0315(nb_path: Path) -> dict:
    g = _orig_load(nb_path)
    g["PERIOD_START"] = PERIOD_START
    g["PERIOD_END"] = PERIOD_END
    g["period_prices"] = g["prices"].loc[PERIOD_START:PERIOD_END, list(g["TICKERS"])].copy()
    return g


seven._load_ns = _load_ns_0315


def main() -> int:
    pest._install_notebook_stubs()
    seven.RESULTS.mkdir(parents=True, exist_ok=True)
    seven.SHORT.mkdir(parents=True, exist_ok=True)
    print("V2 1-day 1-minute study — notebooks already patched/cloned.", flush=True)
    return seven.main()


if __name__ == "__main__":
    raise SystemExit(main())
