#!/usr/bin/env python3
"""Retarget 1d/7d V2 notebooks to monthly expiry Friday 2022-10-21."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPL_7D = [
    ('WINDOW_ID = "2023-03-09_to_2023-03-15"', 'WINDOW_ID = "2022-10-17_to_2022-10-21"'),
    ('PERIOD_START = pd.Timestamp("2023-03-09 09:30:00")', 'PERIOD_START = pd.Timestamp("2022-10-17 09:30:00")'),
    ('PERIOD_END = pd.Timestamp("2023-03-15 15:59:00")', 'PERIOD_END = pd.Timestamp("2022-10-21 15:59:00")'),
    ("2023-03-09 → 2023-03-15", "2022-10-17 → 2022-10-21"),
    ("5 regular sessions: 9, 10, 13, 14, 15 Mar; weekend skipped", "5 regular sessions: 17–21 Oct 2022; weekend skipped"),
    ("(5 RTH sessions: 9, 10, 13, 14, 15 Mar; weekend skipped)", "(5 RTH sessions: 17, 18, 19, 20, 21 Oct)"),
    ("evaluation is **2023-03-09 → 2023-03-15** (5 RTH sessions of 1-minute bars)", "evaluation is **2022-10-17 → 2022-10-21** (5 RTH sessions of 1-minute bars)"),
    ("7-day 1-minute, continuous RTH", "5-weekday 1-minute, continuous RTH (expiry 2022-10-21)"),
    ("Stock price trends — 7-day 1-minute, continuous RTH", "Stock price trends — 2022-10-17 → 2022-10-21, continuous RTH"),
    ("## 1. Stock price trends (7-day 1-minute, continuous RTH)", "## 1. Stock price trends (2022-10-17 → 2022-10-21, continuous RTH)"),
]

REPL_1D = [
    ('PERIOD_START = pd.Timestamp("2023-03-15 09:30:00")', 'PERIOD_START = pd.Timestamp("2022-10-21 09:30:00")'),
    ('PERIOD_END = pd.Timestamp("2023-03-15 15:59:00")', 'PERIOD_END = pd.Timestamp("2022-10-21 15:59:00")'),
    ('WINDOW_ID = "2023-03-09_to_2023-03-15"', 'WINDOW_ID = "2022-10-21"'),
    ("Wednesday 2023-03-15", "Friday 2022-10-21"),
    ("(2023-03-15)", "(2022-10-21)"),
    ("2023-03-15 is an option expiry, so it is a valid 1-day window; a non-expiry day such as 2023-03-13 is not.", "2022-10-21 is the monthly third-Friday expiry, so it is a valid 1-day window."),
    ("Evaluation is **2023-03-15** only.", "Evaluation is **2022-10-21** only."),
    ("Stock price trends — 2023-03-15, continuous RTH", "Stock price trends — 2022-10-21, continuous RTH"),
    ("## 1. Stock price trends (2023-03-15, continuous RTH)", "## 1. Stock price trends (2022-10-21, continuous RTH)"),
    ("## 4. Calibration only — 2023-03-15", "## 4. Calibration only — 2022-10-21"),
    ("## 5. Monte Carlo only — one graph pair per company (2023-03-15)", "## 5. Monte Carlo only — one graph pair per company (2022-10-21)"),
    ("2023-03-15", "2022-10-21"),
]


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [text]


def apply(path: Path, pairs: list[tuple[str, str]]) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb["cells"]:
        raw = src(cell)
        new = raw
        for a, b in pairs:
            if a in new:
                new = new.replace(a, b)
        if new != raw:
            n += 1
            set_src(cell, new)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return n


def main() -> int:
    for p in sorted(ROOT.glob("*/7d_1min*.ipynb")):
        n = apply(p, REPL_7D)
        print(f"7d {p.relative_to(ROOT)}: {n} cells")
    for p in sorted(ROOT.glob("*/1d_1min*.ipynb")):
        n = apply(p, REPL_1D)
        print(f"1d {p.relative_to(ROOT)}: {n} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
