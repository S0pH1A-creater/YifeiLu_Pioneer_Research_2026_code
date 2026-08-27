#!/usr/bin/env python3
"""Switch V3 notebooks/scripts to systematic sampling docs + percentage RMSE labels.

Sampling itself lives in american_lsm.py (Mondays / 15-min / 5-min). This patch
only updates call sites, markdown, and RMSE formulas.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RMSE_OPT = 'rmse = float(np.sqrt(np.mean(df["error"] ** 2)))'
RMSE_OPT_NEW = (
    'rmse = float(100.0 * np.sqrt(np.mean('
    '(df["error"] / np.maximum(np.abs(df["market"]), 1e-8)) ** 2)))'
)
RMSE_PATH = "rmse = float(np.sqrt(np.mean((p50 - _hist) ** 2)))"
RMSE_PATH_NEW = (
    "rmse = float(100.0 * np.sqrt(np.mean("
    "((p50 - _hist) / np.maximum(np.abs(_hist), 1e-8)) ** 2)))"
)
RMSE_PATH2 = "rmse = float(np.sqrt(np.mean((p50 - hist_v) ** 2)))"
RMSE_PATH2_NEW = (
    "rmse = float(100.0 * np.sqrt(np.mean("
    "((p50 - hist_v) / np.maximum(np.abs(hist_v), 1e-8)) ** 2)))"
)

MD_OLD_7D = (
    "Same sampling as the 2-year notebooks: 24 listed American calls, seed 42, "
    "OTM/ATM/ITM × DTE 7–60. **Expiration is the listed expiry** (not the window close). "
    "Quote times are unique RTH 1-minute stamps on those quote days."
)
MD_NEW_7D = (
    "Systematic sample, same rule as every model: **every 15 minutes** in RTH "
    "(~130 observations; 5 sessions × 26). Nearest-ATM listed call, DTE 7–60, "
    "listed expiry. No random dates. Quote times are unique 1-minute stamps. "
    "RMSE is percentage RMSE vs market."
)
MD_OLD_1D = (
    "Same sampling as the 2-year notebooks: 24 listed American calls, seed 42, "
    "OTM/ATM/ITM × DTE 7–60. **Expiration is the listed expiry** (not forced to this session). "
)
MD_NEW_1D = (
    "Systematic sample, same rule as every model: **every 5 minutes** in RTH "
    "(~78 observations; 09:30–15:55). Nearest-ATM listed call, DTE 7–60, listed expiry. "
    "No random dates. Quote times are unique 1-minute stamps. RMSE is percentage RMSE vs market. "
)

REPLACEMENTS = [
    (RMSE_OPT, RMSE_OPT_NEW),
    (RMSE_PATH, RMSE_PATH_NEW),
    (RMSE_PATH2, RMSE_PATH2_NEW),
    ("n_total=24, seed=42", ""),
    ("n_total=24,\n        seed=42,", ""),
    ("n_total=24,\n        seed=42\n", "\n"),
    (MD_OLD_7D, MD_NEW_7D),
    (MD_OLD_1D, MD_NEW_1D),
    ("LSM on 24 listed contracts (seed 42, natural expiry, DTE 7–60)",
     "LSM on the systematic sample (Mondays / 15-min / 5-min; listed expiry, DTE 7–60)"),
    ("24 listed American calls, seed 42, OTM/ATM/ITM × DTE 7–60",
     "systematic ATM calls (Mondays / 15-min / 5-min), DTE 7–60"),
    ("f\"RMSE={rmse:.4f} | MAE={mae:.4f} | \"",
     "f\"RMSE={rmse:.2f}% | MAE={mae:.4f} | \""),
    ("RMSE(p50) = `{rmse:.4f}`", "RMSE%(p50) = `{rmse:.2f}%`"),
    ("f\"RMSE(p50) = `{rmse:.4f}` | seed = `{seed}`\"",
     "f\"RMSE%(p50) = `{rmse:.2f}%` | seed = `{seed}`\""),
    ("RMSE(p50)={rmse:.4f}", "RMSE%(p50)={rmse:.2f}%"),
    ("RMSE = LSM vs market", "percentage RMSE = LSM vs market"),
    ("RMSE = LSM vs market `option_price`", "percentage RMSE = LSM vs market `option_price`"),
]


def _join(src) -> str:
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _split_keep_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] != "" else [])


def patch_text(text: str) -> tuple[str, int]:
    n = 0
    for old, new in REPLACEMENTS:
        if old and old in text:
            text = text.replace(old, new)
            n += 1
    # collapse leftover blank kwarg lists from removing n_total/seed
    text = text.replace("PERIOD_END,\n    )", "PERIOD_END,\n    )")
    text = text.replace("PERIOD_END, )", "PERIOD_END)")
    return text, n


def patch_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb.get("cells", []):
        src = _join(cell.get("source", []))
        new, n = patch_text(src)
        if n:
            hits += n
            cell["source"] = _split_keep_lines(new)
    if hits:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return hits


def main() -> int:
    nbs = list(ROOT.glob("*notebook*/*.ipynb"))
    total = 0
    for p in sorted(nbs):
        if p.name.startswith("."):
            continue
        h = patch_notebook(p)
        if h:
            print(f"  {p.relative_to(ROOT)}  {h} replacements")
            total += h
    print(f"notebooks patched, {total} replacements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
