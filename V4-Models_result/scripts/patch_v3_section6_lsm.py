#!/usr/bin/env python3
"""Patch V3 §6 only: path-wise LSM copy + trading-day discount clock."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_OLD_BLOCK = re.compile(
    r"At each (?:day|step) along each path:\n"
    r"1\. Immediate payoff: \$\\max\(S_t - K, 0\)\$\n"
    r"2\. Continuation value: Longstaff–Schwartz regression on the path cloud \(basis \$1, S, S\^2\$\)\n"
    r"3\. Exercise if payoff \$>\$ continuation\n"
    r"\n"
    r"Paths for pricing are \*\*risk-neutral\*\* \(drift \$\\mu \\rightarrow r\$ from the option panel; "
    r"[^)]+\)\. Not the single expected path — the full Monte Carlo cloud\.",
)

NEW_STEPS_TMPL = """**Do not average simulated stock paths before stopping decisions.** Generate a cloud of individual risk-neutral paths (each keeps its own shocks). Then Longstaff–Schwartz:

1. At each exercise date, compute the immediate payoff $\\max(S_t-K,0)$ **on every path**.
2. Estimate continuation by regression on in-the-money simulated states ($1, S, S^2$, path vol proxy).
3. **Each path** exercises iff payoff $>$ continuation; otherwise it continues.
4. Discount that path's stopping payoff to $t=0$.
5. The American value is the **average of those discounted payoffs** (then $\\max$ with the $t=0$ intrinsic).

Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; {vol_phrase}). §5's expected-vs-history plot is visualization only — it is not the input to LSM."""


def _vol_phrase(nb_path: Path) -> str:
    s = str(nb_path)
    if "heston notebook" in s and "heston merton" not in s:
        return "Heston variance from §4 for that ticker"
    if "garch notebook" in s and "garch merton" not in s:
        return "vol from §4 for that ticker"
    return "vol/jumps from §4 for that ticker"


def _src(cell: dict) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else s


def _set_src(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.split("\n")[:-1]]
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def patch_text(s: str, nb_path: Path) -> str:
    new = NEW_STEPS_TMPL.format(vol_phrase=_vol_phrase(nb_path))
    s = _OLD_BLOCK.sub(lambda _: new, s, count=1)
    if "def _run_optimal_stopping" in s:
        s = s.replace(
            "            dt = 1.0 / N_DAYS\n",
            "            dt = 1.0 / 252.0  # trading-day clock for LSM; never the 1-min N_DAYS\n",
        )
    return s


def main() -> int:
    n = 0
    for nb_path in sorted(ROOT.glob("**/*.ipynb")):
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        raw = nb_path.read_text(encoding="utf-8")
        if "## 6. Optimal stopping" not in raw and "def _run_optimal_stopping" not in raw:
            continue
        nb = json.loads(raw)
        changed = False
        for cell in nb.get("cells", []):
            src = _src(cell)
            if "## 6. Optimal stopping" not in src and "def _run_optimal_stopping" not in src:
                continue
            new = patch_text(src, nb_path)
            if new != src:
                _set_src(cell, new)
                changed = True
        if changed:
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"patched {nb_path.relative_to(ROOT)}")
            n += 1
    print(f"done {n} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
