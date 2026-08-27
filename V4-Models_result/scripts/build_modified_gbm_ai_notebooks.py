#!/usr/bin/env python3
"""Clone Modified GBM yearly notebooks into Modified GBM AI.

Same simulator and LSM hook. estimate_modified_gbm is Adam / PyTorch
simulation matching instead of closed-form counts.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "modified gbm notebook"
DST_DIR = ROOT / "modified gbm ai notebook"

FILES = [
    "2008-2009_modified_gbm.ipynb",
    "2013-2014_modified_gbm.ipynb",
    "2018-2019_modified_gbm.ipynb",
    "2019-2020_modified_gbm.ipynb",
]

ESTIMATE_OLD = "def estimate_modified_gbm(log_rets: pd.Series):"
ESTIMATE_NEW = '''def estimate_modified_gbm(log_rets: pd.Series):
    """Adam / PyTorch fit. Same output keys as closed-form Modified GBM."""
    import sys
    from pathlib import Path as _P
    _scripts = _P.cwd().resolve()
    if _scripts.name.endswith("notebook"):
        _scripts = _scripts.parent / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from modified_gbm_ai import estimate_modified_gbm as _ai_fit
    return _ai_fit(log_rets)


def _estimate_modified_gbm_counts_unused(log_rets: pd.Series):
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n < 1:
        raise SystemExit(f"{label}: expected {old!r}, found {n}")
    return text.replace(old, new, 1) if old == ESTIMATE_OLD else text.replace(old, new)


def convert_notebook(src: Path, dst: Path) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    replaced = False
    for cell in nb["cells"]:
        src_list = cell.get("source", [])
        text = "".join(src_list)
        if "def estimate_modified_gbm" in text and not replaced:
            if ESTIMATE_OLD not in text:
                raise SystemExit(f"no estimate header in {src.name}")
            text = text.replace(ESTIMATE_OLD, ESTIMATE_NEW, 1)
            replaced = True
        text = text.replace("Modified GBM notebooks", "Modified GBM AI notebooks")
        text = text.replace("Rolling Modified GBM parameters", "Rolling Modified GBM AI parameters")
        text = text.replace("risk-neutral Modified GBM paths", "risk-neutral Modified GBM AI paths")
        text = text.replace("Three-stage Modified GBM", "Three-stage Modified GBM AI")
        text = text.replace("# Modified GBM —", "# Modified GBM AI —")
        text = text.replace("Modified GBM (Markov", "Modified GBM AI (Adam + Markov")
        cell["source"] = [line + "\n" for line in text.split("\n")]
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"][-1] = ""
        # keep trailing newline style of original: join later as lines with \n
        if text.endswith("\n"):
            cell["source"] = [ln + "\n" for ln in text.split("\n")[:-1]] + ([text.split("\n")[-1] + "\n"] if text.split("\n")[-1] else [])
        else:
            parts = text.split("\n")
            cell["source"] = [ln + "\n" for ln in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
    if not replaced:
        raise SystemExit(f"did not replace estimate in {src.name}")
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dst.relative_to(ROOT)}", flush=True)


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = SRC_DIR / name
        dst = DST_DIR / name.replace("_modified_gbm.ipynb", "_modified_gbm_ai.ipynb")
        convert_notebook(src, dst)
    spec = DST_DIR / "MODIFIED_GBM_AI.md"
    spec.write_text(
        r"""# Modified GBM AI — model specification

Same price law as Modified GBM: Markov up/down, split |N| sizes, \(S_{t+1}=S_t e^{r_t}\).

**What changes:** parameters are not the lookback counts. Each monthly (or daily) window:

1. Start at those counts (\\(P(U\\mid U), P(D\\mid D), \\mu_U, \\sigma_U, \\mu_D, \\sigma_D\\)).
2. Simulate one-step returns in **PyTorch**.
3. MSE vs the window’s moments (transition rates, up/down sizes, mean/std of \(r\)).
4. **Adam** moves the parameters toward smaller error.
5. `last_up` stays the last observed sign (not learned).

Q-measure paths still use the additive shift so \(E[e^{r}]=e^{r_f\\Delta t}\). LSM is unchanged.

Functions: `estimate_modified_gbm` (Adam), `calibrate_ticker`, `simulate_modified_gbm_rolling`.
""",
        encoding="utf-8",
    )
    print(f"wrote {spec.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
