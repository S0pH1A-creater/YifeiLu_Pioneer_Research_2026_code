#!/usr/bin/env python3
"""Clone original Modified GBM notebooks; fix folded-normal mean only.

Original sets μ = mean(|R|) then draws |N(μ,σ)|, so E[size] > mean(|R|).
Here μ (and σ if needed) are inverted so E[|N(μ,σ)|] equals that sample mean.
Direction coins, |N| simulator, and additive Q are unchanged.

Does not overwrite modified gbm notebook/ or v2.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "modified gbm notebook"
DST_DIR = ROOT / "modified gbm meanfix notebook"

FILES = [
    "2008-2009_modified_gbm.ipynb",
    "2013-2014_modified_gbm.ipynb",
    "2018-2019_modified_gbm.ipynb",
    "2019-2020_modified_gbm.ipynb",
]

MU_OLD = '''    def _mu_sig(arr):
        if arr.size >= 2:
            mu = float(arr.mean())
            sig = float(arr.std(ddof=1))
        elif arr.size == 1:
            mu = float(arr[0])
            sig = float(np.median(mag)) if mag.size else 1e-6
        else:
            mu = float(np.median(mag)) if mag.size else 1e-6
            sig = mu
        if not np.isfinite(mu) or mu < 0:
            mu = abs(mu) if np.isfinite(mu) else 1e-6
        if not np.isfinite(sig) or sig <= 0:
            sig = 1e-6
        return mu, sig
'''

MU_NEW = '''    def _mu_sig(arr):
        """Folded-normal match: E[|N(μ,σ)|] = sample mean of |R|."""
        import math
        if arr.size >= 2:
            m = float(arr.mean())
            s = float(arr.std(ddof=1))
        elif arr.size == 1:
            m = float(arr[0])
            s = float(np.median(mag)) if mag.size else 1e-6
        else:
            m = float(np.median(mag)) if mag.size else 1e-6
            s = m
        if not np.isfinite(m) or m < 0:
            m = abs(m) if np.isfinite(m) else 1e-6
        if not np.isfinite(s) or s <= 0:
            s = 1e-6

        def _eabs(mu, sig):
            sig = max(float(sig), 1e-12)
            a = float(mu) / sig
            return sig * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * a * a) + float(mu) * math.erf(
                a / math.sqrt(2.0)
            )

        e0 = _eabs(0.0, s)
        if e0 >= m:
            mu = 0.0
            sig = m / max(math.sqrt(2.0 / math.pi), 1e-12)
        else:
            lo, hi = 0.0, max(8.0 * m, 1e-6)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if _eabs(mid, s) < m:
                    lo = mid
                else:
                    hi = mid
            mu, sig = 0.5 * (lo + hi), s
        if not np.isfinite(sig) or sig <= 0:
            sig = 1e-6
        if not np.isfinite(mu) or mu < 0:
            mu = 0.0
        return float(mu), float(sig)
'''


def _rewrite_source(text: str) -> str:
    if MU_OLD not in text:
        return text
    text = text.replace(MU_OLD, MU_NEW, 1)
    text = text.replace("Modified GBM notebooks", "Modified GBM meanfix notebooks")
    text = text.replace("# Modified GBM —", "# Modified GBM meanfix —")
    text = text.replace("Three-stage Modified GBM", "Three-stage Modified GBM meanfix")
    text = text.replace("Rolling Modified GBM parameters", "Rolling Modified GBM meanfix parameters")
    text = text.replace("risk-neutral Modified GBM paths", "risk-neutral Modified GBM meanfix paths")
    return text


def convert(src: Path, dst: Path) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    found = False
    for cell in nb["cells"]:
        text = "".join(cell.get("source", []))
        if MU_OLD in text:
            found = True
        text = _rewrite_source(text)
        if text.endswith("\n"):
            lines = text.split("\n")
            cell["source"] = [ln + "\n" for ln in lines[:-1]]
            if lines[-1]:
                cell["source"].append(lines[-1] + "\n")
        else:
            parts = text.split("\n")
            cell["source"] = [ln + "\n" for ln in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
    if not found:
        raise SystemExit(f"no _mu_sig block in {src.name}")
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dst.relative_to(ROOT)}", flush=True)


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        convert(
            SRC_DIR / name,
            DST_DIR / name.replace("_modified_gbm.ipynb", "_modified_gbm_meanfix.ipynb"),
        )
    (DST_DIR / "MODIFIED_GBM_MEANFIX.md").write_text(
        """# Modified GBM meanfix

Same model as original Modified GBM (Markov sign, `|N(μ,σ)|` sizes, additive Q).

Only change: `μ,σ` are chosen so `E[|N(μ,σ)|]` equals the sample mean of `|R|`
in that up/down bucket. Original used `μ = mean(|R|)`, which overstates simulated sizes.

If the half-normal at the sample SD already has mean ≥ sample mean, `μ=0` and `σ` is
shrunk to match the mean.
""",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
