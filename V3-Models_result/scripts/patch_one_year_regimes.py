#!/usr/bin/env python3
"""Shrink V3 2-year regime notebooks to 1-year windows; default lookback 3y / monthly."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# filename stem prefix → new evaluation window
REGIMES = {
    "2008-2009": {
        "start": "2008-08-01",
        "end": "2009-07-31",
        "old_start": "2008-01-01",
        "old_end": "2009-12-31",
        "old_title": "2008–2009",
        "new_title": "August 2008 – July 2009",
        "old_file": "2008-01-01 → 2009-12-31",
        "new_file": "2008-08-01 → 2009-07-31",
    },
    "2013-2014": {
        "start": "2014-01-01",
        "end": "2014-12-31",
        "old_start": "2013-01-01",
        "old_end": "2014-12-31",
        "old_title": "2013–2014",
        "new_title": "2014",
        "old_file": "2013-01-01 → 2014-12-31",
        "new_file": "2014-01-01 → 2014-12-31",
    },
    "2018-2019": {
        "start": "2018-10-01",
        "end": "2019-09-30",
        "old_start": "2018-01-01",
        "old_end": "2019-12-31",
        "old_title": "2018–2019",
        "new_title": "October 2018 – September 2019",
        "old_file": "2018-01-01 → 2019-12-31",
        "new_file": "2018-10-01 → 2019-09-30",
    },
    "2019-2020": {
        "start": "2019-09-01",
        "end": "2020-08-31",
        "old_start": "2019-01-01",
        "old_end": "2020-12-31",
        "old_title": "2019–2020",
        "new_title": "September 2019 – August 2020",
        "old_file": "2019-01-01 → 2020-12-31",
        "new_file": "2019-09-01 → 2020-08-31",
    },
}


def _join(src) -> str:
    return "".join(src) if isinstance(src, list) else str(src)


def _split(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] != "" else [])


def _regime_key(path: Path) -> str | None:
    stem = path.stem.replace("_advanced", "")
    for suffix in ("_heston_merton", "_garch_merton", "_heston", "_garch", "_merton", "_gbm"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def patch_text(text: str, spec: dict) -> str:
    text = text.replace(
        f'PERIOD_START = pd.Timestamp("{spec["old_start"]}")',
        f'PERIOD_START = pd.Timestamp("{spec["start"]}")',
    )
    text = text.replace(
        f'PERIOD_END = pd.Timestamp("{spec["old_end"]}")',
        f'PERIOD_END = pd.Timestamp("{spec["end"]}")',
    )
    text = text.replace(spec["old_file"], spec["new_file"])
    # en-dash display titles (filenames keep ASCII 2008-2009 etc.)
    text = text.replace(spec["old_title"], spec["new_title"])
    if '"3 years"' not in text and '"2 years": pd.DateOffset(years=2)' in text:
        text = re.sub(
            r'("2 years": pd\.DateOffset\(years=2\),)(\s*)("5 years":)',
            r'\1\2"3 years": pd.DateOffset(years=3),\2\3',
            text,
            count=1,
        )
    text = text.replace('value="3 months"', 'value="3 years"')
    text = text.replace('value="6 months"', 'value="3 years"')
    text = text.replace('value="daily"', 'value="monthly"')
    return text


def main() -> int:
    n = 0
    for p in sorted(ROOT.glob("*notebook*/*.ipynb")):
        key = _regime_key(p)
        if key not in REGIMES:
            continue
        spec = REGIMES[key]
        nb = json.loads(p.read_text(encoding="utf-8"))
        changed = False
        for cell in nb.get("cells", []):
            src = _join(cell.get("source", []))
            new = patch_text(src, spec)
            if new != src:
                cell["source"] = _split(new)
                changed = True
        if changed:
            p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  {p.relative_to(ROOT)}")
            n += 1
    print(f"patched {n} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
