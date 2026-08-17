"""Point 7d_1min_*.ipynb at a dated window and fix 1-min DTE → bar steps."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOW_ID = "2023-03-09_to_2023-03-15"
PERIOD_START = "2023-03-09 09:30:00"
PERIOD_END = "2023-03-15 15:59:00"

OLD_LOAD = '''INTRADAY_DIR = DATA / "equity" / "intraday" / "7d_1min"
_frames = []
for _t in TICKERS:
    _p = pd.read_csv(INTRADAY_DIR / f"{_t}.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    _frames.append(_p["Close"].rename(_t))
prices = pd.concat(_frames, axis=1).sort_index()
_days = prices.dropna(how="all").index.normalize().unique().sort_values()
PERIOD_START = prices.loc[prices.index.normalize() >= _days[-7]].index.min()
PERIOD_END = prices.dropna(how="all").index.max()
'''

NEW_LOAD = f'''WINDOW_ID = "{WINDOW_ID}"
INTRADAY_DIR = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID
_frames = []
for _t in TICKERS:
    _p = pd.read_csv(INTRADAY_DIR / f"{{_t}}.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    _frames.append(_p["Close"].rename(_t))
prices = pd.concat(_frames, axis=1).sort_index()
PERIOD_START = pd.Timestamp("{PERIOD_START}")
PERIOD_END = pd.Timestamp("{PERIOD_END}")
'''

OLD_DTE = '''    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(dte, r, dtype=float)
'''

NEW_DTE = '''    dte_days = int(row.dte)
    if dte_days < 2:
        raise ValueError("dte must be >= 2 trading days")
    n_steps = int(dte_days) * int(BARS_PER_DAY)
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(n_steps, r, dtype=float)
'''

MD_REPLACES = [
    ("last 7 regular sessions (1-minute bars)", "2023-03-09 → 2023-03-15 (1-minute bars; 5 RTH sessions)"),
    ("**last 7 regular sessions (1-minute bars)**", "**2023-03-09 → 2023-03-15** (5 RTH sessions, 1-minute bars)"),
    (
        "The evaluation window is **7 regular sessions of 1-minute bars**.",
        "The evaluation window is calendar week **2023-03-09 → 2023-03-15** "
        "(5 regular sessions: 9, 10, 13, 14, 15 Mar; weekend skipped).",
    ),
    (
        "evaluation is **7 days of 1-minute bars**",
        "evaluation is **2023-03-09 → 2023-03-15** (5 RTH sessions of 1-minute bars)",
    ),
]


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [text]


def patch_notebook(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    if WINDOW_ID in raw:
        print(f"skip (already dated) {path.relative_to(ROOT)}", flush=True)
        return
    nb = json.loads(raw)
    changed = False
    for cell in nb["cells"]:
        text = src(cell)
        orig = text
        if cell.get("cell_type") == "markdown":
            for a, b in MD_REPLACES:
                text = text.replace(a, b)
        else:
            if OLD_LOAD in text:
                text = text.replace(OLD_LOAD, NEW_LOAD)
            if OLD_DTE in text:
                text = text.replace(OLD_DTE, NEW_DTE)
                text = text.replace("np.full(dte,", "np.full(n_steps,")
        if text != orig:
            set_src(cell, text)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            changed = True
    if not changed:
        raise RuntimeError(f"no patches applied: {path}")
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}", flush=True)


def main() -> None:
    nbs = sorted({p for p in ROOT.glob("V*-Models_result/**/7d_1min_*.ipynb") if p.is_file()})
    if not nbs:
        raise FileNotFoundError("no 7d_1min notebooks")
    for p in sorted(nbs):
        patch_notebook(p)
    print(f"done ({len(nbs)} notebooks) → {WINDOW_ID}", flush=True)


if __name__ == "__main__":
    main()
