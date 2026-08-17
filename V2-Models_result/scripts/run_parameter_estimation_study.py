#!/usr/bin/env python3
"""Headless §4 parameter-estimation export (V2).

For each of GBM / Merton / Heston / Heston–Merton / GARCH / GARCH–Merton × four regimes:
  lookback = 6 months
  rolling  = none, monthly, daily
  underlyings = AAPL, MSFT, SPY (calibrated separately, plotted together)

Writes under V2-Models_result/results/parameters/:
  {rolling}/{stem}/panel.png
  {rolling}/{stem}/params.csv          (all updates, all tickers)
  {rolling}/{stem}/params_short.csv    (table used in Results_In_Short)

Then builds Results_In_Short/parameters_rolling_{none,monthly,daily}.ipynb
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import traceback
import types
import uuid
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # V2-Models_result/
REPO = ROOT.parent
DATA_ROOT = REPO / "research" / "data"
RESULTS = ROOT / "results" / "parameters"
SHORT = REPO / "Results_In_Short"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

WINDOW_LABEL = "6 months"
ROLLING_MODES = ("none", "monthly", "daily")
TICKERS = ("AAPL", "MSFT", "SPY")
COLORS = {"AAPL": "#1f77b4", "MSFT": "#ff7f0e", "SPY": "#2ca02c"}

STUDIES = [
    ("GBM", "gbm notebook", "*_gbm.ipynb", "gbm"),
    ("Merton", "merton notebook", "*_merton.ipynb", "merton"),
    ("Heston", "heston notebook", "20*_heston.ipynb", "heston"),
    ("Heston–Merton", "heston merton notebook", "*_heston_merton.ipynb", "heston_merton"),
    ("GARCH", "garch notebook", "20*_garch.ipynb", "garch"),
    ("GARCH–Merton", "garch merton notebook", "*_garch_merton.ipynb", "garch_merton"),
]

REGIME_ORDER = ["2008-2009", "2013-2014", "2018-2019", "2019-2020"]
MODEL_ORDER = {"GBM": 0, "Merton": 1, "Heston": 2, "Heston–Merton": 3, "GARCH": 4, "GARCH–Merton": 5}

# Plot panels + short-table value columns per model key
MODEL_PANELS = {
    "gbm": [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("sigma", "σ̂ (annual)", "Estimated volatility"),
    ],
    "merton": [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("sigma", "σ̂ (annual)", "Estimated diffusion volatility"),
        ("lam", "λ̂ (jumps/year)", "Estimated jump intensity"),
        ("mu_j", "μ̂_J (log jump)", "Estimated jump size (mean)"),
        ("kappa", "κ̂", "Jump compensation"),
    ],
    "heston": [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("theta", "θ̂ (var)", "Long-run variance"),
        ("kappa", "κ̂", "Mean-reversion speed"),
        ("xi", "ξ̂", "Vol-of-vol"),
        ("rho", "ρ̂", "Price–vol correlation"),
        ("v0", "v̂₀ (var)", "Initial variance"),
    ],
    "heston_merton": [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("theta", "θ̂ (var)", "Long-run variance"),
        ("kappa", "κ̂", "Mean-reversion speed"),
        ("xi", "ξ̂", "Vol-of-vol"),
        ("rho", "ρ̂", "Price–vol correlation"),
        ("lam", "λ̂ (jumps/year)", "Estimated jump intensity"),
    ],
    "garch": [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("omega", "ω̂", "GARCH intercept"),
        ("alpha", "α̂", "ARCH reaction"),
        ("beta", "β̂", "GARCH persistence"),
        ("sigma0", "σ̂₀", "Initial conditional vol"),
    ],
    "garch_merton": [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("omega", "ω̂", "GARCH intercept"),
        ("alpha", "α̂", "ARCH reaction"),
        ("beta", "β̂", "GARCH persistence"),
        ("lam", "λ̂ (jumps/year)", "Estimated jump intensity"),
        ("kappa", "κ̂", "Jump compensation"),
    ],
}

META_COLS = {"date", "window_start", "window_end", "n_days", "ticker"}


def _install_scipy_minimize_fallback() -> None:
    try:
        import scipy.optimize  # noqa: F401

        return
    except ImportError:
        pass

    opt = types.ModuleType("scipy.optimize")
    scipy_mod = types.ModuleType("scipy")

    class _Res:
        def __init__(self, x, success):
            self.x = x
            self.success = success

    def minimize(fun, x0, args=(), method="L-BFGS-B", **kwargs):
        x = np.asarray(x0, dtype=float).copy()
        best = float(fun(x, *args))
        rng = np.random.default_rng(0)
        for _ in range(40):
            improved = False
            for i in range(len(x)):
                for step in (0.2, 0.05, 0.01):
                    for s in (-step, step):
                        trial = x.copy()
                        trial[i] += s
                        val = float(fun(trial, *args))
                        if val < best:
                            best, x = val, trial
                            improved = True
            if not improved:
                break
        for _ in range(80):
            trial = x + rng.normal(0, 0.05, size=x.shape)
            val = float(fun(trial, *args))
            if val < best:
                best, x = val, trial
        return _Res(x, success=np.isfinite(best))

    opt.minimize = minimize
    scipy_mod.optimize = opt
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.optimize"] = opt


def _install_notebook_stubs() -> None:
    _install_scipy_minimize_fallback()

    class Dummy:
        def __init__(self, *a, **k):
            self.value = k.get("value", a[0] if a else None)
            self.layout = types.SimpleNamespace()
            self.style = {}

        def on_click(self, *a, **k):
            return None

        def __call__(self, *a, **k):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Widgets(types.ModuleType):
        def __init__(self):
            super().__init__("ipywidgets")

        def __getattr__(self, name):
            return Dummy

    w = Widgets()
    for name in (
        "Layout", "Button", "IntSlider", "IntText", "SelectionSlider",
        "Output", "HTML", "HBox", "VBox", "Dropdown", "Label",
    ):
        setattr(w, name, Dummy)
    sys.modules["ipywidgets"] = w

    idisp = types.ModuleType("IPython.display")
    idisp.display = lambda *a, **k: None
    idisp.Markdown = lambda s: s
    idisp.clear_output = lambda *a, **k: None
    idisp.Image = lambda *a, **k: None
    ipy = types.ModuleType("IPython")
    ipy.display = idisp
    ipy.get_ipython = lambda: None
    ipy.version_info = (8, 24, 0)
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = idisp


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _extract_defs(src: str, *, keep_assigns: bool = False) -> str:
    src = src.replace("%matplotlib inline", "\n")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
        ):
            keep.append(node)
            continue
        if keep_assigns and isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            text = ast.get_source_segment(src, node) or ""
            if "widgets." in text or ".children" in text:
                continue
            keep.append(node)
            continue
        if isinstance(node, ast.Expr):
            text = ast.get_source_segment(src, node) or ""
            if text.startswith("plt."):
                keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[])) if keep else ""


def _regime_from_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_heston_merton", "_garch_merton", "_heston", "_garch", "_merton", "_gbm"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _model_key_from_stem(stem: str) -> str:
    for key in ("heston_merton", "garch_merton", "heston", "garch", "merton", "gbm"):
        if stem.endswith(key):
            return key
    return "gbm"


def _load_ns(nb_path: Path) -> dict:
    nb_dir = nb_path.parent
    os.chdir(nb_dir)
    nb = json.loads(nb_path.read_text())
    code_like = []
    for c in nb["cells"]:
        src = _cell_source(c)
        if c["cell_type"] == "code" or "def calibrate_ticker" in src:
            code_like.append(src)

    setup = next(s for s in code_like if "DATA = Path" in s and "PERIOD_START" in s)
    cal = next(s for s in code_like if "def calibrate_ticker" in s)

    g: dict = {"__name__": "__main__", "Path": Path, "np": np, "pd": pd, "plt": plt}
    exec(
        "from pathlib import Path\nimport numpy as np\nimport pandas as pd\n"
        "import matplotlib.pyplot as plt\n",
        g,
    )
    exec(_extract_defs(setup, keep_assigns=True), g)
    g["DATA"] = DATA_ROOT
    prices = pd.read_csv(
        g["DATA"] / "equity" / "prices_clean.csv", parse_dates=["Date"]
    ).set_index("Date").sort_index()
    g["prices"] = prices
    g["period_prices"] = prices.loc[g["PERIOD_START"] : g["PERIOD_END"], g["TICKERS"]].copy()
    g["log_returns_all"] = np.log(prices[g["TICKERS"]]).diff()
    g["rolling"] = {}
    g["cal_meta"] = {}
    exec(_extract_defs(cal, keep_assigns=False), g)
    if "calibrate_ticker" not in g:
        raise RuntimeError(f"Missing calibrate_ticker in {nb_path.name}")
    return g


def _param_cols(df: pd.DataFrame, model_key: str) -> list[str]:
    preferred = [c for c, _, _ in MODEL_PANELS[model_key]]
    # keep preferred that exist, then any other numeric param cols
    cols = [c for c in preferred if c in df.columns]
    for c in df.columns:
        if c in META_COLS or c in cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _save_panel(
    rolling: dict[str, pd.DataFrame],
    model_key: str,
    model: str,
    regime: str,
    rolling_mode: str,
    out_path: Path,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    panels = MODEL_PANELS[model_key]
    n = len(panels)
    fig = Figure(figsize=(11, max(3.2, 2.2 * n)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(n, 1, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (col, ylab, title) in zip(axes, panels):
        for t in TICKERS:
            r = rolling[t]
            if col not in r.columns:
                continue
            x = pd.to_datetime(r["date"])
            mark = "o" if len(r) < 40 else None
            ax.plot(x, r[col], lw=1.2, label=t, color=COLORS[t], marker=mark, ms=3)
        if col in {"mu", "mu_j", "kappa", "rho"}:
            ax.axhline(0, color="0.5", lw=0.8)
        ax.set_ylabel(ylab)
        ax.set_title(f"{title} — {rolling_mode}, lookback {WINDOW_LABEL}")
        ax.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Date")
    fig.suptitle(
        f"{model} | {regime} | rolling={rolling_mode} | lookback={WINDOW_LABEL}",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _pick_daily_key_dates(dates: pd.DatetimeIndex, n_keep: int = 10) -> pd.DatetimeIndex:
    """Evenly spaced key updates including first & last."""
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    if len(dates) <= n_keep:
        return dates
    idx = np.unique(np.round(np.linspace(0, len(dates) - 1, n_keep)).astype(int))
    return dates[idx]


def _make_short_table(
    full: pd.DataFrame,
    rolling_mode: str,
    model_key: str,
) -> pd.DataFrame:
    cols = ["date", "ticker"] + _param_cols(full, model_key)
    df = full[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    if rolling_mode == "none":
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if rolling_mode == "monthly":
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)
    # daily: key dates only
    keys = _pick_daily_key_dates(df["date"], n_keep=10)
    out = df[df["date"].isin(keys)].sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def _fmt_table_md(df: pd.DataFrame, model_key: str) -> str:
    show = df.copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    param_cols = _param_cols(show, model_key)
    # round numeric params
    for c in param_cols:
        show[c] = show[c].map(lambda x: f"{float(x):.4g}")
    headers = ["date", "ticker"] + param_cols
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["------" if h in ("date", "ticker") else "------:" for h in headers]) + "|",
    ]
    for r in show.itertuples(index=False):
        row = {h: getattr(r, h) for h in headers}
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_one(nb_path: Path, model: str, model_key: str) -> list[dict]:
    regime = _regime_from_name(nb_path)
    stem = nb_path.stem
    print(f"\n=== {model} | {regime} ({nb_path.name}) ===", flush=True)
    g = _load_ns(nb_path)
    metas = []
    for mode in ROLLING_MODES:
        t0 = time.time()
        rolling = {
            t: g["calibrate_ticker"](t, WINDOW_LABEL, mode) for t in TICKERS
        }
        out_dir = RESULTS / mode / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # full csv
        parts = []
        for t in TICKERS:
            d = rolling[t].copy()
            d.insert(0, "ticker", t)
            parts.append(d)
        full = pd.concat(parts, ignore_index=True)
        full.to_csv(out_dir / "params.csv", index=False)

        short = _make_short_table(full, mode, model_key)
        short.to_csv(out_dir / "params_short.csv", index=False)

        panel = out_dir / "panel.png"
        _save_panel(rolling, model_key, model, regime, mode, panel)

        n_updates = {t: int(len(rolling[t])) for t in TICKERS}
        print(
            f"  {mode}: updates={n_updates} short_rows={len(short)} "
            f"({time.time()-t0:.1f}s) → {out_dir.relative_to(ROOT)}",
            flush=True,
        )
        metas.append(
            {
                "model": model,
                "model_key": model_key,
                "regime": regime,
                "stem": stem,
                "rolling": mode,
                "n_updates": n_updates,
                "short_rows": int(len(short)),
                "panel_rel": f"../V2-Models_result/results/parameters/{mode}/{stem}/panel.png",
                "table_md": _fmt_table_md(short, model_key),
            }
        )
    return metas


def _md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    source = [ln + "\n" for ln in lines[:-1]]
    if lines[-1] != "":
        source.append(lines[-1] + "\n")
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }


def _build_short_notebook(rolling_mode: str, metas: list[dict]) -> Path:
    SHORT.mkdir(parents=True, exist_ok=True)
    items = [m for m in metas if m["rolling"] == rolling_mode]
    items = sorted(
        items,
        key=lambda m: (
            REGIME_ORDER.index(m["regime"]) if m["regime"] in REGIME_ORDER else 99,
            MODEL_ORDER.get(m["model"], 9),
        ),
    )

    if rolling_mode == "none":
        table_note = "one update per ticker (full parameter set)"
    elif rolling_mode == "monthly":
        table_note = "every monthly update × all tickers"
    else:
        table_note = "key dates only (~10 evenly spaced updates, incl. first & last)"

    cells = []
    cells.append(
        _md_cell(
            f"""# V2 Parameter estimation — rolling = `{rolling_mode}`

**Source:** `V2-Models_result/results/parameters/{rolling_mode}/`  
Lookback fixed at **6 months**. Parameters estimated **separately** for AAPL / MSFT / SPY (same graph).

| Fixed | Value |
|-------|-------|
| Lookback | 6 months |
| Rolling | `{rolling_mode}` only |
| Underlyings | AAPL, MSFT, SPY |

## Layout
Regimes 2008-2009 → 2019-2020; within each GBM → Merton → Heston → Heston–Merton → GARCH → GARCH–Merton.  
Each block: parameter graph + short value table ({table_note}).
"""
        )
    )

    n = len(items)
    for i, m in enumerate(items, start=1):
        n_up = m["n_updates"]
        cells.append(
            _md_cell(
                f"""---

## Study {i}/{n} — {m['model']} · {m['regime']}

> **model=`{m['model']}` · regime=`{m['regime']}` · rolling=`{rolling_mode}` · lookback=6 months**

Updates: AAPL={n_up['AAPL']}, MSFT={n_up['MSFT']}, SPY={n_up['SPY']}

### Parameter graph

![params {m['stem']} {rolling_mode}]({m['panel_rel']})

### Parameter values

{m['table_md']}
"""
            )
        )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": cells,
    }
    out = SHORT / f"parameters_rolling_{rolling_mode}.ipynb"
    out.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _install_notebook_stubs()
    RESULTS.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path]] = []
    for model, folder, glob_pat, key in STUDIES:
        for nb in sorted((ROOT / folder).glob(glob_pat)):
            if "advanced" in nb.name:
                continue
            if argv:
                blob = f"{key} {nb.stem} {model}".lower()
                if not all(a.lower() in blob for a in argv):
                    continue
            jobs.append((model, key, nb))

    if not jobs:
        print("No studies matched.", file=sys.stderr)
        return 1

    print(f"Running {len(jobs)} studies × {len(ROLLING_MODES)} rolling → {RESULTS}", flush=True)
    all_metas: list[dict] = []
    failures: list[str] = []
    t0 = time.time()
    for model, key, nb in jobs:
        try:
            all_metas.extend(run_one(nb, model, key))
        except Exception:
            failures.append(nb.name)
            print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)

    # persist summary
    (RESULTS / "summary.json").write_text(
        json.dumps(all_metas, indent=2), encoding="utf-8"
    )

    for mode in ROLLING_MODES:
        out = _build_short_notebook(mode, all_metas)
        print(f"  wrote {out}", flush=True)

    print(
        f"\nDone {len(all_metas)} mode-exports in {(time.time()-t0)/60:.1f} min",
        flush=True,
    )
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
