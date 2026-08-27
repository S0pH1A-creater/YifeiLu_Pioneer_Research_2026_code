#!/usr/bin/env python3
"""Headless optimal-stopping study across 16 regime notebooks (V2).

For each of GBM / Merton / Heston / Heston–Merton / GARCH / GARCH–Merton × four regimes:
  lookback = 3 years
  rolling  = none, monthly, daily
  underlyings = SPY, AAPL, MSFT

Write per-ticker markdown + figures under V4-Models_result/results/{TICKER}/,
then regenerate the three compare_rolling_*.ipynb notebooks (same layout per ticker).
Equity/options data are read from ../research/data (shared with the repo).
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

# Headless matplotlib before other imports that pull pyplot via notebooks
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

ROOT = Path(__file__).resolve().parents[1]  # V4-Models_result/
REPO = ROOT.parent
DATA_ROOT = REPO / "research" / "data"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from american_lsm import (  # noqa: E402
    STOP_TICKERS,
    lsm_american_call,
    load_calls,
    params_asof,
    pct_rmse,
    sample_calls,
)

WINDOW_LABEL = "3 years"
ROLLING_MODES = ("none", "monthly", "daily")
N_PATHS = 2000
SEED = 42
TICKERS = tuple(STOP_TICKERS)  # SPY, AAPL, MSFT

COLORS = {"AAPL": "#1f77b4", "MSFT": "#ff7f0e", "SPY": "#2ca02c"}

STUDIES = [
    ("GBM", "gbm notebook", "*_gbm.ipynb", "gbm"),
    ("Merton", "merton notebook", "*_merton.ipynb", "merton"),
    ("Heston", "heston notebook", "20*_heston.ipynb", "heston"),
    ("Heston–Merton", "heston merton notebook", "*_heston_merton.ipynb", "heston_merton"),
    ("GARCH", "garch notebook", "20*_garch.ipynb", "garch"),
    ("GARCH–Merton", "garch merton notebook", "*_garch_merton.ipynb", "garch_merton"),
]

# Display order in compare notebooks (regime then model)
REGIME_ORDER = ["2008-2009", "2013-2014", "2018-2019", "2019-2020"]
MODEL_ORDER = {"GBM": 0, "Merton": 1, "Heston": 2, "Heston–Merton": 3, "GARCH": 4, "GARCH–Merton": 5}


def _patch_lbfgsb_nelder_mead() -> None:
    """Route GARCH L-BFGS-B through Nelder-Mead.

    SciPy's L-BFGS-B Fortran step can segfault on the GARCH NLL (overflowing
    unconstrained omega / huge finite-difference gradients). Nelder-Mead uses
    the same likelihood and does not need gradients.
    """
    try:
        import scipy.optimize as so
    except ImportError:
        return
    orig = getattr(so, "minimize", None)
    if orig is None or getattr(orig, "_v3_nelder_patch", False):
        return

    def minimize(fun, x0, args=(), method=None, **kwargs):
        if method in (None, "L-BFGS-B", "l-bfgs-b"):
            method = "Nelder-Mead"
            opts = dict(kwargs.pop("options", None) or {})
            opts.setdefault("maxiter", 800)
            kwargs["options"] = opts
            kwargs.pop("jac", None)
            kwargs.pop("bounds", None)
        return orig(fun, x0, args=args, method=method, **kwargs)

    minimize._v3_nelder_patch = True
    so.minimize = minimize


def _install_scipy_minimize_fallback() -> None:
    """If scipy is missing, provide a tiny optimize.minimize used by GARCH notebooks."""
    try:
        import scipy.optimize  # noqa: F401

        _patch_lbfgsb_nelder_mead()
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
        """Coordinate + random search; enough for GARCH NLL in this study."""
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
    """Stub IPython / ipywidgets so notebook cells import cleanly headless."""
    _install_scipy_minimize_fallback()

    class Dummy:
        def __init__(self, *a, **k):
            self.value = k.get("value", a[0] if a else None)
            self.layout = types.SimpleNamespace()
            self.style = {}
            self.outputs = []

        def on_click(self, *a, **k):
            return None

        def __call__(self, *a, **k):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter([])

    class Widgets(types.ModuleType):
        def __init__(self):
            super().__init__("ipywidgets")

        def __getattr__(self, name):
            return Dummy

    w = Widgets()
    for name in (
        "Layout",
        "Button",
        "IntSlider",
        "IntText",
        "SelectionSlider",
        "Output",
        "HTML",
        "HBox",
        "VBox",
        "Dropdown",
        "Label",
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
    """Keep function defs (+ optional setup assigns). Drop widget UI / auto-reestimate."""
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
    for suffix in ("_heston_merton", "_garch_merton", "_modified_gbm_v3", "_modified_gbm_v2", "_modified_gbm_ai", "_modified_gbm_meanfix", "_modified_gbm", "_heston", "_garch", "_merton", "_gbm"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_ns(nb_path: Path) -> dict:
    nb_dir = nb_path.parent
    os.chdir(nb_dir)
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    code_like = []
    for c in cells:
        src = _cell_source(c)
        if c["cell_type"] == "code" or "def _rn_paths_for_contract" in src or "def calibrate_ticker" in src:
            code_like.append(src)

    setup = next(s for s in code_like if "DATA = Path" in s and "PERIOD_START" in s)
    cal = next(s for s in code_like if "def calibrate_ticker" in s)
    sim = next(s for s in code_like if "def simulate_" in s and "def calibrate_ticker" not in s)
    stop_cands = [s for s in code_like if "def _rn_paths_for_contract" in s]
    # Prefer multi-ticker builder (rolling[ticker]); fall back to first match
    stop = next((s for s in stop_cands if "rolling[ticker]" in s), None)
    if stop is None:
        stop = next(iter(stop_cands), None)
    if stop is None:
        raise RuntimeError(f"Missing _rn_paths_for_contract in {nb_path.name}")

    g: dict = {"__name__": "__main__", "Path": Path, "np": np, "pd": pd, "plt": plt}
    exec("from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n", g)
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
    exec(_extract_defs(sim, keep_assigns=False), g)
    g.update(
        {
            "lsm_american_call": lsm_american_call,
            "load_calls": load_calls,
            "load_spy_calls": lambda d: load_calls(d, "SPY"),
            "params_asof": params_asof,
            "sample_calls": sample_calls,
            "sample_spy_calls": sample_calls,
            "STOP_TICKERS": STOP_TICKERS,
            "sys": sys,
        }
    )
    exec(_extract_defs(stop, keep_assigns=False), g)
    if "_rn_paths_for_contract" not in g:
        raise RuntimeError(f"Missing _rn_paths_for_contract in {nb_path.name}")
    return g


def _run_mode(g: dict, rolling_mode: str, contracts: pd.DataFrame, ticker: str) -> dict:
    t_cal0 = time.time()
    cal = g["calibrate_ticker"](ticker, WINDOW_LABEL, rolling_mode)
    g["rolling"] = {ticker: cal}
    g["cal_meta"] = {"window_label": WINDOW_LABEL, "rolling_mode": rolling_mode}
    t_cal = time.time() - t_cal0

    dt = 1.0 / float(g["N_DAYS"])
    rows = []
    example = None
    t_lsm0 = time.time()
    for i, row in enumerate(contracts.itertuples(index=False)):
        paths = g["_rn_paths_for_contract"](row, N_PATHS, SEED + i)
        res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
        err = res.price - float(row.option_price)
        rows.append(
            {
                "ticker": ticker,
                "trading_date": row.trading_date,
                "S_t": float(row.S_t),
                "K": float(row.K),
                "dte": int(row.dte),
                "r": float(row.r),
                "market": float(row.option_price),
                "model_price": res.price,
                "error": err,
                "early_ex_frac": res.early_exercise_frac,
                "mean_ex_day": res.mean_exercise_step,
            }
        )
        if example is None:
            example = (row, paths, res)
    t_lsm = time.time() - t_lsm0

    df = pd.DataFrame(rows)
    rmse = pct_rmse(df["model_price"], df["market"])
    mae = float(np.mean(np.abs(df["error"])))
    bias = float(np.mean(df["error"]))
    early = float(df["early_ex_frac"].mean())
    return {
        "df": df,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "early": early,
        "n_updates": int(len(cal)),
        "t_cal": t_cal,
        "t_lsm": t_lsm,
        "example": example,
    }


def _save_figures(
    model: str,
    regime: str,
    rolling_mode: str,
    ticker: str,
    out: dict,
    fig_dir: Path,
) -> tuple[str, str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = out["df"]
    color = COLORS.get(ticker, "#2ca02c")

    fig = Figure(figsize=(13.5, 4.0))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    ax = axes[0]
    ax.scatter(df["market"], df["model_price"], alpha=0.75, color=color)
    lo = min(df["market"].min(), df["model_price"].min())
    hi = max(df["market"].max(), df["model_price"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("market option_price")
    ax.set_ylabel("model LSM price")
    ax.set_title("Price: model vs market")

    axes[1].bar(
        ["model", "market"],
        [df["model_price"].mean(), df["market"].mean()],
        color=[color, "#7f7f7f"],
    )
    axes[1].set_title("Mean option value")
    axes[1].set_ylabel("price")

    axes[2].hist(df["mean_ex_day"], bins=12, color=color, alpha=0.85, edgecolor="white")
    axes[2].set_xlabel("mean exercise day (by contract)")
    axes[2].set_title("Optimal exercise timing")
    fig.suptitle(
        f"{model} | {ticker} | {regime} | rolling={rolling_mode} | lookback={WINDOW_LABEL}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    panel_name = f"{rolling_mode}_panel.png"
    fig.savefig(fig_dir / panel_name, dpi=110, bbox_inches="tight")
    plt.close(fig)

    row, paths, res = out["example"]
    j = int(np.argmin(np.abs(res.exercise_steps - res.mean_exercise_step)))
    t_ex = int(res.exercise_steps[j])
    fig2 = Figure(figsize=(10, 3.8))
    FigureCanvasAgg(fig2)
    ax = fig2.subplots()
    ax.plot(paths[j], color=color, lw=1.5, label="one RN path")
    ax.axhline(float(row.K), color="gray", ls="--", lw=1, label=f"K={row.K:g}")
    ax.scatter(
        [t_ex],
        [paths[j, t_ex]],
        color="crimson",
        zorder=5,
        s=50,
        label=f"exercise day {t_ex}",
    )
    ax.set_xlabel("day")
    ax.set_ylabel("S")
    ax.set_title(
        f"{ticker} example path | trade {pd.Timestamp(row.trading_date).date()} | "
        f"dte={int(row.dte)} | model={res.price:.3f} vs mkt={float(row.option_price):.3f}"
    )
    ax.legend(frameon=False, loc="best")
    fig2.tight_layout()
    line_name = f"{rolling_mode}_path.png"
    fig2.savefig(fig_dir / line_name, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    return panel_name, line_name


def _write_study_md(
    model: str,
    regime: str,
    stem: str,
    ticker: str,
    modes: dict[str, dict],
    fig_rel: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.md"

    lines = [
        f"# Optimal stopping — {model} — {regime} — {ticker}",
        "",
        f"**Model:** {model}  ",
        f"**Underlying:** {ticker}  ",
        f"**Regime / period:** {regime}  ",
        f"**Lookback:** {WINDOW_LABEL} (fixed)  ",
        f"**Rolling modes:** {', '.join(ROLLING_MODES)}  ",
        f"**Pricing:** LSM American calls on {ticker} | n_paths={N_PATHS} | seed={SEED} | systematic Monday sample",
        "",
        "## Comparison (same contracts, same lookback)",
        "",
        "| Rolling | RMSE | MAE | Bias (model−mkt) | Mean early-ex frac | Cal updates | Cal s | LSM s |",
        "|---------|-----:|----:|-----------------:|-------------------:|------------:|------:|------:|",
    ]
    for mode in ROLLING_MODES:
        m = modes[mode]
        lines.append(
            f"| `{mode}` | {m['rmse']:.2f}% | {m['mae']:.4f} | {m['bias']:.4f} | "
            f"{m['early']:.3f} | {m['n_updates']} | {m['t_cal']:.1f} | {m['t_lsm']:.1f} |"
        )

    best = min(ROLLING_MODES, key=lambda k: modes[k]["rmse"])
    lines += [
        "",
        f"**Lowest RMSE% (this study):** `{best}` = {modes[best]['rmse']:.2f}%",
        "",
    ]

    for mode in ROLLING_MODES:
        m = modes[mode]
        df = m["df"]
        panel, line = m["panel"], m["line"]
        lines += [
            f"## Rolling = `{mode}`",
            "",
            f"- **RMSE%:** {m['rmse']:.2f}%  ",
            f"- **MAE:** {m['mae']:.4f}  ",
            f"- **Bias:** {m['bias']:.4f}  ",
            f"- **Mean early-exercise fraction:** {m['early']:.3f}  ",
            f"- **Calibration updates ({ticker}):** {m['n_updates']}",
            "",
            f"![panel {mode}]({fig_rel}/{panel})",
            "",
            f"![path {mode}]({fig_rel}/{line})",
            "",
            "<details><summary>Contract errors (compact)</summary>",
            "",
            "| date | K | dte | market | model | error | early_ex |",
            "|------|--:|----:|-------:|------:|------:|---------:|",
        ]
        for r in df.itertuples(index=False):
            lines.append(
                f"| {pd.Timestamp(r.trading_date).date()} | {r.K:g} | {r.dte} | "
                f"{r.market:.3f} | {r.model_price:.3f} | {r.error:.3f} | {r.early_ex_frac:.3f} |"
            )
        lines += ["", "</details>", ""]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _slim_modes(modes: dict[str, dict]) -> dict:
    slim = {}
    for k, v in modes.items():
        slim[k] = {
            kk: vv
            for kk, vv in v.items()
            if kk not in ("df", "example")
        }
        slim[k]["rmse"] = v["rmse"]
        slim[k]["mae"] = v["mae"]
        slim[k]["bias"] = v["bias"]
        slim[k]["early"] = v["early"]
    return slim


def _write_ticker_index(ticker: str, completed: list[tuple[str, str, str, dict]]) -> None:
    """Per-ticker index page."""
    out_dir = RESULTS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "INDEX.md"
    completed = sorted(
        completed,
        key=lambda r: (REGIME_ORDER.index(r[1]) if r[1] in REGIME_ORDER else 99, MODEL_ORDER.get(r[0], 9)),
    )
    lines = [
        f"# Optimal stopping — {ticker} — index (3-year lookback)",
        "",
        f"16 studies × 3 rolling modes (`none` / `monthly` / `daily`). "
        f"Same {ticker} systematic Monday sample, n_paths={N_PATHS}. Lower percentage RMSE better.",
        "",
        "| Regime | Model | none | monthly | daily | best rolling | file |",
        "|--------|-------|-----:|--------:|------:|--------------|------|",
    ]
    for model, regime, stem, modes in completed:
        best = min(ROLLING_MODES, key=lambda k: modes[k]["rmse"])
        lines.append(
            f"| {regime} | {model} | {modes['none']['rmse']:.2f}% | "
            f"{modes['monthly']['rmse']:.2f}% | {modes['daily']['rmse']:.2f}% | "
            f"`{best}` | [{stem}.md]({stem}.md) |"
        )
    lines += ["", "## Best model by regime (min RMSE across rolling modes)", ""]
    for regime in REGIME_ORDER:
        cands = [r for r in completed if r[1] == regime]
        if not cands:
            continue
        pick = min(cands, key=lambda r: min(r[3][m]["rmse"] for m in ROLLING_MODES))
        br = min(ROLLING_MODES, key=lambda k: pick[3][k]["rmse"])
        lines.append(
            f"- **{regime}:** {pick[0]} (`{br}`, RMSE%={pick[3][br]['rmse']:.2f}%)"
        )
    lines += [
        "",
        f"Figures: `{ticker}/figures/<study>/{{none,monthly,daily}}_{{panel,path}}.png`.",
        "Generated by `scripts/run_optimal_stopping_study.py`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_index(by_ticker: dict[str, list]) -> None:
    path = RESULTS / "INDEX.md"
    lines = [
        "# V2 Optimal stopping — index (3-year lookback)",
        "",
        "Underlyings: **SPY**, **AAPL**, **MSFT**. Same LSM harness and systematic Monday sample "
        f"per ticker, n_paths={N_PATHS}. Metric is percentage RMSE.",
        "",
        "## Per-ticker indexes",
        "",
    ]
    for t in TICKERS:
        lines.append(f"- [{t}]({t}/INDEX.md)")
    lines += [
        "",
        "## Comparison notebooks",
        "",
        "| Notebook | Rolling |",
        "|----------|---------|",
        "| [compare_rolling_none.ipynb](../compare_rolling_none.ipynb) | none |",
        "| [compare_rolling_monthly.ipynb](../compare_rolling_monthly.ipynb) | monthly |",
        "| [compare_rolling_daily.ipynb](../compare_rolling_daily.ipynb) | daily |",
        "",
        "Each compare notebook has three identical sections (SPY → AAPL → MSFT): RMSE table + 16 study blocks.",
        "",
        "See `../V3_CHANGES.md` and `V1_vs_V2_RMSE.md`.",
        "",
    ]
    # compact SPY summary for continuity
    if "SPY" in by_ticker and by_ticker["SPY"]:
        lines += ["## Quick SPY RMSE% (daily)", "", "| Regime | Model | daily RMSE% |", "|--------|-------|-----------:|"]
        for model, regime, stem, modes in sorted(
            by_ticker["SPY"],
            key=lambda r: (REGIME_ORDER.index(r[1]) if r[1] in REGIME_ORDER else 99, MODEL_ORDER.get(r[0], 9)),
        ):
            lines.append(f"| {regime} | {model} | {modes['daily']['rmse']:.2f}% |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]],
    }


def _build_compare_notebook(
    rolling_mode: str,
    by_ticker: dict[str, list[tuple[str, str, str, dict]]],
) -> Path:
    """Rebuild compare_rolling_{mode}.ipynb with identical SPY / AAPL / MSFT sections."""
    cells = []
    title = f"""# V2 Optimal stopping comparison — rolling = `{rolling_mode}`

**Source:** `V4-Models_result/results/{{TICKER}}/` after methodological fixes (see `V3_CHANGES.md`).  
Same systematic Monday sample / LSM harness for **SPY**, **AAPL**, and **MSFT** (n_paths={N_PATHS}). Metric is percentage RMSE.

| Fixed | Value |
|-------|-------|
| Lookback | 3 years |
| Rolling | `{rolling_mode}` only |
| Underlyings | SPY, AAPL, MSFT |

## Layout
For each ticker: regimes 2008-2009 → 2019-2020; within each GBM → Merton → Heston → Heston–Merton → GARCH → GARCH–Merton.  
Each block: label + RMSE + three-panel graph.
"""
    cells.append(_md_cell(title))

    for ticker in TICKERS:
        completed = by_ticker.get(ticker, [])
        completed = sorted(
            completed,
            key=lambda r: (
                REGIME_ORDER.index(r[1]) if r[1] in REGIME_ORDER else 99,
                MODEL_ORDER.get(r[0], 9),
            ),
        )
        cells.append(_md_cell(f"# Underlying: `{ticker}`\n"))

        # Quick RMSE table
        table_lines = [
            f"## Quick RMSE table — `{ticker}` · `{rolling_mode}`",
            "",
            "| # | Regime | Model | RMSE% | MAE | Bias |",
            "|---|--------|-------|-----:|----:|-----:|",
        ]
        for i, (model, regime, stem, modes) in enumerate(completed, start=1):
            m = modes[rolling_mode]
            table_lines.append(
                f"| {i} | {regime} | {model} | **{m['rmse']:.2f}%** | "
                f"{m['mae']:.4f} | {m['bias']:.4f} |"
            )
        cells.append(_md_cell("\n".join(table_lines)))

        n = len(completed)
        for i, (model, regime, stem, modes) in enumerate(completed, start=1):
            m = modes[rolling_mode]
            fig_rel = f"results/{ticker}/figures/{stem}/{rolling_mode}_panel.png"
            block = f"""---

## Study {i}/{n} — {model} · {regime} · {ticker}

> **underlying=`{ticker}` · model=`{model}` · regime=`{regime}` · rolling=`{rolling_mode}` · lookback=3 years**

| Metric | Value |
|--------|------:|
| **RMSE%** | **{m['rmse']:.2f}%** |
| MAE | {m['mae']:.4f} |
| Bias (model − market) | {m['bias']:.4f} |
| Mean early-exercise frac | {m['early']:.3f} |

### Three-panel graph — {model} · {regime} · {ticker} · `{rolling_mode}`

![Study {i} {ticker}]({fig_rel})
"""
            cells.append(_md_cell(block))

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
    out = ROOT / f"compare_rolling_{rolling_mode}.ipynb"
    out.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run_one_ticker(
    nb_path: Path,
    model: str,
    ticker: str,
    g: dict,
) -> tuple[str, str, str, dict]:
    regime = _regime_from_name(nb_path)
    stem = nb_path.stem
    print(f"  [{ticker}] rolling modes …", flush=True)

    contracts = sample_calls(
        load_calls(g["DATA"], ticker),
        g["PERIOD_START"],
        g["PERIOD_END"],
    )
    if len(contracts) == 0:
        raise RuntimeError(f"No {ticker} contracts for {regime}")

    # Ensure underlying column present for path builder
    if "underlying" not in contracts.columns:
        contracts = contracts.copy()
        contracts["underlying"] = ticker
    else:
        contracts = contracts.copy()
        contracts["underlying"] = ticker

    t_dir = RESULTS / ticker
    fig_dir = t_dir / "figures" / stem
    fig_rel = f"figures/{stem}"
    modes: dict[str, dict] = {}
    for mode in ROLLING_MODES:
        out = _run_mode(g, mode, contracts, ticker)
        panel, line = _save_figures(model, regime, mode, ticker, out, fig_dir)
        out["panel"], out["line"] = panel, line
        out["df"].to_csv(fig_dir / f"{mode}_contracts.csv", index=False)
        modes[mode] = out
        print(
            f"    {ticker}/{mode}: RMSE%={out['rmse']:.2f}% MAE={out['mae']:.4f} "
            f"updates={out['n_updates']} cal={out['t_cal']:.1f}s lsm={out['t_lsm']:.1f}s",
            flush=True,
        )

    md = _write_study_md(model, regime, stem, ticker, modes, fig_rel, t_dir)
    print(f"    wrote {md.relative_to(ROOT)}", flush=True)
    return model, regime, stem, _slim_modes(modes)


def run_one(nb_path: Path, model: str, tickers: tuple[str, ...]) -> dict[str, tuple]:
    regime = _regime_from_name(nb_path)
    print(f"\n=== {model} | {regime} ({nb_path.name}) ===", flush=True)
    g = _load_ns(nb_path)
    out: dict[str, tuple] = {}
    for ticker in tickers:
        out[ticker] = run_one_ticker(nb_path, model, ticker, g)
    return out


def _match_filters(model: str, key: str, regime: str, stem: str, filters: list[str]) -> bool:
    """AND across tokens: each token must match model key, regime, stem, or ticker."""
    if not filters:
        return True
    # ticker filters are handled separately
    non_ticker = [t for t in filters if t.upper() not in TICKERS]
    if not non_ticker:
        return True
    blob = {key, regime, stem, model.lower(), model.lower().replace("–", "-")}
    for tok in non_ticker:
        t = tok.lower().replace("–", "-")
        if not any(t == b.lower() or t in b.lower() for b in blob):
            return False
    return True


def _parse_ticker_filters(argv: list[str]) -> tuple[list[str], tuple[str, ...]]:
    tickers = []
    rest = []
    for a in argv:
        if a.upper() in TICKERS:
            tickers.append(a.upper())
        else:
            rest.append(a)
    return rest, tuple(tickers) if tickers else TICKERS


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    filters, tickers = _parse_ticker_filters(argv)

    _install_notebook_stubs()
    RESULTS.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    for model, folder, glob_pat, key in STUDIES:
        folder_path = ROOT / folder
        for nb in sorted(folder_path.glob(glob_pat)):
            if "advanced" in nb.name:
                continue
            regime = _regime_from_name(nb)
            if not _match_filters(model, key, regime, nb.stem, filters):
                continue
            jobs.append((model, nb))

    if not jobs:
        print("No studies matched.", file=sys.stderr)
        return 1

    print(
        f"Running {len(jobs)} studies × {len(tickers)} tickers → {RESULTS}",
        flush=True,
    )
    by_ticker: dict[str, list] = {t: [] for t in tickers}
    failures: list[str] = []
    t0 = time.time()
    for model, nb in jobs:
        try:
            per = run_one(nb, model, tickers)
            for t, row in per.items():
                by_ticker[t].append(row)
        except Exception:
            failures.append(nb.name)
            print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)

    for t, completed in by_ticker.items():
        if completed:
            _write_ticker_index(t, completed)

    # Persist / merge summary.json per ticker, then rebuild compare notebooks from disk.
    for t in TICKERS:
        t_dir = RESULTS / t
        t_dir.mkdir(parents=True, exist_ok=True)
        summary_path = t_dir / "summary.json"
        existing: dict[str, dict] = {}
        if summary_path.exists():
            try:
                for r in json.loads(summary_path.read_text()):
                    existing[r["stem"]] = r
            except Exception:
                existing = {}
        for model, regime, stem, modes in by_ticker.get(t, []):
            existing[stem] = {
                "model": model,
                "regime": regime,
                "stem": stem,
                "modes": modes,
            }
        summary = sorted(
            existing.values(),
            key=lambda r: (
                REGIME_ORDER.index(r["regime"]) if r["regime"] in REGIME_ORDER else 99,
                MODEL_ORDER.get(r["model"], 9),
            ),
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    full_by_ticker: dict[str, list] = {}
    for t in TICKERS:
        summary_path = RESULTS / t / "summary.json"
        if not summary_path.exists():
            continue
        raw = json.loads(summary_path.read_text())
        full_by_ticker[t] = [
            (r["model"], r["regime"], r["stem"], r["modes"]) for r in raw
        ]
        _write_ticker_index(t, full_by_ticker[t])

    if full_by_ticker:
        for mode in ROLLING_MODES:
            out = _build_compare_notebook(mode, full_by_ticker)
            print(f"  wrote {out.relative_to(ROOT)}", flush=True)

    _write_root_index(full_by_ticker if full_by_ticker else by_ticker)

    n_done = sum(len(v) for v in by_ticker.values())
    print(
        f"\nDone {n_done} ticker-studies in {(time.time()-t0)/60:.1f} min",
        flush=True,
    )
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
    print(f"Index: {RESULTS / 'INDEX.md'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
