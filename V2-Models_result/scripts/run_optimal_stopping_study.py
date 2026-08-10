#!/usr/bin/env python3
"""Headless optimal-stopping study across 16 regime notebooks (V2).

For each of GBM / Merton / Heston–Merton / GARCH–Merton × four regimes:
  lookback = 6 months
  rolling  = none, monthly, daily
Write comparison markdown + figures under V2-Models_result/results/.
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

ROOT = Path(__file__).resolve().parents[1]  # V2-Models_result/
REPO = ROOT.parent
DATA_ROOT = REPO / "research" / "data"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from american_lsm import (  # noqa: E402
    lsm_american_call,
    load_spy_calls,
    params_asof,
    sample_spy_calls,
)

WINDOW_LABEL = "6 months"
ROLLING_MODES = ("none", "monthly", "daily")
N_PATHS = 2000
SEED = 42
N_CONTRACTS = 24

STUDIES = [
    ("GBM", "gbm notebook", "*_gbm.ipynb", "gbm"),
    ("Merton", "merton notebook", "*_merton.ipynb", "merton"),
    ("Heston–Merton", "heston merton notebook", "*_heston_merton.ipynb", "heston_merton"),
    ("GARCH–Merton", "garch merton notebook", "*_garch_merton.ipynb", "garch_merton"),
]


def _install_scipy_minimize_fallback() -> None:
    """If scipy is missing, provide a tiny optimize.minimize used by GARCH notebooks."""
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
        """Coordinate + random search; enough for GARCH NLL in this study."""
        x = np.asarray(x0, dtype=float).copy()
        best = float(fun(x, *args))
        rng = np.random.default_rng(0)
        # coordinate descent
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
        # random polish
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
    # 2008-2009_gbm.ipynb → 2008-2009  (longest suffixes first)
    stem = path.stem
    for suffix in ("_heston_merton", "_garch_merton", "_merton", "_gbm"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_ns(nb_path: Path) -> dict:
    nb_dir = nb_path.parent
    os.chdir(nb_dir)
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    # Prefer code cells; allow mis-typed markdown that contains stopping code
    code_like = []
    for c in cells:
        src = _cell_source(c)
        if c["cell_type"] == "code" or "def _rn_paths_for_contract" in src or "def calibrate_ticker" in src:
            code_like.append(src)

    # Heuristic: setup has DATA=; cal has calibrate_ticker; sim has simulate_*; stop has _rn_paths
    setup = next(s for s in code_like if "DATA = Path" in s and "PERIOD_START" in s)
    cal = next(s for s in code_like if "def calibrate_ticker" in s)
    sim = next(s for s in code_like if "def simulate_" in s and "def calibrate_ticker" not in s)
    stop = next(s for s in code_like if "def _rn_paths_for_contract" in s)

    g: dict = {"__name__": "__main__", "Path": Path, "np": np, "pd": pd, "plt": plt}
    # Also expose common notebook imports
    exec("from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n", g)
    exec(_extract_defs(setup, keep_assigns=True), g)
    # Fix DATA to absolute research/data regardless of cwd quirks
    g["DATA"] = DATA_ROOT
    # Reload prices with absolute DATA (setup already loaded relative — refresh)
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
    # RN path builder may import from american_lsm; also inject helpers
    g.update(
        {
            "lsm_american_call": lsm_american_call,
            "load_spy_calls": load_spy_calls,
            "params_asof": params_asof,
            "sample_spy_calls": sample_spy_calls,
            "sys": sys,
        }
    )
    exec(_extract_defs(stop, keep_assigns=False), g)
    if "_rn_paths_for_contract" not in g:
        raise RuntimeError(f"Missing _rn_paths_for_contract in {nb_path.name}")
    return g


def _run_mode(g: dict, rolling_mode: str, contracts: pd.DataFrame) -> dict:
    t_cal0 = time.time()
    spy_cal = g["calibrate_ticker"]("SPY", WINDOW_LABEL, rolling_mode)
    g["rolling"] = {"SPY": spy_cal}
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
    rmse = float(np.sqrt(np.mean(df["error"] ** 2)))
    mae = float(np.mean(np.abs(df["error"])))
    bias = float(np.mean(df["error"]))
    early = float(df["early_ex_frac"].mean())
    return {
        "df": df,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "early": early,
        "n_updates": int(len(spy_cal)),
        "t_cal": t_cal,
        "t_lsm": t_lsm,
        "example": example,
    }


def _save_figures(
    model: str,
    regime: str,
    rolling_mode: str,
    out: dict,
    fig_dir: Path,
) -> tuple[str, str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = out["df"]
    color = "#2ca02c"

    # Panel: model vs market + mean bar + exercise timing (notebook §6 style)
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
        f"{model} | {regime} | rolling={rolling_mode} | lookback={WINDOW_LABEL}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    panel_name = f"{rolling_mode}_panel.png"
    fig.savefig(fig_dir / panel_name, dpi=110, bbox_inches="tight")

    # Line graph: one RN path + exercise
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
        f"Example path | trade {pd.Timestamp(row.trading_date).date()} | "
        f"dte={int(row.dte)} | model={res.price:.3f} vs mkt={float(row.option_price):.3f}"
    )
    ax.legend(frameon=False, loc="best")
    fig2.tight_layout()
    line_name = f"{rolling_mode}_path.png"
    fig2.savefig(fig_dir / line_name, dpi=110, bbox_inches="tight")
    return panel_name, line_name


def _write_study_md(
    model: str,
    regime: str,
    stem: str,
    modes: dict[str, dict],
    fig_rel: str,
) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{stem}.md"

    lines = [
        f"# Optimal stopping — {model} — {regime}",
        "",
        f"**Model:** {model}  ",
        f"**Regime / period:** {regime}  ",
        f"**Lookback:** {WINDOW_LABEL} (fixed)  ",
        f"**Rolling modes:** {', '.join(ROLLING_MODES)}  ",
        f"**Pricing:** LSM American calls on SPY | n_paths={N_PATHS} | seed={SEED} | contracts={N_CONTRACTS}",
        "",
        "## Comparison (same contracts, same lookback)",
        "",
        "| Rolling | RMSE | MAE | Bias (model−mkt) | Mean early-ex frac | Cal updates | Cal s | LSM s |",
        "|---------|-----:|----:|-----------------:|-------------------:|------------:|------:|------:|",
    ]
    for mode in ROLLING_MODES:
        m = modes[mode]
        lines.append(
            f"| `{mode}` | {m['rmse']:.4f} | {m['mae']:.4f} | {m['bias']:.4f} | "
            f"{m['early']:.3f} | {m['n_updates']} | {m['t_cal']:.1f} | {m['t_lsm']:.1f} |"
        )

    best = min(ROLLING_MODES, key=lambda k: modes[k]["rmse"])
    lines += [
        "",
        f"**Lowest RMSE (this study):** `{best}` = {modes[best]['rmse']:.4f}",
        "",
    ]

    for mode in ROLLING_MODES:
        m = modes[mode]
        df = m["df"]
        panel, line = m["panel"], m["line"]
        lines += [
            f"## Rolling = `{mode}`",
            "",
            f"- **RMSE:** {m['rmse']:.4f}  ",
            f"- **MAE:** {m['mae']:.4f}  ",
            f"- **Bias:** {m['bias']:.4f}  ",
            f"- **Mean early-exercise fraction:** {m['early']:.3f}  ",
            f"- **Calibration updates (SPY):** {m['n_updates']}",
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


def _write_index(completed: list[tuple[str, str, str, dict]]) -> None:
    """One-page cross-model index for quick comparison."""
    path = RESULTS / "INDEX.md"
    order = {"GBM": 0, "Merton": 1, "Heston–Merton": 2, "GARCH–Merton": 3}
    completed = sorted(completed, key=lambda r: (r[1], order.get(r[0], 9)))
    lines = [
        "# Optimal stopping — index (6-month lookback)",
        "",
        "16 studies × 3 rolling modes (`none` / `monthly` / `daily`). "
        "Same SPY American-call sample (n=24, seed=42), n_paths=2000. Lower RMSE better.",
        "",
        "| Regime | Model | none | monthly | daily | best rolling | file |",
        "|--------|-------|-----:|--------:|------:|--------------|------|",
    ]
    for model, regime, stem, modes in completed:
        best = min(ROLLING_MODES, key=lambda k: modes[k]["rmse"])
        lines.append(
            f"| {regime} | {model} | {modes['none']['rmse']:.4f} | "
            f"{modes['monthly']['rmse']:.4f} | {modes['daily']['rmse']:.4f} | "
            f"`{best}` | [{stem}.md]({stem}.md) |"
        )
    lines += ["", "## Best model by regime (min RMSE across rolling modes)", ""]
    for regime in sorted({r[1] for r in completed}):
        cands = [r for r in completed if r[1] == regime]
        pick = min(cands, key=lambda r: min(r[3][m]["rmse"] for m in ROLLING_MODES))
        br = min(ROLLING_MODES, key=lambda k: pick[3][k]["rmse"])
        lines.append(
            f"- **{regime}:** {pick[0]} (`{br}`, RMSE={pick[3][br]['rmse']:.4f})"
        )
    lines += [
        "",
        "Figures: `figures/<study>/{none,monthly,daily}_{panel,path}.png`.",
        "Generated by `scripts/run_optimal_stopping_study.py`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(nb_path: Path, model: str) -> tuple[str, str, str, dict]:
    regime = _regime_from_name(nb_path)
    stem = nb_path.stem  # e.g. 2008-2009_gbm
    print(f"\n=== {model} | {regime} ({nb_path.name}) ===", flush=True)
    g = _load_ns(nb_path)

    contracts = sample_spy_calls(
        load_spy_calls(g["DATA"]),
        g["PERIOD_START"],
        g["PERIOD_END"],
        n_total=N_CONTRACTS,
        seed=SEED,
    )
    if len(contracts) == 0:
        raise RuntimeError(f"No SPY contracts for {regime}")

    fig_dir = RESULTS / "figures" / stem
    fig_rel = f"figures/{stem}"
    modes: dict[str, dict] = {}
    for mode in ROLLING_MODES:
        print(f"  rolling={mode} …", flush=True)
        out = _run_mode(g, mode, contracts)
        panel, line = _save_figures(model, regime, mode, out, fig_dir)
        out["panel"], out["line"] = panel, line
        # persist csv for the mode
        out["df"].to_csv(fig_dir / f"{mode}_contracts.csv", index=False)
        modes[mode] = out
        print(
            f"    RMSE={out['rmse']:.4f} MAE={out['mae']:.4f} "
            f"updates={out['n_updates']} cal={out['t_cal']:.1f}s lsm={out['t_lsm']:.1f}s",
            flush=True,
        )

    md = _write_study_md(model, regime, stem, modes, fig_rel)
    print(f"  wrote {md.relative_to(ROOT)}", flush=True)
    # slim modes for index (drop heavy frames)
    slim = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("df", "example")}
        for k, v in modes.items()
    }
    # keep rmse etc
    for k in slim:
        slim[k]["rmse"] = modes[k]["rmse"]
    return model, regime, stem, slim


def _match_filters(model: str, key: str, regime: str, stem: str, filters: list[str]) -> bool:
    """AND across tokens: each token must match model key, regime, or stem."""
    if not filters:
        return True
    blob = {key, regime, stem, model.lower(), model.lower().replace("–", "-")}
    for tok in filters:
        t = tok.lower().replace("–", "-")
        if not any(t == b.lower() or t in b.lower() for b in blob):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    _install_notebook_stubs()
    RESULTS.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    for model, folder, glob_pat, key in STUDIES:
        folder_path = ROOT / folder
        for nb in sorted(folder_path.glob(glob_pat)):
            if "advanced" in nb.name:
                continue
            regime = _regime_from_name(nb)
            if not _match_filters(model, key, regime, nb.stem, argv):
                continue
            jobs.append((model, nb))

    if not jobs:
        print("No studies matched.", file=sys.stderr)
        return 1

    print(f"Running {len(jobs)} studies → {RESULTS}", flush=True)
    completed = []
    failures: list[str] = []
    t0 = time.time()
    for model, nb in jobs:
        try:
            completed.append(run_one(nb, model))
        except Exception:
            failures.append(nb.name)
            print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)

    if completed:
        _write_index(completed)
    print(f"\nDone {len(completed)}/{len(jobs)} studies in {(time.time()-t0)/60:.1f} min", flush=True)
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
    print(f"Index: {RESULTS / 'INDEX.md'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
