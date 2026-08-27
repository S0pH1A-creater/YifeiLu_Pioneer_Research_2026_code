#!/usr/bin/env python3
"""V2 RMSE report — no matplotlib (avoids font-cache lock).

Lookback = 1 hour, rolling = minutely.
Windows: 5-weekday 2022-10-17→21 and 1-day 2022-10-21 (monthly expiry).
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

print("boot", flush=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v2_rmse_mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
print("imports ok", flush=True)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA_ROOT = REPO / "research" / "data"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from american_lsm import (  # noqa: E402
    lsm_american_call,
    load_calls,
    params_asof,
    pct_rmse,
    sample_listed_minute_calls,
    session_gap,
    stitch_continuous,
)

WINDOW_LABEL = "1 hour"
ROLLING = "minutely"
SEED = 42
N_PATHS_STOCK = 1000
N_PATHS_STOP = 500
TICKERS = ("SPY", "AAPL", "MSFT")
WINDOW_ID = "2022-10-17_to_2022-10-21"
RATES_PATH = DATA_ROOT / "rates" / "risk_free_dgs3mo_short_interval.csv"
INTRADAY_DIR = DATA_ROOT / "equity" / "short_interval" / "prices_1min_rth"
OUT = ROOT / "results" / "rmse_report_1h_minutely.json"

WINDOWS = [
    {
        "name": "7-day",
        "period_start": pd.Timestamp("2022-10-17 09:30:00"),
        "period_end": pd.Timestamp("2022-10-21 15:59:00"),
        "nbs": [
            ("GBM", "gbm notebook", "7d_1min_gbm.ipynb"),
            ("Modified GBM", "modified gbm notebook", "7d_1min_modified_gbm.ipynb"),
            ("Merton", "merton notebook", "7d_1min_merton.ipynb"),
            ("Heston", "heston notebook", "7d_1min_heston.ipynb"),
            ("Heston–Merton", "heston merton notebook", "7d_1min_heston_merton.ipynb"),
            ("GARCH", "garch notebook", "7d_1min_garch.ipynb"),
            ("GARCH–Merton", "garch merton notebook", "7d_1min_garch_merton.ipynb"),
        ],
    },
    {
        "name": "1-day",
        "period_start": pd.Timestamp("2022-10-21 09:30:00"),
        "period_end": pd.Timestamp("2022-10-21 15:59:00"),
        "nbs": [
            ("GBM", "gbm notebook", "1d_1min_gbm.ipynb"),
            ("Modified GBM", "modified gbm notebook", "1d_1min_modified_gbm.ipynb"),
            ("Merton", "merton notebook", "1d_1min_merton.ipynb"),
            ("Heston", "heston notebook", "1d_1min_heston.ipynb"),
            ("Heston–Merton", "heston merton notebook", "1d_1min_heston_merton.ipynb"),
            ("GARCH", "garch notebook", "1d_1min_garch.ipynb"),
            ("GARCH–Merton", "garch merton notebook", "1d_1min_garch_merton.ipynb"),
        ],
    },
]


def _stub_notebook_modules() -> None:
    class Dummy:
        def __init__(self, *a, **k):
            self.value = k.get("value", a[0] if a else None)

        def on_click(self, *a, **k):
            return None

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, name):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    w = types.ModuleType("ipywidgets")
    for name in (
        "Layout", "Button", "IntSlider", "IntText", "SelectionSlider",
        "Output", "HTML", "HBox", "VBox", "Dropdown", "Label",
    ):
        setattr(w, name, Dummy)
    w.__getattr__ = lambda self, n: Dummy  # type: ignore
    sys.modules["ipywidgets"] = w

    mpl = types.ModuleType("matplotlib")
    mpl.use = lambda *a, **k: None
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.ioff = lambda: None
    pyplot.close = lambda *a, **k: None
    pyplot.subplots = lambda *a, **k: (Dummy(), Dummy())
    pyplot.rcParams = {}
    mpl.pyplot = pyplot
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = pyplot

    idisp = types.ModuleType("IPython.display")
    idisp.display = lambda *a, **k: None
    idisp.Markdown = lambda s: s
    idisp.clear_output = lambda *a, **k: None
    idisp.Image = lambda *a, **k: None
    ipy = types.ModuleType("IPython")
    ipy.display = idisp
    ipy.get_ipython = lambda: None
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = idisp


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _extract_defs(src: str, *, keep_assigns: bool = False) -> str:
    src = src.replace("%matplotlib inline", "\n")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.get_source_segment(src, node) or ""
            if "matplotlib" in text or "ipywidgets" in text or "IPython" in text:
                continue
            keep.append(node)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
            continue
        if keep_assigns and isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            text = ast.get_source_segment(src, node) or ""
            if "widgets." in text or ".children" in text:
                continue
            keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[])) if keep else ""


def _load_rates() -> pd.Series:
    r = pd.read_csv(RATES_PATH, parse_dates=["observation_date"])
    s = r.set_index("observation_date")["DGS3MO"].astype(float) / 100.0
    return s.sort_index()


def _load_g(nb_path: Path, period_start, period_end) -> dict:
    os_chdir = Path.cwd()
    try:
        os_mod = __import__("os")
        os_mod.chdir(nb_path.parent)
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_like = []
        for c in nb["cells"]:
            src = _cell_source(c)
            if c["cell_type"] == "code" or "def calibrate_ticker" in src or "def _rn_paths_for_contract" in src:
                code_like.append(src)
        setup = next(s for s in code_like if "DATA = Path" in s and "PERIOD_START" in s)
        cal = next(s for s in code_like if "def calibrate_ticker" in s)
        sim = next(s for s in code_like if "def simulate_" in s and "def calibrate_ticker" not in s)
        stop = next(s for s in code_like if "def _rn_paths_for_contract" in s)

        g: dict = {"__name__": "__main__", "Path": Path, "np": np, "pd": pd}
        exec("from pathlib import Path\nimport numpy as np\nimport pandas as pd\n", g)
        exec(_extract_defs(setup, keep_assigns=False), g)
        if "_with_option_days" not in g:
            def _with_option_days(fn, *args, **kwargs):
                saved = g["N_DAYS"]
                g["N_DAYS"] = 252
                try:
                    return fn(*args, **kwargs)
                finally:
                    g["N_DAYS"] = saved

            g["_with_option_days"] = _with_option_days
        g.update(
            {
                "DATA": DATA_ROOT,
                "TICKERS": ["AAPL", "MSFT", "SPY"],
                "BARS_PER_DAY": 390,
                "N_DAYS": 252 * 390,
                "N_STEPS": 5000,
                "WINDOW_OPTIONS": {"1 hour": 60, "1 day": 390},
                "ROLLING_OPTIONS": ["minutely", "hourly"],
                "WINDOW_ID": WINDOW_ID,
                "JUMP_THRESH": 3.0,
                "MIN_WINDOW": 60,
                "GAP_MIN": pd.Timedelta(minutes=2),
                "stitch_continuous": stitch_continuous,
                "_session_gap": session_gap,
                "session_gap": session_gap,
                "params_asof": params_asof,
            }
        )
        intra = INTRADAY_DIR
        frames = []
        for t in g["TICKERS"]:
            p = pd.read_csv(intra / f"{t}.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
            frames.append(p["Close"].rename(t))
        prices_raw = pd.concat(frames, axis=1).sort_index()
        stitch = g.get("stitch_continuous", stitch_continuous)
        prices = pd.concat([stitch(prices_raw[t]).rename(t) for t in g["TICKERS"]], axis=1).sort_index()
        g["prices"] = prices
        g["PERIOD_START"] = period_start
        g["PERIOD_END"] = period_end
        g["period_prices"] = prices.loc[period_start:period_end, list(g["TICKERS"])].copy()
        log_returns_all = np.log(prices[list(g["TICKERS"])]).diff()
        gap_fn = g.get("_session_gap", session_gap)
        for t in g["TICKERS"]:
            gg = gap_fn(prices[t].dropna().index)
            log_returns_all.loc[gg.reindex(log_returns_all.index, fill_value=False), t] = np.nan
        g["log_returns_all"] = log_returns_all
        g["rolling"] = {}
        g["cal_meta"] = {}
        exec(_extract_defs(cal, keep_assigns=True), g)
        exec(_extract_defs(sim, keep_assigns=False), g)
        exec(_extract_defs(stop, keep_assigns=False), g)
        if "calibrate_ticker" not in g or "_rn_paths_for_contract" not in g:
            raise RuntimeError(f"missing defs in {nb_path.name}")
        return g
    finally:
        __import__("os").chdir(os_chdir)


def _stock_rmse(g: dict, ticker: str, cal: pd.DataFrame) -> float:
    sched = g["param_schedule_for_steps"](ticker, cal)
    if "simulate_modified_gbm_rolling" in g:
        _dates, steps, S0, hist = sched
        paths = g["simulate_modified_gbm_rolling"](steps, S0, N_PATHS_STOCK, SEED)
    elif "simulate_gbm_rolling" in g:
        _dates, mu, sig, S0, hist = sched
        paths = g["simulate_gbm_rolling"](mu, sig, S0, N_PATHS_STOCK, SEED)
    elif "simulate_garch_merton_rolling" in g:
        _dates, steps, S0, hist = sched
        paths = g["simulate_garch_merton_rolling"](steps, S0, N_PATHS_STOCK, SEED)
    elif "simulate_garch_rolling" in g:
        _dates, steps, S0, hist = sched
        paths = g["simulate_garch_rolling"](steps, S0, N_PATHS_STOCK, SEED)
    elif "simulate_heston_rolling" in g:
        (
            _dates, mu, kappa, theta, xi, rho, v0, S0, hist
        ) = sched
        paths = g["simulate_heston_rolling"](
            mu, kappa, theta, xi, rho, v0, S0, N_PATHS_STOCK, SEED
        )
    elif "simulate_heston_merton_rolling" in g:
        (
            _dates, mu, kappa, theta, xi, rho, v0, lam, muj, sj, kapj, S0, hist
        ) = sched
        paths = g["simulate_heston_merton_rolling"](
            mu, kappa, theta, xi, rho, v0, lam, muj, sj, kapj, S0, N_PATHS_STOCK, SEED
        )
    else:
        _dates, mu, sig, lam, muj, sj, kap, S0, hist = sched
        paths = g["simulate_merton_rolling"](mu, sig, lam, muj, sj, kap, S0, N_PATHS_STOCK, SEED)
    expected = paths.mean(axis=0)
    hist_v = np.asarray(hist.values, dtype=float)
    n = min(len(expected), len(hist_v))
    return float(pct_rmse(expected[:n], hist_v[:n]))


def _stop_metrics(g: dict, ticker: str, cal: pd.DataFrame, contracts: pd.DataFrame) -> dict:
    g["rolling"] = {ticker: cal}
    g["cal_meta"] = {"window_label": WINDOW_LABEL, "rolling_mode": ROLLING}
    dt = 1.0 / 252
    rows = []
    for i, row in enumerate(contracts.itertuples(index=False)):
        paths = g["_rn_paths_for_contract"](row, N_PATHS_STOP, SEED + i)
        res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
        err = res.price - float(row.option_price)
        rows.append({"error": err, "early": res.early_exercise_frac, "model": res.price, "bs": float(row.option_price)})
    df = pd.DataFrame(rows)
    return {
        "rmse_stop": pct_rmse(df["model"], df["bs"]),
        "mae_stop": float(np.mean(np.abs(df["error"]))),
        "bias_stop": float(np.mean(df["error"])),
        "early_ex_frac": float(df["early"].mean()),
        "n_contracts": int(len(df)),
        "mean_model": float(df["model"].mean()),
        "mean_bs": float(df["bs"].mean()),
    }


def _dump(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def main() -> int:
    print("stubbing notebook modules…", flush=True)
    _stub_notebook_modules()
    rows: list[dict] = []
    t_all = time.time()
    print(
        f"V2 RMSE | lookback={WINDOW_LABEL} | rolling={ROLLING} | "
        f"stock_paths={N_PATHS_STOCK} | stop_paths={N_PATHS_STOP} | systematic sample",
        flush=True,
    )
    for win in WINDOWS:
        print(f"\n===== {win['name']} {win['period_start'].date()} → {win['period_end'].date()} =====", flush=True)
        contracts_by = None
        for model, folder, nb_name in win["nbs"]:
            nb = ROOT / folder / nb_name
            print(f"\n--- {model} | {nb_name} ---", flush=True)
            try:
                t0 = time.time()
                g = _load_g(nb, win["period_start"], win["period_end"])
                print(f"  loaded ({time.time()-t0:.1f}s)", flush=True)
                if contracts_by is None:
                    contracts_by = {}
                    for t in TICKERS:
                        df = sample_listed_minute_calls(
                            load_calls(DATA_ROOT, t, panel="short_interval"),
                            g["prices"][t],
                            win["period_start"],
                            win["period_end"],
                        )
                        if df.empty:
                            raise RuntimeError(f"No listed contracts for {t}")
                        if not df["trading_date"].is_unique:
                            raise RuntimeError(f"{t}: duplicate trading minutes")
                        contracts_by[t] = df.reset_index(drop=True)
                        exps = sorted({pd.Timestamp(x).normalize().date() for x in contracts_by[t]["expiration"]})
                        print(
                            f"  {t}: {len(contracts_by[t])} contracts | "
                            f"unique minutes={contracts_by[t]['trading_date'].nunique()} | "
                            f"listed expiries={exps}",
                            flush=True,
                        )
                for ticker in TICKERS:
                    t1 = time.time()
                    print(f"  calibrate {ticker}…", flush=True)
                    cal = g["calibrate_ticker"](ticker, WINDOW_LABEL, ROLLING)
                    print(f"    n_cal={len(cal)} ({time.time()-t1:.1f}s)  stock RMSE…", flush=True)
                    rmse_s = _stock_rmse(g, ticker, cal)
                    print(f"    RMSE(S)={rmse_s:.4f}  stopping…", flush=True)
                    stop = _stop_metrics(g, ticker, cal, contracts_by[ticker])
                    rec = {
                        "window": win["name"],
                        "model": model,
                        "ticker": ticker,
                        "lookback": WINDOW_LABEL,
                        "rolling": ROLLING,
                        "n_cal": int(len(cal)),
                        "rmse_st": rmse_s,
                        **stop,
                        "seconds": round(time.time() - t1, 1),
                    }
                    rows.append(rec)
                    _dump(rows)
                    print(
                        f"  {ticker:4s}  RMSE(S)={rmse_s:8.4f}  RMSE(stop)={stop['rmse_stop']:8.4f}  "
                        f"early={stop['early_ex_frac']:.3f}  {rec['seconds']:.0f}s",
                        flush=True,
                    )
                print(f"  {model} done in {time.time()-t0:.0f}s", flush=True)
            except Exception:
                print(f"FAILED {model}:\n{traceback.format_exc()}", flush=True)
                rows.append({"window": win["name"], "model": model, "ticker": "*", "error": traceback.format_exc()})
                _dump(rows)
    print(f"\nAll done in {(time.time()-t_all)/60:.1f} min → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
