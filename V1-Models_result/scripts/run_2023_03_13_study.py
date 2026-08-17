#!/usr/bin/env python3
"""Build + run V1 single-day 1-minute notebooks for Monday 2023-03-13.

Clones the four 7d_1min notebooks (GBM / Merton / Heston–Merton / GARCH–Merton),
restricts evaluation to one RTH session, then runs headless optimal stopping.

No 2023 listed quotes exist in the project options panel. Stopping RMSE is
LSM American vs a Black–Scholes European benchmark on synthetic 2-day
ATM/OTM/ITM SPY calls (same contracts/seeds across models).
"""
from __future__ import annotations

import ast
import json
import math
import os
import shutil
import sys
import time
import traceback
import types
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

ROOT = Path(__file__).resolve().parents[1]  # V1-Models_result/
REPO = ROOT.parent
DATA_ROOT = REPO / "research" / "data"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from american_lsm import lsm_american_call, params_asof  # noqa: E402
from run_optimal_stopping_study import (  # noqa: E402
    _cell_source,
    _extract_defs,
    _install_notebook_stubs,
)

WINDOW_ID = "2023-03-09_to_2023-03-15"
PERIOD_START = pd.Timestamp("2023-03-13 09:30:00")
PERIOD_END = pd.Timestamp("2023-03-13 15:59:00")
WINDOW_LABEL = "1 day"
ROLLING_MODES = ("hourly", "minutely")
N_PATHS = 2000
SEED = 42
DTE_DAYS = 2
TICKER = "SPY"
RATES_PATH = DATA_ROOT / "equity" / "intraday" / "7d_1min" / WINDOW_ID / "dgs3mo.csv"
RESULTS = ROOT / "results" / "1d_1min" / "2023-03-13"

NOTEBOOKS = [
    ("GBM", "gbm notebook", "7d_1min_gbm.ipynb", "2023-03-13_gbm.ipynb"),
    ("Merton", "merton notebook", "7d_1min_merton.ipynb", "2023-03-13_merton.ipynb"),
    ("Heston", "heston notebook", "7d_1min_heston.ipynb", "2023-03-13_heston.ipynb"),
    ("Heston–Merton", "heston merton notebook", "7d_1min_heston_merton.ipynb", "2023-03-13_heston_merton.ipynb"),
    ("GARCH", "garch notebook", "7d_1min_garch.ipynb", "2023-03-13_garch.ipynb"),
    ("GARCH–Merton", "garch merton notebook", "7d_1min_garch_merton.ipynb", "2023-03-13_garch_merton.ipynb"),
]

REPLACEMENTS = [
    (
        "The evaluation window is calendar week **2023-03-09 → 2023-03-15** "
        "(5 regular sessions: 9, 10, 13, 14, 15 Mar; weekend skipped).",
        "The evaluation window is **Monday 2023-03-13** "
        "(one RTH session; lookback uses prior 1-minute bars on file).",
    ),
    (
        "**Period file:** **2023-03-09 → 2023-03-15 (1-minute bars; 5 RTH sessions)**.",
        "**Period file:** **Monday 2023-03-13 (1-minute bars; 1 RTH session)**.",
    ),
    (
        "PERIOD_START = pd.Timestamp(\"2023-03-09 09:30:00\")",
        "PERIOD_START = pd.Timestamp(\"2023-03-13 09:30:00\")",
    ),
    (
        "PERIOD_END = pd.Timestamp(\"2023-03-15 15:59:00\")",
        "PERIOD_END = pd.Timestamp(\"2023-03-13 15:59:00\")",
    ),
    ("period 7-day 1-minute", "period 1-day 1-minute (2023-03-13)"),
    ("7-day 1-minute window", "1-day 1-minute window (2023-03-13)"),
    ("— 7-day 1-minute", "— 2023-03-13"),
    ("(7-day 1-minute)", "(2023-03-13)"),
    ("1-minute close, 7-day 1-minute", "1-minute close, 2023-03-13"),
    (
        "trading_date` in **2023-03-09 → 2023-03-15 (1-minute bars; 5 RTH sessions)**.",
        "trading_date` in **2023-03-13 (1-minute bars; 1 RTH session)**.",
    ),
    (
        "The daily options panel is 2008–2020, so it will not overlap this 7-day 1-minute window. "
        "§6 is kept for the same workflow; expect an empty sample.",
        "The daily options panel is 2008–2020, so it does not overlap 2023-03-13. "
        "§6 falls back to synthetic 2-day SPY calls; RMSE is LSM vs a Black–Scholes European benchmark.",
    ),
    (
        "(ATM band / DTE 7–60 as in the panel).",
        "(synthetic 2-day ATM/OTM/ITM vs BS European if the listed panel is empty).",
    ),
]


def _rewrite_source(src: str) -> str:
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)
    return src


def _to_source_lines(text: str) -> list[str]:
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
        return [p + "\n" for p in parts]
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts else [])


def build_notebooks() -> list[Path]:
    written = []
    for _model, folder, src_name, dst_name in NOTEBOOKS:
        src = ROOT / folder / src_name
        dst = ROOT / folder / dst_name
        nb = json.loads(src.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            raw = _cell_source(cell)
            if not raw:
                continue
            new = _rewrite_source(raw)
            if new != raw:
                cell["source"] = _to_source_lines(new)
        dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(dst)
        print(f"  wrote {dst.relative_to(REPO)}", flush=True)
    return written


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))


def _load_rates() -> pd.Series:
    r = pd.read_csv(RATES_PATH, parse_dates=["observation_date"])
    s = r.set_index("observation_date")["DGS3MO"].astype(float) / 100.0
    return s.sort_index()


def make_synthetic_spy_calls(g: dict) -> pd.DataFrame:
    px = g["period_prices"][TICKER].dropna()
    rets = g["log_returns_all"][TICKER].dropna()
    rates = _load_rates()
    n_days = float(g["N_DAYS"])
    day = px.loc[px.index.normalize() == PERIOD_START.normalize()]
    targets = []
    for h, m in ((9, 30), (11, 0), (13, 0), (15, 0)):
        ts = PERIOD_START.normalize() + pd.Timedelta(hours=h, minutes=m)
        if ts in day.index:
            targets.append(ts)
        else:
            later = day.index[day.index >= ts]
            if len(later):
                targets.append(later[0])
    rows = []
    for ts in pd.DatetimeIndex(sorted(set(targets))):
        S = float(px.loc[ts])
        if not np.isfinite(S) or S <= 0:
            continue
        win = rets.loc[(rets.index > ts - pd.Timedelta(days=1)) & (rets.index <= ts)]
        if len(win) < 30:
            continue
        sigma = float(win.std(ddof=1) * np.sqrt(n_days))
        r = float(rates.asof(ts.normalize()))
        if not np.isfinite(r):
            r = float(rates.dropna().iloc[-1])
        T = DTE_DAYS / 252.0
        for mny in (0.97, 1.00, 1.03):
            K = round(S * float(mny), 2)
            rows.append(
                {
                    "underlying": TICKER,
                    "trading_date": ts,
                    "S_t": S,
                    "K": K,
                    "expiration": ts + pd.Timedelta(days=DTE_DAYS),
                    "dte": DTE_DAYS,
                    "r": r,
                    "moneyness": S / K,
                    "option_price": bs_call(S, K, T, r, sigma),
                    "sigma_bs": sigma,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No synthetic SPY contracts for 2023-03-13")
    return df.reset_index(drop=True)


def _load_ns(nb_path: Path) -> dict:
    nb_dir = nb_path.parent
    os.chdir(nb_dir)
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

    g: dict = {"__name__": "__main__", "Path": Path, "np": np, "pd": pd, "plt": plt}
    exec(
        "from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n",
        g,
    )
    exec(_extract_defs(setup, keep_assigns=False), g)
    g.update(
        {
            "DATA": DATA_ROOT,
            "TICKERS": ["AAPL", "MSFT", "SPY"],
            "BARS_PER_DAY": 390,
            "N_DAYS": 252 * 390,
            "N_STEPS": 5000,
            "COLORS": {"AAPL": "#1f77b4", "MSFT": "#ff7f0e", "SPY": "#2ca02c"},
            "WINDOW_OPTIONS": {"1 day": pd.Timedelta(days=1)},
            "ROLLING_OPTIONS": ["minutely", "hourly"],
            "WINDOW_ID": WINDOW_ID,
            "JUMP_THRESH": 3.0,
            "MIN_WINDOW": 60,
        }
    )
    intra = DATA_ROOT / "equity" / "intraday" / "7d_1min" / WINDOW_ID
    frames = []
    for t in g["TICKERS"]:
        p = pd.read_csv(intra / f"{t}.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
        frames.append(p["Close"].rename(t))
    prices = pd.concat(frames, axis=1).sort_index()
    g["prices"] = prices
    g["PERIOD_START"] = PERIOD_START
    g["PERIOD_END"] = PERIOD_END
    g["period_prices"] = prices.loc[PERIOD_START:PERIOD_END, list(g["TICKERS"])].copy()
    g["log_returns_all"] = np.log(prices[list(g["TICKERS"])]).diff()
    g["rolling"] = {}
    g["cal_meta"] = {}
    exec(_extract_defs(cal, keep_assigns=True), g)
    exec(_extract_defs(sim, keep_assigns=False), g)
    g.update({"lsm_american_call": lsm_american_call, "params_asof": params_asof, "sys": sys})
    exec(_extract_defs(stop, keep_assigns=False), g)
    if "calibrate_ticker" not in g or "_rn_paths_for_contract" not in g:
        raise RuntimeError(f"Missing calibrate/_rn_paths in {nb_path.name}")
    return g


def run_stopping(model: str, g: dict, contracts: pd.DataFrame) -> dict:
    dt = 1.0 / float(g["N_DAYS"])
    out = {}
    for mode in ROLLING_MODES:
        t0 = time.time()
        cal = g["calibrate_ticker"](TICKER, WINDOW_LABEL, mode)
        if cal is None or len(cal) == 0:
            raise RuntimeError(f"Empty {TICKER} calibration for {model} / {mode}")
        g["rolling"] = {TICKER: cal}
        g["cal_meta"] = {"window_label": WINDOW_LABEL, "rolling_mode": mode}
        rows = []
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
                    "market": float(row.option_price),
                    "model_price": res.price,
                    "error": err,
                    "early_ex_frac": res.early_exercise_frac,
                }
            )
        df = pd.DataFrame(rows)
        packed = {
            "rmse": float(np.sqrt(np.mean(df["error"] ** 2))),
            "mae": float(np.mean(np.abs(df["error"]))),
            "bias": float(np.mean(df["error"])),
            "early": float(df["early_ex_frac"].mean()),
            "n_updates": int(len(cal)),
            "n_contracts": int(len(df)),
        }
        fig_dir = RESULTS / "figures" / model.replace("–", "-").replace(" ", "_")
        fig_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(fig_dir / f"{mode}_contracts.csv", index=False)
        out[mode] = packed
        print(
            f"    {TICKER}/{mode}: RMSE={packed['rmse']:.4f} "
            f"n={packed['n_contracts']} updates={packed['n_updates']} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    return out


def main() -> int:
    _install_notebook_stubs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Building 2023-03-13 notebooks…", flush=True)
    build_notebooks()

    summary = []
    failures = []
    contracts = None
    t0 = time.time()
    for model, folder, _src, dst_name in NOTEBOOKS:
        nb = ROOT / folder / dst_name
        print(f"\n=== {model} | 2023-03-13 ({nb.name}) ===", flush=True)
        try:
            g = _load_ns(nb)
            if contracts is None:
                contracts = make_synthetic_spy_calls(g)
                contracts.to_csv(RESULTS / "synthetic_calls_SPY.csv", index=False)
                print(f"  synthetic SPY: {len(contracts)} contracts", flush=True)
            modes = run_stopping(model, g, contracts)
            summary.append({"model": model, "stem": nb.stem, "modes": modes})
        except Exception:
            failures.append(nb.name)
            print(f"FAILED {nb}:\n{traceback.format_exc()}", flush=True)

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRMSE (SPY LSM vs BS European) — Monday 2023-03-13", flush=True)
    print(f"{'Model':<16} {'hourly':>10} {'minutely':>10}", flush=True)
    for row in summary:
        h = row["modes"]["hourly"]["rmse"]
        u = row["modes"]["minutely"]["rmse"]
        print(f"{row['model']:<16} {h:10.4f} {u:10.4f}", flush=True)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min → {RESULTS}", flush=True)
    if failures:
        print("Failures: " + ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
