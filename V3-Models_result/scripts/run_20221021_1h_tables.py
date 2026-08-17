#!/usr/bin/env python3
"""2022-10-21 study tables: 5 models, 1h lookback, hourly/minutely, 1-day & 5-day.

Stock path: MAE / RMSE of the 50th percentile path, ICP, average 25–75 band width.
Options: LSM American-call RMSE vs market (same listed-minute sample, seed 42).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v2_rmse_mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_v2_rmse_report as rpt  # noqa: E402
from american_lsm import (  # noqa: E402
    lsm_american_call,
    load_calls,
    pct_rmse,
    sample_listed_minute_calls,
)

WINDOW_LABEL = "1 hour"
ROLLING_MODES = ("hourly", "minutely")
SEED = 42
N_PATHS_STOCK = 1000
N_PATHS_STOP = 500
TICKERS = ("SPY", "AAPL", "MSFT")
MODELS = ("GBM", "Merton", "Heston", "Heston–Merton", "GARCH", "GARCH–Merton")
OUT = ROOT / "results" / "rmse_report_20221021_1h.json"


def _simulate_paths(g: dict, ticker: str, cal: pd.DataFrame):
    sched = g["param_schedule_for_steps"](ticker, cal)
    if "simulate_gbm_rolling" in g:
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
    return paths, hist


def _stock_metrics(paths, hist) -> dict:
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    hist_v = np.asarray(hist.values, dtype=float)
    n = min(len(p50), len(hist_v))
    p25, p50, p75, hist_v = p25[:n], p50[:n], p75[:n], hist_v[:n]
    return {
        "rmse_st": pct_rmse(p50, hist_v),
        "mae_st": float(np.mean(np.abs(p50 - hist_v))),
        "icp_st": float(np.mean((hist_v >= p25) & (hist_v <= p75))),
        "abw_st": float(np.mean(p75 - p25)),
    }


def _stop_rmse(g: dict, ticker: str, cal: pd.DataFrame, contracts: pd.DataFrame) -> dict:
    g["rolling"] = {ticker: cal}
    g["cal_meta"] = {"window_label": WINDOW_LABEL, "rolling_mode": g.get("_rolling_mode", "")}
    dt = 1.0 / 252
    err = []
    mkt = []
    for i, row in enumerate(contracts.itertuples(index=False)):
        paths = g["_rn_paths_for_contract"](row, N_PATHS_STOP, SEED + i)
        res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
        err.append(res.price - float(row.option_price))
        mkt.append(float(row.option_price))
    err = np.asarray(err, dtype=float)
    mkt = np.asarray(mkt, dtype=float)
    return {
        "rmse_stop": pct_rmse(mkt + err, mkt),
        "mae_stop": float(np.mean(np.abs(err))),
        "n_contracts": int(len(err)),
    }


def main() -> int:
    print("stubbing notebook modules…", flush=True)
    rpt._stub_notebook_modules()
    rpt.WINDOW_LABEL = WINDOW_LABEL
    rpt.SEED = SEED
    rpt.N_PATHS_STOCK = N_PATHS_STOCK
    rpt.N_PATHS_STOP = N_PATHS_STOP

    rows: list[dict] = []
    t_all = time.time()
    print(
        f"2022-10-21 | lookback={WINDOW_LABEL} | rolling={list(ROLLING_MODES)} | "
        f"stock_paths={N_PATHS_STOCK} | stop_paths={N_PATHS_STOP} | systematic sample",
        flush=True,
    )
    for win in rpt.WINDOWS:
        print(f"\n===== {win['name']} {win['period_start'].date()} → {win['period_end'].date()} =====", flush=True)
        contracts_by = None
        for model, folder, nb_name in win["nbs"]:
            nb = ROOT / folder / nb_name
            print(f"\n--- {model} | {nb_name} ---", flush=True)
            try:
                t0 = time.time()
                g = rpt._load_g(nb, win["period_start"], win["period_end"])
                print(f"  loaded ({time.time()-t0:.1f}s)", flush=True)
                if contracts_by is None:
                    contracts_by = {}
                    for t in TICKERS:
                        df = sample_listed_minute_calls(
                            load_calls(rpt.DATA_ROOT, t, panel="short_interval"),
                            g["prices"][t],
                            win["period_start"],
                            win["period_end"],
                        )
                        if df.empty:
                            raise RuntimeError(f"No listed contracts for {t}")
                        contracts_by[t] = df.reset_index(drop=True)
                        print(f"  {t}: {len(contracts_by[t])} contracts", flush=True)
                for rolling in ROLLING_MODES:
                    g["_rolling_mode"] = rolling
                    for ticker in TICKERS:
                        t1 = time.time()
                        print(f"  {rolling:9s} {ticker} calibrate…", flush=True)
                        cal = g["calibrate_ticker"](ticker, WINDOW_LABEL, rolling)
                        paths, hist = _simulate_paths(g, ticker, cal)
                        st = _stock_metrics(paths, hist)
                        stop = _stop_rmse(g, ticker, cal, contracts_by[ticker])
                        rec = {
                            "window": win["name"],
                            "model": model,
                            "ticker": ticker,
                            "lookback": WINDOW_LABEL,
                            "rolling": rolling,
                            "n_cal": int(len(cal)),
                            **st,
                            **stop,
                            "seconds": round(time.time() - t1, 1),
                        }
                        rows.append(rec)
                        OUT.parent.mkdir(parents=True, exist_ok=True)
                        OUT.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
                        print(
                            f"    RMSE(S)={st['rmse_st']:.4f} MAE={st['mae_st']:.4f} "
                            f"ICP={100*st['icp_st']:.1f}% width={st['abw_st']:.4f} "
                            f"RMSE(opt)={stop['rmse_stop']:.4f}  {rec['seconds']:.0f}s",
                            flush=True,
                        )
                print(f"  {model} done in {time.time()-t0:.0f}s", flush=True)
            except Exception:
                print(f"FAILED {model}:\n{traceback.format_exc()}", flush=True)
                rows.append(
                    {
                        "window": win["name"],
                        "model": model,
                        "ticker": "*",
                        "error": traceback.format_exc(),
                    }
                )
                OUT.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nAll done in {(time.time()-t_all)/60:.1f} min → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
