#!/usr/bin/env python3
"""Rename 2023-03-15_* → 1d_1min_* and point 7d/1d V2 notebooks at short_interval data.

Makes 7-day and 1-day notebooks use the same research workflow as the 2-year
files: listed American calls, LSM vs market option_price, plus §5 RMSE(S_t).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RENAME = [
    ("gbm notebook", "2023-03-15_gbm.ipynb", "1d_1min_gbm.ipynb"),
    ("merton notebook", "2023-03-15_merton.ipynb", "1d_1min_merton.ipynb"),
    ("heston merton notebook", "2023-03-15_heston_merton.ipynb", "1d_1min_heston_merton.ipynb"),
    ("garch merton notebook", "2023-03-15_garch_merton.ipynb", "1d_1min_garch_merton.ipynb"),
    (
        "heston merton advanced notebook",
        "2023-03-15_heston_merton_advanced.ipynb",
        "1d_1min_heston_merton_advanced.ipynb",
    ),
]

NOTEBOOKS = [
    ROOT / "gbm notebook" / "7d_1min_gbm.ipynb",
    ROOT / "gbm notebook" / "1d_1min_gbm.ipynb",
    ROOT / "merton notebook" / "7d_1min_merton.ipynb",
    ROOT / "merton notebook" / "1d_1min_merton.ipynb",
    ROOT / "heston merton notebook" / "7d_1min_heston_merton.ipynb",
    ROOT / "heston merton notebook" / "1d_1min_heston_merton.ipynb",
    ROOT / "garch merton notebook" / "7d_1min_garch_merton.ipynb",
    ROOT / "garch merton notebook" / "1d_1min_garch_merton.ipynb",
    ROOT / "heston merton advanced notebook" / "7d_1min_heston_merton_advanced.ipynb",
    ROOT / "heston merton advanced notebook" / "1d_1min_heston_merton_advanced.ipynb",
]

OLD_INTRADAY = '''WINDOW_ID = "2023-03-09_to_2023-03-15"
INTRADAY_DIR = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID'''

NEW_INTRADAY = '''WINDOW_ID = "2023-03-09_to_2023-03-15"
INTRADAY_DIR = DATA / "equity" / "short_interval" / "prices_1min_rth"
RATES_PATH = DATA / "rates" / "risk_free_dgs3mo_short_interval.csv"
OPT_DIR = DATA / "options" / "processed" / "short_interval"

def _with_option_days(fn, *args, **kwargs):
    """§6 uses a trading-day clock (N=252), same as the 2-year notebooks."""
    global N_DAYS
    saved = N_DAYS
    N_DAYS = 252
    try:
        return fn(*args, **kwargs)
    finally:
        N_DAYS = saved'''

MD2_7D = """## 2. Strike prices in this period

Listed American-option quotes come from `research/data/options/processed/short_interval/`
(2022-09-30 → 2023-09-29; same ATM ±10% / DTE 7–60 filters as the 2-year panels).

Unique listed strikes during **2023-03-09 → 2023-03-15** (SPY primary; AAPL / MSFT secondary).
§6 prices those contracts with LSM vs **market** `option_price`, same workflow as the 2-year files.
"""

MD2_1D = """## 2. Strike prices in this period

Listed American-option quotes come from `research/data/options/processed/short_interval/`
(2022-09-30 → 2023-09-29; same ATM ±10% / DTE 7–60 filters as the 2-year panels).

Unique listed strikes on **2023-03-15** (SPY primary; AAPL / MSFT secondary).
§6 prices those contracts with LSM vs **market** `option_price`, same workflow as the 2-year files.
"""

CODE2 = '''for ticker in TICKERS:
    path = OPT_DIR / f"{ticker}_options_panel.csv"
    opt = pd.read_csv(path, usecols=["trading_date", "K"], parse_dates=["trading_date"])
    m = (opt["trading_date"] >= PERIOD_START.normalize()) & (
        opt["trading_date"] <= PERIOD_END.normalize()
    )
    sub = opt.loc[m]
    uniq = np.sort(sub["K"].dropna().unique())
    dmin, dmax = sub["trading_date"].min(), sub["trading_date"].max()
    display(Markdown(
        f"### {ticker} — {len(uniq)} unique strikes "
        f"(options quotes {dmin.date() if pd.notna(dmin) else 'n/a'} → "
        f"{dmax.date() if pd.notna(dmax) else 'n/a'})"
    ))
    print("Strikes K:", ", ".join(f"{x:g}" for x in uniq[:80]), ("…" if len(uniq) > 80 else ""))
    display(pd.DataFrame({"K": uniq}).T)
'''

OLD_N_STEPS = '''    n_steps = n_steps_to_expiry(
        row.trading_date, getattr(row, "expiration", PERIOD_END)
    )'''

NEW_N_STEPS = '''    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    n_steps = dte'''

OLD_RATES_BLOCK = '''_rates_path = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID / "dgs3mo.csv"
_rates = (
    pd.read_csv(_rates_path, parse_dates=["observation_date"])
    .set_index("observation_date")["DGS3MO"].astype(float) / 100.0
)
_contracts_by_ticker = {}
for _t in _STOP_TICKERS:
    _panel = load_calls(DATA, _t)
    _c = sample_calls(_panel, PERIOD_START, PERIOD_END, n_total=24, seed=42)
    if _c is None or len(_c) == 0:
        _c = make_synthetic_intraday_calls(
            period_prices[_t],
            log_returns_all[_t],
            n_days=float(N_DAYS),
            period_start=PERIOD_START,
            period_end=PERIOD_END,
            rates=_rates,
            ticker=_t,
            expiration=PERIOD_END,
            bars_per_day=int(BARS_PER_DAY),
        )
    _contracts_by_ticker[_t] = _c'''

NEW_RATES_BLOCK = '''_contracts_by_ticker = {}
for _t in _STOP_TICKERS:
    _panel = load_calls(DATA, _t, panel="short_interval")
    _contracts_by_ticker[_t] = sample_calls(
        _panel, PERIOD_START.normalize(), PERIOD_END.normalize(), n_total=24, seed=42
    )'''


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [text]


def rename_notebooks() -> None:
    for folder, old, new in RENAME:
        src_p = ROOT / folder / old
        dst_p = ROOT / folder / new
        if dst_p.exists() and not src_p.exists():
            print(f"  already {dst_p.relative_to(ROOT.parent)}")
            continue
        if not src_p.exists():
            raise FileNotFoundError(src_p)
        src_p.rename(dst_p)
        print(f"  renamed {old} → {new}")


def patch_nb(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    is_1d = "1d_1min" in path.name
    hits: list[str] = []
    for cell in nb["cells"]:
        raw = src(cell)
        new = raw

        if OLD_INTRADAY in new:
            new = new.replace(OLD_INTRADAY, NEW_INTRADAY)
            hits.append("intraday→short_interval")

        if cell.get("cell_type") == "markdown" and "Listed option quotes in this repo end in **2020**" in new:
            new = MD2_1D if is_1d else MD2_7D
            hits.append("§2 markdown")

        if "from american_lsm import make_synthetic_intraday_calls" in new and "for ticker in TICKERS:" in new:
            new = CODE2
            hits.append("§2 listed strikes")

        if "make_synthetic_intraday_calls," in new:
            new = new.replace("    make_synthetic_intraday_calls,\n", "")
            new = new.replace("    n_steps_to_expiry as _n_steps_lib,\n", "")
            hits.append("drop synthetic import")

        if OLD_N_STEPS in new:
            new = new.replace(OLD_N_STEPS, NEW_N_STEPS)
            hits.append("dte steps")

        if "return simulate_gbm_rolling(" in new and "_with_option_days" not in new:
            new = new.replace(
                "return simulate_gbm_rolling(",
                "return _with_option_days(simulate_gbm_rolling, ",
            )
            hits.append("wrap gbm")
        if "return simulate_merton_rolling(" in new and "_with_option_days" not in new:
            new = new.replace(
                "return simulate_merton_rolling(",
                "return _with_option_days(simulate_merton_rolling, ",
            )
            hits.append("wrap merton")
        if "return simulate_heston_merton_rolling(" in new and "_with_option_days" not in new:
            new = new.replace(
                "return simulate_heston_merton_rolling(",
                "return _with_option_days(simulate_heston_merton_rolling, ",
            )
            hits.append("wrap heston")
        if "return simulate_garch_merton_rolling(" in new and "_with_option_days" not in new:
            new = new.replace(
                "return simulate_garch_merton_rolling(",
                "return _with_option_days(simulate_garch_merton_rolling, ",
            )
            hits.append("wrap garch")

        if OLD_RATES_BLOCK in new:
            new = new.replace(OLD_RATES_BLOCK, NEW_RATES_BLOCK)
            hits.append("listed contracts")

        if "(synthetic ATM/OTM/ITM until expiry" in new:
            new = new.replace(
                '        f"(synthetic ATM/OTM/ITM until expiry {PERIOD_END.date()}; RMSE = LSM vs BS European)."',
                '        f"(ATM band / DTE 7–60 as in the short_interval panel; RMSE = LSM vs market)."',
            )
            hits.append("§6 sample caption")

        if "RMSE(stopping vs BS)=" in new:
            new = new.replace("RMSE(stopping vs BS)=", "RMSE=")
            hits.append("RMSE vs market label")

        if 'ax.set_xlabel("BS European benchmark")' in new:
            new = new.replace(
                'ax.set_xlabel("BS European benchmark")',
                'ax.set_xlabel("market option_price")',
            )
            new = new.replace(
                'ax.set_title("Price: model vs BS benchmark")',
                'ax.set_title("Price: model vs market")',
            )
            new = new.replace(
                '["model", "BS bench"]',
                '["model", "market"]',
            )
            hits.append("§6 market axes")

        if "No {ticker} call contracts after synthetic fallback." in new:
            new = new.replace(
                "No {ticker} call contracts after synthetic fallback.",
                "No {ticker} call contracts in this period panel slice.",
            )
            hits.append("empty-panel message")

        if "            dt = 1.0 / N_DAYS\n            stopping_results = {}" in new:
            new = new.replace(
                "            dt = 1.0 / N_DAYS\n            stopping_results = {}",
                "            dt = 1.0 / 252\n            stopping_results = {}",
            )
            hits.append("§6 dt=1/252")

        # intro: listed DTE instead of forced session expiry
        if "Contracts expire **2023-03-15 15:59**." in new:
            new = new.replace(
                "Prices are stitched to **continuous RTH trading time** (overnight/weekend removed). Contracts expire **2023-03-15 15:59**.",
                "Prices are stitched to **continuous RTH trading time** (overnight/weekend removed). §6 uses **listed** American calls (DTE 7–60) from `short_interval`, same filters as the 2-year notebooks.",
            )
            hits.append("intro listed options")
        if "Contracts still **expire at 2023-03-15 15:59**." in new:
            new = new.replace(
                "Contracts still **expire at 2023-03-15 15:59**. Calibration lookback default = **1 hour** (60 trading minutes). Rolling default = **minutely**. Evaluation is **2023-03-15** only. Contracts expire **2023-03-15 15:59** (remaining RTH minutes that session).",
                "Calibration lookback default = **1 hour** (60 trading minutes). Rolling default = **minutely**. Evaluation is **2023-03-15** only. §6 uses **listed** American calls (DTE 7–60) from `short_interval`, same filters as the 2-year notebooks.",
            )
            hits.append("1d intro listed")

        if new != raw:
            set_src(cell, new)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return hits


def main() -> int:
    print("rename 2023-03-15_* → 1d_1min_*")
    rename_notebooks()
    print("patch 7d_1min / 1d_1min notebooks")
    for p in NOTEBOOKS:
        if not p.exists():
            raise FileNotFoundError(p)
        hits = patch_nb(p)
        print(f"  {p.relative_to(ROOT)}: {', '.join(hits) if hits else 'NO HITS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
