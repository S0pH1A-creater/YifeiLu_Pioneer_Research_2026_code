#!/usr/bin/env python3
"""Force 1d/7d §6 expiry to the window's last bar; unique 1-minute quote times."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    ROOT / "gbm notebook" / "7d_1min_gbm.ipynb",
    ROOT / "gbm notebook" / "1d_1min_gbm.ipynb",
    ROOT / "merton notebook" / "7d_1min_merton.ipynb",
    ROOT / "merton notebook" / "1d_1min_merton.ipynb",
    ROOT / "heston merton notebook" / "7d_1min_heston_merton.ipynb",
    ROOT / "heston merton notebook" / "1d_1min_heston_merton.ipynb",
    ROOT / "garch merton notebook" / "7d_1min_garch_merton.ipynb",
    ROOT / "garch merton notebook" / "1d_1min_garch_merton.ipynb",
]
ADVANCED = [
    ROOT / "heston merton advanced notebook" / "7d_1min_heston_merton_advanced.ipynb",
    ROOT / "heston merton advanced notebook" / "1d_1min_heston_merton_advanced.ipynb",
]

OLD_STEPS = '''    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    n_steps = dte'''

NEW_STEPS = '''    n_steps = int(getattr(row, "n_steps", 0)) or n_steps_to_expiry(
        prices.index, row.trading_date, PERIOD_END
    )
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")'''

OLD_LOAD = '''from american_lsm import (
    STOP_TICKERS,
    lsm_american_call,
    load_calls,
    params_asof,
    sample_calls,
)'''

NEW_LOAD = '''from american_lsm import (
    STOP_TICKERS,
    lsm_american_call,
    load_calls,
    n_steps_to_expiry,
    params_asof,
    sample_forced_expiry_minute_calls,
)'''

OLD_CONTRACTS = '''_STOP_TICKERS = list(STOP_TICKERS)
_contracts_by_ticker = {}
for _t in _STOP_TICKERS:
    _panel = load_calls(DATA, _t, panel="short_interval")
    _contracts_by_ticker[_t] = sample_calls(
        _panel, PERIOD_START.normalize(), PERIOD_END.normalize(), n_total=24, seed=42
    )'''

NEW_CONTRACTS = '''_STOP_TICKERS = list(STOP_TICKERS)
_rates = (
    pd.read_csv(RATES_PATH, parse_dates=["observation_date"])
    .set_index("observation_date")["DGS3MO"].astype(float) / 100.0
)
_contracts_by_ticker = {}
for _t in _STOP_TICKERS:
    _listed = load_calls(DATA, _t, panel="short_interval")
    _c = sample_forced_expiry_minute_calls(
        prices[_t],
        log_returns_all[_t],
        rates=_rates,
        listed=_listed,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        ticker=_t,
        n_days=float(N_DAYS),
        n_total=24,
        seed=42,
    )
    if not _c["trading_date"].is_unique:
        raise RuntimeError(f"{_t}: duplicate trading minutes in §6 sample")
    _minute = _c["trading_date"].dt.floor("min")
    if not (_minute == _c["trading_date"]).all():
        raise RuntimeError(f"{_t}: §6 quote times are not on the 1-minute grid")
    print(
        f"{_t}: {len(_c)} contracts | unique minutes={_c['trading_date'].nunique()} "
        f"| minute-grid=True | expiry={PERIOD_END}"
    )
    _contracts_by_ticker[_t] = _c'''

MD2_7D = """## 2. Strike prices in this period

Same sampling logic as the 2-year notebooks (24 contracts, seed 42, OTM/ATM/ITM × remaining-life buckets), on the **1-minute** clock.

Every contract **expires at the last bar of this window** (`PERIOD_END`). Quote times are actual RTH 1-minute stamps; no two contracts share a minute.
"""

MD2_1D = """## 2. Strike prices in this period

Same sampling logic as the 2-year notebooks (24 contracts, seed 42, OTM/ATM/ITM × remaining-life buckets), on the **1-minute** clock.

Every contract **expires at the last bar of this session** (`PERIOD_END` = 2023-03-15 15:59). Quote times are actual RTH 1-minute stamps; no two contracts share a minute.
"""

CODE2 = '''import sys
_SCRIPTS = Path("..") / "scripts"
if str(_SCRIPTS.resolve()) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.resolve()))
from american_lsm import load_calls, sample_forced_expiry_minute_calls

_rates = (
    pd.read_csv(RATES_PATH, parse_dates=["observation_date"])
    .set_index("observation_date")["DGS3MO"].astype(float) / 100.0
)
_section2 = {}
for ticker in TICKERS:
    listed = load_calls(DATA, ticker, panel="short_interval")
    df = sample_forced_expiry_minute_calls(
        prices[ticker],
        log_returns_all[ticker],
        rates=_rates,
        listed=listed,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        ticker=ticker,
        n_days=float(N_DAYS),
        n_total=24,
        seed=42,
    )
    _section2[ticker] = df
    uniq_t = df["trading_date"].nunique() if len(df) else 0
    uniq_k = df["K"].nunique() if len(df) else 0
    display(Markdown(
        f"### {ticker} — {len(df)} contracts, {uniq_t} unique 1-minute times, "
        f"{uniq_k} unique strikes (expiry {PERIOD_END})"
    ))
    if len(df):
        print("times unique:", bool(df["trading_date"].is_unique),
              "| minute grid:", bool((df["trading_date"].dt.floor("min") == df["trading_date"]).all()))
        cols = [c for c in ["trading_date", "S_t", "K", "expiration", "dte", "n_steps", "r", "moneyness", "option_price"] if c in df.columns]
        display(df[cols].round(4))
    else:
        print("No contracts.")
'''


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [text]


def patch(path: Path, *, advanced: bool = False) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    is_1d = "1d_1min" in path.name
    hits: list[str] = []
    for cell in nb["cells"]:
        raw = src(cell)
        new = raw

        if cell.get("cell_type") == "markdown" and new.lstrip().startswith("## 2. Strike"):
            new = MD2_1D if is_1d else MD2_7D
            hits.append("§2 md")

        if "for ticker in TICKERS:" in new and "OPT_DIR" in new and "unique strikes" in new:
            new = CODE2
            hits.append("§2 sample")

        if OLD_LOAD in new:
            new = new.replace(OLD_LOAD, NEW_LOAD)
            hits.append("imports")
        if OLD_STEPS in new:
            new = new.replace(OLD_STEPS, NEW_STEPS)
            hits.append("n_steps remaining minutes")
        if "return _with_option_days(simulate_gbm_rolling, " in new:
            new = new.replace("return _with_option_days(simulate_gbm_rolling, ", "return simulate_gbm_rolling(")
            hits.append("unwrap gbm")
        if "return _with_option_days(simulate_merton_rolling, " in new:
            new = new.replace("return _with_option_days(simulate_merton_rolling, ", "return simulate_merton_rolling(")
            hits.append("unwrap merton")
        if "return _with_option_days(simulate_heston_merton_rolling, " in new:
            new = new.replace(
                "return _with_option_days(simulate_heston_merton_rolling, ",
                "return simulate_heston_merton_rolling(",
            )
            hits.append("unwrap heston")
        if "return _with_option_days(simulate_garch_merton_rolling, " in new:
            new = new.replace(
                "return _with_option_days(simulate_garch_merton_rolling, ",
                "return simulate_garch_merton_rolling(",
            )
            hits.append("unwrap garch")
        if OLD_CONTRACTS in new:
            new = new.replace(OLD_CONTRACTS, NEW_CONTRACTS)
            hits.append("forced-expiry sample")
        if "(ATM band / DTE 7–60 as in the short_interval panel; RMSE = LSM vs market)." in new:
            new = new.replace(
                '        f"(ATM band / DTE 7–60 as in the short_interval panel; RMSE = LSM vs market)."',
                '        f"(forced expiry {PERIOD_END}; unique 1-minute times; RMSE = LSM vs contract price)."',
            )
            hits.append("caption")
        if "            dt = 1.0 / 252\n            stopping_results = {}" in new:
            new = new.replace(
                "            dt = 1.0 / 252\n            stopping_results = {}",
                "            dt = 1.0 / N_DAYS\n            stopping_results = {}",
            )
            hits.append("dt 1-min")

        drop_helper = (
            "def _with_option_days(fn, *args, **kwargs):\n"
            '    """§6 uses a trading-day clock (N=252), same as the 2-year notebooks."""\n'
            "    global N_DAYS\n"
            "    saved = N_DAYS\n"
            "    N_DAYS = 252\n"
            "    try:\n"
            "        return fn(*args, **kwargs)\n"
            "    finally:\n"
            "        N_DAYS = saved\n"
        )
        if drop_helper in new:
            new = new.replace(drop_helper, "")
            hits.append("drop N=252 wrap")

        new = new.replace(
            "§6 uses **listed** American calls (DTE 7–60) from `short_interval`, same filters as the 2-year notebooks.",
            "§6 uses the **same sampling/LSM/RMSE logic as the 2-year files**, with expiry forced to this window's last 1-minute bar and unique minute-accurate quote times.",
        )
        new = new.replace(
            "LSM exercise decision on SPY American calls (risk-neutral paths from the same simulator); results in `stopping_results`.",
            "LSM on 24 contracts (seed 42, same stratification as 2-year), expiry forced to this window's last 1-minute bar, unique minute-accurate quote times; results in `stopping_results`.",
        )
        new = new.replace("At each day along each path:", "At each step along each path:")
        if "forced to this window's last 1-minute bar" in new and "intro" not in hits:
            hits.append("intro")

        if new != raw:
            set_src(cell, new)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return hits


def main() -> int:
    for p in NOTEBOOKS:
        hits = patch(p)
        print(f"{p.relative_to(ROOT)}: {', '.join(hits) if hits else 'NO HITS'}")
    for p in ADVANCED:
        hits = patch(p, advanced=True)
        print(f"{p.relative_to(ROOT)}: {', '.join(hits) if hits else 'NO HITS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
