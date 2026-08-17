#!/usr/bin/env python3
"""Patch V1 7d_1min notebooks: trading-time prices, synthetic strikes, 1h/minutely."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "gbm notebook" / "7d_1min_gbm.ipynb",
    ROOT / "merton notebook" / "7d_1min_merton.ipynb",
    ROOT / "heston merton notebook" / "7d_1min_heston_merton.ipynb",
    ROOT / "garch merton notebook" / "7d_1min_garch_merton.ipynb",
]

STITCH = '''
GAP_MIN = pd.Timedelta(minutes=2)

def _session_gap(idx) -> pd.Series:
    return pd.Series(idx, index=idx).diff() > GAP_MIN

def stitch_continuous(close: pd.Series) -> pd.Series:
    """Rebuild a gapless trading-time price: overnight/weekend returns are not applied."""
    s = close.dropna()
    r = np.log(s).diff()
    r = r.mask(_session_gap(s.index), 0.0)
    r.iloc[0] = 0.0
    return pd.Series(float(s.iloc[0]) * np.exp(r.cumsum()), index=s.index, name=s.name)

def trading_x(idx) -> np.ndarray:
    return np.arange(len(idx))

def session_starts(idx) -> np.ndarray:
    g = _session_gap(idx).fillna(False).to_numpy()
    return np.flatnonzero(g)
'''.lstrip("\n")

REPLACEMENTS = [
    (
        '''WINDOW_OPTIONS = {
    "1 day": pd.Timedelta(days=1),
}''',
        '''WINDOW_OPTIONS = {
    "1 hour": 60,   # trading minutes (not calendar hours)
    "1 day": 390,
}''',
    ),
    (
        '''prices = pd.concat(_frames, axis=1).sort_index()
PERIOD_START = pd.Timestamp("2023-03-09 09:30:00")
PERIOD_END = pd.Timestamp("2023-03-15 15:59:00")
period_prices = prices.loc[PERIOD_START:PERIOD_END, TICKERS].copy()
log_returns_all = np.log(prices[TICKERS]).diff()''',
        STITCH
        + '''
prices_raw = pd.concat(_frames, axis=1).sort_index()
PERIOD_START = pd.Timestamp("2023-03-09 09:30:00")
PERIOD_END = pd.Timestamp("2023-03-15 15:59:00")
prices = pd.concat(
    [stitch_continuous(prices_raw[t]).rename(t) for t in TICKERS],
    axis=1,
).sort_index()
period_prices = prices.loc[PERIOD_START:PERIOD_END, TICKERS].copy()
log_returns_all = np.log(prices[TICKERS]).diff()
# Do not treat the stitched 0-return at session joins as a 1-minute observation
for _t in TICKERS:
    _g = _session_gap(prices[_t].dropna().index)
    log_returns_all.loc[_g.reindex(log_returns_all.index, fill_value=False), _t] = np.nan
'''.lstrip("\n"),
    ),
    (
        '''fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, ticker in zip(axes, TICKERS):
    s = period_prices[ticker].dropna()
    ax.plot(s.index, s.values, color=COLORS[ticker], lw=1.4)
    role = "primary" if ticker == "SPY" else "secondary"
    ax.set_ylabel("Close")
    ax.set_title(f"{ticker} ({role}) — 1-minute close, 7-day 1-minute")
axes[-1].set_xlabel("Datetime")
fig.suptitle("Stock price trends — period 7-day 1-minute", fontsize=13, y=1.01)''',
        '''fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, ticker in zip(axes, TICKERS):
    s = period_prices[ticker].dropna()
    x = trading_x(s.index)
    ax.plot(x, s.values, color=COLORS[ticker], lw=1.4)
    for br in session_starts(s.index):
        ax.axvline(br, color="0.75", lw=0.8, ls=":")
    role = "primary" if ticker == "SPY" else "secondary"
    ax.set_ylabel("Close")
    ax.set_title(f"{ticker} ({role}) — continuous trading-time close (RTH only)")
axes[-1].set_xlabel("trading minute (overnight/weekend removed)")
fig.suptitle("Stock price trends — 7-day 1-minute, continuous RTH", fontsize=13, y=1.01)''',
    ),
    (
        '''    r = log_returns_all[ticker].loc[PERIOD_START:PERIOD_END].dropna()
    rv = r.rolling(30, min_periods=10).std() * np.sqrt(N_DAYS)
    ax.plot(rv.index, rv.values, color=COLORS[ticker], lw=1.0)''',
        '''    r = log_returns_all[ticker].loc[PERIOD_START:PERIOD_END].dropna()
    rv = r.rolling(30, min_periods=10).std() * np.sqrt(N_DAYS)
    ax.plot(trading_x(rv.index), rv.values, color=COLORS[ticker], lw=1.0)''',
    ),
    (
        'axes[-1].set_xlabel("Datetime")\nfig.suptitle("Minute-level realized volatility — 7-day 1-minute window"',
        'axes[-1].set_xlabel("trading minute")\nfig.suptitle("Minute-level realized volatility — continuous RTH"',
    ),
    (
        '''def _slice_window(rets: pd.Series, end: pd.Timestamp, offset: pd.DateOffset) -> pd.Series:
    start = end - offset
    return rets.loc[(rets.index > start) & (rets.index <= end)]''',
        '''def _slice_window(rets: pd.Series, end: pd.Timestamp, n_bars) -> pd.Series:
    """Last n trading-time bars ending at `end` (calendar gaps already removed)."""
    sub = rets.loc[rets.index <= pd.Timestamp(end)]
    n = int(n_bars)
    return sub.iloc[-n:] if len(sub) else sub''',
    ),
    (
        '            x = pd.to_datetime(r["date"])\n',
        '            x = np.array([\n'
        '                period_prices.index.get_indexer([pd.Timestamp(d)], method="pad")[0]\n'
        '                for d in r["date"]\n'
        '            ])\n',
    ),
    (
        '        axes[1].set_xlabel("Date")\n',
        '        axes[1].set_xlabel("trading minute")\n',
    ),
    ('value="1 day"', 'value="1 hour"'),
    ('value="hourly"', 'value="minutely"'),
    (
        '"<b>§4 Calibration (graphs only)</b> — 1-day lookback; hourly or minutely rolling, then <b>Reestimate</b>. "',
        '"<b>§4 Calibration (graphs only)</b> — default lookback <b>1 hour</b> (60 trading minutes); default rolling <b>minutely</b>, then <b>Reestimate</b>. "',
    ),
    (
        '''            axes[0].plot(dates_now, paths.T, color=COLORS[ticker], alpha=0.12, lw=0.7)
            axes[0].plot(dates_now, expected, color="black", lw=2.2, label="expected path (MC mean)")''',
        '''            tt = trading_x(dates_now)
            axes[0].plot(tt, paths.T, color=COLORS[ticker], alpha=0.12, lw=0.7)
            axes[0].plot(tt, expected, color="black", lw=2.2, label="expected path (MC mean)")''',
    ),
    (
        '''            axes[1].plot(dates_now, hist_now.values, color=COLORS[ticker], lw=1.8, label="historical")
            axes[1].plot(dates_now, expected, color="black", lw=2.0, ls="--", label="expected path")''',
        '''            axes[1].plot(tt, hist_now.values, color=COLORS[ticker], lw=1.8, label="historical")
            axes[1].plot(tt, expected, color="black", lw=2.0, ls="--", label="expected path")''',
    ),
    (
        '            for ax in axes:\n                ax.set_xlabel("date")\n',
        '            for ax in axes:\n                ax.set_xlabel("trading minute")\n',
    ),
    (
        '''from american_lsm import (
    lsm_american_call,
    load_spy_calls,
    params_asof,
    sample_spy_calls,
)''',
        '''from american_lsm import (
    lsm_american_call,
    load_spy_calls,
    make_synthetic_intraday_calls,
    params_asof,
    sample_spy_calls,
)''',
    ),
    (
        '''_spy_calls_all = load_spy_calls(DATA)
_contracts = sample_spy_calls(
    _spy_calls_all, PERIOD_START, PERIOD_END, n_total=24, seed=42
)
stopping_results = None

display(Markdown(
    f"Sampled **{len(_contracts)}** SPY American calls in "
    f"{PERIOD_START.date()} → {PERIOD_END.date()} "
    f"(ATM band / DTE 7–60 as in the panel)."
))''',
        '''_spy_calls_all = load_spy_calls(DATA)
_contracts = sample_spy_calls(
    _spy_calls_all, PERIOD_START, PERIOD_END, n_total=24, seed=42
)
if _contracts is None or len(_contracts) == 0:
    _rates_path = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID / "dgs3mo.csv"
    _rates = (
        pd.read_csv(_rates_path, parse_dates=["observation_date"])
        .set_index("observation_date")["DGS3MO"].astype(float) / 100.0
    )
    _contracts = make_synthetic_intraday_calls(
        period_prices["SPY"],
        log_returns_all["SPY"],
        n_days=float(N_DAYS),
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        rates=_rates,
        ticker="SPY",
        dte_days=2,
    )
stopping_results = None

display(Markdown(
    f"Using **{len(_contracts)}** SPY American calls in "
    f"{PERIOD_START.date()} → {PERIOD_END.date()} "
    f"(listed panel empty → synthetic 2-trading-day ATM/OTM/ITM vs BS European)."
))''',
    ),
    (
        'display(Markdown("No SPY call contracts in this period panel slice."))',
        'display(Markdown("No SPY call contracts after synthetic fallback."))',
    ),
    (
        'ax.set_xlabel("market option_price")',
        'ax.set_xlabel("BS European benchmark")',
    ),
    (
        'ax.set_title("Price: model vs market")',
        'ax.set_title("Price: model vs BS benchmark")',
    ),
    (
        '["model", "market"]',
        '["model", "BS bench"]',
    ),
    (
        '''    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    steps = {
        "mu": np.full(dte, r, dtype=float),
        "omega": np.full(dte, float(p["omega"]), dtype=float),
        "alpha": np.full(dte, float(p["alpha"]), dtype=float),
        "beta": np.full(dte, float(p["beta"]), dtype=float),
        "sigma0": np.full(dte, float(p["sigma0"]), dtype=float),
        "lam": np.full(dte, float(p["lam"]), dtype=float),
        "mu_j": np.full(dte, float(p["mu_j"]), dtype=float),
        "sigma_j": np.full(dte, float(p["sigma_j"]), dtype=float),
        "kappa": np.full(dte, float(p["kappa"]), dtype=float),
    }''',
        '''    dte_days = int(row.dte)
    if dte_days < 1:
        raise ValueError("dte must be >= 1 trading day")
    n_steps = int(dte_days) * int(BARS_PER_DAY)
    r = float(row.r)
    S0 = float(row.S_t)
    steps = {
        "mu": np.full(n_steps, r, dtype=float),
        "omega": np.full(n_steps, float(p["omega"]), dtype=float),
        "alpha": np.full(n_steps, float(p["alpha"]), dtype=float),
        "beta": np.full(n_steps, float(p["beta"]), dtype=float),
        "sigma0": np.full(n_steps, float(p["sigma0"]), dtype=float),
        "lam": np.full(n_steps, float(p["lam"]), dtype=float),
        "mu_j": np.full(n_steps, float(p["mu_j"]), dtype=float),
        "sigma_j": np.full(n_steps, float(p["sigma_j"]), dtype=float),
        "kappa": np.full(n_steps, float(p["kappa"]), dtype=float),
    }''',
    ),
]

MD2 = """## 2. Strike prices in this period

Listed option quotes in this repo end in **2020**, so they do not cover **2023-03-09 → 2023-03-15**.

§2 / §6 use **synthetic 2-trading-day ATM / OTM / ITM calls** on each RTH session (09:30, 11:00, 13:00, 15:00).
Benchmark = Black–Scholes European; rate = 3-month T-bill. Prices are the **continuous trading-time** series (overnight/weekend removed). SPY is the stopping underlying.
"""

CODE2 = '''import sys
_SCRIPTS = Path("..") / "scripts"
if str(_SCRIPTS.resolve()) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.resolve()))
from american_lsm import make_synthetic_intraday_calls

_rates_path = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID / "dgs3mo.csv"
_rates = (
    pd.read_csv(_rates_path, parse_dates=["observation_date"])
    .set_index("observation_date")["DGS3MO"].astype(float) / 100.0
)
_synth = {}
for ticker in TICKERS:
    _synth[ticker] = make_synthetic_intraday_calls(
        period_prices[ticker],
        log_returns_all[ticker],
        n_days=float(N_DAYS),
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        rates=_rates,
        ticker=ticker,
        dte_days=2,
    )
    df = _synth[ticker]
    uniq = np.sort(df["K"].dropna().unique()) if len(df) else np.array([])
    display(Markdown(
        f"### {ticker} — {len(uniq)} synthetic strikes "
        f"(2-trading-day ATM/OTM/ITM; continuous RTH prices)"
    ))
    if len(df):
        print("Strikes K:", ", ".join(f"{x:g}" for x in uniq))
        display(df[["trading_date", "S_t", "K", "dte", "r", "moneyness", "option_price"]].round(4))
    else:
        print("No synthetic contracts.")
_contracts_synth_spy = _synth.get("SPY", pd.DataFrame())
'''

INTRO_BITS = [
    (
        "Calibration lookback = **1 day**. Rolling = **hourly** or **minutely**",
        "Calibration lookback default = **1 hour** (60 trading minutes). Rolling default = **minutely**",
    ),
    (
        "lookback is 1 day; choose hourly or minutely rolling",
        "lookback default is 1 hour; rolling default is minutely",
    ),
]


def to_lines(text: str) -> list[str]:
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
        return [x + "\n" for x in parts]
    return [x + "\n" for x in parts[:-1]] + [parts[-1]]


def patch_nb(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    hits = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        new = src
        if cell.get("cell_type") == "markdown" and src.startswith("## 2. Strike prices"):
            new = MD2
            hits.append("md2")
        elif "for ticker in TICKERS:" in src and "_options_panel.csv" in src and "unique strikes" in src:
            new = CODE2
            hits.append("code2")
        else:
            for old, nxt in REPLACEMENTS + INTRO_BITS:
                if old in new:
                    new = new.replace(old, nxt)
                    hits.append("repl")
        if new != src:
            cell["source"] = to_lines(new)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: {len(hits)} edits")


def main() -> None:
    for p in NOTEBOOKS:
        patch_nb(p)


if __name__ == "__main__":
    main()
