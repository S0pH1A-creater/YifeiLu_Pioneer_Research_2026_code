#!/usr/bin/env python3
"""Patch V2 7d_1min notebooks and clone 2023-03-15 daily notebooks.

- Continuous RTH prices (drop overnight/weekend straight lines)
- Synthetic strikes until expiration 2023-03-15 15:59
- RMSE(S_t) on §5 figures; stopping RMSE stays in §6
- Clone 7d → 2023-03-15_*.ipynb (same expiry, one RTH session)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS_7D = [
    ROOT / "gbm notebook" / "7d_1min_gbm.ipynb",
    ROOT / "merton notebook" / "7d_1min_merton.ipynb",
    ROOT / "heston merton notebook" / "7d_1min_heston_merton.ipynb",
    ROOT / "garch merton notebook" / "7d_1min_garch_merton.ipynb",
    ROOT / "heston merton advanced notebook" / "7d_1min_heston_merton_advanced.ipynb",
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

def n_steps_to_expiry(quote_ts, expiry_ts=None) -> int:
    """Remaining 1-minute RTH bars from quote to expiration (PERIOD_END)."""
    q = pd.Timestamp(quote_ts)
    e = pd.Timestamp(expiry_ts) if expiry_ts is not None else PERIOD_END
    idx = prices.index
    n = int(((idx > q) & (idx <= e)).sum())
    return max(n, 2)
'''.lstrip("\n")

MD2 = """## 2. Strike prices in this period

Listed option quotes in this repo end in **2020**, so they do not cover **2023-03-09 → 2023-03-15**.

§2 / §6 use **synthetic ATM / OTM / ITM calls** on each RTH session (09:30, 11:00, 13:00, 15:00).
Every contract **expires at 2023-03-15 15:59** (session close). DTE / Monte Carlo steps = remaining RTH 1-minute bars until that expiry.
Benchmark = Black–Scholes European; rate = 3-month T-bill. Prices are the **continuous trading-time** series (overnight/weekend removed).
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
        expiration=PERIOD_END,
        bars_per_day=int(BARS_PER_DAY),
    )
    df = _synth[ticker]
    uniq = np.sort(df["K"].dropna().unique()) if len(df) else np.array([])
    display(Markdown(
        f"### {ticker} — {len(uniq)} synthetic strikes "
        f"(ATM/OTM/ITM until expiry {PERIOD_END.date()}; continuous RTH prices)"
    ))
    if len(df):
        print("Strikes K:", ", ".join(f"{x:g}" for x in uniq))
        cols = [c for c in ["trading_date", "S_t", "K", "expiration", "dte", "n_steps", "r", "moneyness", "option_price"] if c in df.columns]
        display(df[cols].round(4))
    else:
        print("No synthetic contracts.")
'''

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
        '''            fig.suptitle(
                f"{ticker} | seed={seed} | {cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')}",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
        print(f"RMSE(expected vs historical) = {rmse:.4f} | seed = {seed}")''',
        '''            rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
            fig.suptitle(
                f"{ticker} | RMSE(S_t)={rmse:.4f} | seed={seed} | "
                f"{cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')}",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        display(Markdown(
            f"**RMSE (expected vs historical \(S_t\))** = `{rmse:.4f}` "
            f"| seed = `{seed}`"
        ))''',
    ),
    (
        '''            fig.suptitle(
                f"{ticker} | seed={seed} | {cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')} | Method A",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
        print(f"RMSE(expected vs historical) = {rmse:.4f} | seed = {seed}")''',
        '''            rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
            fig.suptitle(
                f"{ticker} | RMSE(S_t)={rmse:.4f} | seed={seed} | "
                f"{cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')} | Method A",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        display(Markdown(
            f"**RMSE (expected vs historical \(S_t\))** = `{rmse:.4f}` "
            f"| seed = `{seed}`"
        ))''',
    ),
    (
        '''            fig.suptitle(
                f"{ticker} | seed={seed} | {cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')} | Method B",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
        print(f"RMSE(expected vs historical) = {rmse:.4f} | seed = {seed}")''',
        '''            rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
            fig.suptitle(
                f"{ticker} | RMSE(S_t)={rmse:.4f} | seed={seed} | "
                f"{cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')} | Method B",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        display(Markdown(
            f"**RMSE (expected vs historical \(S_t\))** = `{rmse:.4f}` "
            f"| seed = `{seed}`"
        ))''',
    ),
    (
        '''from american_lsm import (
    STOP_TICKERS,
    lsm_american_call,
    load_calls,
    params_asof,
    sample_calls,
)''',
        '''from american_lsm import (
    STOP_TICKERS,
    lsm_american_call,
    load_calls,
    make_synthetic_intraday_calls,
    n_steps_to_expiry as _n_steps_lib,
    params_asof,
    sample_calls,
)''',
    ),
    (
        '''    dte_days = int(row.dte)
    if dte_days < 2:
        raise ValueError("dte must be >= 2 trading days")
    n_steps = int(dte_days) * int(BARS_PER_DAY)''',
        '''    n_steps = n_steps_to_expiry(
        row.trading_date, getattr(row, "expiration", PERIOD_END)
    )''',
    ),
    (
        '''_STOP_TICKERS = list(STOP_TICKERS)
_contracts_by_ticker = {}
for _t in _STOP_TICKERS:
    _panel = load_calls(DATA, _t)
    _contracts_by_ticker[_t] = sample_calls(
        _panel, PERIOD_START, PERIOD_END, n_total=24, seed=42
    )''',
        '''_STOP_TICKERS = list(STOP_TICKERS)
_rates_path = DATA / "equity" / "intraday" / "7d_1min" / WINDOW_ID / "dgs3mo.csv"
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
    _contracts_by_ticker[_t] = _c''',
    ),
    (
        'f"(ATM band / DTE 7–60 as in the panel)."',
        'f"(synthetic ATM/OTM/ITM until expiry {PERIOD_END.date()}; RMSE = LSM vs BS European)."',
    ),
    (
        'display(Markdown(f"No {ticker} call contracts in this period panel slice."))',
        'display(Markdown(f"No {ticker} call contracts after synthetic fallback."))',
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
        'f"RMSE={rmse:.4f} | MAE={mae:.4f} | "',
        'f"RMSE(stopping vs BS)={rmse:.4f} | MAE={mae:.4f} | "',
    ),
    (
        '["trading_date", "S_t", "K", "dte", "market", "model_price",\n                         "error", "early_ex_frac", "mean_ex_day"]',
        '["trading_date", "S_t", "K", "dte", "market", "model_price",\n                         "error", "early_ex_frac", "mean_ex_day"]',
    ),
]

INTRO_BITS = [
    (
        "Calibration lookback = **1 day**. Rolling = **hourly** or **minutely** (60-minute / 2–30-minute / 1-hour files are on disk but unused here).",
        "Calibration lookback default = **1 hour** (60 trading minutes). Rolling default = **minutely**. Prices are stitched to **continuous RTH trading time** (overnight/weekend removed). Contracts expire **2023-03-15 15:59**.",
    ),
    (
        "lookback is 1 day; choose hourly or minutely rolling",
        "lookback default is 1 hour; rolling default is minutely",
    ),
    (
        "## 1. Stock price trends (7-day 1-minute)\n\nAdjusted close for AAPL, MSFT, and SPY (primary).\n",
        "## 1. Stock price trends (7-day 1-minute, continuous RTH)\n\nOvernight/weekend hours are dropped and prices are stitched into a continuous trading-time series. Dotted lines mark session joins. The return/vol panel uses 1-minute log returns on that same series.\n",
    ),
]


def to_lines(text: str) -> list[str]:
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
        return [x + "\n" for x in parts]
    return [x + "\n" for x in parts[:-1]] + [parts[-1]]


def patch_nb(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        new = src
        if cell.get("cell_type") == "markdown" and src.startswith("## 2. Strike prices"):
            new = MD2
            hits += 1
        elif (
            cell.get("cell_type") == "code"
            and "for ticker in TICKERS:" in src
            and "_options_panel.csv" in src
            and "unique strikes" in src
        ):
            new = CODE2
            hits += 1
        else:
            for old, nxt in REPLACEMENTS + INTRO_BITS:
                if old in new:
                    new = new.replace(old, nxt)
                    hits += 1
        if new != src:
            cell["source"] = to_lines(new)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {hits} edits")
    return hits


DAILY_REPLACES = [
    (
        "The evaluation window is calendar week **2023-03-09 → 2023-03-15** "
        "(5 regular sessions: 9, 10, 13, 14, 15 Mar; weekend skipped).",
        "The evaluation window is **Wednesday 2023-03-15** "
        "(one RTH session; lookback uses prior 1-minute bars on file). "
        "Contracts still **expire at 2023-03-15 15:59**.",
    ),
    (
        "**Period file:** **2023-03-09 → 2023-03-15 (1-minute bars; 5 RTH sessions)**.",
        "**Period file:** **Wednesday 2023-03-15 (1-minute bars; 1 RTH session)**.",
    ),
    (
        "PERIOD_START = pd.Timestamp(\"2023-03-09 09:30:00\")",
        "PERIOD_START = pd.Timestamp(\"2023-03-15 09:30:00\")",
    ),
    ("period 7-day 1-minute", "period 1-day 1-minute (2023-03-15)"),
    ("7-day 1-minute window", "1-day 1-minute window (2023-03-15)"),
    ("— 7-day 1-minute", "— 2023-03-15"),
    ("(7-day 1-minute)", "(2023-03-15)"),
    (
        "Stock price trends — 7-day 1-minute, continuous RTH",
        "Stock price trends — 2023-03-15, continuous RTH",
    ),
    (
        "## 1. Stock price trends (7-day 1-minute, continuous RTH)",
        "## 1. Stock price trends (2023-03-15, continuous RTH)",
    ),
    (
        "do not cover **2023-03-09 → 2023-03-15**",
        "do not cover **2023-03-15**",
    ),
    (
        "Calibration lookback default = **1 hour** (60 trading minutes). Rolling default = **minutely**. "
        "Prices are stitched to **continuous RTH trading time** (overnight/weekend removed). "
        "Contracts expire **2023-03-15 15:59**.",
        "Calibration lookback default = **1 hour** (60 trading minutes). Rolling default = **minutely**. "
        "Evaluation is **2023-03-15** only. Contracts expire **2023-03-15 15:59** "
        "(remaining RTH minutes that session).",
    ),
]


DAILY_MAP = [
    ("gbm notebook", "7d_1min_gbm.ipynb", "1d_1min_gbm.ipynb"),
    ("merton notebook", "7d_1min_merton.ipynb", "1d_1min_merton.ipynb"),
    ("heston merton notebook", "7d_1min_heston_merton.ipynb", "1d_1min_heston_merton.ipynb"),
    ("garch merton notebook", "7d_1min_garch_merton.ipynb", "1d_1min_garch_merton.ipynb"),
    (
        "heston merton advanced notebook",
        "7d_1min_heston_merton_advanced.ipynb",
        "1d_1min_heston_merton_advanced.ipynb",
    ),
]


def clone_daily(src: Path, dst: Path) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        raw = "".join(cell.get("source", []))
        new = raw
        for old, nxt in DAILY_REPLACES:
            new = new.replace(old, nxt)
        if new != raw:
            cell["source"] = to_lines(new)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {dst.relative_to(ROOT)}")


def main() -> None:
    for p in NOTEBOOKS_7D:
        if not p.is_file():
            print(f"skip missing {p}")
            continue
        patch_nb(p)
    print("Cloning 2023-03-15 daily notebooks…")
    for folder, src_name, dst_name in DAILY_MAP:
        src = ROOT / folder / src_name
        dst = ROOT / folder / dst_name
        if not src.is_file():
            print(f"skip missing {src}")
            continue
        clone_daily(src, dst)


if __name__ == "__main__":
    main()
