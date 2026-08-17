"""Clone each 2019–2020 model notebook into a 7-day 1-minute version."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

MODELS = [
    ("gbm notebook", "2019-2020_gbm.ipynb", "7d_1min_gbm.ipynb"),
    ("merton notebook", "2019-2020_merton.ipynb", "7d_1min_merton.ipynb"),
    ("heston notebook", "2019-2020_heston.ipynb", "7d_1min_heston.ipynb"),
    ("heston merton notebook", "2019-2020_heston_merton.ipynb", "7d_1min_heston_merton.ipynb"),
    (
        "heston merton advanced notebook",
        "2019-2020_heston_merton_advanced.ipynb",
        "7d_1min_heston_merton_advanced.ipynb",
    ),
    ("garch notebook", "2019-2020_garch.ipynb", "7d_1min_garch.ipynb"),
    ("garch merton notebook", "2019-2020_garch_merton.ipynb", "7d_1min_garch_merton.ipynb"),
]

VERSIONS = ["V1-Models_result", "V2-Models_result"]

NEW_LOAD = '''
INTRADAY_DIR = DATA / "equity" / "intraday" / "7d_1min"
_frames = []
for _t in TICKERS:
    _p = pd.read_csv(INTRADAY_DIR / f"{_t}.csv", parse_dates=["Datetime"]).set_index("Datetime").sort_index()
    _frames.append(_p["Close"].rename(_t))
prices = pd.concat(_frames, axis=1).sort_index()
_days = prices.dropna(how="all").index.normalize().unique().sort_values()
PERIOD_START = prices.loc[prices.index.normalize() >= _days[-7]].index.min()
PERIOD_END = prices.dropna(how="all").index.max()
period_prices = prices.loc[PERIOD_START:PERIOD_END, TICKERS].copy()
log_returns_all = np.log(prices[TICKERS]).diff()
'''.lstrip("\n")

NEW_ROLLING = '''    period_idx = rets.loc[(rets.index >= PERIOD_START) & (rets.index <= PERIOD_END)].index
    if rolling_mode == "minutely":
        update_dates = period_idx
    elif rolling_mode == "hourly":
        hours = period_idx.floor("h")
        update_dates = pd.DatetimeIndex(
            [period_idx[hours == h].max() for h in hours.unique()]
        ).sort_values()
    else:
        update_dates = pd.DatetimeIndex([period_idx[0]])
'''

OLD_ROLLING = '''    if rolling_mode == "daily":
        update_dates = rets.loc[(rets.index >= PERIOD_START) & (rets.index <= PERIOD_END)].index
    elif rolling_mode == "monthly":
        t0 = period_prices[ticker].dropna().index[0]
        month_ends = pd.date_range(PERIOD_START, PERIOD_END, freq="ME")
        update_dates = pd.DatetimeIndex([t0]).append(month_ends).unique().sort_values()
    else:
        update_dates = pd.DatetimeIndex([period_prices[ticker].dropna().index[0]])
'''

OLD_WINDOWS = '''WINDOW_OPTIONS = {
    "3 months": pd.DateOffset(months=3),
    "6 months": pd.DateOffset(months=6),
    "1 year": pd.DateOffset(years=1),
    "2 years": pd.DateOffset(years=2),
    "5 years": pd.DateOffset(years=5),
}
ROLLING_OPTIONS = ["daily", "monthly", "none"]
'''

NEW_WINDOWS = '''WINDOW_OPTIONS = {
    "1 day": pd.Timedelta(days=1),
}
ROLLING_OPTIONS = ["minutely", "hourly"]
'''

VOL_APPENDIX = '''

# Realized volatility from 1-minute log returns (annualized with 252 × 390)
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
vol_rows = []
for ax, ticker in zip(axes, TICKERS):
    r = log_returns_all[ticker].loc[PERIOD_START:PERIOD_END].dropna()
    rv = r.rolling(30, min_periods=10).std() * np.sqrt(N_DAYS)
    ax.plot(rv.index, rv.values, color=COLORS[ticker], lw=1.0)
    role = "primary" if ticker == "SPY" else "secondary"
    ax.set_ylabel("σ̂ (ann.)")
    ax.set_title(f"{ticker} ({role}) — 30-min rolling vol from 1-min returns")
    vol_rows.append({
        "ticker": ticker,
        "n_bars": int(r.shape[0]),
        "sigma_1min": float(r.std(ddof=1)),
        "sigma_ann": float(r.std(ddof=1) * np.sqrt(N_DAYS)),
        "mu_ann": float(r.mean() * N_DAYS),
    })
axes[-1].set_xlabel("Datetime")
fig.suptitle("Minute-level realized volatility — 7-day 1-minute window", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
display(pd.DataFrame(vol_rows).set_index("ticker").round(6))
'''


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [text]
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def adapt_markdown(text: str) -> str:
    import re

    text = text.replace("2019–2020", "7-day 1-minute")
    text = text.replace("2019-2020", "7-day 1-minute")
    text = text.replace("2019-01-01 → 2020-12-31", "last 7 regular sessions (1-minute bars)")
    text = text.replace("**2019-01-01 → 2020-12-31**", "**last 7 regular sessions (1-minute bars)**")
    text = text.replace("Extra window covering COVID 2020 — useful for comparing jump intensity / fat tails.",
                        "Not a 2-year regime: evaluation is **7 days of 1-minute bars**; lookback is **1 day**.")
    text = text.replace("from daily log returns", "from 1-minute log returns")
    text = text.replace("Jump days", "Jump bars")
    text = text.replace("daily σ", "1-min σ")
    text = text.replace("daily variance proxy", "1-minute variance proxy")
    text = text.replace("on the daily grid", "on the 1-minute grid")
    text = text.replace("trading-day grid", "1-minute grid")
    text = text.replace("Default rolling = **monthly**", "Default rolling = **hourly**")
    text = text.replace("Default rolling=monthly", "Default rolling=hourly")
    text = text.replace("with `daily`/`monthly`", "with `minutely`/`hourly`")
    text = text.replace("Rolling = daily / monthly", "Rolling = minutely / hourly")
    text = text.replace("(\\(N=252\\))", "(\\(N=252\\times 390\\) 1-min bars/year)")
    text = text.replace("\\times 252", "\\times (252\\times 390)")
    text = text.replace("× 252", "× (252×390)")
    text = text.replace("\\(\\hat\\mu_{\\mathrm{day}}\\times 252\\)",
                        "\\(\\hat\\mu_{\\mathrm{bar}}\\times (252\\times 390)\\)")
    text = text.replace("≤21 days", "≤21 bars")
    text = text.replace("`none` / `monthly` / `daily`", "`minutely` / `hourly`")
    text = text.replace("choose lookback / rolling", "lookback is 1 day; choose hourly or minutely rolling")
    text = re.sub(r"\\sqrt\{252\}(?!\\times)", r"\\sqrt{252\\times 390}", text)
    text = text.replace("jump-day returns", "jump-bar returns")
    text = text.replace("\\hat\\sigma_{\\text{day}}", "\\hat\\sigma_{\\text{1min}}")
    if text.lstrip().startswith("## 2. Strike"):
        text = text.rstrip() + (
            "\n\nThe daily options panel is 2008–2020, so it will not overlap this "
            "7-day 1-minute window. §6 is kept for the same workflow; expect an empty sample.\n"
        )
    if text.lstrip().startswith("# ") and "7-day 1-minute" in text and "not a 2-year" not in text.lower():
        lines = text.splitlines()
        insert = (
            "\n> **This is not a 2-year regime file.** "
            "The evaluation window is **7 regular sessions of 1-minute bars**. "
            "Calibration lookback = **1 day**. Rolling = **hourly** or **minutely** "
            "(60-minute / 2–30-minute / 1-hour files are on disk but unused here).\n"
        )
        lines.insert(1, insert.rstrip("\n"))
        text = "\n".join(lines)
        if not text.endswith("\n"):
            text += "\n"
    return text


def adapt_setup(text: str) -> str:
    text = text.replace('DATA = Path("..") / ".." / "research" / "data"',
                        'DATA = Path("..") / ".." / "research" / "data"')
    text = text.replace('DATA = Path("..") / "data"',
                        'DATA = Path("..") / ".." / "research" / "data"')
    text = text.replace(
        'PERIOD_START = pd.Timestamp("2019-01-01")\nPERIOD_END = pd.Timestamp("2020-12-31")\n',
        "",
    )
    text = text.replace("N_DAYS = 252\n",
                        "BARS_PER_DAY = 390  # regular session 09:30–16:00 ET\n"
                        "N_DAYS = 252 * BARS_PER_DAY  # annualize 1-minute log returns\n")
    text = text.replace("N_STEPS = 500  # Monte Carlo time steps per path",
                        "N_STEPS = 5000  # keep the full 1-minute grid (no daily-style subsample)")
    text = text.replace("N_STEPS = 500  # research Monte Carlo steps (not estimation)",
                        "N_STEPS = 5000  # keep the full 1-minute grid (no daily-style subsample)")
    text = text.replace("flag |r| > c * daily σ as a jump",
                        "flag |r| > c * 1-min σ as a jump")
    if OLD_WINDOWS not in text:
        raise RuntimeError("WINDOW_OPTIONS block not found")
    text = text.replace(OLD_WINDOWS, NEW_WINDOWS)
    old_load = (
        'prices = pd.read_csv(DATA / "equity" / "prices_clean.csv", parse_dates=["Date"]).set_index("Date").sort_index()\n'
        "period_prices = prices.loc[PERIOD_START:PERIOD_END, TICKERS].copy()\n"
        "log_returns_all = np.log(prices[TICKERS]).diff()\n"
    )
    if old_load not in text:
        raise RuntimeError("price-load block not found")
    text = text.replace(old_load, NEW_LOAD)
    text = text.replace(
        'print(f"Price sample: {prices.index.min().date()} → {prices.index.max().date()}")\n'
        "print(\n"
        '    f"Period rows: {len(period_prices)} trading days "\n'
        '    f"({period_prices.index.min().date()} → {period_prices.index.max().date()})"\n'
        ")",
        'print(f"1-min sample: {prices.index.min()} → {prices.index.max()}")\n'
        "print(\n"
        '    f"Evaluation window: {len(period_prices)} 1-minute bars "\n'
        '    f"({period_prices.index.min()} → {period_prices.index.max()}) — NOT a 2-year regime"\n'
        ")",
    )
    return text


def adapt_code(text: str) -> str:
    if OLD_ROLLING in text:
        text = text.replace(OLD_ROLLING, NEW_ROLLING)
    text = text.replace('value="3 months"', 'value="1 day"')
    text = text.replace('value="daily"', 'value="hourly"')
    text = text.replace('value="monthly"', 'value="hourly"')
    text = text.replace("Evenly spaced trading-day grid", "Evenly spaced 1-minute grid")
    text = text.replace(
        "With `none`, parameters stay fixed; with `daily`/`monthly`, they refresh at each update for the next MC segment.",
        "With `hourly` / `minutely`, parameters refresh at each update for the next MC segment. Lookback is 1 day.",
    )
    text = text.replace("Default rolling=monthly.", "Default rolling=hourly.")
    text = text.replace(
        "<b>§4 Calibration (graphs only)</b> — lookback + rolling, then <b>Reestimate</b>. ",
        "<b>§4 Calibration (graphs only)</b> — 1-day lookback; hourly or minutely rolling, then <b>Reestimate</b>. ",
    )
    return text


def adapt_price_plot(text: str) -> str:
    text = text.replace("2019–2020", "7-day 1-minute")
    text = text.replace("2019-2020", "7-day 1-minute")
    text = text.replace('ax.set_ylabel("Adj close")', 'ax.set_ylabel("Close")')
    text = text.replace("adjusted close", "1-minute close")
    text = text.replace('axes[-1].set_xlabel("Date")', 'axes[-1].set_xlabel("Datetime")')
    if "Minute-level realized volatility" not in text:
        text = text.rstrip() + VOL_APPENDIX
    return text


def transform_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        text = src(cell)
        if cell.get("cell_type") == "markdown":
            set_src(cell, adapt_markdown(text))
            continue
        if "WINDOW_OPTIONS = {" in text and "prices = pd.read_csv" in text:
            set_src(cell, adapt_code(adapt_setup(text)))
            continue
        if text.lstrip().startswith("fig, axes = plt.subplots(3, 1"):
            set_src(cell, adapt_price_plot(text))
            continue
        set_src(cell, adapt_code(text) if cell.get("cell_type") == "code" else text)
    nb.setdefault("metadata", {}).setdefault("kernelspec", {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    })
    return nb


def main() -> None:
    written = []
    for version in VERSIONS:
        for folder, src_name, dst_name in MODELS:
            src_path = ROOT / version / folder / src_name
            dst_path = ROOT / version / folder / dst_name
            if not src_path.exists():
                raise FileNotFoundError(src_path)
            nb = transform_notebook(src_path)
            dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
            written.append(str(dst_path.relative_to(ROOT)))
            print(f"wrote {dst_path.relative_to(ROOT)}", flush=True)
    v1_fetch = ROOT / "V1-Models_result" / "scripts" / "data_fetch_intraday.py"
    v2_fetch = ROOT / "V2-Models_result" / "scripts" / "data_fetch_intraday.py"
    if v2_fetch.exists():
        shutil.copy2(v2_fetch, v1_fetch)
        print(f"copied fetch script → {v1_fetch.relative_to(ROOT)}", flush=True)
    print(f"done ({len(written)} notebooks)", flush=True)


if __name__ == "__main__":
    main()
