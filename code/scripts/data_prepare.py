"""
Step 1 — Equity preparation (DataCollection §1 / ResearchProposal-v2).

Loads cleaned adjusted closes, computes log returns for all tickers,
splits into volatility regimes, writes summary statistics, and figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from data_fetch import (
        DATA_DIR,
        EQUITY_DIR,
        EXTRA_STUDY_TICKERS,
        PRIMARY_TICKER,
        PRICES_PATH,
        SECONDARY_TICKERS,
        TICKERS,
        load_or_download,
    )
except ImportError:
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    EQUITY_DIR = DATA_DIR / "equity"
    PRICES_PATH = EQUITY_DIR / "prices_clean.csv"
    PRIMARY_TICKER = "SPY"
    SECONDARY_TICKERS = ["AAPL", "MSFT"]
    EXTRA_STUDY_TICKERS = ["AMZN"]
    TICKERS = [PRIMARY_TICKER, *SECONDARY_TICKERS, *EXTRA_STUDY_TICKERS]

    def load_or_download(force: bool = False) -> pd.DataFrame:
        df = pd.read_csv(PRICES_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
        df.index.name = "Date"
        return df

# ---------------------------------------------------------------------------
# Paths and regime definitions
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

LOG_RETURNS_ALL = EQUITY_DIR / "log_returns_all.csv"
LOG_RETURNS_BY_REGIME = EQUITY_DIR / "log_returns_by_regime.csv"
SUMMARY_STATS = EQUITY_DIR / "summary_stats.csv"

REGIMES: dict[str, tuple[str, str]] = {
    "crisis": ("2008-01-01", "2009-12-31"),
    "normal": ("2013-01-01", "2014-12-31"),
    "late": ("2018-01-01", "2019-12-31"),
    "covid": ("2019-01-01", "2020-12-31"),
}

# 1.5-year study evaluation windows — used only for figure highlighters.
FIGURE_REGIMES: dict[str, tuple[str, str]] = {
    "crisis": ("2008-08-01", "2009-07-31"),
    "normal": ("2014-01-01", "2014-12-31"),
    "late": ("2018-10-01", "2019-09-30"),
    "covid": ("2019-09-01", "2020-08-31"),
}

REGIME_ORDER = ["crisis", "normal", "late", "covid"]
REGIME_LABELS = {
    "crisis": "Crisis (2008–2009)",
    "normal": "Normal (2013–2014)",
    "late": "Late (2018–2019)",
    "covid": "COVID (2019–2020)",
}
FIGURE_REGIME_LABELS = {
    name: f"regime {i}: {FIGURE_REGIMES[name][0]} to {FIGURE_REGIMES[name][1]}"
    for i, name in enumerate(REGIME_ORDER, start=1)
}
REGIME_COLORS = {
    "crisis": "#c44e52",
    "normal": "#4c72b0",
    "late": "#dd8452",
    "covid": "#8172b3",  # purple — distinct from late/orange
}

MIN_OBS_PER_REGIME = 200


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(S_t / S_{t-1}) — primary return measure for modeling."""
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.index.name = "Date"
    return returns


def regimes_for_date(date: pd.Timestamp) -> list[str]:
    """All regimes whose windows contain `date` (allows late ∩ covid in 2019)."""
    matched: list[str] = []
    for name, (start, end) in REGIMES.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            matched.append(name)
    return matched


def assign_regime(date: pd.Timestamp) -> str | None:
    """Primary regime label (first match in REGIME_ORDER)."""
    matched = regimes_for_date(date)
    return matched[0] if matched else None


def build_regime_long(returns: pd.DataFrame) -> pd.DataFrame:
    """Long-format table: ticker, regime, date, log_return.

    Overlapping windows (late ∩ covid in 2019) emit one row per matching regime
    so each evaluation period keeps its full sample for stats/figures.
    """
    rows: list[dict] = []
    for ticker in returns.columns:
        series = returns[ticker]
        for date, value in series.items():
            for regime in regimes_for_date(pd.Timestamp(date)):
                rows.append(
                    {
                        "ticker": ticker,
                        "regime": regime,
                        "date": date,
                        "log_return": float(value),
                    }
                )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "regime", "date"]).reset_index(drop=True)


def figure_regimes_for_date(date: pd.Timestamp) -> list[str]:
    """Evaluation-window matches used for figure highlighters."""
    matched: list[str] = []
    for name, (start, end) in FIGURE_REGIMES.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            matched.append(name)
    return matched


def build_figure_regime_long(returns: pd.DataFrame) -> pd.DataFrame:
    """Same long table as `build_regime_long`, but on the 1-year evaluation windows."""
    rows: list[dict] = []
    for ticker in returns.columns:
        series = returns[ticker]
        for date, value in series.items():
            for regime in figure_regimes_for_date(pd.Timestamp(date)):
                rows.append(
                    {
                        "ticker": ticker,
                        "regime": regime,
                        "date": date,
                        "log_return": float(value),
                    }
                )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "regime", "date"]).reset_index(drop=True)


def summary_statistics(regime_long: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker × regime with moments used for jump-model motivation."""
    records: list[dict] = []
    for (ticker, regime), group in regime_long.groupby(["ticker", "regime"]):
        r = group["log_return"]
        records.append(
            {
                "ticker": ticker,
                "regime": regime,
                "n_obs": int(len(r)),
                "start_date": group["date"].min().date().isoformat(),
                "end_date": group["date"].max().date().isoformat(),
                "mean": float(r.mean()),
                "std": float(r.std()),
                "ann_vol": float(r.std() * np.sqrt(252)),
                "skewness": float(r.skew()),
                "kurtosis": float(r.kurtosis()),  # excess kurtosis (Fisher)
                "min": float(r.min()),
                "max": float(r.max()),
            }
        )
    stats = pd.DataFrame(records)
    regime_cat = pd.Categorical(stats["regime"], categories=REGIME_ORDER, ordered=True)
    ticker_cat = pd.Categorical(stats["ticker"], categories=TICKERS, ordered=True)
    stats = (
        stats.assign(regime=regime_cat, ticker=ticker_cat)
        .sort_values(["ticker", "regime"])
        .reset_index(drop=True)
    )
    stats["regime"] = stats["regime"].astype(str)
    stats["ticker"] = stats["ticker"].astype(str)
    return stats


# ---------------------------------------------------------------------------
# Figures (save only — no plt.show)
# ---------------------------------------------------------------------------
def plot_price_by_regime(prices: pd.DataFrame, ticker: str) -> Path:
    """Price path with shaded regime bands (same style as SPY figure 01)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    series = prices[ticker].dropna()
    ax.plot(
        series.index,
        series.values,
        color="#2f2f2f",
        linewidth=1.2,
        label=f"{ticker} Adj Close",
    )

    for name in REGIME_ORDER:
        start, end = FIGURE_REGIMES[name]
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=REGIME_COLORS[name],
            alpha=0.18,
            label=FIGURE_REGIME_LABELS[name],
        )

    ax.set_title(f"{ticker} Price Path with Volatility Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close (USD)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = FIGURES_DIR / f"01_{ticker.lower()}_price_by_regime.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"✓ Saved: {path.name}")
    return path


def plot_spy_price_by_regime(prices: pd.DataFrame) -> Path:
    """Backward-compatible wrapper for the primary ticker figure."""
    return plot_price_by_regime(prices, PRIMARY_TICKER)


def plot_aapl_msft_amzn_price_by_regime(prices: pd.DataFrame) -> Path:
    """Overlay AAPL / MSFT / AMZN with the same evaluation-window bands."""
    fig, ax = plt.subplots(figsize=(12, 5))
    line_colors = {"AAPL": "#4c72b0", "MSFT": "#55a868", "AMZN": "#c44e52"}
    for ticker in ("AAPL", "MSFT", "AMZN"):
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        ax.plot(
            series.index,
            series.values,
            color=line_colors[ticker],
            linewidth=1.2,
            label=f"{ticker} Adj Close",
        )

    for name in REGIME_ORDER:
        start, end = FIGURE_REGIMES[name]
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=REGIME_COLORS[name],
            alpha=0.18,
            label=FIGURE_REGIME_LABELS[name],
        )

    ax.set_title("AAPL, MSFT, AMZN Price Paths with Volatility Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close (USD)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = FIGURES_DIR / "01_aapl_msft_amzn_price_by_regime.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"✓ Saved: {path.name}")
    return path


def plot_return_distributions(regime_long: pd.DataFrame) -> Path:
    """Histograms of SPY log returns by regime (density)."""
    spy = regime_long[regime_long["ticker"] == PRIMARY_TICKER]
    n = len(REGIME_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)

    for ax, name in zip(axes, REGIME_ORDER):
        data = spy.loc[spy["regime"] == name, "log_return"]
        ax.hist(
            data,
            bins=50,
            density=True,
            color=REGIME_COLORS[name],
            alpha=0.75,
            edgecolor="none",
        )
        ax.set_title(FIGURE_REGIME_LABELS[name], fontsize=9)
        ax.set_xlabel("Daily log return")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density")
    fig.suptitle("SPY Log-Return Distributions by Regime", y=1.02)
    fig.tight_layout()

    path = FIGURES_DIR / "02_return_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {path.name}")
    return path


def plot_annualized_volatility(stats: pd.DataFrame) -> Path:
    """Grouped bar chart: annualized vol by ticker and regime."""
    fig, ax = plt.subplots(figsize=(11, 5))
    tickers = [t for t in TICKERS if t in set(stats["ticker"])]
    x = np.arange(len(tickers))
    n = len(REGIME_ORDER)
    width = 0.8 / n

    for i, name in enumerate(REGIME_ORDER):
        vals = []
        for t in tickers:
            match = stats.loc[(stats["ticker"] == t) & (stats["regime"] == name), "ann_vol"]
            vals.append(float(match.iloc[0]) if len(match) else 0.0)
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            vals,
            width=width,
            color=REGIME_COLORS[name],
            edgecolor="none",
            label=FIGURE_REGIME_LABELS[name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Annualized volatility")
    ax.set_title("Annualized Volatility by Ticker and Regime")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = FIGURES_DIR / "03_annualized_volatility_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"✓ Saved: {path.name}")
    return path


def plot_rolling_volatility_spy(returns: pd.DataFrame) -> Path:
    """30-day rolling annualized vol for SPY across full sample."""
    spy_r = returns[PRIMARY_TICKER]
    rolling = spy_r.rolling(window=30).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling.index, rolling.values, color="#4c72b0", linewidth=1.0, alpha=0.85)

    for name in REGIME_ORDER:
        start, end = FIGURE_REGIMES[name]
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=REGIME_COLORS[name],
            alpha=0.15,
            label=FIGURE_REGIME_LABELS[name],
        )

    ax.set_title("SPY 30-Day Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = FIGURES_DIR / "04_rolling_volatility_spy.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"✓ Saved: {path.name}")
    return path


def plot_all_tickers_normalized(prices: pd.DataFrame) -> Path:
    """Normalized price paths for SPY / AAPL / MSFT / AMZN."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in TICKERS:
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        norm = s / s.iloc[0] * 100.0
        ax.plot(norm.index, norm.values, linewidth=1.1, label=ticker)

    for name in REGIME_ORDER:
        start, end = FIGURE_REGIMES[name]
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=REGIME_COLORS[name],
            alpha=0.12,
            label=FIGURE_REGIME_LABELS[name],
        )

    ax.set_title("Normalized Adj Close (100 = first observation)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = FIGURES_DIR / "05_all_tickers_normalized.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"✓ Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def run_checks(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_long: pd.DataFrame,
    stats: pd.DataFrame,
) -> dict[str, bool]:
    results: dict[str, bool] = {}

    results["prices_no_nan"] = int(prices.isna().sum().sum()) == 0
    results["returns_no_nan"] = int(returns.isna().sum().sum()) == 0
    results["has_all_tickers"] = all(t in prices.columns for t in TICKERS)

    spy = regime_long[regime_long["ticker"] == PRIMARY_TICKER]
    spy_counts = spy.groupby("regime")["log_return"].count().reindex(REGIME_ORDER)
    results["sufficient_obs_per_regime"] = bool((spy_counts >= MIN_OBS_PER_REGIME).all())

    spy_stats = stats[stats["ticker"] == PRIMARY_TICKER].set_index("regime")
    crisis_vol = float(spy_stats.loc["crisis", "ann_vol"])
    normal_vol = float(spy_stats.loc["normal", "ann_vol"])
    results["crisis_vol_gt_normal"] = crisis_vol > normal_vol

    simple = prices[PRIMARY_TICKER].pct_change().dropna()
    aligned = pd.concat([returns[PRIMARY_TICKER], simple], axis=1, join="inner").dropna()
    aligned.columns = ["log", "simple"]
    median_abs_diff = float((aligned["log"] - aligned["simple"]).abs().median())
    results["log_approx_simple"] = median_abs_diff < 1e-3

    # Every ticker has all evaluation regimes
    for ticker in TICKERS:
        if ticker not in stats["ticker"].values:
            results[f"{ticker}_has_regimes"] = False
            continue
        n_regimes = stats.loc[stats["ticker"] == ticker, "regime"].nunique()
        results[f"{ticker}_has_regimes"] = n_regimes == len(REGIME_ORDER)

    return results


def print_data_description(prices: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Concise description suitable for the paper Data section."""
    tickers = list(prices.columns)
    print("\n" + "=" * 70)
    print("DATA DESCRIPTION (for paper section)")
    print("=" * 70)
    print(
        f"Tickers: {', '.join(tickers)} | Primary: {PRIMARY_TICKER}\n"
        f"Full sample: {prices.index.min().date()} → {prices.index.max().date()} "
        f"({len(prices)} trading days)\n"
        "Sources: State Street NAV (SPY); Yahoo adj-close mirror (AAPL/MSFT/AMZN)\n"
        "Transform: daily logarithmic returns r_t = ln(S_t / S_{t-1})"
    )
    print(f"\nRegime summary ({PRIMARY_TICKER}):")
    spy = stats[stats["ticker"] == PRIMARY_TICKER]
    for _, row in spy.iterrows():
        print(
            f"  {REGIME_LABELS[row['regime']]}: "
            f"n={row['n_obs']}, ann_vol={row['ann_vol']:.2%}, "
            f"skew={row['skewness']:.2f}, excess_kurt={row['kurtosis']:.2f}"
        )
    print(f"\nFull summary written to {SUMMARY_STATS.relative_to(DATA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_research_figures(prices: pd.DataFrame | None = None) -> None:
    """Write 01–05 into docs/figures, including AMZN and evaluation-window bands."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if prices is None:
        prices = load_or_download(force=False)
    returns = compute_log_returns(prices)
    fig_long = build_figure_regime_long(returns)
    fig_stats = summary_statistics(fig_long)
    print(f"\nGenerating figures → {FIGURES_DIR}")
    for ticker in (PRIMARY_TICKER, *SECONDARY_TICKERS, *EXTRA_STUDY_TICKERS):
        plot_price_by_regime(prices, ticker)
    plot_aapl_msft_amzn_price_by_regime(prices)
    plot_return_distributions(fig_long)
    plot_annualized_volatility(fig_stats)
    plot_rolling_volatility_spy(returns)
    plot_all_tickers_normalized(prices)


def main() -> None:
    print("=" * 70)
    print("STEP 1 — DATA PREPARE (returns, regimes, figures)")
    print("=" * 70)

    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_or_download(force=False)
    returns = compute_log_returns(prices)
    regime_long = build_regime_long(returns)
    stats = summary_statistics(regime_long)

    returns.to_csv(LOG_RETURNS_ALL)
    print(f"✓ Saved: {LOG_RETURNS_ALL.relative_to(DATA_DIR)}")

    regime_long.to_csv(LOG_RETURNS_BY_REGIME, index=False)
    print(f"✓ Saved: {LOG_RETURNS_BY_REGIME.relative_to(DATA_DIR)}")

    stats.to_csv(SUMMARY_STATS, index=False)
    print(f"✓ Saved: {SUMMARY_STATS.relative_to(DATA_DIR)}")

    print("\nGenerating figures...")
    write_research_figures(prices)

    print("\nSanity checks:")
    checks = run_checks(prices, returns, regime_long, stats)
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    if not all(checks.values()):
        raise SystemExit("One or more sanity checks failed — inspect data before modeling.")

    print_data_description(prices, stats)
    print("\n✓ Step 1 equity prepare complete")


if __name__ == "__main__":
    main()
