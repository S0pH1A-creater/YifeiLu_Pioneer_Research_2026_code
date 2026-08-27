#!/usr/bin/env python3
"""Patch V4 GARCH notebooks for Duan (1995) GARCH-in-mean + LRNVR Q-paths."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "garch notebook"

FORMULAS_MD = r"""## 3. Estimation formulas (Duan GARCH(1,1) + LRNVR)

Source: Duan, J.-C. (1995), “The GARCH Option Pricing Model,” *Mathematical Finance* 5(1). Time unit is this notebook’s trading clock \(\Delta t=1/N_{\mathrm{days}}\). The one-period risk-free rate \(r_{f,t}\) is **observed** (FRED DGS3MO, decimal, divided by \(N_{\mathrm{days}}\)); it is not estimated.

### Physical measure \(P\) (GARCH-in-mean)

$$
\ln\frac{S_t}{S_{t-1}}=r_{f,t}+\lambda\sigma_t-\tfrac12\sigma_t^2+\sigma_t\varepsilon_t,\qquad \varepsilon_t\mid\mathcal F_{t-1}\sim N(0,1)
$$

$$
\sigma_t^2=\omega+\alpha\sigma_{t-1}^2\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2
$$

MLE on the lookback window jointly estimates \((\lambda,\omega,\alpha,\beta)\). \(\sigma_0\) is the last filtered conditional vol (else \(\sqrt{\omega/(1-\alpha-\beta)}\)). \(\lambda\) is the unit risk premium from returns + DGS3MO — not from option quotes.

### LRNVR

A measure \(Q\sim P\) satisfies the locally risk-neutral valuation relationship iff, one step ahead,

1. \(\mathbb E^Q[S_t/S_{t-1}\mid\mathcal F_{t-1}]=e^{r_{f,t}}\)
2. \(\mathrm{Var}^Q[\ln(S_t/S_{t-1})\mid\mathcal F_{t-1}]=\sigma_t^2\)

Then \(\xi_t=\varepsilon_t+\lambda\) with \(\xi_t\mid\mathcal F_{t-1}\sim N(0,1)\) under \(Q\).

### Risk-neutral measure \(Q\)

$$
\ln\frac{S_t}{S_{t-1}}=r_{f,t}-\tfrac12\sigma_t^2+\sigma_t\xi_t
$$

$$
\sigma_t^2=\omega+\alpha\sigma_{t-1}^2(\xi_{t-1}-\lambda)^2+\beta\sigma_{t-1}^2
$$

| Object | Under \(P\) | Under \(Q\) |
|--------|-------------|-------------|
| \(\omega,\alpha,\beta\) | estimated | unchanged |
| \(\sigma_0\) | last filter vol | unchanged (LRNVR 2) |
| \(\lambda\) | estimated in the mean | same number; used as \((\xi-\lambda)\) in the variance |
| Conditional mean | \(r_{f,t}+\lambda\sigma_t-\tfrac12\sigma_t^2\) | \(r_{f,t}-\tfrac12\sigma_t^2\) |

§4 graphs: \(\lambda,\omega,\alpha,\beta,\sigma_0\). §5 Monte Carlo uses **\(P\)**. §6 LSM uses **\(Q\)** only.
"""

CAL_HEAD = '''import sys
from pathlib import Path as _CalPath
_CAL_SCRIPTS = str((_CalPath("..") / "scripts").resolve())
if _CAL_SCRIPTS not in sys.path:
    sys.path.insert(0, _CAL_SCRIPTS)
from garch_duan_lrnvr import (
    estimate_garch_params as _estimate_garch_duan,
    load_rf_annual,
    q_persist,
    report_p_and_q,
)


def _rf_annual_series():
    return load_rf_annual(DATA, short_interval=int(N_DAYS) > 400)


def estimate_garch_params(log_rets: pd.Series, warm=None):
    """Duan GARCH-in-mean MLE: (λ, ω, α, β, σ0) from lookback returns + DGS3MO."""
    hat = _estimate_garch_duan(
        log_rets,
        _rf_annual_series(),
        n_days=int(N_DAYS),
        min_window=MIN_WINDOW,
        warm=warm,
    )
    n = int(log_rets.dropna().shape[0])
    if hat is None:
        return None
    hat["n"] = n
    return hat

'''

SIM_HEAD = '''import sys
from pathlib import Path as _SimPath
_SIM_SCRIPTS = str((_SimPath("..") / "scripts").resolve())
if _SIM_SCRIPTS not in sys.path:
    sys.path.insert(0, _SIM_SCRIPTS)
from garch_duan_lrnvr import simulate_garch_p, simulate_garch_q


def simulate_garch_rolling(steps, S0, n_paths, seed):
    """§5 physical-measure paths (Duan GARCH-in-mean)."""
    return simulate_garch_p(steps, S0, n_paths, seed, n_days=int(N_DAYS))


'''

RN_DAILY = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Duan LRNVR Q-paths to expiry (not μ → r)."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    steps = {
        "lambda": np.full(dte, float(p["lambda"]), dtype=float),
        "omega": np.full(dte, float(p["omega"]), dtype=float),
        "alpha": np.full(dte, float(p["alpha"]), dtype=float),
        "beta": np.full(dte, float(p["beta"]), dtype=float),
        "sigma0": np.full(dte, float(p["sigma0"]), dtype=float),
        "rf": np.full(dte, r, dtype=float),
    }
    return simulate_garch_q(steps, S0, n_paths, seed, n_days=int(N_DAYS))
'''

RN_MINUTE = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Duan LRNVR Q-paths to expiry (not μ → r). Daily LSM clock via _with_option_days."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    n_steps = int(getattr(row, "n_steps", 0)) or dte
    r = float(row.r)
    S0 = float(row.S_t)
    steps = {
        "lambda": np.full(n_steps, float(p["lambda"]), dtype=float),
        "omega": np.full(n_steps, float(p["omega"]), dtype=float),
        "alpha": np.full(n_steps, float(p["alpha"]), dtype=float),
        "beta": np.full(n_steps, float(p["beta"]), dtype=float),
        "sigma0": np.full(n_steps, float(p["sigma0"]), dtype=float),
        "rf": np.full(n_steps, r, dtype=float),
    }
    return simulate_garch_q(steps, S0, n_paths, seed, n_days=252)
'''


def _src(cell) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else str(s)


def _set(cell, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.split("\n")[:-1]] + (
        [text.split("\n")[-1] + "\n"] if text.endswith("\n") else []
    )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing block: {label}")
    return text.replace(old, new, 1)


def patch_intro(src: str) -> str:
    old = (
        "1. In each lookback window, estimate GARCH(1,1) parameters by **maximum likelihood** "
        "(returns → conditional \\(\\sigma_t\\) path → likelihood → best \\(\\hat\\mu,\\hat\\omega,\\hat\\alpha,\\hat\\beta\\). No jump parameters."
    )
    new = (
        "1. In each lookback window, estimate Duan GARCH-in-mean "
        "\\((\\lambda,\\omega,\\alpha,\\beta)\\) by **maximum likelihood** on returns with observed "
        "\\(r_{f,t}\\) from DGS3MO. No jump parameters."
    )
    if old in src:
        src = src.replace(old, new, 1)
    src = src.replace(
        "risk-neutral paths from the same simulator",
        "Duan LRNVR $Q$-paths from the same simulator",
    )
    return src


def patch_calibrate_daily(src: str) -> str:
    start = src.index("def _unpack_garch")
    end = src.index("def _slice_window")
    src = src[:start] + CAL_HEAD + src[end:]

    src = _replace_once(
        src,
        """        mu, omega, alpha, beta, sigma0, n = estimate_garch_params(
            window, warm=warm
        )
        if n < MIN_WINDOW or not np.isfinite(mu):
            continue
        warm = (mu / N_DAYS, omega, alpha, beta)
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "n_days": n,
            "mu": mu,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "sigma0": sigma0,
        })""",
        """        hat = estimate_garch_params(window, warm=warm)
        if hat is None or hat["n"] < MIN_WINDOW or not np.isfinite(hat["lambda"]):
            continue
        if not hat.get("q_stationary", True):
            print(
                f"warning: Q-GARCH not stationary at {pd.Timestamp(t_u).date()} "
                f"(α(1+λ²)+β={q_persist(hat['alpha'], hat['beta'], hat['lambda']):.4f})",
                flush=True,
            )
        warm = (hat["lambda"], hat["omega"], hat["alpha"], hat["beta"])
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "n_days": hat["n"],
            "lambda": hat["lambda"],
            "omega": hat["omega"],
            "alpha": hat["alpha"],
            "beta": hat["beta"],
            "sigma0": hat["sigma0"],
            "mu_p": hat["mu_p"],
            "rf_mean": hat["rf_mean"],
            "persist_p": hat["persist_p"],
            "persist_q": hat["persist_q"],
            "uncond_var_p": hat["uncond_var_p"],
            "uncond_var_q": hat["uncond_var_q"],
            "q_stationary": hat["q_stationary"],
        })""",
        "daily calibrate_ticker rows",
    )

    src = _replace_once(
        src,
        '    cols = ["mu", "omega", "alpha", "beta", "sigma0"]',
        '    cols = ["lambda", "omega", "alpha", "beta", "sigma0", "rf"]',
        "param_schedule cols",
    )
    src = _replace_once(
        src,
        """    arrs = {c: cal[c].to_numpy(dtype=float) for c in cols}
    steps = {c: np.empty(n_steps, dtype=float) for c in cols}

    for i in range(n_steps):
        idx = np.searchsorted(cal_dates, np.datetime64(dates[i]), side="right") - 1
        if idx < 0:
            idx = 0
        for c in cols:
            steps[c][i] = arrs[c][idx]""",
        """    param_cols = ["lambda", "omega", "alpha", "beta", "sigma0"]
    arrs = {c: cal[c].to_numpy(dtype=float) for c in param_cols}
    steps = {c: np.empty(n_steps, dtype=float) for c in param_cols}
    steps["rf"] = np.empty(n_steps, dtype=float)
    rf_ann = _rf_annual_series()
    rf_on_dates = rf_ann.reindex(pd.DatetimeIndex(dates), method="ffill").bfill()
    rf_vals = rf_on_dates.to_numpy(dtype=float)

    for i in range(n_steps):
        idx = np.searchsorted(cal_dates, np.datetime64(dates[i]), side="right") - 1
        if idx < 0:
            idx = 0
        for c in param_cols:
            steps[c][i] = arrs[c][idx]
        steps["rf"][i] = float(rf_vals[i]) if np.isfinite(rf_vals[i]) else float(rf_ann.dropna().iloc[-1])""",
        "param_schedule loop",
    )

    src = src.replace(
        '"""Rolling-parameter graphs: μ, ω, α, β, σ₀."""',
        '"""Rolling-parameter graphs: λ, ω, α, β, σ₀."""',
    )
    src = _replace_once(
        src,
        """    panels = [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("omega", "ω̂", "GARCH intercept"),
        ("alpha", "α̂", "ARCH reaction"),
        ("beta", "β̂", "GARCH persistence"),
        ("sigma0", "σ̂₀", "Initial conditional vol"),
    ]""",
        """    panels = [
        ("lambda", "λ̂ (unit risk premium)", "Duan unit risk premium"),
        ("omega", "ω̂", "GARCH intercept"),
        ("alpha", "α̂", "ARCH reaction"),
        ("beta", "β̂", "GARCH persistence"),
        ("sigma0", "σ̂₀", "Initial conditional vol"),
    ]""",
        "daily panels",
    )
    src = src.replace('            if col == "mu":', '            if col == "lambda":')
    src = src.replace(
        "Shows μ̂, ω̂, α̂, β̂, σ̂₀. No Monte Carlo here.",
        "Shows λ̂, ω̂, α̂, β̂, σ̂₀. No Monte Carlo here. P and Q tables after Reestimate.",
    )

    extra = '''
        p_rows, q_rows = [], []
        for t in TICKERS:
            if rolling[t] is None or len(rolling[t]) == 0:
                continue
            last = rolling[t].iloc[-1]
            p_tbl, q_tbl = report_p_and_q(last)
            p_tbl["ticker"] = t
            q_tbl["ticker"] = t
            p_rows.append(p_tbl)
            q_rows.append(q_tbl)
        if p_rows:
            display(Markdown("### Physical-measure (P) parameters"))
            display(pd.DataFrame(p_rows))
            display(Markdown("### Risk-neutral (Q) dynamics — Duan LRNVR"))
            display(pd.DataFrame(q_rows))
'''
    marker = "        plot_rolling_paths(rolling, window_label, rolling_mode)\n"
    if marker not in src:
        raise RuntimeError("plot_rolling_paths call missing")
    src = src.replace(marker, marker + extra, 1)
    return src


def patch_calibrate_minute(src: str) -> str:
    start = src.index("def _unpack_garch")
    end = src.index("def _slice_window")
    src = src[:start] + CAL_HEAD + src[end:]
    src = _replace_once(
        src,
        """        mu, omega, alpha, beta, sigma0, n = estimate_garch_params(
            window, warm=warm
        )
        if n < MIN_WINDOW or not np.isfinite(mu):
            continue
        warm = (mu / N_DAYS, omega, alpha, beta)
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "n_days": n,
            "mu": mu,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "sigma0": sigma0,
        })""",
        """        hat = estimate_garch_params(window, warm=warm)
        if hat is None or hat["n"] < MIN_WINDOW or not np.isfinite(hat["lambda"]):
            continue
        if not hat.get("q_stationary", True):
            print(
                f"warning: Q-GARCH not stationary at {pd.Timestamp(t_u)} "
                f"(α(1+λ²)+β={q_persist(hat['alpha'], hat['beta'], hat['lambda']):.4f})",
                flush=True,
            )
        warm = (hat["lambda"], hat["omega"], hat["alpha"], hat["beta"])
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "n_days": hat["n"],
            "lambda": hat["lambda"],
            "omega": hat["omega"],
            "alpha": hat["alpha"],
            "beta": hat["beta"],
            "sigma0": hat["sigma0"],
            "mu_p": hat["mu_p"],
            "rf_mean": hat["rf_mean"],
            "persist_p": hat["persist_p"],
            "persist_q": hat["persist_q"],
            "uncond_var_p": hat["uncond_var_p"],
            "uncond_var_q": hat["uncond_var_q"],
            "q_stationary": hat["q_stationary"],
        })""",
        "minute calibrate_ticker rows",
    )
    src = _replace_once(
        src,
        '    cols = ["mu", "omega", "alpha", "beta", "sigma0"]',
        '    cols = ["lambda", "omega", "alpha", "beta", "sigma0", "rf"]',
        "minute param_schedule cols",
    )
    src = _replace_once(
        src,
        """    arrs = {c: cal[c].to_numpy(dtype=float) for c in cols}
    steps = {c: np.empty(n_steps, dtype=float) for c in cols}

    for i in range(n_steps):
        idx = np.searchsorted(cal_dates, np.datetime64(dates[i]), side="right") - 1
        if idx < 0:
            idx = 0
        for c in cols:
            steps[c][i] = arrs[c][idx]""",
        """    param_cols = ["lambda", "omega", "alpha", "beta", "sigma0"]
    arrs = {c: cal[c].to_numpy(dtype=float) for c in param_cols}
    steps = {c: np.empty(n_steps, dtype=float) for c in param_cols}
    steps["rf"] = np.empty(n_steps, dtype=float)
    rf_ann = _rf_annual_series()
    rf_on_dates = rf_ann.reindex(pd.DatetimeIndex(dates).normalize(), method="ffill").bfill()
    rf_vals = rf_on_dates.to_numpy(dtype=float)

    for i in range(n_steps):
        idx = np.searchsorted(cal_dates, np.datetime64(dates[i]), side="right") - 1
        if idx < 0:
            idx = 0
        for c in param_cols:
            steps[c][i] = arrs[c][idx]
        steps["rf"][i] = float(rf_vals[min(i, len(rf_vals) - 1)]) if len(rf_vals) else 0.0""",
        "minute param_schedule loop",
    )
    src = src.replace(
        '"""Rolling-parameter graphs: μ, ω, α, β, σ₀."""',
        '"""Rolling-parameter graphs: λ, ω, α, β, σ₀."""',
    )
    src = src.replace('("mu", "μ̂ (annual)", "Estimated drift")', '("lambda", "λ̂ (unit risk premium)", "Duan unit risk premium")')
    src = src.replace('            if col == "mu":', '            if col == "lambda":')
    src = src.replace(
        "Shows μ̂, ω̂, α̂, β̂, σ̂₀. No Monte Carlo here.",
        "Shows λ̂, ω̂, α̂, β̂, σ̂₀. No Monte Carlo here. P and Q tables after Reestimate.",
    )
    extra = '''
        p_rows, q_rows = [], []
        for t in TICKERS:
            if rolling[t] is None or len(rolling[t]) == 0:
                continue
            last = rolling[t].iloc[-1]
            p_tbl, q_tbl = report_p_and_q(last)
            p_tbl["ticker"] = t
            q_tbl["ticker"] = t
            p_rows.append(p_tbl)
            q_rows.append(q_tbl)
        if p_rows:
            display(Markdown("### Physical-measure (P) parameters"))
            display(pd.DataFrame(p_rows))
            display(Markdown("### Risk-neutral (Q) dynamics — Duan LRNVR"))
            display(pd.DataFrame(q_rows))
'''
    marker = "        plot_rolling_paths(rolling, window_label, rolling_mode)\n"
    if marker in src:
        src = src.replace(marker, marker + extra, 1)
    return src


def patch_sim(src: str) -> str:
    start = src.index("def simulate_garch_rolling")
    # cut through the function, keep _show_fig onward
    rest_marker = "def _show_fig"
    end = src.index(rest_marker, start)
    return src[:start] + SIM_HEAD + src[end:]


def patch_rn(src: str, minute: bool) -> str:
    start = src.index("def _rn_paths_for_contract")
    # next top-level after the function: _STOP_TICKERS
    end = src.index("\n_STOP_TICKERS", start)
    body = RN_MINUTE if minute else RN_DAILY
    return src[:start] + body + src[end:]


def patch_s6_md(src: str) -> str:
    return src.replace(
        "Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; vol from §4 for that ticker).",
        "Paths for pricing are **risk-neutral under Duan (1995) LRNVR** "
        "(mean $r_f-\\tfrac12\\sigma_t^2$, variance shock $\\xi_t-\\lambda$; "
        "$(\\omega,\\alpha,\\beta,\\lambda,\\sigma_0)$ from §4).",
    ).replace(
        "risk-neutral paths from the same simulator",
        "Duan LRNVR $Q$-paths from the same simulator",
    )


def patch_nb(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    minute = "1min" in path.name
    for cell in nb["cells"]:
        src = _src(cell)
        if cell["cell_type"] == "markdown" and src.startswith("# GARCH"):
            _set(cell, patch_intro(src))
        elif cell["cell_type"] == "markdown" and src.startswith("## 3. Estimation formulas"):
            _set(cell, FORMULAS_MD)
        elif cell["cell_type"] == "markdown" and src.startswith("## 6. Optimal stopping"):
            _set(cell, patch_s6_md(src))
        elif cell["cell_type"] == "code" and "def calibrate_ticker" in src and "def _unpack_garch" in src:
            _set(cell, patch_calibrate_minute(src) if minute else patch_calibrate_daily(src))
        elif cell["cell_type"] == "code" and "def simulate_garch_rolling" in src:
            _set(cell, patch_sim(src))
        elif cell["cell_type"] == "code" and "def _rn_paths_for_contract" in src:
            _set(cell, patch_rn(src, minute=minute))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"patched {path.name}", flush=True)


def main() -> int:
    nbs = sorted(NB_DIR.glob("*_garch.ipynb"))
    if not nbs:
        raise SystemExit(f"no notebooks in {NB_DIR}")
    for p in nbs:
        patch_nb(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
