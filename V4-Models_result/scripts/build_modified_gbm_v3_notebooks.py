#!/usr/bin/env python3
"""Clone current Modified GBM notebooks into Modified GBM v3.

v3 replaces the 1-lag {U, D} chain with an order-3 sign history:
UUU, UUD, UDU, UDD, DUU, DUD, DDU, DDD (8 states). The next bar is still
U or D. Magnitudes stay the original split-normal |N(μ, σ)| draws.

Does not overwrite modified gbm notebook/ or modified gbm v2 notebook/.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "modified gbm notebook"
DST_DIR = ROOT / "modified gbm v3 notebook"

SRC_FILES = [
    "2008-2009_modified_gbm.ipynb",
    "2013-2014_modified_gbm.ipynb",
    "2018-2019_modified_gbm.ipynb",
    "2019-2020_modified_gbm.ipynb",
    "7d_1min_modified_gbm.ipynb",
    "1d_1min_modified_gbm.ipynb",
]

# Bit pack: 4*oldest + 2*mid + newest, U=1, D=0.
SIGN_STATES = ("DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU")
P_U_COLS = tuple(f"p_u_{name.lower()}" for name in SIGN_STATES)
STEP_COLS = list(P_U_COLS) + ["mu_u", "sig_u", "mu_d", "sig_d", "last_state"]

FORMULAS_MD = r"""## 3. Estimation formulas (Modified GBM v3)

Same magnitude stage as Modified GBM. Direction is an order-3 Markov chain on
the last three non-zero signs.

**Stage 1 — direction.** Let \(U=\{R_t>0\}\) and \(D=\{R_t<0\}\). The state is the last
three signs,

\[
s_{t-1}\in\{\mathrm{UUU},\mathrm{UUD},\mathrm{UDU},\mathrm{UDD},\mathrm{DUU},\mathrm{DUD},\mathrm{DDU},\mathrm{DDD}\}.
\]

From consecutive non-zero returns in the lookback, Laplace-smooth the eight coins

\[
\hat P(U\mid s),\qquad \hat P(D\mid s)=1-\hat P(U\mid s).
\]

After the next sign \(X\in\{U,D\}\) is drawn, the state slides: drop the oldest sign
and append \(X\). Example: \(\mathrm{UUD}\) then \(U\) becomes \(\mathrm{UDU}\).

**Stage 2 — magnitude.** After the direction is chosen, draw the size from a separate
normal for up and down moves and take it positive:

\[
m_t\mid U \sim \bigl|N(\mu_U,\sigma_U^2)\bigr|,\qquad
m_t\mid D \sim \bigl|N(\mu_D,\sigma_D^2)\bigr|.
\]

\(\mu_U,\sigma_U\) (resp. \(\mu_D,\sigma_D\)) are the sample mean and standard deviation of \(|R|\)
on up (resp. down) bars.

**Stage 3 — price.** \(R_t=+m_t\) on \(U\) and \(-m_t\) on \(D\), then \(S_{t+1}=S_t e^{R_t}\).

| Parameter | Role |
|-----------|------|
| \(P(U\mid \mathrm{UUU}),\ldots,P(U\mid \mathrm{DDD})\) | Next-bar up probability given the last three signs |
| \(\mu_U,\sigma_U\) | Mean / SD of up magnitudes |
| \(\mu_D,\sigma_D\) | Mean / SD of down magnitudes |

Risk-neutral paths (§6) keep the calibrated transitions and magnitudes, then shift each step so \(E[e^{R_t}]=\exp(r\,\Delta t)\).
"""

ESTIMATE_FN = '''def estimate_modified_gbm(log_rets: pd.Series):
    """Order-3 sign history (UUU..DDD) + split-normal magnitudes."""
    x = log_rets.dropna().astype(float)
    x = x[np.isfinite(x)]
    n = int(x.shape[0])
    nz = x[x != 0.0]
    if n < 4 or int(nz.shape[0]) < 4:
        return None
    signed = nz.to_numpy(dtype=float)
    up = signed > 0.0
    bits = up.astype(np.intp)
    mag = np.abs(signed)
    n_from = np.zeros(8, dtype=float)
    n_to_u = np.zeros(8, dtype=float)
    for i in range(2, bits.size - 1):
        state = int((bits[i - 2] << 2) | (bits[i - 1] << 1) | bits[i])
        n_from[state] += 1.0
        n_to_u[state] += float(bits[i + 1])
    names = ("ddd", "ddu", "dud", "duu", "udd", "udu", "uud", "uuu")
    coins = {}
    for k, name in enumerate(names):
        coins[f"p_u_{name}"] = float((n_to_u[k] + 0.5) / (n_from[k] + 1.0))

    def _mu_sig(arr):
        if arr.size >= 2:
            mu = float(arr.mean())
            sig = float(arr.std(ddof=1))
        elif arr.size == 1:
            mu = float(arr[0])
            sig = float(np.median(mag)) if mag.size else 1e-6
        else:
            mu = float(np.median(mag)) if mag.size else 1e-6
            sig = mu
        if not np.isfinite(mu) or mu < 0:
            mu = abs(mu) if np.isfinite(mu) else 1e-6
        if not np.isfinite(sig) or sig <= 0:
            sig = 1e-6
        return mu, sig

    mu_u, sig_u = _mu_sig(mag[up])
    mu_d, sig_d = _mu_sig(mag[~up])
    last_state = float((int(bits[-3]) << 2) | (int(bits[-2]) << 1) | int(bits[-1]))
    out = {
        "n_days": n,
        **coins,
        "mu_u": mu_u,
        "sig_u": sig_u,
        "mu_d": mu_d,
        "sig_d": sig_d,
        "last_state": last_state,
        "last_up": 1.0 if bool(up[-1]) else 0.0,
        "p_u": float(up.mean()),
    }
    return out


'''

PARAM_SCHEDULE_NEW = '''def param_schedule_for_steps(ticker: str, cal_table: pd.DataFrame):
    hist = _mc_time_grid(period_prices[ticker], N_STEPS)
    dates = hist.index
    n_steps = len(dates) - 1
    cal = cal_table.sort_values("date").reset_index(drop=True)
    cal_dates = pd.to_datetime(cal["date"]).to_numpy()
    cols = [
        "p_u_ddd", "p_u_ddu", "p_u_dud", "p_u_duu",
        "p_u_udd", "p_u_udu", "p_u_uud", "p_u_uuu",
        "mu_u", "sig_u", "mu_d", "sig_d", "last_state",
    ]
    arrs = {c: cal[c].to_numpy(dtype=float) for c in cols}
    steps = {c: np.empty(n_steps, dtype=float) for c in cols}
    for i in range(n_steps):
        idx = np.searchsorted(cal_dates, np.datetime64(dates[i]), side="right") - 1
        if idx < 0:
            idx = 0
        for c in cols:
            steps[c][i] = arrs[c][idx]
    return dates, steps, float(hist.iloc[0]), hist
'''

PLOT_YEARLY = '''def plot_rolling_paths(rolling_dict: dict, window_label: str, rolling_mode: str):
    """Rolling Modified GBM v3 parameters (graphs only)."""
    panels = [
        (("p_u_uuu", "p_u_ddd"), r"$\\hat P(U|UUU)$ / $\\hat P(U|DDD)$", "After a three-bar run"),
        (("p_u_udu", "p_u_dud"), r"$\\hat P(U|UDU)$ / $\\hat P(U|DUD)$", "After an alternating triple"),
        (("mu_u", "mu_d"), r"$\\hat\\mu_U$ / $\\hat\\mu_D$", "Magnitude means (per bar)"),
        (("sig_u", "sig_d"), r"$\\hat\\sigma_U$ / $\\hat\\sigma_D$", "Magnitude SDs (per bar)"),
    ]
    with plt.ioff():
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for ax, ((c1, c2), ylab, title) in zip(axes, panels):
            for t in TICKERS:
                r = rolling_dict[t]
                x = pd.to_datetime(r["date"])
                mark = "o" if len(r) < 40 else None
                ax.plot(x, r[c1], lw=1.2, label=f"{t} {c1}", color=COLORS[t], marker=mark, ms=3)
                ax.plot(x, r[c2], lw=1.0, ls="--", color=COLORS[t], alpha=0.75, marker=mark, ms=3)
            ax.set_ylabel(ylab)
            ax.set_title(f"{title} — {rolling_mode}, lookback {window_label}")
            ax.legend(frameon=False, ncol=3, fontsize=8)
        axes[-1].set_xlabel("Date")
        fig.tight_layout()
    _show_fig(fig)
'''

PLOT_INTRADAY = '''def plot_rolling_paths(rolling_dict: dict, window_label: str, rolling_mode: str):
    """Rolling Modified GBM v3 parameters (graphs only)."""
    panels = [
        (("p_u_uuu", "p_u_ddd"), r"$\\hat P(U|UUU)$ / $\\hat P(U|DDD)$", "After a three-bar run"),
        (("p_u_udu", "p_u_dud"), r"$\\hat P(U|UDU)$ / $\\hat P(U|DUD)$", "After an alternating triple"),
        (("mu_u", "mu_d"), r"$\\hat\\mu_U$ / $\\hat\\mu_D$", "Magnitude means (per bar)"),
        (("sig_u", "sig_d"), r"$\\hat\\sigma_U$ / $\\hat\\sigma_D$", "Magnitude SDs (per bar)"),
    ]
    with plt.ioff():
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for ax, ((c1, c2), ylab, title) in zip(axes, panels):
            for t in TICKERS:
                r = rolling_dict[t]
                x = np.array([
                    period_prices.index.get_indexer([pd.Timestamp(d)], method="pad")[0]
                    for d in r["date"]
                ])
                mark = "o" if len(r) < 40 else None
                ax.plot(x, r[c1], lw=1.2, label=f"{t} {c1}", color=COLORS[t], marker=mark, ms=3)
                ax.plot(x, r[c2], lw=1.0, ls="--", color=COLORS[t], alpha=0.75, marker=mark, ms=3)
            ax.set_ylabel(ylab)
            ax.set_title(f"{title} — {rolling_mode}, lookback {window_label}")
            ax.legend(frameon=False, ncol=3, fontsize=8)
        axes[-1].set_xlabel("trading minute")
        fig.tight_layout()
    _show_fig(fig)
'''

SIMULATE_FN = '''def simulate_modified_gbm_rolling(steps, S0, n_paths, seed, rf=None):
    """Modified GBM v3: order-3 sign history, split-normal size, S * exp(r).

    `rf` is an annual risk-free rate. When set, each step is shifted so
    E[exp(r_t)] = exp(rf / N_DAYS). Leave `rf=None` for P-measure paths.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(steps["p_u_uuu"])
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    state = np.full(n_paths, int(np.clip(round(float(steps["last_state"][0])), 0, 7)), dtype=np.intp)
    rf_step = None if rf is None else float(rf) / float(N_DAYS)
    p_keys = (
        "p_u_ddd", "p_u_ddu", "p_u_dud", "p_u_duu",
        "p_u_udd", "p_u_udu", "p_u_uud", "p_u_uuu",
    )

    for i in range(n_steps):
        p_u = np.array([float(np.clip(steps[k][i], 0.0, 1.0)) for k in p_keys], dtype=float)
        mu_u = float(steps["mu_u"][i])
        mu_d = float(steps["mu_d"][i])
        sig_u = max(float(steps["sig_u"][i]), 1e-12)
        sig_d = max(float(steps["sig_d"][i]), 1e-12)
        p_up = p_u[state]
        up = rng.random(n_paths) < p_up
        state = ((state << 1) | up.astype(np.intp)) & 7
        mag = np.empty(n_paths, dtype=float)
        n_up = int(np.count_nonzero(up))
        n_dn = n_paths - n_up
        if n_up:
            mag[up] = np.abs(rng.normal(mu_u, sig_u, size=n_up))
        if n_dn:
            mag[~up] = np.abs(rng.normal(mu_d, sig_d, size=n_dn))
        mag = np.maximum(mag, 1e-16)
        r = np.where(up, mag, -mag)
        if rf_step is not None:
            mx = float(np.mean(np.exp(r)))
            r = r + (rf_step - np.log(max(mx, 1e-300)))
        paths[:, i + 1] = paths[:, i] * np.exp(r)
    return paths
'''

RN_STEPS = '''    n_steps = int(getattr(row, "n_steps", 0)) or int(dte)
    r = float(row.r)
    S0 = float(row.S_t)
    steps = {
        "p_u_ddd": np.full(n_steps, float(p["p_u_ddd"]), dtype=float),
        "p_u_ddu": np.full(n_steps, float(p["p_u_ddu"]), dtype=float),
        "p_u_dud": np.full(n_steps, float(p["p_u_dud"]), dtype=float),
        "p_u_duu": np.full(n_steps, float(p["p_u_duu"]), dtype=float),
        "p_u_udd": np.full(n_steps, float(p["p_u_udd"]), dtype=float),
        "p_u_udu": np.full(n_steps, float(p["p_u_udu"]), dtype=float),
        "p_u_uud": np.full(n_steps, float(p["p_u_uud"]), dtype=float),
        "p_u_uuu": np.full(n_steps, float(p["p_u_uuu"]), dtype=float),
        "mu_u": np.full(n_steps, float(p["mu_u"]), dtype=float),
        "sig_u": np.full(n_steps, float(p["sig_u"]), dtype=float),
        "mu_d": np.full(n_steps, float(p["mu_d"]), dtype=float),
        "sig_d": np.full(n_steps, float(p["sig_d"]), dtype=float),
        "last_state": np.full(n_steps, float(p["last_state"]), dtype=float),
    }
    return simulate_modified_gbm_rolling(steps, S0, n_paths, seed, rf=r)
'''

ROLLING_MD = r"""# Modified GBM v3 notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Model

1. **Direction** — order-3 Markov chain on the last three signs. Eight states: UUU, UUD, UDU, UDD, DUU, DUD, DDU, DDD. The next bar is U with \(P(U\mid s)\) and D with \(1-P(U\mid s)\). The state then slides one step.
2. **Magnitude** — positive size from \(N(\mu_U,\sigma_U^2)\) or \(N(\mu_D,\sigma_D^2)\), taken absolute. Same as original Modified GBM.
3. **Price** — \(S_{t+1}=S_t e^{R_t}\) with \(R_t=+m_t\) (up) or \(R_t=-m_t\) (down).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** v3 parameters using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Files

- `2008-2009_modified_gbm_v3.ipynb`
- `2013-2014_modified_gbm_v3.ipynb`
- `2018-2019_modified_gbm_v3.ipynb`
- `2019-2020_modified_gbm_v3.ipynb`
- `7d_1min_modified_gbm_v3.ipynb`
- `1d_1min_modified_gbm_v3.ipynb`
"""

SPEC_MD = r"""# Modified GBM v3 — model specification

**This notebook is the documentation source of truth.**

Modified GBM v3 sits beside the original Modified GBM and v2. Sizes are unchanged from v1 (\(|N(\mu,\sigma)|\)). The only change is the sign chain: last three signs, not last one.

## Symbols

| Symbol | Meaning |
|--------|---------|
| \(R_t=\ln(S_t/S_{t-1})\) | log-return |
| \(U,D\) | up / down |
| \(s\in\{\mathrm{UUU},\ldots,\mathrm{DDD}\}\) | last three non-zero signs (8 states) |
| \(P(U\mid s)\) | next-bar up coin given \(s\) |
| \(\mu_U,\sigma_U,\mu_D,\sigma_D\) | split-normal size parameters |

## State update

Encode \(U=1\), \(D=0\). Pack \(s=4s_{t-2}+2s_{t-1}+s_t\). After drawing \(X\in\{0,1\}\),

\[
s \leftarrow (2s+X)\bmod 8.
\]

Laplace: \(\hat P(U\mid s)=(n_{s\to U}+0.5)/(n_s+1)\).

## Price

\[
R_t=\begin{cases}+|N(\mu_U,\sigma_U^2)| & U\\-|N(\mu_D,\sigma_D^2)| & D\end{cases},\qquad S_{t+1}=S_t e^{R_t}
\]

\(Q\) still shifts each step so \(E[e^{R_t}]=e^{r_f\Delta t}\).

## Code

`V4-Models_result/modified gbm v3 notebook/20*_modified_gbm_v3.ipynb`

Functions: `estimate_modified_gbm`, `calibrate_ticker`, `simulate_modified_gbm_rolling`.
"""


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_src(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.split("\n")[:-1]] + (
        [text.split("\n")[-1] + "\n"] if text.endswith("\n") else []
    )
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def _replace_fn(text: str, name: str, new_fn: str) -> str:
    token = f"def {name}("
    start = text.index(token)
    nxt = text.find("\ndef ", start + 1)
    if nxt < 0:
        raise RuntimeError(f"could not find end of {name}")
    body = new_fn if new_fn.endswith("\n") else new_fn + "\n"
    return text[:start] + body + "\n" + text[nxt + 1 :]


def _replace_rn(text: str) -> str:
    marker = '    if dte < 2:\n        raise ValueError("dte must be >= 2")\n'
    start = text.index(marker) + len(marker)
    end = text.index("\n\n_STOP_TICKERS")
    return text[:start] + RN_STEPS + text[end:]


def transform(nb: dict, *, intraday: bool) -> dict:
    cells = nb["cells"]
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    c0 = _src(cells[0])
    c0 = c0.replace("Modified GBM calibration", "Modified GBM v3 calibration")
    c0 = c0.replace(
        r"re-estimate the four transition probabilities and \((\mu_U,\sigma_U,\mu_D,\sigma_D)\) from the current window",
        r"re-estimate the eight \(P(U\mid s)\) coins for \(s\in\{\mathrm{UUU},\ldots,\mathrm{DDD}\}\) and \((\mu_U,\sigma_U,\mu_D,\sigma_D)\) from the current window",
    )
    _set_src(cells[0], c0)

    _set_src(cells[7], FORMULAS_MD)

    for idx in (8, 10, 12):
        t = _src(cells[idx])
        t = t.replace("Modified GBM", "Modified GBM v3")
        _set_src(cells[idx], t)

    cal = _src(cells[9])
    if "def estimate_modified_gbm" not in cal:
        raise RuntimeError("estimate_modified_gbm missing in source notebook")
    cal = _replace_fn(cal, "estimate_modified_gbm", ESTIMATE_FN)
    cal = _replace_fn(cal, "param_schedule_for_steps", PARAM_SCHEDULE_NEW)
    plot = PLOT_INTRADAY if intraday else PLOT_YEARLY
    cal = _replace_fn(cal, "plot_rolling_paths", plot)
    _set_src(cells[9], cal)

    mc = _src(cells[11])
    mc = _replace_fn(mc, "simulate_modified_gbm_rolling", SIMULATE_FN)
    _set_src(cells[11], mc)

    stop = _src(cells[13])
    stop = stop.replace("### Modified GBM — LSM", "### Modified GBM v3 — LSM")
    stop = stop.replace("Modified GBM optimal stopping", "Modified GBM v3 optimal stopping")
    stop = _replace_rn(stop)
    _set_src(cells[13], stop)
    return nb


def write_spec_notebook(path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "mgbm-v3-spec",
                "metadata": {},
                "source": [ln + "\n" for ln in SPEC_MD.strip().split("\n")],
            }
        ],
    }
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for name in SRC_FILES:
        src = SRC_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = DST_DIR / name.replace("_modified_gbm.ipynb", "_modified_gbm_v3.ipynb")
        nb = json.loads(src.read_text())
        transform(nb, intraday=("1min" in name))
        dst.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {dst.relative_to(ROOT)}")
    (DST_DIR / "ROLLING_CALIBRATION.md").write_text(ROLLING_MD)
    (DST_DIR / "MODIFIED_GBM_V3.md").write_text(SPEC_MD)
    write_spec_notebook(DST_DIR / "modified_gbm_v3_model.ipynb")
    print("wrote modified gbm v3 notebook/ROLLING_CALIBRATION.md")
    print("wrote modified gbm v3 notebook/MODIFIED_GBM_V3.md")
    print("wrote modified gbm v3 notebook/modified_gbm_v3_model.ipynb")


if __name__ == "__main__":
    main()
