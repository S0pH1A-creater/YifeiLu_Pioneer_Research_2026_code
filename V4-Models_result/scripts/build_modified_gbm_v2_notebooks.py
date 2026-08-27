#!/usr/bin/env python3
"""Clone current Modified GBM notebooks into Modified GBM v2.

v2 keeps the 1-lag sign chain. Sizes are moment-matched lognormals (Way B).
A second 1-lag chain (calm / wild) selects which (μ, σ) pair is used.
Does not overwrite modified gbm notebook/.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "modified gbm notebook"
DST_DIR = ROOT / "modified gbm v2 notebook"

SRC_FILES = [
    "2008-2009_modified_gbm.ipynb",
    "2013-2014_modified_gbm.ipynb",
    "2018-2019_modified_gbm.ipynb",
    "2019-2020_modified_gbm.ipynb",
    "7d_1min_modified_gbm.ipynb",
    "1d_1min_modified_gbm.ipynb",
]

STEP_COLS = [
    "p_uu",
    "p_du",
    "p_ud",
    "p_dd",
    "p_hh",
    "p_ll",
    "p_lh",
    "p_hl",
    "mu_u_l",
    "sig_u_l",
    "mu_u_h",
    "sig_u_h",
    "mu_d_l",
    "sig_d_l",
    "mu_d_h",
    "sig_d_h",
    "last_up",
    "last_wild",
]

FORMULAS_MD = r"""## 3. Estimation formulas (Modified GBM v2)

Same Markov direction as Modified GBM. Magnitudes are lognormal (always positive) and the
lognormal \((\mu,\sigma)\) pair is selected by a second 1-lag calm / wild chain.

**Stage 1 — direction.** Let \(U=\{R_t>0\}\) and \(D=\{R_t<0\}\). Laplace-smoothed

\[
\hat P(U\mid U),\ \hat P(D\mid U),\ \hat P(U\mid D),\ \hat P(D\mid D).
\]

**Stage 2 — calm / wild.** \(R_t=\ln(S_t/S_{t-1})\). A bar is wild (\(H\)) if \(|R|\) is above the
lookback median of \(|R|\), else calm (\(L\)). Laplace-smoothed

\[
\hat P(H\mid H),\ \hat P(L\mid H),\ \hat P(H\mid L),\ \hat P(L\mid L).
\]

**Stage 3 — size (Way B).** In each of the four buckets (up-calm, up-wild, down-calm, down-wild)
let \(m=\mathrm{mean}(|R|)\) and \(v=\mathrm{var}(|R|)\). Then

\[
\sigma^2=\log(1+v/m^2),\qquad \mu=\log(m)-\tfrac12\sigma^2,\qquad
\mathrm{size}=\exp\bigl(N(\mu,\sigma^2)\bigr).
\]

A bucket with fewer than two observations reuses the same-sign other regime, else the pooled same-sign moments.

**Stage 4 — price.** \(R_t=+\mathrm{size}\) on \(U\) and \(-\mathrm{size}\) on \(D\), then \(S_{t+1}=S_t e^{R_t}\).

**Stage 5 — \(Q\) (size-only).** Risk-neutral paths keep the \(P\)-measure sign coins and calm/wild coins.
If \(R^P=\pm\mathrm{size}\), choose one scale \(\lambda>0\) on that step so

\[
R^Q=\lambda R^P,\qquad \widehat{\mathbb{E}}[e^{R^Q}]=e^{r_f\Delta t}.
\]

Signs cannot flip. If no \(\lambda>0\) exists (a path cloud with no up move), fall back to the old additive shift.
"""

ESTIMATE_FN = '''def estimate_modified_gbm(log_rets: pd.Series):
    """Markov direction + lognormal sizes with a calm/wild Markov selector."""
    eps = 1e-8
    x = log_rets.dropna().astype(float)
    x = x[np.isfinite(x)]
    n = int(x.shape[0])
    nz = x[x != 0.0]
    if n < 3 or int(nz.shape[0]) < 3:
        return None
    signed = nz.to_numpy(dtype=float)
    mag = np.maximum(np.abs(signed), eps)
    up = signed > 0.0
    wild = mag > float(np.median(mag))

    prev_u, curr_u = up[:-1], up[1:]
    n_from_u = int(prev_u.sum())
    n_from_d = int((~prev_u).sum())
    n_uu = int((prev_u & curr_u).sum())
    n_dd = int((~prev_u & ~curr_u).sum())
    p_uu = (n_uu + 0.5) / (n_from_u + 1.0)
    p_dd = (n_dd + 0.5) / (n_from_d + 1.0)
    p_du = 1.0 - p_uu
    p_ud = 1.0 - p_dd

    prev_h, curr_h = wild[:-1], wild[1:]
    n_from_h = int(prev_h.sum())
    n_from_l = int((~prev_h).sum())
    n_hh = int((prev_h & curr_h).sum())
    n_ll = int((~prev_h & ~curr_h).sum())
    p_hh = (n_hh + 0.5) / (n_from_h + 1.0)
    p_ll = (n_ll + 0.5) / (n_from_l + 1.0)
    p_hl = 1.0 - p_hh
    p_lh = 1.0 - p_ll

    def _lognormal_params(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return None
        m = float(np.mean(arr))
        if not np.isfinite(m) or m < eps:
            m = eps
        v = float(np.var(arr, ddof=1))
        if not np.isfinite(v) or v < 0.0:
            v = 0.0
        sig2 = float(np.log(1.0 + v / (m * m)))
        sig = float(np.sqrt(max(sig2, 0.0)))
        if sig <= 0.0:
            sig = 1e-12
        mu = float(np.log(m) - 0.5 * sig2)
        return mu, sig

    ul = _lognormal_params(mag[up & ~wild])
    uh = _lognormal_params(mag[up & wild])
    dl = _lognormal_params(mag[(~up) & ~wild])
    dh = _lognormal_params(mag[(~up) & wild])
    u_pool = _lognormal_params(mag[up])
    d_pool = _lognormal_params(mag[~up])
    all_pool = _lognormal_params(mag)

    def _fill(primary, same_sign, pooled):
        if primary is not None:
            return primary
        if same_sign is not None:
            return same_sign
        if pooled is not None:
            return pooled
        return (float(np.log(eps)), 1e-12)

    mu_u_l, sig_u_l = _fill(ul, uh, u_pool if u_pool is not None else all_pool)
    mu_u_h, sig_u_h = _fill(uh, ul, u_pool if u_pool is not None else all_pool)
    mu_d_l, sig_d_l = _fill(dl, dh, d_pool if d_pool is not None else all_pool)
    mu_d_h, sig_d_h = _fill(dh, dl, d_pool if d_pool is not None else all_pool)
    return {
        "n_days": n,
        "p_uu": float(p_uu),
        "p_du": float(p_du),
        "p_ud": float(p_ud),
        "p_dd": float(p_dd),
        "p_hh": float(p_hh),
        "p_ll": float(p_ll),
        "p_lh": float(p_lh),
        "p_hl": float(p_hl),
        "mu_u_l": float(mu_u_l),
        "sig_u_l": float(sig_u_l),
        "mu_u_h": float(mu_u_h),
        "sig_u_h": float(sig_u_h),
        "mu_d_l": float(mu_d_l),
        "sig_d_l": float(sig_d_l),
        "mu_d_h": float(mu_d_h),
        "sig_d_h": float(sig_d_h),
        "last_up": 1.0 if bool(up[-1]) else 0.0,
        "last_wild": 1.0 if bool(wild[-1]) else 0.0,
        "p_u": float(up.mean()),
        "p_h": float(wild.mean()),
    }


'''

PARAM_SCHEDULE_NEW = '''def param_schedule_for_steps(ticker: str, cal_table: pd.DataFrame):
    hist = _mc_time_grid(period_prices[ticker], N_STEPS)
    dates = hist.index
    n_steps = len(dates) - 1
    cal = cal_table.sort_values("date").reset_index(drop=True)
    cal_dates = pd.to_datetime(cal["date"]).to_numpy()
    cols = [
        "p_uu", "p_du", "p_ud", "p_dd", "p_hh", "p_ll", "p_lh", "p_hl",
        "mu_u_l", "sig_u_l", "mu_u_h", "sig_u_h",
        "mu_d_l", "sig_d_l", "mu_d_h", "sig_d_h",
        "last_up", "last_wild",
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
    """Rolling Modified GBM v2 parameters (graphs only)."""
    panels = [
        (("p_uu", "p_dd"), r"$\\hat P(U|U)$ / $\\hat P(D|D)$", "Direction persistence"),
        (("p_hh", "p_ll"), r"$\\hat P(H|H)$ / $\\hat P(L|L)$", "Calm / wild persistence"),
        (("mu_u_l", "mu_u_h"), r"$\\mu_{U,L}$ / $\\mu_{U,H}$", "Up log-size means (calm / wild)"),
        (("mu_d_l", "mu_d_h"), r"$\\mu_{D,L}$ / $\\mu_{D,H}$", "Down log-size means (calm / wild)"),
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
    """Rolling Modified GBM v2 parameters (graphs only)."""
    panels = [
        (("p_uu", "p_dd"), r"$\\hat P(U|U)$ / $\\hat P(D|D)$", "Direction persistence"),
        (("p_hh", "p_ll"), r"$\\hat P(H|H)$ / $\\hat P(L|L)$", "Calm / wild persistence"),
        (("mu_u_l", "mu_u_h"), r"$\\mu_{U,L}$ / $\\mu_{U,H}$", "Up log-size means (calm / wild)"),
        (("mu_d_l", "mu_d_h"), r"$\\mu_{D,L}$ / $\\mu_{D,H}$", "Down log-size means (calm / wild)"),
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

SIMULATE_FN = '''def _rn_size_scale(r_p, target):
    """Positive size scale λ with mean(exp(λ r_p)) = target. Keeps sign(r_p)."""
    r_p = np.asarray(r_p, dtype=float)
    target = float(target)
    if (not np.isfinite(target)) or target <= 0.0:
        return None
    log_t = float(np.log(target))
    if (not np.any(r_p > 0.0)) and log_t > 0.0:
        return None
    lam = 1.0
    for _ in range(40):
        x = lam * r_p
        m = float(np.max(x))
        e = np.exp(x - m)
        s = float(e.sum())
        lm = float(np.log(s / float(e.size)) + m)
        f = lm - log_t
        if abs(f) < 1e-13:
            return float(lam) if lam > 0.0 else None
        deriv = float(np.sum(r_p * e) / s)
        if (not np.isfinite(deriv)) or abs(deriv) < 1e-18:
            break
        lam = lam - f / deriv
        if (not np.isfinite(lam)) or lam <= 0.0:
            lam = 1e-8
    if lam > 0.0 and np.isfinite(lam):
        x = lam * r_p
        m = float(np.max(x))
        e = np.exp(x - m)
        lm = float(np.log(float(np.mean(e))) + m)
        if abs(lm - log_t) < 1e-10:
            return float(lam)
    return None


def simulate_modified_gbm_rolling(steps, S0, n_paths, seed, rf=None):
    """Modified GBM v2: Markov sign, Markov calm/wild, lognormal size, S * exp(r).

    `rf` is an annual risk-free rate. When set, sizes are scaled (not shifted)
    so E[exp(r_t)] = exp(rf / N_DAYS) and signs stay as drawn under P.
    Leave `rf=None` for P-measure paths.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(steps["p_uu"])
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    up = np.full(n_paths, float(steps["last_up"][0]) >= 0.5, dtype=bool)
    wild = np.full(n_paths, float(steps["last_wild"][0]) >= 0.5, dtype=bool)
    rf_step = None if rf is None else float(rf) / float(N_DAYS)

    for i in range(n_steps):
        p_uu = float(np.clip(steps["p_uu"][i], 0.0, 1.0))
        p_ud = float(np.clip(steps["p_ud"][i], 0.0, 1.0))
        p_hh = float(np.clip(steps["p_hh"][i], 0.0, 1.0))
        p_ll = float(np.clip(steps["p_ll"][i], 0.0, 1.0))
        mu_u_l = float(steps["mu_u_l"][i])
        mu_u_h = float(steps["mu_u_h"][i])
        mu_d_l = float(steps["mu_d_l"][i])
        mu_d_h = float(steps["mu_d_h"][i])
        sig_u_l = max(float(steps["sig_u_l"][i]), 1e-12)
        sig_u_h = max(float(steps["sig_u_h"][i]), 1e-12)
        sig_d_l = max(float(steps["sig_d_l"][i]), 1e-12)
        sig_d_h = max(float(steps["sig_d_h"][i]), 1e-12)
        p_up = np.where(up, p_uu, p_ud)
        p_wild = np.where(wild, p_hh, 1.0 - p_ll)
        up = rng.random(n_paths) < p_up
        wild = rng.random(n_paths) < p_wild
        mu = np.where(up, np.where(wild, mu_u_h, mu_u_l), np.where(wild, mu_d_h, mu_d_l))
        sig = np.where(up, np.where(wild, sig_u_h, sig_u_l), np.where(wild, sig_d_h, sig_d_l))
        mag = np.exp(rng.normal(mu, sig))
        mag = np.maximum(mag, 1e-16)
        r_p = np.where(up, mag, -mag)
        if rf_step is None:
            r = r_p
        else:
            lam = _rn_size_scale(r_p, float(np.exp(rf_step)))
            if lam is None:
                mx = float(np.mean(np.exp(r_p)))
                r = r_p + (rf_step - np.log(max(mx, 1e-300)))
            else:
                r = lam * r_p
        paths[:, i + 1] = paths[:, i] * np.exp(r)
    return paths
'''

RN_STEPS = '''    n_steps = int(getattr(row, "n_steps", 0)) or int(dte)
    r = float(row.r)
    S0 = float(row.S_t)
    steps = {
        "p_uu": np.full(n_steps, float(p["p_uu"]), dtype=float),
        "p_du": np.full(n_steps, float(p["p_du"]), dtype=float),
        "p_ud": np.full(n_steps, float(p["p_ud"]), dtype=float),
        "p_dd": np.full(n_steps, float(p["p_dd"]), dtype=float),
        "p_hh": np.full(n_steps, float(p["p_hh"]), dtype=float),
        "p_ll": np.full(n_steps, float(p["p_ll"]), dtype=float),
        "p_lh": np.full(n_steps, float(p["p_lh"]), dtype=float),
        "p_hl": np.full(n_steps, float(p["p_hl"]), dtype=float),
        "mu_u_l": np.full(n_steps, float(p["mu_u_l"]), dtype=float),
        "sig_u_l": np.full(n_steps, float(p["sig_u_l"]), dtype=float),
        "mu_u_h": np.full(n_steps, float(p["mu_u_h"]), dtype=float),
        "sig_u_h": np.full(n_steps, float(p["sig_u_h"]), dtype=float),
        "mu_d_l": np.full(n_steps, float(p["mu_d_l"]), dtype=float),
        "sig_d_l": np.full(n_steps, float(p["sig_d_l"]), dtype=float),
        "mu_d_h": np.full(n_steps, float(p["mu_d_h"]), dtype=float),
        "sig_d_h": np.full(n_steps, float(p["sig_d_h"]), dtype=float),
        "last_up": np.full(n_steps, float(p["last_up"]), dtype=float),
        "last_wild": np.full(n_steps, float(p["last_wild"]), dtype=float),
    }
    return simulate_modified_gbm_rolling(steps, S0, n_paths, seed, rf=r)
'''

ROLLING_MD = r"""# Modified GBM v2 notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Model

1. **Direction** — two-state Markov chain \(P(U\mid U),\ P(D\mid U),\ P(U\mid D),\ P(D\mid D)\).
2. **Calm / wild** — two-state Markov chain on size. A lookback bar is wild if \(|R|\) exceeds the window median of \(|R|\).
3. **Magnitude** — \(\mathrm{size}=\exp(N(\mu_{d,s},\sigma_{d,s}^2))\) (Way B, always positive). Four pairs: up/down × calm/wild.
4. **Price** — \(S_{t+1}=S_t e^{R_t}\) with \(R_t=+\mathrm{size}\) (up) or \(-\mathrm{size}\) (down).
5. **Q** — size-only: \(R^Q=\lambda R^P\) with \(\lambda\) chosen so \(E[e^{R^Q}]=e^{r_f\Delta t}\). Signs stay as drawn under \(P\).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** v2 parameters using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Files

- `2008-2009_modified_gbm_v2.ipynb`
- `2013-2014_modified_gbm_v2.ipynb`
- `2018-2019_modified_gbm_v2.ipynb`
- `2019-2020_modified_gbm_v2.ipynb`
- `7d_1min_modified_gbm_v2.ipynb`
- `1d_1min_modified_gbm_v2.ipynb`
"""

SPEC_MD = r"""# Modified GBM v2 — model specification

**This notebook is the documentation source of truth.**

Modified GBM v2 sits beside the original Modified GBM. Direction under \(P\) is unchanged. Three upgrades:

1. **Way B sizes.** \(\mathrm{size}=\exp(N(\mu,\sigma^2))\), always positive. \(\mu,\sigma\) are set so \(E[\mathrm{size}]=m\) and \(\mathrm{Var}(\mathrm{size})=v\), with \(m,v\) the mean and variance of \(|R|\) in that bucket.
2. **Calm / wild memory.** A second 1-lag Markov chain on whether the last \(|R|\) was above the lookback median. That chain picks which lognormal \((\mu,\sigma)\) is used. It does not change the up/down coins.
3. **Size-only \(Q\).** LSM does not add a constant to every return. It scales the drawn sizes by one \(\lambda>0\) so \(E[e^{R}]=e^{r_f\Delta t}\) and the sign of each path-step is unchanged.

## Symbols

| Symbol | Meaning |
|--------|---------|
| \(R_t=\ln(S_t/S_{t-1})\) | log-return |
| \(U,D\) | up / down |
| \(L,H\) | calm / wild |
| \(P(U\mid U),\ P(D\mid U),\ P(U\mid D),\ P(D\mid D)\) | sign coins |
| \(P(H\mid H),\ P(L\mid H),\ P(H\mid L),\ P(L\mid L)\) | calm/wild coins |
| \(m,v\) | mean and variance of \(\|R\|\) in one bucket |
| \(\mu,\sigma\) | mean and SD of \(\log(\mathrm{size})\) |

## Equations

\[
\sigma^2=\log(1+v/m^2),\qquad \mu=\log(m)-\tfrac12\sigma^2,\qquad
\mathrm{size}=\exp\bigl(N(\mu,\sigma^2)\bigr)
\]

\[
R_t=\begin{cases}+\mathrm{size}&U\\-\mathrm{size}&D\end{cases},\qquad S_{t+1}=S_t e^{R_t}
\]

Under \(Q\), \(R^Q=\lambda R^P\) with \(\lambda\) solved from \(\widehat{\mathbb{E}}[e^{R^Q}]=e^{r_f\Delta t}\). No additive shift and no direction premium. If a step has no up move, the old additive fallback is used.

## Code

`V4-Models_result/modified gbm v2 notebook/20*_modified_gbm_v2.ipynb`

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
    c0 = c0.replace("Modified GBM calibration", "Modified GBM v2 calibration")
    c0 = c0.replace(
        r"re-estimate the four transition probabilities and \((\mu_U,\sigma_U,\mu_D,\sigma_D)\) from the current window",
        r"re-estimate the sign coins, the calm/wild coins, and the four lognormal \((\mu,\sigma)\) pairs from the current window",
    )
    _set_src(cells[0], c0)

    _set_src(cells[7], FORMULAS_MD)

    for idx in (8, 10, 12):
        t = _src(cells[idx])
        t = t.replace("Modified GBM", "Modified GBM v2")
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
    stop = stop.replace("### Modified GBM — LSM", "### Modified GBM v2 — LSM")
    stop = stop.replace("Modified GBM optimal stopping", "Modified GBM v2 optimal stopping")
    stop = stop.replace(
        "mean-corrected E[e^{r_t}]=e^{r Δt}",
        "size-only Q: R ← λR so E[e^{r_t}]=e^{r Δt}, signs unchanged",
    )
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
                "id": "mgbm-v2-spec",
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
        dst = DST_DIR / name.replace("_modified_gbm.ipynb", "_modified_gbm_v2.ipynb")
        nb = json.loads(src.read_text())
        transform(nb, intraday=("1min" in name))
        dst.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {dst.relative_to(ROOT)}")
    (DST_DIR / "ROLLING_CALIBRATION.md").write_text(ROLLING_MD)
    (DST_DIR / "MODIFIED_GBM_V2.md").write_text(SPEC_MD)
    write_spec_notebook(DST_DIR / "modified_gbm_v2_model.ipynb")
    print("wrote modified gbm v2 notebook/ROLLING_CALIBRATION.md")
    print("wrote modified gbm v2 notebook/MODIFIED_GBM_V2.md")
    print("wrote modified gbm v2 notebook/modified_gbm_v2_model.ipynb")


if __name__ == "__main__":
    main()
