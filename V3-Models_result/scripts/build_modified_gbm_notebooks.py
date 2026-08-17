#!/usr/bin/env python3
"""Clone V3 GBM notebooks into Modified GBM (Markov direction + split normals)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "gbm notebook"
DST_DIR = ROOT / "modified gbm notebook"

GBM_FILES = [
    "2008-2009_gbm.ipynb",
    "2013-2014_gbm.ipynb",
    "2018-2019_gbm.ipynb",
    "2019-2020_gbm.ipynb",
    "7d_1min_gbm.ipynb",
    "1d_1min_gbm.ipynb",
]

FORMULAS_MD = r"""## 3. Estimation formulas (Modified GBM)

A three-stage discrete replacement for GBM. Direction is a two-state Markov chain; magnitude is normal and taken positive; the signed return drives the price.

**Stage 1 — direction.** Let \(U=\{r_t>0\}\) and \(D=\{r_t<0\}\). From consecutive non-zero returns in the lookback window, calibrate

\[
\hat P(U\mid U),\ \hat P(D\mid U),\ \hat P(U\mid D),\ \hat P(D\mid D)
\]

(Laplace-smoothed). The previous bar's direction selects which pair is used, so a high \(\hat P(U\mid U)\) or \(\hat P(D\mid D)\) produces upward or downward clustering.

**Stage 2 — magnitude.** After the direction is chosen, draw the size from a separate normal for up and down moves and take it positive:

\[
m_t\mid U \sim \bigl|N(\mu_U,\sigma_U^2)\bigr|,\qquad
m_t\mid D \sim \bigl|N(\mu_D,\sigma_D^2)\bigr|.
\]

\(\mu_U,\sigma_U\) (resp. \(\mu_D,\sigma_D\)) are the sample mean and standard deviation of \(|r|\) on up (resp. down) bars, on the same scale as the series (daily or 1-minute).

**Stage 3 — price.** The signed return is \(r_t=+m_t\) on an up move and \(r_t=-m_t\) on a down move:

\[
S_{t+1}=S_t\,e^{r_t}.
\]

| Parameter | Role |
|-----------|------|
| \(P(U\mid U),\ P(D\mid U)\) | Next-bar direction after an up move |
| \(P(U\mid D),\ P(D\mid D)\) | Next-bar direction after a down move |
| \(\mu_U,\sigma_U\) | Mean / SD of up magnitudes |
| \(\mu_D,\sigma_D\) | Mean / SD of down magnitudes |

Risk-neutral paths (§6) keep the calibrated transitions and magnitudes, then shift each step so \(E[e^{r_t}]=\exp(r\,\Delta t)\).

**True rolling:** at each update date, re-estimate from the lookback window ending there; those params drive the next Monte Carlo segment.
"""

ESTIMATE_FN = '''def estimate_modified_gbm(log_rets: pd.Series):
    """Markov direction + split-normal magnitudes on the bar's log returns."""
    x = log_rets.dropna().astype(float)
    x = x[np.isfinite(x)]
    n = int(x.shape[0])
    nz = x[x != 0.0]
    if n < 3 or int(nz.shape[0]) < 3:
        return None
    signed = nz.to_numpy(dtype=float)
    up = signed > 0.0
    mag = np.abs(signed)
    prev, curr = up[:-1], up[1:]
    n_from_u = int(prev.sum())
    n_from_d = int((~prev).sum())
    n_uu = int((prev & curr).sum())
    n_dd = int((~prev & ~curr).sum())
    p_uu = (n_uu + 0.5) / (n_from_u + 1.0)
    p_dd = (n_dd + 0.5) / (n_from_d + 1.0)
    p_du = 1.0 - p_uu
    p_ud = 1.0 - p_dd

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
    return {
        "n_days": n,
        "p_uu": float(p_uu),
        "p_du": float(p_du),
        "p_ud": float(p_ud),
        "p_dd": float(p_dd),
        "mu_u": mu_u,
        "sig_u": sig_u,
        "mu_d": mu_d,
        "sig_d": sig_d,
        "last_up": 1.0 if bool(up[-1]) else 0.0,
        "p_u": float(up.mean()),
    }


'''

CALIBRATE_INNER_OLD = '''        mu, sigma, n = estimate_mu_sigma(window)
        if n < 2:
            continue
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            "n_days": n,
            "mu": mu,
            "sigma": sigma,
        })
'''

CALIBRATE_INNER_NEW = '''        est = estimate_modified_gbm(window)
        if est is None:
            continue
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min(),
            "window_end": window.index.max(),
            **est,
        })
'''

PARAM_SCHEDULE_NEW = '''def param_schedule_for_steps(ticker: str, cal_table: pd.DataFrame):
    hist = _mc_time_grid(period_prices[ticker], N_STEPS)
    dates = hist.index
    n_steps = len(dates) - 1
    cal = cal_table.sort_values("date").reset_index(drop=True)
    cal_dates = pd.to_datetime(cal["date"]).to_numpy()
    cols = ["p_uu", "p_du", "p_ud", "p_dd", "mu_u", "sig_u", "mu_d", "sig_d", "last_up"]
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
    """Rolling Modified GBM parameters (graphs only)."""
    panels = [
        (("p_uu", "p_dd"), r"$\\hat P(U|U)$ / $\\hat P(D|D)$", "Direction persistence"),
        (("p_ud", "p_du"), r"$\\hat P(U|D)$ / $\\hat P(D|U)$", "Direction reversal"),
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
    """Rolling Modified GBM parameters (graphs only)."""
    panels = [
        (("p_uu", "p_dd"), r"$\\hat P(U|U)$ / $\\hat P(D|D)$", "Direction persistence"),
        (("p_ud", "p_du"), r"$\\hat P(U|D)$ / $\\hat P(D|U)$", "Direction reversal"),
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
    """Three-stage Modified GBM: Markov direction, split-normal size, S * exp(r).

    `rf` is an annual risk-free rate. When set, each step is shifted so
    E[exp(r_t)] = exp(rf / N_DAYS). Leave `rf=None` for P-measure paths.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(steps["p_uu"])
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    up = np.full(n_paths, float(steps["last_up"][0]) >= 0.5, dtype=bool)
    rf_step = None if rf is None else float(rf) / float(N_DAYS)

    for i in range(n_steps):
        p_uu = float(np.clip(steps["p_uu"][i], 0.0, 1.0))
        p_ud = float(np.clip(steps["p_ud"][i], 0.0, 1.0))
        mu_u = float(steps["mu_u"][i])
        mu_d = float(steps["mu_d"][i])
        sig_u = max(float(steps["sig_u"][i]), 1e-12)
        sig_d = max(float(steps["sig_d"][i]), 1e-12)
        p_up = np.where(up, p_uu, p_ud)
        up = rng.random(n_paths) < p_up
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
        "p_uu": np.full(n_steps, float(p["p_uu"]), dtype=float),
        "p_du": np.full(n_steps, float(p["p_du"]), dtype=float),
        "p_ud": np.full(n_steps, float(p["p_ud"]), dtype=float),
        "p_dd": np.full(n_steps, float(p["p_dd"]), dtype=float),
        "mu_u": np.full(n_steps, float(p["mu_u"]), dtype=float),
        "sig_u": np.full(n_steps, float(p["sig_u"]), dtype=float),
        "mu_d": np.full(n_steps, float(p["mu_d"]), dtype=float),
        "sig_d": np.full(n_steps, float(p["sig_d"]), dtype=float),
        "last_up": np.full(n_steps, float(p["last_up"]), dtype=float),
    }
    return simulate_modified_gbm_rolling(steps, S0, n_paths, seed, rf=r)
'''

ROLLING_MD = r"""# Modified GBM notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Model

Three stages, estimated on the lookback window of log returns:

1. **Direction** — two-state Markov chain with \(P(U\\mid U),\ P(D\\mid U),\ P(U\\mid D),\ P(D\\mid D)\). The previous bar's sign selects the pair, which creates up/down clustering when persistence is high.
2. **Magnitude** — positive size from \(N(\\mu_U,\\sigma_U^2)\) or \(N(\\mu_D,\\sigma_D^2)\), taken absolute. Separate up and down distributions.
3. **Price** — \(S_{t+1}=S_t e^{r_t}\) with \(r_t=+m_t\) (up) or \(r_t=-m_t\) (down).

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** Modified GBM parameters using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \\to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

Intraday notebooks use `minutely` / `hourly` on the 1-minute grid.

## Lookback choices

3 months · 6 months · 1 year · 2 years · 3 years · 5 years (calendar lookback; uses available history if the series is shorter). Default on 1-year regime notebooks: **3 years**, rolling **monthly**.

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, rolling parameter charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; MC paths and expected path vs historical prices (one pair per ticker).
- **§6 Optimal stopping:** LSM American calls on risk-neutral Modified GBM paths.

## Files

- `2008-2009_modified_gbm.ipynb`
- `2013-2014_modified_gbm.ipynb`
- `2018-2019_modified_gbm.ipynb`
- `2019-2020_modified_gbm.ipynb`
- `7d_1min_modified_gbm.ipynb`
- `1d_1min_modified_gbm.ipynb`

## Reopen + Run All

Use plain `display(ui)`. Re-running a cell replaces that cell’s output. Workflow: open notebook → Restart kernel → Run All.

**After reopen:** Kernel → **Restart Kernel**, then **Run All**.
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
    c0 = c0.replace("GBM calibration", "Modified GBM calibration")
    c0 = c0.replace(
        r"re-estimate \(\hat\mu,\hat\sigma\) from the current window",
        r"re-estimate the four transition probabilities and \((\mu_U,\sigma_U,\mu_D,\sigma_D)\) from the current window",
    )
    _set_src(cells[0], c0)

    _set_src(cells[7], FORMULAS_MD)

    for idx in (8, 10, 12):
        t = _src(cells[idx])
        t = t.replace("GBM", "Modified GBM")
        _set_src(cells[idx], t)

    cal = _src(cells[9])
    if "def estimate_mu_sigma" not in cal:
        raise RuntimeError("estimate_mu_sigma missing")
    start = cal.index("def estimate_mu_sigma")
    end = cal.index("def _slice_window")
    cal = cal[:start] + ESTIMATE_FN + cal[end:]
    if CALIBRATE_INNER_OLD not in cal:
        raise RuntimeError("calibrate inner block missing")
    cal = cal.replace(CALIBRATE_INNER_OLD, CALIBRATE_INNER_NEW)
    cal = _replace_fn(cal, "param_schedule_for_steps", PARAM_SCHEDULE_NEW)
    plot = PLOT_INTRADAY if intraday else PLOT_YEARLY
    cal = _replace_fn(cal, "plot_rolling_paths", plot)
    _set_src(cells[9], cal)

    mc = _src(cells[11])
    mc = _replace_fn(mc, "simulate_gbm_rolling", SIMULATE_FN)
    old_draw = """        dates_now, mu_now, sig_now, S0_now, hist_now = param_schedule_for_steps(
            ticker, rolling[ticker]
        )
        paths = simulate_gbm_rolling(mu_now, sig_now, S0_now, n_paths, seed)"""
    new_draw = """        dates_now, steps_now, S0_now, hist_now = param_schedule_for_steps(
            ticker, rolling[ticker]
        )
        paths = simulate_modified_gbm_rolling(steps_now, S0_now, n_paths, seed)"""
    if old_draw not in mc:
        raise RuntimeError("§5 draw unpack missing")
    mc = mc.replace(old_draw, new_draw)
    _set_src(cells[11], mc)

    stop = _src(cells[13])
    stop = stop.replace("### GBM — LSM", "### Modified GBM — LSM")
    stop = stop.replace("GBM optimal stopping", "Modified GBM optimal stopping")
    stop = stop.replace(
        "Risk-neutral paths to expiry using §5 simulator (μ → r).",
        "Risk-neutral paths to expiry using §5 simulator (mean-corrected E[e^{r_t}]=e^{r Δt}).",
    )
    stop = _replace_rn(stop)
    _set_src(cells[13], stop)
    return nb


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for name in GBM_FILES:
        src = SRC_DIR / name
        dst = DST_DIR / name.replace("_gbm.ipynb", "_modified_gbm.ipynb")
        nb = json.loads(src.read_text())
        transform(nb, intraday=("1min" in name))
        dst.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {dst.relative_to(ROOT)}")
    (DST_DIR / "ROLLING_CALIBRATION.md").write_text(ROLLING_MD)
    print("wrote modified gbm notebook/ROLLING_CALIBRATION.md")


if __name__ == "__main__":
    main()
