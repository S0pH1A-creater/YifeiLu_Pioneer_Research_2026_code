#!/usr/bin/env python3
"""Patch V4 GARCH–Merton notebooks: Duan (1995) LRNVR + Pan (2002) μ_J*."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "garch merton notebook"

FORMULAS_MD = r"""## 3. Estimation formulas (GARCH–Merton — Duan LRNVR + Pan jump premium)

**\(P\):** Duan GARCH-in-mean on the continuous block; Merton 3σ jumps from standardized residuals.

$$
\ln\frac{S_t}{S_{t-1}}=r_{f,t}+\lambda\sigma_t-\tfrac12\sigma_t^2+\sigma_t\varepsilon_t+J_t,\qquad
\sigma_t^2=\omega+\alpha\sigma_{t-1}^2\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2
$$

\(J_t\) is compound-Poisson with intensity \(\lambda_{\mathrm{jump}}\) and sizes \(N(\mu_J,\sigma_J^2)\). \(\lambda\) above is Duan’s unit equity premium (not jump intensity).

**\(Q\):** Duan (1995) LRNVR for \((\omega,\alpha,\beta,\lambda,\sigma_0)\); Pan (2002) jump-size premium \(\mu_J^*\) from listed calls with \(\lambda_{\mathrm{jump}}^*=\lambda_{\mathrm{jump}}\), \(\sigma_J^*=\sigma_J\).

$$
\ln\frac{S_t}{S_{t-1}}=r_{f,t}-\lambda_{\mathrm{jump}}\kappa^*\,\Delta t-\tfrac12\sigma_t^2+\sigma_t\xi_t+J_t^*,
\qquad
\sigma_t^2=\omega+\alpha\sigma_{t-1}^2(\xi_{t-1}-\lambda)^2+\beta\sigma_{t-1}^2
$$

§4 graphs: \(P\). §5 uses \(P\). §6 LSM uses \(Q\) only.
"""

CAL_HEAD = '''import sys
from pathlib import Path as _GmPath
_GM_SCRIPTS = str((_GmPath("..") / "scripts").resolve())
if _GM_SCRIPTS not in sys.path:
    sys.path.insert(0, _GM_SCRIPTS)
from heston_option_calibration import load_calls_panel, select_quotes_asof, quotes_fingerprint
from garch_duan_lrnvr import load_rf_annual, q_persist
from garch_merton_pq import (
    estimate_garch_merton_p,
    calibrate_garch_merton_q,
    report_garch_merton_pq,
)


def _rf_annual_series():
    return load_rf_annual(DATA, short_interval=int(N_DAYS) > 400)


def _opt_dir():
    if "OPT_DIR" in globals():
        return Path(OPT_DIR)
    data = Path(DATA)
    if int(N_DAYS) > 400:
        return data / "options" / "processed" / "short_interval"
    return data / "options" / "processed"


def estimate_garch_merton_params(log_rets: pd.Series, jump_thresh: float = JUMP_THRESH, warm=None):
    """Duan GARCH-in-mean + 3σ jumps. Returns dict or None."""
    return estimate_garch_merton_p(
        log_rets,
        _rf_annual_series(),
        n_days=int(N_DAYS),
        min_window=MIN_WINDOW,
        jump_thresh=float(jump_thresh),
        warm=warm,
    )

'''

SIM_HEAD = '''import sys
from pathlib import Path as _GmSimPath
_GM_SIM = str((_GmSimPath("..") / "scripts").resolve())
if _GM_SIM not in sys.path:
    sys.path.insert(0, _GM_SIM)
from garch_merton_pq import simulate_garch_merton_p, simulate_garch_merton_q


def simulate_garch_merton_rolling(steps, S0, n_paths, seed):
    """§5 physical-measure paths (Duan GARCH-in-mean + jumps)."""
    return simulate_garch_merton_p(steps, S0, n_paths, seed, n_days=int(N_DAYS))


'''

RN_DAILY = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Duan LRNVR + Pan μ_J* Q-paths (not μ → r)."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None or not np.isfinite(p.get("mu_j_q", np.nan)):
        raise RuntimeError(f"No Q jump premium for {ticker} — need option quotes in §4.")
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
        "lam": np.full(dte, float(p["lam"]), dtype=float),
        "mu_j": np.full(dte, float(p["mu_j"]), dtype=float),
        "mu_j_q": np.full(dte, float(p["mu_j_q"]), dtype=float),
        "sigma_j": np.full(dte, float(p["sigma_j"]), dtype=float),
        "kappa": np.full(dte, float(p["kappa"]), dtype=float),
        "kappa_q": np.full(dte, float(p["kappa_q"]), dtype=float),
    }
    return simulate_garch_merton_q(steps, S0, n_paths, seed, n_days=int(N_DAYS))
'''


def _src(cell) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else str(s)


def _set(cell, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if text.endswith("\n") else [])


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing block: {label}")
    return text.replace(old, new, 1)


def patch_intro(src: str) -> str:
    src = src.replace(
        "estimate GARCH(1,1)–Merton parameters by **maximum likelihood** (returns → conditional \(\\sigma_t\) path → likelihood → best \(\\hat\\mu,\\hat\\omega,\\hat\\alpha,\\hat\\beta\)), then jump params from standardized residuals.",
        "estimate Duan GARCH-in-mean \((\\lambda,\\omega,\\alpha,\\beta)\) by **maximum likelihood** on returns + DGS3MO, then Merton jump params from standardized residuals; fit Pan \(\\mu_J^*\) from listed calls.",
    )
    src = src.replace(
        "risk-neutral paths from the same simulator",
        "Duan LRNVR + Pan $Q$-paths from the same simulator",
    )
    return src


def patch_calibrate(src: str) -> str:
    start = src.index("def _unpack_garch")
    end = src.index("def _slice_window")
    src = src[:start] + CAL_HEAD + src[end:]

    old_loop = """    for t_u in update_dates:
        window = _slice_window(rets, pd.Timestamp(t_u), offset)
        mu, omega, alpha, beta, sigma0, lam, mu_j, sigma_j, kappa, n = estimate_garch_merton_params(
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
            "lam": lam,
            "mu_j": mu_j,
            "sigma_j": sigma_j,
            "kappa": kappa,
        })
    return pd.DataFrame(rows)"""

    new_loop = """    panel = load_calls_panel(Path(DATA), ticker, opt_dir=_opt_dir())
    prev_q = None
    prev_fp = None
    for t_u in update_dates:
        window = _slice_window(rets, pd.Timestamp(t_u), offset)
        hat = estimate_garch_merton_params(window, warm=warm)
        if hat is None or hat["n"] < MIN_WINDOW or not np.isfinite(hat["lambda"]):
            continue
        if not hat.get("q_stationary", True):
            print(
                f"warning: Q-GARCH not stationary at {pd.Timestamp(t_u).date()} "
                f"(α(1+λ²)+β={q_persist(hat['alpha'], hat['beta'], hat['lambda']):.4f})",
                flush=True,
            )
        quotes = select_quotes_asof(panel, pd.Timestamp(t_u), offset)
        fp = quotes_fingerprint(quotes)
        if prev_q is not None and fp == prev_fp and fp:
            q = prev_q
        else:
            q = calibrate_garch_merton_q(hat, quotes)
            if q["success"]:
                prev_q, prev_fp = q, fp
            elif prev_q is not None:
                q = prev_q
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
            "lam": hat["lam"],
            "mu_j": hat["mu_j"],
            "sigma_j": hat["sigma_j"],
            "kappa": hat["kappa"],
            "sigma": hat["sigma"],
            "mu_p": hat["mu_p"],
            "rf_mean": hat["rf_mean"],
            "persist_p": hat["persist_p"],
            "persist_q": hat["persist_q"],
            "uncond_var_p": hat["uncond_var_p"],
            "uncond_var_q": hat["uncond_var_q"],
            "q_stationary": hat["q_stationary"],
            "mu_j_q": q.get("mu_j_q", np.nan),
            "kappa_q": q.get("kappa_q", np.nan),
            "n_quotes": q.get("n_quotes", 0),
            "q_success": bool(q.get("success", False)),
            "jump_premium_identified": bool(q.get("jump_premium_identified", False)),
        })
    return pd.DataFrame(rows)"""

    src = _replace_once(src, old_loop, new_loop, "calibrate_ticker loop")

    src = _replace_once(
        src,
        '    cols = ["mu", "omega", "alpha", "beta", "sigma0", "lam", "mu_j", "sigma_j", "kappa"]',
        '    cols = ["lambda", "omega", "alpha", "beta", "sigma0", "lam", "mu_j", "sigma_j", "kappa", "rf"]',
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
            steps[c][i] = arrs[c][idx]

    return dates, steps, float(hist.iloc[0]), hist""",
        """    param_cols = ["lambda", "omega", "alpha", "beta", "sigma0", "lam", "mu_j", "sigma_j", "kappa"]
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
        steps["rf"][i] = float(rf_vals[i]) if np.isfinite(rf_vals[i]) else float(rf_ann.dropna().iloc[-1])

    return dates, steps, float(hist.iloc[0]), hist""",
        "param_schedule loop",
    )

    src = src.replace(
        '"""Rolling-parameter graphs: μ, ω, α, β, λ, κ."""',
        '"""Rolling-parameter graphs: λ_duan, ω, α, β, λ_jump, κ."""',
    )
    src = _replace_once(
        src,
        """    panels = [
        ("mu", "μ̂ (annual)", "Estimated drift"),
        ("omega", "ω̂", "GARCH intercept"),
        ("alpha", "α̂", "ARCH reaction"),
        ("beta", "β̂", "GARCH persistence"),
        ("lam", "λ̂ (jumps/year)", "Estimated jump intensity"),
        ("kappa", "κ̂", "Jump compensation"),
    ]""",
        """    panels = [
        ("lambda", "λ̂ (Duan unit premium)", "Duan unit risk premium"),
        ("omega", "ω̂", "GARCH intercept"),
        ("alpha", "α̂", "ARCH reaction"),
        ("beta", "β̂", "GARCH persistence"),
        ("lam", "λ̂_jump (jumps/year)", "Estimated jump intensity"),
        ("kappa", "κ̂", "Jump compensation (P)"),
    ]""",
        "panels",
    )
    src = src.replace('            if col in {"mu", "kappa"}:', '            if col in {"lambda", "kappa"}:')
    src = src.replace(
        "Shows μ̂, ω̂, α̂, β̂, λ̂, κ̂. No Monte Carlo here.",
        "Shows λ̂, ω̂, α̂, β̂, λ̂_jump, κ̂. No Monte Carlo here. P/Q tables after Reestimate.",
    )

    extra = '''
        p_rows, q_rows = [], []
        for t in TICKERS:
            if rolling[t] is None or len(rolling[t]) == 0:
                continue
            last = rolling[t].iloc[-1]
            p_tbl, prem, q_tbl = report_garch_merton_pq(last)
            p_tbl["ticker"] = t
            q_tbl["ticker"] = t
            q_tbl["mu_j_q"] = prem.get("mu_j_q")
            p_rows.append(p_tbl)
            q_rows.append(q_tbl)
        if p_rows:
            display(Markdown("### Physical-measure (P) parameters"))
            display(pd.DataFrame(p_rows))
            display(Markdown("### Risk-neutral (Q) — Duan LRNVR + Pan μ_J*"))
            display(pd.DataFrame(q_rows))
'''
    marker = "        plot_rolling_paths(rolling, window_label, rolling_mode)\n"
    if marker not in src:
        raise RuntimeError("plot_rolling_paths call missing")
    src = src.replace(marker, marker + extra, 1)
    return src


def patch_sim(src: str) -> str:
    start = src.index("def simulate_garch_merton_rolling")
    end = src.index("def _show_fig", start)
    return src[:start] + SIM_HEAD + src[end:]


def patch_rn(src: str) -> str:
    # ensure simulate_garch_merton_q is imported in the stop cell
    if "from garch_merton_pq import" not in src and "simulate_garch_merton_q" not in src:
        needle = "from american_lsm import ("
        if needle in src:
            src = src.replace(
                needle,
                "from garch_merton_pq import simulate_garch_merton_q\nfrom american_lsm import (",
                1,
            )
    start = src.index("def _rn_paths_for_contract")
    end = src.index("\n_STOP_TICKERS", start)
    return src[:start] + RN_DAILY + src[end:]


def patch_s6_md(src: str) -> str:
    return src.replace(
        "Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; vol/jumps from §4 for that ticker).",
        "Paths for pricing are **risk-neutral under Duan (1995) LRNVR + Pan (2002) $\\mu_J^*$** "
        "(GARCH mean $r_f-\\tfrac12\\sigma_t^2$, variance shock $\\xi_t-\\lambda$; jump size $\\mu_J^*$ from listed calls).",
    )


def patch_nb(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = _src(cell)
        if cell["cell_type"] == "markdown" and src.startswith("# GARCH–Merton"):
            _set(cell, patch_intro(src))
        elif cell["cell_type"] == "markdown" and src.startswith("## 3. Estimation formulas"):
            _set(cell, FORMULAS_MD)
        elif cell["cell_type"] == "markdown" and src.startswith("## 6. Optimal stopping"):
            _set(cell, patch_s6_md(src))
        elif cell["cell_type"] == "code" and "def calibrate_ticker" in src and "def _unpack_garch" in src:
            _set(cell, patch_calibrate(src))
        elif cell["cell_type"] == "code" and "def simulate_garch_merton_rolling" in src:
            _set(cell, patch_sim(src))
        elif cell["cell_type"] == "code" and "def _rn_paths_for_contract" in src:
            _set(cell, patch_rn(src))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"patched {path.name}", flush=True)


def main() -> int:
    nbs = sorted(NB_DIR.glob("20*_garch_merton.ipynb"))
    if not nbs:
        raise SystemExit(f"no notebooks in {NB_DIR}")
    for p in nbs:
        patch_nb(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
