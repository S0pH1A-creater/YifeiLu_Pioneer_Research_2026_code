#!/usr/bin/env python3
"""Patch V4 Heston / Merton / Heston–Merton notebooks for Pan–Bates P→Q.

Does not touch GBM, GARCH, GARCH–Merton, Modified GBM, or V1–V3.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PQ_IMPORT_HESTON = '''from pq_risk_premium import (
    estimate_heston_p,
    calibrate_heston_q,
    report_heston_pq,
    simulate_heston_q,
)

'''

PQ_IMPORT_MERTON = '''import sys
from pathlib import Path as _PqPath
_PQ_SCRIPTS = str((_PqPath("..") / "scripts").resolve())
if _PQ_SCRIPTS not in sys.path:
    sys.path.insert(0, _PQ_SCRIPTS)
from heston_option_calibration import load_calls_panel, select_quotes_asof, quotes_fingerprint
from pq_risk_premium import (
    estimate_merton_p,
    calibrate_merton_q,
    report_merton_pq,
    simulate_merton_q,
)

def _heston_opt_dir():
    if "OPT_DIR" in globals():
        return Path(OPT_DIR)
    data = Path(DATA)
    if int(N_DAYS) > 400:
        return data / "options" / "processed" / "short_interval"
    return data / "options" / "processed"

'''

PQ_IMPORT_BATES = '''from pq_risk_premium import (
    estimate_bates_p,
    calibrate_bates_q,
    report_bates_pq,
    simulate_bates_q,
)

'''

HESTON_FORMULAS = r"""## 3. Estimation formulas (Heston — P from returns, Q via Pan 2002)

Physical \(P\) (Method A on lookback returns):

$$dS_t/S_t=\mu\,dt+\sqrt{v_t}\,dW^{P,S},\qquad dv_t=\kappa(\theta-v_t)\,dt+\xi\sqrt{v_t}\,dW^{P,v}$$

Volatility risk is **not** identified from returns (variance is not a traded asset). Pan (2002) eqs. (2.5) and Appendix D: premium \(\eta_v\) from listed calls, \(\xi,\rho\) held at \(P\),

$$\kappa^*=\kappa-\eta_v,\qquad \bar v^*=\kappa\bar v/\kappa^*,\qquad dv=[\kappa(\bar v-v)+\eta_v v]\,dt+\xi\sqrt{v}\,dW^{Q,v}$$

$$dS_t/S_t=r_f\,dt+\sqrt{v_t}\,dW^{Q,S}$$

| Object | Under \(P\) | Risk premium | Under \(Q\) |
|--------|-------------|--------------|-------------|
| \(\mu\) | lookback mean | equity (Girsanov) | \(r_f\) |
| \(\kappa,\theta\) | Method A | \(\eta_v\) from options | \(\kappa-\eta_v\), \(\kappa\theta/\kappa^*\) |
| \(\xi,\rho\) | Method A | — | unchanged |
| \(v_0\) | recent RV | — | option-implied \(v_0^Q\) |

§4 graphs: \(P\) parameters. §5 Monte Carlo uses **\(P\)**. §6 LSM uses **\(Q\)** only.
"""

MERTON_FORMULAS_NOTE = (
    "P from returns (3σ jumps). Jump-size premium \(\\mu_J^*\\) from listed calls "
    "(Pan 2002); \(\\lambda^*=\\lambda\) (timing not separately identified). "
    "Do not set \(P=Q\) for jumps when \(\\lambda>0\). §5 uses \(P\); §6 LSM uses \(Q\)."
)

BATES_FORMULAS = r"""## 3. Estimation formulas (Heston–Merton / Bates — Pan 2002 P→Q)

One SVJ framework (Bates 1996 dynamics; Pan 2002 premia). Do **not** glue a standalone Heston map to a standalone Merton map.

**\(P\):** Method A \((\mu,\kappa,\theta,\xi,\rho,v_0)\) plus 3σ jump block \((\lambda,\mu_J,\sigma_J)\).

**\(Q\):** \(\eta_v\) and \(\mu_J^*\) from listed Bates/Fourier calls; \(\lambda^*=\lambda\); \(\xi,\rho,\sigma_J\) from \(P\).

$$dS/S=(r_f-\lambda\kappa^Q)\,dt+\sqrt{v}\,dW^Q+(e^J-1)\,dN,\qquad J\sim N(\mu_J^*,\sigma_J^2)$$

$$dv=[\kappa(\theta-v)+\eta_v v]\,dt+\xi\sqrt{v}\,dW^{Q,v}$$

§4 graphs: \(P\). §5 uses \(P\). §6 LSM uses \(Q\).
"""

HESTON_LOOP = '''    panel = load_calls_panel(Path(DATA), ticker, opt_dir=_heston_opt_dir())
    prev_q = None
    prev_fp = None
    for t_u in update_dates:
        window = _slice_window(rets, pd.Timestamp(t_u), offset)
        quotes = select_quotes_asof(panel, pd.Timestamp(t_u), offset)
        fp = quotes_fingerprint(quotes)
        p = estimate_heston_p(window, N_DAYS)
        if not p["success"]:
            continue
        if prev_q is not None and fp == prev_fp and fp:
            q = prev_q
        else:
            q = calibrate_heston_q(p, quotes)
            if q["success"]:
                prev_q, prev_fp = q, fp
            elif prev_q is not None:
                q = prev_q
        mu = p["mu"] if np.isfinite(p["mu"]) else 0.0
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min() if len(window) else pd.Timestamp(t_u),
            "window_end": window.index.max() if len(window) else pd.Timestamp(t_u),
            "n_days": p["n"],
            "mu": mu,
            "kappa": p["kappa"],
            "theta": p["theta"],
            "xi": p["xi"],
            "rho": p["rho"],
            "v0": p["v0"],
            "eta_v": q.get("eta_v", np.nan),
            "kappa_q": q.get("kappa_q", np.nan),
            "theta_q": q.get("theta_q", np.nan),
            "v0_q": q.get("v0_q", np.nan),
            "n_quotes": q.get("n_quotes", 0),
            "q_success": bool(q.get("success", False)),
            "q_explosive": bool(q.get("q_explosive", False)),
        })
'''

BATES_LOOP = '''    panel = load_calls_panel(Path(DATA), ticker, opt_dir=_heston_opt_dir())
    prev_q = None
    prev_fp = None
    thresh = float(globals().get("JUMP_THRESH", 3.0))
    for t_u in update_dates:
        window = _slice_window(rets, pd.Timestamp(t_u), offset)
        quotes = select_quotes_asof(panel, pd.Timestamp(t_u), offset)
        fp = quotes_fingerprint(quotes)
        p = estimate_bates_p(window, N_DAYS, thresh)
        if not p["success"]:
            continue
        if prev_q is not None and fp == prev_fp and fp:
            q = prev_q
        else:
            q = calibrate_bates_q(p, quotes)
            if q["success"]:
                prev_q, prev_fp = q, fp
            elif prev_q is not None:
                q = prev_q
        mu = p["mu"] if np.isfinite(p["mu"]) else 0.0
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min() if len(window) else pd.Timestamp(t_u),
            "window_end": window.index.max() if len(window) else pd.Timestamp(t_u),
            "n_days": p["n"],
            "mu": mu,
            "kappa": p["kappa"],
            "theta": p["theta"],
            "xi": p["xi"],
            "rho": p["rho"],
            "v0": p["v0"],
            "lam": p["lam"],
            "mu_j": p["mu_j"],
            "sigma_j": p["sigma_j"],
            "kappa_j": p["kappa_j"],
            "eta_v": q.get("eta_v", np.nan),
            "kappa_q": q.get("kappa_q", np.nan),
            "theta_q": q.get("theta_q", np.nan),
            "v0_q": q.get("v0_q", np.nan),
            "mu_j_q": q.get("mu_j_q", np.nan),
            "kappa_j_q": q.get("kappa_j_q", np.nan),
            "n_quotes": q.get("n_quotes", 0),
            "q_success": bool(q.get("success", False)),
            "q_explosive": bool(q.get("q_explosive", False)),
            "jump_premium_identified": bool(q.get("jump_premium_identified", False)),
        })
'''

MERTON_INNER = '''        quotes = select_quotes_asof(panel, pd.Timestamp(t_u), offset)
        fp = quotes_fingerprint(quotes)
        p = estimate_merton_p(window, N_DAYS, float(globals().get("JUMP_THRESH", 3.0)))
        if not p["success"] or p["n"] < 2 or not np.isfinite(p["mu"]):
            continue
        if prev_q is not None and fp == prev_fp and fp:
            q = prev_q
        else:
            q = calibrate_merton_q(p, quotes)
            if q["success"]:
                prev_q, prev_fp = q, fp
            elif prev_q is not None:
                q = prev_q
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min() if len(window) else pd.Timestamp(t_u),
            "window_end": window.index.max() if len(window) else pd.Timestamp(t_u),
            "n_days": p["n"],
            "mu": p["mu"],
            "sigma": p["sigma"],
            "lam": p["lam"],
            "mu_j": p["mu_j"],
            "sigma_j": p["sigma_j"],
            "kappa": p["kappa"],
            "mu_j_q": q.get("mu_j_q", np.nan),
            "kappa_q": q.get("kappa_q", np.nan),
            "n_quotes": q.get("n_quotes", 0),
            "q_success": bool(q.get("success", False)),
            "jump_premium_identified": bool(q.get("jump_premium_identified", False)),
        })
'''

PQ_TABLES = {
    "heston": '''
        p_rows, prem_rows, q_rows = [], [], []
        for t in TICKERS:
            if rolling[t] is None or len(rolling[t]) == 0:
                continue
            last = rolling[t].iloc[-1]
            p_tbl, prem, q_tbl = report_heston_pq(last)
            p_tbl["ticker"] = t
            prem["ticker"] = t
            q_tbl["ticker"] = t
            p_rows.append(p_tbl)
            prem_rows.append(prem)
            q_rows.append(q_tbl)
        if p_rows:
            display(Markdown("### Physical-measure (P) parameters"))
            display(pd.DataFrame(p_rows))
            display(Markdown("### Volatility risk premium (Pan $\\\\eta_v$, from listed calls)"))
            display(pd.DataFrame(prem_rows))
            display(Markdown("### Risk-neutral (Q) dynamics"))
            display(pd.DataFrame(q_rows))
''',
    "merton": '''
        p_rows, prem_rows, q_rows = [], [], []
        for t in TICKERS:
            if rolling[t] is None or len(rolling[t]) == 0:
                continue
            last = rolling[t].iloc[-1]
            p_tbl, prem, q_tbl = report_merton_pq(last)
            p_tbl["ticker"] = t
            prem["ticker"] = t
            q_tbl["ticker"] = t
            p_rows.append(p_tbl)
            prem_rows.append(prem)
            q_rows.append(q_tbl)
        if p_rows:
            display(Markdown("### Physical-measure (P) parameters"))
            display(pd.DataFrame(p_rows))
            display(Markdown("### Jump-size risk premium (Pan $\\\\mu-\\\\mu^*$, from listed calls)"))
            display(pd.DataFrame(prem_rows))
            display(Markdown("### Risk-neutral (Q) dynamics"))
            display(pd.DataFrame(q_rows))
''',
    "bates": '''
        p_rows, prem_rows, q_rows = [], [], []
        for t in TICKERS:
            if rolling[t] is None or len(rolling[t]) == 0:
                continue
            last = rolling[t].iloc[-1]
            p_tbl, prem, q_tbl = report_bates_pq(last)
            p_tbl["ticker"] = t
            prem["ticker"] = t
            q_tbl["ticker"] = t
            p_rows.append(p_tbl)
            prem_rows.append(prem)
            q_rows.append(q_tbl)
        if p_rows:
            display(Markdown("### Physical-measure (P) parameters"))
            display(pd.DataFrame(p_rows))
            display(Markdown("### Volatility and jump-size premia (Pan, from listed calls)"))
            display(pd.DataFrame(prem_rows))
            display(Markdown("### Risk-neutral (Q) dynamics"))
            display(pd.DataFrame(q_rows))
''',
}

RN_HESTON_DAILY = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Pan Q-Heston paths (not μ → r on P variance)."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None or not np.isfinite(p.get("eta_v", np.nan)):
        raise RuntimeError(f"No Q premium for {ticker} — need option quotes in §4.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    n = dte
    v0q = float(p["v0_q"]) if np.isfinite(p.get("v0_q", np.nan)) else float(p["v0"])
    steps = {
        "rf": np.full(n, r, dtype=float),
        "kappa": np.full(n, float(p["kappa"]), dtype=float),
        "theta": np.full(n, float(p["theta"]), dtype=float),
        "xi": np.full(n, float(p["xi"]), dtype=float),
        "rho": np.full(n, float(p["rho"]), dtype=float),
        "v0": np.full(n, v0q, dtype=float),
        "v0_q": np.full(n, v0q, dtype=float),
        "eta_v": np.full(n, float(p["eta_v"]), dtype=float),
    }
    return simulate_heston_q(steps, S0, n_paths, seed, n_days=int(N_DAYS) if int(N_DAYS) <= 400 else 252)
'''

RN_HESTON_MINUTE = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Pan Q-Heston paths on the daily LSM clock."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None or not np.isfinite(p.get("eta_v", np.nan)):
        raise RuntimeError(f"No Q premium for {ticker} — need option quotes in §4.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    n = int(getattr(row, "n_steps", 0)) or dte
    r = float(row.r)
    S0 = float(row.S_t)
    v0q = float(p["v0_q"]) if np.isfinite(p.get("v0_q", np.nan)) else float(p["v0"])
    steps = {
        "rf": np.full(n, r, dtype=float),
        "kappa": np.full(n, float(p["kappa"]), dtype=float),
        "theta": np.full(n, float(p["theta"]), dtype=float),
        "xi": np.full(n, float(p["xi"]), dtype=float),
        "rho": np.full(n, float(p["rho"]), dtype=float),
        "v0": np.full(n, v0q, dtype=float),
        "v0_q": np.full(n, v0q, dtype=float),
        "eta_v": np.full(n, float(p["eta_v"]), dtype=float),
    }
    return simulate_heston_q(steps, S0, n_paths, seed, n_days=252)
'''

RN_MERTON_DAILY = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Pan Q-Merton paths (λ* = λ, μ_J* from options)."""
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
    n = dte
    steps = {
        "rf": np.full(n, r, dtype=float),
        "sigma": np.full(n, float(p["sigma"]), dtype=float),
        "lam": np.full(n, float(p["lam"]), dtype=float),
        "mu_j": np.full(n, float(p["mu_j"]), dtype=float),
        "mu_j_q": np.full(n, float(p["mu_j_q"]), dtype=float),
        "sigma_j": np.full(n, float(p["sigma_j"]), dtype=float),
        "kappa": np.full(n, float(p["kappa"]), dtype=float),
        "kappa_q": np.full(n, float(p["kappa_q"]), dtype=float),
    }
    return simulate_merton_q(steps, S0, n_paths, seed, n_days=int(N_DAYS) if int(N_DAYS) <= 400 else 252)
'''

RN_MERTON_MINUTE = RN_MERTON_DAILY.replace("n = dte", "n = int(getattr(row, \"n_steps\", 0)) or dte").replace(
    "int(N_DAYS) if int(N_DAYS) <= 400 else 252", "252"
)

RN_BATES_DAILY = '''def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Pan–Bates Q paths (vol premium η_v and jump-size μ_J*)."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None or not np.isfinite(p.get("eta_v", np.nan)):
        raise RuntimeError(f"No Q premia for {ticker} — need option quotes in §4.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    n = dte
    v0q = float(p["v0_q"]) if np.isfinite(p.get("v0_q", np.nan)) else float(p["v0"])
    mujq = float(p["mu_j_q"]) if np.isfinite(p.get("mu_j_q", np.nan)) else float(p["mu_j"])
    kapjq = float(p["kappa_j_q"]) if np.isfinite(p.get("kappa_j_q", np.nan)) else float(p["kappa_j"])
    steps = {
        "rf": np.full(n, r, dtype=float),
        "kappa": np.full(n, float(p["kappa"]), dtype=float),
        "theta": np.full(n, float(p["theta"]), dtype=float),
        "xi": np.full(n, float(p["xi"]), dtype=float),
        "rho": np.full(n, float(p["rho"]), dtype=float),
        "v0": np.full(n, v0q, dtype=float),
        "v0_q": np.full(n, v0q, dtype=float),
        "eta_v": np.full(n, float(p["eta_v"]), dtype=float),
        "lam": np.full(n, float(p["lam"]), dtype=float),
        "mu_j": np.full(n, float(p["mu_j"]), dtype=float),
        "mu_j_q": np.full(n, mujq, dtype=float),
        "sigma_j": np.full(n, float(p["sigma_j"]), dtype=float),
        "kappa_j": np.full(n, float(p["kappa_j"]), dtype=float),
        "kappa_j_q": np.full(n, kapjq, dtype=float),
    }
    return simulate_bates_q(steps, S0, n_paths, seed, n_days=int(N_DAYS) if int(N_DAYS) <= 400 else 252)
'''

RN_BATES_MINUTE = RN_BATES_DAILY.replace("n = dte", "n = int(getattr(row, \"n_steps\", 0)) or dte").replace(
    "int(N_DAYS) if int(N_DAYS) <= 400 else 252", "252"
)


def _src(cell) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else str(s)


def _set(cell, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1] + "\n"]


def _replace_loop(src: str, new_loop: str) -> str:
    start = src.index("    panel = load_calls_panel")
    end = src.index("    return pd.DataFrame(rows)")
    return src[:start] + new_loop + "\n" + src[end:]


def _replace_merton_loop(src: str) -> str:
    start = src.index("    for t_u in update_dates:")
    end = src.index("    return pd.DataFrame(rows)")
    head = src[src.index("def calibrate_ticker"):start]
    if "panel = load_calls_panel" not in head:
        inject = (
            "    panel = load_calls_panel(Path(DATA), ticker, opt_dir=_heston_opt_dir())\n"
            "    prev_q = None\n"
            "    prev_fp = None\n"
        )
        src = src[:start] + inject + src[start:]
        start = src.index("    for t_u in update_dates:")
        end = src.index("    return pd.DataFrame(rows)")
    inner_start = src.index("        window = _slice_window", start)
    return src[:inner_start] + "        window = _slice_window(rets, pd.Timestamp(t_u), offset)\n" + MERTON_INNER + "\n" + src[end:]


def _replace_rn(src: str, new_fn: str) -> str:
    start = src.index("def _rn_paths_for_contract")
    rest = src[start:]
    nxt = rest.find("\n\n\n")
    if nxt < 0:
        nxt = rest.find("\n_STOP_TICKERS")
    if nxt < 0:
        raise RuntimeError("cannot find end of _rn_paths_for_contract")
    return src[:start] + new_fn + src[start + nxt :]


def _add_pq_tables(src: str, kind: str) -> str:
    marker = "        plot_rolling_paths(rolling, window_label, rolling_mode)\n"
    if marker not in src:
        return src
    extra = PQ_TABLES[kind]
    if "Physical-measure (P) parameters" in src:
        return src
    return src.replace(marker, marker + extra, 1)


def _patch_md_formulas(cell, kind: str) -> None:
    src = _src(cell)
    if kind == "heston" and "Estimation formulas (Heston" in src:
        _set(cell, HESTON_FORMULAS)
    elif kind == "bates" and "Estimation formulas (Heston" in src:
        _set(cell, BATES_FORMULAS)
    elif kind == "merton" and "Estimation formulas (Merton" in src:
        if "Pan 2002" not in src:
            _set(cell, src.rstrip() + "\n\n" + MERTON_FORMULAS_NOTE + "\n")


def _patch_md_lsm(cell) -> None:
    src = _src(cell)
    if "μ \\rightarrow r" not in src and r"$\mu \rightarrow r$" not in src and "mu → r" not in src:
        src = src.replace(
            "drift $\\mu \\rightarrow r$ from the option panel",
            "Pan/Bates **$Q$** dynamics (risk premia from listed calls; not $\\mu\\to r$ on $P$ jumps/variance)",
        )
        _set(cell, src)
        return
    src = src.replace(
        "Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; Heston variance from §4 for that ticker).",
        "Paths for pricing are **$Q$** (Pan volatility risk premium; Heston $P$ from returns). §5 is $P$ visualization only.",
    )
    src = src.replace(
        "Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; vol/jumps from §4 for that ticker).",
        "Paths for pricing are **$Q$** (Pan jump-size and/or volatility premia). §5 is $P$ visualization only.",
    )
    _set(cell, src)


def patch_heston(nb: dict, minute: bool) -> None:
    for cell in nb["cells"]:
        src = _src(cell)
        if cell["cell_type"] == "markdown":
            _patch_md_formulas(cell, "heston")
            _patch_md_lsm(cell)
            if "option-implied calibration of" in src:
                cell["source"] = [
                    s.replace(
                        "V3 option-implied calibration of $(\\kappa,\\theta,\\xi,\\rho,v_0)$ from listed call prices (Fourier / characteristic function). Monte Carlo and LSM unchanged.",
                        "P: Method A on returns. Q: Pan (2002) $\\eta_v$ from listed calls. §5 uses P; §6 LSM uses Q.",
                    )
                    for s in (cell["source"] if isinstance(cell["source"], list) else [cell["source"]])
                ]
            continue
        if "from heston_option_calibration import" in src and "estimate_heston_p" not in src:
            src = src.replace(
                "from heston_option_calibration import (\n",
                PQ_IMPORT_HESTON + "from heston_option_calibration import (\n",
                1,
            )
        if "def estimate_heston_params" in src and "panel = load_calls_panel" in src:
            src = _replace_loop(src, HESTON_LOOP)
        if "plot_rolling_paths(rolling" in src:
            src = _add_pq_tables(src, "heston")
            src = src.replace("option-implied", "P + Pan Q")
        if "def _rn_paths_for_contract" in src:
            src = _replace_rn(src, RN_HESTON_MINUTE if minute else RN_HESTON_DAILY)
        _set(cell, src)


def patch_merton(nb: dict, minute: bool) -> None:
    for cell in nb["cells"]:
        src = _src(cell)
        if cell["cell_type"] == "markdown":
            _patch_md_formulas(cell, "merton")
            _patch_md_lsm(cell)
            continue
        if "def estimate_merton_params" in src and "def calibrate_ticker" in src:
            if "from pq_risk_premium import" not in src:
                src = PQ_IMPORT_MERTON + src
            src = _replace_merton_loop(src)
        if "plot_rolling_paths(rolling" in src:
            src = _add_pq_tables(src, "merton")
        if "def _rn_paths_for_contract" in src:
            src = _replace_rn(src, RN_MERTON_MINUTE if minute else RN_MERTON_DAILY)
        _set(cell, src)


def patch_bates(nb: dict, minute: bool) -> None:
    for cell in nb["cells"]:
        src = _src(cell)
        if cell["cell_type"] == "markdown":
            _patch_md_formulas(cell, "bates")
            _patch_md_lsm(cell)
            continue
        if "from heston_option_calibration import" in src and "estimate_bates_p" not in src:
            src = src.replace(
                "from heston_option_calibration import (\n",
                PQ_IMPORT_BATES + "from heston_option_calibration import (\n",
                1,
            )
        if "def estimate_heston_merton_params" in src and "panel = load_calls_panel" in src:
            src = _replace_loop(src, BATES_LOOP)
        if "plot_rolling_paths(rolling" in src:
            src = _add_pq_tables(src, "bates")
            src = src.replace("option-implied", "P + Pan Q")
        if "def _rn_paths_for_contract" in src:
            src = _replace_rn(src, RN_BATES_MINUTE if minute else RN_BATES_DAILY)
        _set(cell, src)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}", flush=True)


def main() -> int:
    for path in sorted((ROOT / "heston notebook").glob("*.ipynb")):
        minute = "1min" in path.name
        nb = _load(path)
        patch_heston(nb, minute)
        _save(path, nb)
    for path in sorted((ROOT / "merton notebook").glob("*.ipynb")):
        minute = "1min" in path.name
        nb = _load(path)
        patch_merton(nb, minute)
        _save(path, nb)
    for path in sorted((ROOT / "heston merton notebook").glob("*.ipynb")):
        minute = "1min" in path.name
        nb = _load(path)
        patch_bates(nb, minute)
        _save(path, nb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
