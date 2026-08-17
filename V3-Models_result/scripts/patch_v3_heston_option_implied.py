#!/usr/bin/env python3
"""Replace Method A Heston / Heston–Merton estimation with option-implied NLS (V3).

Leaves Monte Carlo SDEs, LSM, GBM/Merton/GARCH notebooks, and V1/V2 untouched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMPORT_BLOCK = '''import sys
from pathlib import Path as _CalPath
_CAL_SCRIPTS = str((_CalPath("..") / "scripts").resolve())
if _CAL_SCRIPTS not in sys.path:
    sys.path.insert(0, _CAL_SCRIPTS)
from heston_option_calibration import (
    load_calls_panel,
    select_quotes_asof,
    calibrate_heston_from_quotes,
    physical_mu,
    merton_jump_params,
    quotes_fingerprint,
)


def _heston_opt_dir():
    if "OPT_DIR" in globals():
        return Path(OPT_DIR)
    data = Path(DATA)
    if int(N_DAYS) > 400:
        return data / "options" / "processed" / "short_interval"
    return data / "options" / "processed"

'''

NEW_ESTIMATE_HESTON = '''def estimate_heston_params(log_rets: pd.Series, quotes=None, x0=None):
    """Option-implied (κ, θ, ξ, ρ, v0); μ from lookback stock returns (P-measure)."""
    mu = physical_mu(log_rets, N_DAYS)
    n = int(log_rets.dropna().shape[0])
    cal = calibrate_heston_from_quotes(quotes, x0=x0, max_nfev=50)
    if not cal["success"] or not np.isfinite(cal["kappa"]):
        return mu, np.nan, np.nan, np.nan, np.nan, np.nan, n
    return mu, cal["kappa"], cal["theta"], cal["xi"], cal["rho"], cal["v0"], n

'''

NEW_ESTIMATE_HM = '''def estimate_heston_merton_params(log_rets: pd.Series, quotes=None, x0=None, jump_thresh: float = 3.0):
    """Bates option-implied (κ, θ, ξ, ρ, v0); jumps from a return threshold; μ from returns."""
    mu = physical_mu(log_rets, N_DAYS)
    n = int(log_rets.dropna().shape[0])
    lam, mu_j, sigma_j, kappa_j = merton_jump_params(log_rets, N_DAYS, jump_thresh)
    cal = calibrate_heston_from_quotes(
        quotes, x0=x0, lam=lam, mu_j=mu_j, sigma_j=sigma_j, max_nfev=50
    )
    if not cal["success"] or not np.isfinite(cal["kappa"]):
        return mu, np.nan, np.nan, np.nan, np.nan, np.nan, lam, mu_j, sigma_j, kappa_j, n
    return mu, cal["kappa"], cal["theta"], cal["xi"], cal["rho"], cal["v0"], lam, mu_j, sigma_j, kappa_j, n

'''

HESTON_LOOP = '''    panel = load_calls_panel(Path(DATA), ticker, opt_dir=_heston_opt_dir())
    prev_x = None
    prev_fp = None
    prev_heston = None
    for t_u in update_dates:
        window = _slice_window(rets, pd.Timestamp(t_u), offset)
        quotes = select_quotes_asof(panel, pd.Timestamp(t_u), offset)
        fp = quotes_fingerprint(quotes)
        n = int(window.dropna().shape[0]) if len(window) else 0
        mu = physical_mu(window, N_DAYS)
        if prev_heston is not None and fp == prev_fp and fp:
            kappa, theta, xi, rho, v0 = prev_heston
        else:
            cal = calibrate_heston_from_quotes(quotes, x0=prev_x, max_nfev=50)
            if cal["success"] and np.isfinite(cal["kappa"]):
                kappa, theta, xi, rho, v0 = (
                    cal["kappa"], cal["theta"], cal["xi"], cal["rho"], cal["v0"]
                )
                prev_x = [kappa, theta, xi, rho, v0]
                prev_heston = (kappa, theta, xi, rho, v0)
                prev_fp = fp
            elif prev_heston is not None:
                kappa, theta, xi, rho, v0 = prev_heston
            else:
                continue
        if not np.isfinite(mu):
            mu = 0.0
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min() if len(window) else pd.Timestamp(t_u),
            "window_end": window.index.max() if len(window) else pd.Timestamp(t_u),
            "n_days": n,
            "mu": mu,
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "v0": v0,
        })
'''

HM_LOOP = '''    panel = load_calls_panel(Path(DATA), ticker, opt_dir=_heston_opt_dir())
    prev_x = None
    prev_fp = None
    prev_heston = None
    thresh = float(globals().get("JUMP_THRESH", 3.0))
    for t_u in update_dates:
        window = _slice_window(rets, pd.Timestamp(t_u), offset)
        quotes = select_quotes_asof(panel, pd.Timestamp(t_u), offset)
        fp = quotes_fingerprint(quotes)
        n = int(window.dropna().shape[0]) if len(window) else 0
        mu = physical_mu(window, N_DAYS)
        lam, mu_j, sigma_j, kappa_j = merton_jump_params(window, N_DAYS, thresh)
        if prev_heston is not None and fp == prev_fp and fp:
            kappa, theta, xi, rho, v0 = prev_heston
        else:
            cal = calibrate_heston_from_quotes(
                quotes, x0=prev_x, lam=lam, mu_j=mu_j, sigma_j=sigma_j, max_nfev=50
            )
            if cal["success"] and np.isfinite(cal["kappa"]):
                kappa, theta, xi, rho, v0 = (
                    cal["kappa"], cal["theta"], cal["xi"], cal["rho"], cal["v0"]
                )
                prev_x = [kappa, theta, xi, rho, v0]
                prev_heston = (kappa, theta, xi, rho, v0)
                prev_fp = fp
            elif prev_heston is not None:
                kappa, theta, xi, rho, v0 = prev_heston
            else:
                continue
        if not np.isfinite(mu):
            mu = 0.0
        rows.append({
            "date": pd.Timestamp(t_u),
            "window_start": window.index.min() if len(window) else pd.Timestamp(t_u),
            "window_end": window.index.max() if len(window) else pd.Timestamp(t_u),
            "n_days": n,
            "mu": mu,
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "v0": v0,
            "lam": lam,
            "mu_j": mu_j,
            "sigma_j": sigma_j,
            "kappa_j": kappa_j,
        })
'''

HESTON_TABLE = r'''| Parameter | Estimator (V3 — option-implied nonlinear least squares) |
|-----------|----------------------------------------------------------|
| \(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) | \(\min\sum_i w_i\big(C^{\mathrm{Heston}}(K_i,T_i)-C^{\mathrm{mkt}}_i\big)^2\) using the Fourier / characteristic-function European call |
| Market quotes | listed calls in the lookback ending at the update date (moneyness \(0.8\)–\(1.2\), DTE \(5\)–\(365\)) |
| \(\hat\mu\) | \(\bar r \times N_{\mathrm{days}}\) (physical drift for §5; LSM uses \(\mu\to r\)) |
'''

HM_TABLE = r'''| Parameter | Estimator (V3 — option-implied Bates + jump threshold) |
|-----------|----------------------------------------------------------|
| \(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) | \(\min\sum_i w_i\big(C^{\mathrm{Bates}}(K_i,T_i)-C^{\mathrm{mkt}}_i\big)^2\) with jumps held from the return window |
| Market quotes | listed calls in the lookback ending at the update date (moneyness \(0.8\)–\(1.2\), DTE \(5\)–\(365\)) |
| \(\hat\mu\) | \(\bar r \times N_{\mathrm{days}}\) (physical drift for §5; LSM uses \(\mu\to r\)) |
| Jump bars | \(|r_t| > c\cdot\hat\sigma\) with \(c=3\) |
| \(\hat\lambda\) | \(n_{\mathrm{jumps}} / Y\) |
| \(\hat\mu_J,\hat\sigma_J\) | mean / std of jump-bar returns |
| \(\kappa_J\) | \(e^{\hat\mu_J + \hat\sigma_J^2/2}-1\) (derived) |
'''


def _src(cell: dict) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else s


def _set_src(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line + "\n" for line in text.split("\n")[:-1]]
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def _replace_estimate(s: str, with_jumps: bool) -> str:
    new = NEW_ESTIMATE_HM if with_jumps else NEW_ESTIMATE_HESTON
    name = "estimate_heston_merton_params" if with_jumps else "estimate_heston_params"
    pat = rf"def {name}\([\s\S]*?return .*\n"
    if not re.search(pat, s):
        raise RuntimeError(f"could not find {name}")
    s = re.sub(pat, new, s, count=1)
    if "def _heston_opt_dir" not in s:
        s = IMPORT_BLOCK + s
    return s


def _replace_loop(s: str, with_jumps: bool) -> str:
    new_loop = HM_LOOP if with_jumps else HESTON_LOOP
    pat = (
        r"    for t_u in update_dates:\n"
        r"        window = _slice_window\(rets, pd\.Timestamp\(t_u\), offset\)\n"
        r"[\s\S]*?"
        r"        \}\)\n"
        r"    return pd\.DataFrame\(rows\)"
    )
    if not re.search(pat, s):
        raise RuntimeError("could not find calibrate_ticker loop")
    s = re.sub(pat, new_loop + "    return pd.DataFrame(rows)", s, count=1)
    return s


def _replace_table(s: str, with_jumps: bool) -> str:
    new_table = HM_TABLE if with_jumps else HESTON_TABLE
    pat = (
        r"\| Parameter \| Estimator \(Method A — historical moments; no RNG\) \|\n"
        r"\|-----------\|-+\|\n"
        r"(?:\|.*\|\n)+"
    )
    if re.search(pat, s):
        s = re.sub(pat, lambda _m: new_table, s, count=1)
    return s


def _strip_dead_method_a(s: str) -> str:
    """Remove unreachable Method A body left after the new estimate function."""
    s = re.sub(
        r'(return mu, cal\["kappa"\], cal\["theta"\], cal\["xi"\], cal\["rho"\], cal\["v0"\]'
        r'(?:, lam, mu_j, sigma_j, kappa_j)?, n\n)'
        r'[\s\S]*?'
        r'(def _slice_window)',
        r"\1\n\2",
        s,
        count=1,
    )
    return s


def convert_text(s: str, with_jumps: bool) -> str:
    s = s.replace(
        "**Estimation:** Method A only (realized-variance moments; no jump threshold). No MC in calibration.",
        "**Estimation:** V3 option-implied calibration of \\(\\kappa,\\theta,\\xi,\\rho,v_0\\) from listed call prices (Fourier / characteristic function). Monte Carlo and LSM unchanged.",
    )
    s = s.replace(
        "**Estimation:** Method A only (realized-variance moments + jump threshold). No MC in calibration.",
        "**Estimation:** V3 option-implied Bates calibration of \\(\\kappa,\\theta,\\xi,\\rho,v_0\\) from listed call prices; jump block from a return threshold. Monte Carlo and LSM unchanged.",
    )
    s = s.replace("## 3. Estimation formulas (Heston — Method A)", "## 3. Estimation formulas (Heston — option-implied)")
    s = s.replace(
        "## 3. Estimation formulas (Heston–Merton — Method A)",
        "## 3. Estimation formulas (Heston–Merton — option-implied)",
    )
    s = s.replace(
        r"re-estimate \(\hat\mu,\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0,\hat\lambda,\hat\mu_J,\hat\kappa_J\) (and \(\hat\sigma_J\)) from the current window",
        r"re-calibrate \(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) to listed calls in the current lookback (jumps from the return window)",
    )
    s = s.replace(
        r"re-estimate \(\hat\mu,\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) from the current window",
        r"re-calibrate \(\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\) to listed calls in the current lookback",
    )
    s = s.replace(
        "MIN_WINDOW = 60  # trading days required for Method A moments",
        "MIN_WINDOW = 60  # unused by option-implied Heston block; kept for return-window helpers",
    )
    s = s.replace(
        "        \"Method A moments: μ̂, θ̂, κ̂, ξ̂, ρ̂, λ̂. No Monte Carlo here.\"",
        "        \"Option-implied: κ̂, θ̂, ξ̂, ρ̂, v̂₀ from listed calls (Fourier). μ̂ from lookback returns.\"",
    )
    s = s.replace(
        '            f"**Calibration updated (Method A):** lookback=`{window_label}`, rolling=`{rolling_mode}` "',
        '            f"**Calibration updated (option-implied):** lookback=`{window_label}`, rolling=`{rolling_mode}` "',
    )
    s = s.replace(
        '                f"{cal_meta.get(\'rolling_mode\')} / {cal_meta.get(\'window_label\')} | Method A",',
        '                f"{cal_meta.get(\'rolling_mode\')} / {cal_meta.get(\'window_label\')} | option-implied",',
    )
    s = _replace_table(s, with_jumps)
    if "def estimate_heston" in s:
        if "from heston_option_calibration import" not in s:
            s = _replace_estimate(s, with_jumps)
            s = _replace_loop(s, with_jumps)
        s = _strip_dead_method_a(s)
    return s


def patch_notebook(path: Path, with_jumps: bool) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        _set_src(cell, convert_text(_src(cell), with_jumps))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}")


def main() -> int:
    heston_dir = ROOT / "heston notebook"
    hm_dir = ROOT / "heston merton notebook"
    for p in sorted(heston_dir.glob("*_heston.ipynb")):
        patch_notebook(p, with_jumps=False)
    for p in sorted(hm_dir.glob("*_heston_merton.ipynb")):
        patch_notebook(p, with_jumps=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
