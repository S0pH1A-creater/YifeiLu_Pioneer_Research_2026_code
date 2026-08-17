#!/usr/bin/env python3
"""Clone Heston–Merton notebooks into Heston-only (no jump block)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

NEW_SIM = '''def simulate_heston_rolling(
    mu_step, kappa_step, theta_step, xi_step, rho_step, v0_step,
    S0, n_paths, seed,
):
    """Heston Euler MC (no jumps); params may change by step (rolling schedule)."""
    rng = np.random.default_rng(seed)
    n_steps = len(mu_step)
    dt = 1.0 / N_DAYS
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    v = np.full(n_paths, float(v0_step[0]), dtype=float)

    for i in range(n_steps):
        mu = mu_step[i]
        kappa = kappa_step[i]
        theta = theta_step[i]
        xi = xi_step[i]
        rho = float(np.clip(rho_step[i], -0.999, 0.999))

        z_v = rng.standard_normal(n_paths)
        z_indep = rng.standard_normal(n_paths)
        z_s = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * z_indep

        v_pos = np.maximum(v, 0.0)
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * z_v
        v = np.maximum(v, 0.0)
        v_pos = np.maximum(v, 0.0)

        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * v_pos) * dt
            + np.sqrt(v_pos * dt) * z_s
        )
    return paths
'''

PAIRS = [
    (
        REPO / "V2-Models_result" / "heston merton notebook",
        REPO / "V2-Models_result" / "heston notebook",
    ),
    (
        REPO / "V1-Models_result" / "heston merton notebook",
        REPO / "V1-Models_result" / "heston notebook",
    ),
]


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


def convert_text(s: str) -> str:
    s = re.sub(
        r"def simulate_heston_merton_rolling\([\s\S]*?return paths\n",
        NEW_SIM,
        s,
        count=1,
    )

    s = s.replace(
        "def estimate_heston_merton_params(log_rets: pd.Series, jump_thresh: float = JUMP_THRESH):",
        "def estimate_heston_params(log_rets: pd.Series):",
    )
    s = s.replace(
        "Method A: μ, κ, θ, ξ, ρ, v0, λ, μ_J, σ_J, κ_J from a lookback window (no RNG).",
        "Method A: μ, κ, θ, ξ, ρ, v0 from a lookback window (no RNG, no jumps).",
    )
    s = s.replace(
        "    - Continuous variance (θ, v0) from non-jump days only so jump second\n"
        "      moments are not also loaded into the Heston variance level.\n",
        "    - Variance (θ, v0) from all returns in the window (no jump filter).\n",
    )
    s = s.replace("nan = (np.nan,) * 10 + (n,)", "nan = (np.nan,) * 6 + (n,)")
    s = s.replace(
        "    jump_mask = np.abs(x.values) > jump_thresh * sigma_day\n"
        "    jumps = x.iloc[jump_mask]\n"
        "    normal = x.iloc[~jump_mask]\n"
        "    base = normal if int(normal.shape[0]) >= 2 else x\n"
        "    mu = float(base.mean() * N_DAYS)\n",
        "    base = x\n"
        "    mu = float(base.mean() * N_DAYS)\n",
    )
    s = re.sub(
        r"\n    years = n / float\(N_DAYS\)\n"
        r"    n_jumps = int\(jump_mask\.sum\(\)\)\n"
        r"    lam = float\(n_jumps / years\) if years > 0 else 0\.0\n\n"
        r"    if n_jumps >= 2:\n"
        r"        mu_j = float\(jumps\.mean\(\)\)\n"
        r"        sigma_j = float\(jumps\.std\(ddof=1\)\)\n"
        r"    elif n_jumps == 1:\n"
        r"        mu_j = float\(jumps\.iloc\[0\]\)\n"
        r"        sigma_j = 0\.0\n"
        r"    else:\n"
        r"        mu_j = 0\.0\n"
        r"        sigma_j = 0\.0\n"
        r"    if not np\.isfinite\(sigma_j\) or sigma_j < 0:\n"
        r"        sigma_j = 0\.0\n\n"
        r"    kappa_j = float\(np\.exp\(mu_j \+ 0\.5 \* sigma_j\*\*2\) - 1\.0\)\n"
        r"    return mu, kappa, theta, xi, rho, v0, lam, mu_j, sigma_j, kappa_j, n\n",
        "\n    return mu, kappa, theta, xi, rho, v0, n\n",
        s,
        count=1,
    )
    s = s.replace(
        "        mu, kappa, theta, xi, rho, v0, lam, mu_j, sigma_j, kappa_j, n = estimate_heston_merton_params(window)",
        "        mu, kappa, theta, xi, rho, v0, n = estimate_heston_params(window)",
    )
    s = s.replace(
        '            "v0": v0,\n'
        '            "lam": lam,\n'
        '            "mu_j": mu_j,\n'
        '            "sigma_j": sigma_j,\n'
        '            "kappa_j": kappa_j,\n',
        '            "v0": v0,\n',
    )
    s = s.replace(
        '    cols = ["mu", "kappa", "theta", "xi", "rho", "v0", "lam", "mu_j", "sigma_j", "kappa_j"]',
        '    cols = ["mu", "kappa", "theta", "xi", "rho", "v0"]',
    )
    s = s.replace(
        '        steps["v0"], steps["lam"], steps["mu_j"], steps["sigma_j"], steps["kappa_j"],\n'
        "        float(hist.iloc[0]), hist,\n",
        '        steps["v0"],\n'
        "        float(hist.iloc[0]), hist,\n",
    )
    s = s.replace(
        '    """Six rolling-parameter graphs: μ, θ, κ, ξ, ρ, λ."""\n'
        "    panels = [\n"
        '        ("mu", "μ̂ (annual)", "Estimated drift"),\n'
        '        ("theta", "θ̂ (var)", "Long-run variance"),\n'
        '        ("kappa", "κ̂", "Mean-reversion speed"),\n'
        '        ("xi", "ξ̂", "Vol-of-vol"),\n'
        '        ("rho", "ρ̂", "Price–vol correlation"),\n'
        '        ("lam", "λ̂ (jumps/year)", "Estimated jump intensity"),\n'
        "    ]",
        '    """Six rolling-parameter graphs: μ, θ, κ, ξ, ρ, v0."""\n'
        "    panels = [\n"
        '        ("mu", "μ̂ (annual)", "Estimated drift"),\n'
        '        ("theta", "θ̂ (var)", "Long-run variance"),\n'
        '        ("kappa", "κ̂", "Mean-reversion speed"),\n'
        '        ("xi", "ξ̂", "Vol-of-vol"),\n'
        '        ("rho", "ρ̂", "Price–vol correlation"),\n'
        '        ("v0", "v̂₀ (var)", "Initial variance"),\n'
        "    ]",
    )
    s = s.replace(
        "        Method A moments: μ̂, θ̂, κ̂, ξ̂, ρ̂, λ̂. No Monte Carlo here.",
        "        Method A moments: μ̂, θ̂, κ̂, ξ̂, ρ̂, v̂₀. No Monte Carlo here.",
    )
    s = s.replace(
        '            "Also estimated in the same windows (for simulation): "\n'
        '            r"$v_0$, $\\mu_J$, $\\sigma_J$, $\\kappa_J$."\n',
        '            "Also estimated in the same windows (for simulation): "\n'
        '            r"$v_0$ initializes each path; $v_t$ then follows the Heston SDE."\n',
    )
    s = s.replace(
        "            dates_now, mu_now, kappa_now, theta_now, xi_now, rho_now, v0_now,\n"
        "            lam_now, muj_now, sj_now, kapj_now, S0_now, hist_now,\n",
        "            dates_now, mu_now, kappa_now, theta_now, xi_now, rho_now, v0_now,\n"
        "            S0_now, hist_now,\n",
    )
    s = s.replace(
        "            mu_now, kappa_now, theta_now, xi_now, rho_now, v0_now,\n"
        "            lam_now, muj_now, sj_now, kapj_now, S0_now, n_paths, seed,\n",
        "            mu_now, kappa_now, theta_now, xi_now, rho_now, v0_now,\n"
        "            S0_now, n_paths, seed,\n",
    )
    s = s.replace(
        '    lam_step = np.full(dte, float(p["lam"]), dtype=float)\n'
        '    muj_step = np.full(dte, float(p["mu_j"]), dtype=float)\n'
        '    sj_step = np.full(dte, float(p["sigma_j"]), dtype=float)\n'
        '    kapj_step = np.full(dte, float(p["kappa_j"]), dtype=float)\n',
        "",
    )
    s = s.replace(
        '    lam_step = np.full(n_steps, float(p["lam"]), dtype=float)\n'
        '    muj_step = np.full(n_steps, float(p["mu_j"]), dtype=float)\n'
        '    sj_step = np.full(n_steps, float(p["sigma_j"]), dtype=float)\n'
        '    kapj_step = np.full(n_steps, float(p["kappa_j"]), dtype=float)\n',
        "",
    )
    s = s.replace(
        r"| \(\hat\mu\) | \(\bar r_{\text{non-jump}} \times (252\times 390)\) |",
        r"| \(\hat\mu\) | \(\bar r \times (252\times 390)\) |",
    )
    s = s.replace(
        "| Jump bars | \\(|r_t| > c\\cdot\\hat\\sigma_{\\text{1min}}\\) with \\(c=3\\) |\n"
        "| \\(\\hat\\lambda\\) | \\(n_{\\text{jumps}} / Y\\) |\n"
        "| \\(\\hat\\mu_J,\\hat\\sigma_J\\) | mean / std of jump-bar returns |\n"
        "| \\(\\kappa_J\\) | \\(e^{\\hat\\mu_J + \\hat\\sigma_J^2/2}-1\\) (derived) |\n",
        "",
    )
    s = s.replace(
        "        mu_step, kappa_step, theta_step, xi_step, rho_step, v0_step,\n"
        "        lam_step, muj_step, sj_step, kapj_step, S0, n_paths, seed,\n",
        "        mu_step, kappa_step, theta_step, xi_step, rho_step, v0_step,\n"
        "        S0, n_paths, seed,\n",
    )
    s = s.replace(
        "JUMP_THRESH = 3.0  # flag |r| > c * daily σ as a jump\n",
        "",
    )
    s = s.replace(
        "JUMP_THRESH = 3.0  # flag |r| > c * 1-min σ as a jump\n",
        "",
    )

    # formulas / copy
    s = s.replace(
        r"$$\frac{dS_t}{S_{t-}} = (\mu - \lambda\kappa_J)\, dt + \sqrt{v_t}\, dW_t^S + (e^J - 1)\, dN_t$$",
        r"$$\frac{dS_t}{S_t} = \mu\, dt + \sqrt{v_t}\, dW_t^S$$",
    )
    s = s.replace(
        r"$$S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\lambda\kappa_J-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S+\sum_{i=1}^{N_{\Delta t}}J_i\Big)$$",
        r"$$S_{t+\Delta t}=S_t\exp\Big(\big(\mu-\tfrac12 v_t\big)\Delta t+\sqrt{v_t}\sqrt{\Delta t}\,Z_S\Big)$$",
    )
    s = s.replace(
        r"with \(\mathrm{Corr}(Z_S,Z_v)=\rho\), \(N_{\Delta t}\sim\mathrm{Poisson}(\lambda\Delta t)\), \(J_i\sim N(\mu_J,\sigma_J^2)\), and \(\kappa_J=e^{\mu_J+\sigma_J^2/2}-1\).",
        r"with \(\mathrm{Corr}(Z_S,Z_v)=\rho\).",
    )
    s = s.replace(
        r"| \(\hat\mu\) | \(\bar r_{\text{non-jump}} \times 252\) |",
        r"| \(\hat\mu\) | \(\bar r \times N_{\mathrm{days}}\) |",
    )
    s = s.replace(
        r"| Jump days | \(|r_t| > c\cdot\hat\sigma_{\text{day}}\) with \(c=3\) |" + "\n"
        r"| \(\hat\lambda\) | \(n_{\text{jumps}} / Y\) |" + "\n"
        r"| \(\hat\mu_J,\hat\sigma_J\) | mean / std of jump-day returns |" + "\n"
        r"| \(\kappa_J\) | \(e^{\hat\mu_J + \hat\sigma_J^2/2}-1\) (derived) |" + "\n",
        "",
    )
    s = s.replace(
        r"**Six calibrated quantities shown in §4:** \(\mu,\theta,\kappa,\xi,\rho,\lambda\). \(v_0,\mu_J,\sigma_J,\kappa_J\) are also rolled for simulation.",
        r"**Six calibrated quantities shown in §4:** \(\mu,\theta,\kappa,\xi,\rho,v_0\).",
    )
    s = s.replace(
        r"re-estimate \(\hat\mu,\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0,\hat\lambda,\hat\mu_J,\hat\kappa_J\) (and \(\hat\sigma_J\))",
        r"re-estimate \(\hat\mu,\hat\kappa,\hat\theta,\hat\xi,\hat\rho,\hat v_0\)",
    )
    s = s.replace(
        "Method A only (realized-variance moments + jump threshold).",
        "Method A only (realized-variance moments; no jump threshold).",
    )
    s = s.replace(
        "**Estimation:** Method A only (realized-variance moments + jump threshold). No MC in calibration.",
        "**Estimation:** Method A only (realized-variance moments; no jumps). No MC in calibration.",
    )
    s = s.replace("vol/jumps from §4", "Heston variance from §4")
    s = s.replace("Heston–Merton Euler MC", "Heston Euler MC")

    # names last
    s = s.replace("estimate_heston_merton_params", "estimate_heston_params")
    s = s.replace("simulate_heston_merton_rolling", "simulate_heston_rolling")
    s = s.replace("heston_merton", "heston")
    s = s.replace("Heston–Merton", "Heston")
    s = s.replace("Heston-Merton", "Heston")
    return s


def convert_notebook(src: Path, dst: Path) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        _set_src(cell, convert_text(_src(cell)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for src_dir, dst_dir in PAIRS:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.ipynb")):
            if src.name.startswith("."):
                continue
            dst = dst_dir / src.name.replace("_heston_merton.ipynb", "_heston.ipynb")
            convert_notebook(src, dst)
            print(f"wrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
