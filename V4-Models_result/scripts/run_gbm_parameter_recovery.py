#!/usr/bin/env python3
"""GBM parameter-recovery study (implementation check).

Simulate daily prices from known (μ, σ) with the V3 Euler step, hide the
true parameters, and recover them with the notebook estimator:

    μ̂ = mean(r) × 252
    σ̂ = std(r, ddof=1) × √252

where r_t = ln(S_t / S_{t-1}). Repeat over independent seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gbm_parameter_recovery"

TRUE = dict(mu=0.08, sigma=0.20)
S0 = 100.0
N_DAYS = 252
DT = 1.0 / N_DAYS
N_STEPS = 10 * N_DAYS
SEEDS = (42, 7, 123, 2024, 99, 314, 2718, 8675309)


def simulate_gbm_path(mu: float, sigma: float, *, seed: int) -> np.ndarray:
    """Same Euler as V3 `simulate_gbm_rolling` (one path)."""
    rng = np.random.default_rng(seed)
    s = np.empty(N_STEPS + 1, dtype=float)
    s[0] = S0
    drift = (mu - 0.5 * sigma * sigma) * DT
    vol = sigma * np.sqrt(DT)
    for i in range(N_STEPS):
        s[i + 1] = s[i] * np.exp(drift + vol * float(rng.standard_normal()))
    return s


def estimate_mu_sigma(log_rets: np.ndarray) -> dict:
    """V3 notebook estimator: annualized mean and sample std of log returns."""
    x = np.asarray(log_rets, dtype=float)
    x = x[np.isfinite(x)]
    mu = float(x.mean() * N_DAYS)
    sigma = float(x.std(ddof=1) * np.sqrt(N_DAYS))
    return {"mu": mu, "sigma": sigma, "n": int(x.size)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ito_mu = TRUE["mu"] - 0.5 * TRUE["sigma"] ** 2
    rows = []
    print("True (hidden): μ={mu}  σ={sigma}  log-drift μ−½σ²={ito:.4f}".format(
        ito=ito_mu, **TRUE
    ))
    print(f"  {N_STEPS} daily steps, {len(SEEDS)} seeds\n")

    for seed in SEEDS:
        s = simulate_gbm_path(TRUE["mu"], TRUE["sigma"], seed=int(seed))
        r = np.diff(np.log(s))
        hat = estimate_mu_sigma(r)
        mu_ito = hat["mu"] + 0.5 * hat["sigma"] ** 2
        row = {
            "seed": int(seed),
            "S_T": float(s[-1]),
            "mu_hat": hat["mu"],
            "sigma_hat": hat["sigma"],
            "mu_ito_corrected": float(mu_ito),
            "mu_abs": hat["mu"] - TRUE["mu"],
            "mu_rel": (hat["mu"] - TRUE["mu"]) / TRUE["mu"],
            "sigma_abs": hat["sigma"] - TRUE["sigma"],
            "sigma_rel": (hat["sigma"] - TRUE["sigma"]) / TRUE["sigma"],
            "mu_vs_logdrift_abs": hat["mu"] - ito_mu,
            "ito_corrected_abs": mu_ito - TRUE["mu"],
        }
        rows.append(row)
        print(
            f"seed {seed:8d}  μ̂={hat['mu']:+.4f}  σ̂={hat['sigma']:.4f}  "
            f"μ̂+½σ̂²={mu_ito:.4f}  S_T={s[-1]:.2f}"
        )

    def _summ(key_hat, tru):
        hats = np.array([r[key_hat] for r in rows], dtype=float)
        err = hats - tru
        rel = err / tru
        return {
            "true": float(tru),
            "mean": float(hats.mean()),
            "median": float(np.median(hats)),
            "std": float(hats.std(ddof=1)),
            "bias": float(err.mean()),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mape": float(np.mean(np.abs(rel))),
            "median_abs_rel": float(np.median(np.abs(rel))),
        }

    summary = {
        "mu_vs_sde": _summ("mu_hat", TRUE["mu"]),
        "mu_vs_logdrift": _summ("mu_hat", ito_mu),
        "mu_ito_corrected": _summ("mu_ito_corrected", TRUE["mu"]),
        "sigma": _summ("sigma_hat", TRUE["sigma"]),
        "n_seeds": len(rows),
    }

    def _ok(s, tight, loose):
        med = s["median_abs_rel"]
        if med <= tight:
            return "recovered"
        if med <= loose:
            return "partial"
        return "not-recovered"

    summary["verdict_sigma"] = _ok(summary["sigma"], 0.10, 0.25)
    summary["verdict_mu_sde"] = _ok(summary["mu_vs_sde"], 0.20, 0.50)
    summary["verdict_mu_logdrift"] = _ok(summary["mu_vs_logdrift"], 0.20, 0.50)
    summary["verdict_mu_ito"] = _ok(summary["mu_ito_corrected"], 0.20, 0.50)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            x = float(obj)
            return None if not np.isfinite(x) else x
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    payload = {
        "true": TRUE,
        "log_drift": ito_mu,
        "design": {
            "s0": S0,
            "dt": DT,
            "n_steps": N_STEPS,
            "n_years": N_STEPS / N_DAYS,
            "seeds": list(SEEDS),
            "estimator": "μ̂ = mean(r)×252, σ̂ = std(r, ddof=1)×√252",
            "simulator": "S ← S exp((μ−½σ²)Δt + σ√Δt Z)",
        },
        "summary": summary,
        "seeds": rows,
    }
    out = OUT / "recovery.json"
    out.write_text(json.dumps(_clean(payload), indent=2))
    print(f"\nwrote {out}")
    print(f"  σ:     {summary['verdict_sigma']:16s}  mean={summary['sigma']['mean']:.4f}  "
          f"mae={summary['sigma']['mae']:.4f}  med|rel|={summary['sigma']['median_abs_rel']:.1%}")
    print(f"  μ vs SDE μ: {summary['verdict_mu_sde']:12s}  mean={summary['mu_vs_sde']['mean']:.4f}  "
          f"bias={summary['mu_vs_sde']['bias']:+.4f}  med|rel|={summary['mu_vs_sde']['median_abs_rel']:.1%}")
    print(f"  μ vs log-drift: {summary['verdict_mu_logdrift']:8s}  mean={summary['mu_vs_logdrift']['mean']:.4f}  "
          f"bias={summary['mu_vs_logdrift']['bias']:+.4f}")
    print(f"  μ̂+½σ̂² vs SDE μ: {summary['verdict_mu_ito']:6s}  mean={summary['mu_ito_corrected']['mean']:.4f}  "
          f"bias={summary['mu_ito_corrected']['bias']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
