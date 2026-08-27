"""Modified GBM with Adam / PyTorch parameter estimation.

Same three-stage price model as Modified GBM (Markov sign, split |N| sizes,
S ← S e^r). Parameters are not the closed-form counts. They start at those
counts, then Adam shrinks the gap between simulated one-step returns and the
lookback window.

Used by `modified gbm ai notebook/` and the 1.5-year monthly 10k OS study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SIM = 4096
N_ADAM = 80
ADAM_LR = 0.08


def _moment_init(log_rets: pd.Series) -> dict | None:
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
        "p_du": float(1.0 - p_uu),
        "p_ud": float(1.0 - p_dd),
        "p_dd": float(p_dd),
        "mu_u": mu_u,
        "sig_u": sig_u,
        "mu_d": mu_d,
        "sig_d": sig_d,
        "last_up": 1.0 if bool(up[-1]) else 0.0,
        "p_u": float(up.mean()),
        "signed": signed,
        "up": up,
        "prev": prev,
        "curr": curr,
        "mag": mag,
    }


def _targets(init: dict) -> dict:
    signed = init["signed"]
    up = init["up"]
    mag = init["mag"]
    prev = init["prev"]
    curr = init["curr"]
    return {
        "p_uu": float(curr[prev].mean()) if prev.any() else init["p_uu"],
        "p_dd": float((~curr[~prev]).mean()) if (~prev).any() else init["p_dd"],
        "mu_u": float(mag[up].mean()) if up.any() else init["mu_u"],
        "sig_u": float(mag[up].std(ddof=1)) if up.sum() >= 2 else init["sig_u"],
        "mu_d": float(mag[~up].mean()) if (~up).any() else init["mu_d"],
        "sig_d": float(mag[~up].std(ddof=1)) if (~up).sum() >= 2 else init["sig_d"],
        "mean_r": float(signed.mean()),
        "std_r": float(signed.std(ddof=1)) if signed.size >= 2 else float(np.abs(signed).mean()),
        "p_u": float(up.mean()),
    }


def estimate_modified_gbm(log_rets: pd.Series):
    """Adam fit of Modified GBM on lookback log returns.

    Start at the usual count / split-normal moments. Simulate one-step
    returns in PyTorch, MSE vs those moments, Adam on unconstrained
    (logit p, log μ, log σ). last_up stays the last observed sign.
    """
    init = _moment_init(log_rets)
    if init is None:
        return None
    tgt = _targets(init)
    try:
        import torch
    except ImportError:
        return _fit_adam_numpy(init, tgt)

    torch.manual_seed(0)
    device = torch.device("cpu")
    logit = lambda p: torch.tensor(
        float(np.log(np.clip(p, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(p, 1e-4, 1.0 - 1e-4)))),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    logp = lambda x: torch.tensor(
        float(np.log(max(x, 1e-8))),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    z_uu = logit(init["p_uu"])
    z_dd = logit(init["p_dd"])
    z_mu_u = logp(init["mu_u"])
    z_sig_u = logp(init["sig_u"])
    z_mu_d = logp(init["mu_d"])
    z_sig_d = logp(init["sig_d"])
    opt = torch.optim.Adam([z_uu, z_dd, z_mu_u, z_sig_u, z_mu_d, z_sig_d], lr=ADAM_LR)

    t_p_uu = torch.tensor(tgt["p_uu"], dtype=torch.float64, device=device)
    t_p_dd = torch.tensor(tgt["p_dd"], dtype=torch.float64, device=device)
    t_mu_u = torch.tensor(tgt["mu_u"], dtype=torch.float64, device=device)
    t_sig_u = torch.tensor(max(tgt["sig_u"], 1e-8), dtype=torch.float64, device=device)
    t_mu_d = torch.tensor(tgt["mu_d"], dtype=torch.float64, device=device)
    t_sig_d = torch.tensor(max(tgt["sig_d"], 1e-8), dtype=torch.float64, device=device)
    t_mean = torch.tensor(tgt["mean_r"], dtype=torch.float64, device=device)
    t_std = torch.tensor(max(tgt["std_r"], 1e-8), dtype=torch.float64, device=device)
    t_p_u = torch.tensor(tgt["p_u"], dtype=torch.float64, device=device)
    p_u0 = float(init["p_u"])
    last_loss = None

    for _ in range(N_ADAM):
        opt.zero_grad()
        p_uu = torch.sigmoid(z_uu)
        p_dd = torch.sigmoid(z_dd)
        mu_u = torch.exp(z_mu_u)
        sig_u = torch.exp(z_sig_u).clamp(min=1e-8)
        mu_d = torch.exp(z_mu_d)
        sig_d = torch.exp(z_sig_d).clamp(min=1e-8)

        prev = (torch.rand(N_SIM, dtype=torch.float64, device=device) < p_u0).to(torch.float64)
        p_up = prev * p_uu + (1.0 - prev) * (1.0 - p_dd)
        # Relaxed Bernoulli so direction has a gradient
        u = torch.rand(N_SIM, dtype=torch.float64, device=device)
        temp = 0.25
        up = torch.sigmoid((torch.log(u.clamp(1e-6, 1 - 1e-6)) - torch.log(1.0 - u.clamp(1e-6, 1 - 1e-6)) + torch.log(p_up.clamp(1e-6, 1 - 1e-6)) - torch.log(1.0 - p_up.clamp(1e-6, 1 - 1e-6))) / temp)

        z = torch.randn(N_SIM, dtype=torch.float64, device=device)
        mag = torch.abs(up * (mu_u + sig_u * z) + (1.0 - up) * (mu_d + sig_d * z)).clamp(min=1e-16)
        r = (2.0 * up - 1.0) * mag

        sim_p_uu = (prev * up).sum() / prev.sum().clamp(min=1.0)
        sim_p_dd = ((1.0 - prev) * (1.0 - up)).sum() / (1.0 - prev).sum().clamp(min=1.0)
        w_u = up
        w_d = 1.0 - up
        sim_mu_u = (w_u * mag).sum() / w_u.sum().clamp(min=1.0)
        sim_mu_d = (w_d * mag).sum() / w_d.sum().clamp(min=1.0)
        sim_sig_u = torch.sqrt(((w_u * (mag - sim_mu_u) ** 2).sum() / w_u.sum().clamp(min=2.0)).clamp(min=1e-16))
        sim_sig_d = torch.sqrt(((w_d * (mag - sim_mu_d) ** 2).sum() / w_d.sum().clamp(min=2.0)).clamp(min=1e-16))
        sim_mean = r.mean()
        sim_std = r.std(unbiased=True).clamp(min=1e-8)
        sim_p_u = up.mean()

        scale = t_std.detach().clamp(min=1e-4)
        loss = (
            (sim_p_uu - t_p_uu) ** 2
            + (sim_p_dd - t_p_dd) ** 2
            + ((sim_mu_u - t_mu_u) / scale) ** 2
            + ((sim_mu_d - t_mu_d) / scale) ** 2
            + ((sim_sig_u - t_sig_u) / scale) ** 2
            + ((sim_sig_d - t_sig_d) / scale) ** 2
            + ((sim_mean - t_mean) / scale) ** 2
            + ((sim_std - t_std) / scale) ** 2
            + (sim_p_u - t_p_u) ** 2
        )
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())

    with torch.no_grad():
        p_uu = float(torch.sigmoid(z_uu))
        p_dd = float(torch.sigmoid(z_dd))
        mu_u = float(torch.exp(z_mu_u))
        sig_u = float(torch.exp(z_sig_u))
        mu_d = float(torch.exp(z_mu_d))
        sig_d = float(torch.exp(z_sig_d))

    return {
        "n_days": init["n_days"],
        "p_uu": p_uu,
        "p_du": 1.0 - p_uu,
        "p_ud": 1.0 - p_dd,
        "p_dd": p_dd,
        "mu_u": max(mu_u, 1e-8),
        "sig_u": max(sig_u, 1e-8),
        "mu_d": max(mu_d, 1e-8),
        "sig_d": max(sig_d, 1e-8),
        "last_up": init["last_up"],
        "p_u": init["p_u"],
        "ai_loss": last_loss,
        "ai_adam": 1.0,
    }


def _pack(init: dict) -> np.ndarray:
    def logit(p):
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return np.log(p / (1.0 - p))

    return np.array(
        [
            logit(init["p_uu"]),
            logit(init["p_dd"]),
            np.log(max(init["mu_u"], 1e-8)),
            np.log(max(init["sig_u"], 1e-8)),
            np.log(max(init["mu_d"], 1e-8)),
            np.log(max(init["sig_d"], 1e-8)),
        ],
        dtype=float,
    )


def _unpack(z: np.ndarray) -> dict:
    sig = 1.0 / (1.0 + np.exp(-z[:2]))
    mag = np.exp(z[2:])
    return {
        "p_uu": float(sig[0]),
        "p_dd": float(sig[1]),
        "mu_u": float(mag[0]),
        "sig_u": float(max(mag[1], 1e-8)),
        "mu_d": float(mag[2]),
        "sig_d": float(max(mag[3], 1e-8)),
    }


def _sim_loss(z: np.ndarray, tgt: dict, p_u0: float, rng: np.random.Generator) -> float:
    p = _unpack(z)
    prev = rng.random(N_SIM) < p_u0
    p_up = np.where(prev, p["p_uu"], 1.0 - p["p_dd"])
    up = rng.random(N_SIM) < p_up
    mag = np.empty(N_SIM, dtype=float)
    n_u = int(up.sum())
    n_d = N_SIM - n_u
    if n_u:
        mag[up] = np.abs(rng.normal(p["mu_u"], p["sig_u"], size=n_u))
    if n_d:
        mag[~up] = np.abs(rng.normal(p["mu_d"], p["sig_d"], size=n_d))
    mag = np.maximum(mag, 1e-16)
    r = np.where(up, mag, -mag)
    scale = max(tgt["std_r"], 1e-4)
    sim_p_uu = float(up[prev].mean()) if prev.any() else p["p_uu"]
    sim_p_dd = float((~up[~prev]).mean()) if (~prev).any() else p["p_dd"]
    sim_mu_u = float(mag[up].mean()) if n_u else p["mu_u"]
    sim_mu_d = float(mag[~up].mean()) if n_d else p["mu_d"]
    sim_sig_u = float(mag[up].std(ddof=1)) if n_u >= 2 else p["sig_u"]
    sim_sig_d = float(mag[~up].std(ddof=1)) if n_d >= 2 else p["sig_d"]
    terms = [
        sim_p_uu - tgt["p_uu"],
        sim_p_dd - tgt["p_dd"],
        (sim_mu_u - tgt["mu_u"]) / scale,
        (sim_mu_d - tgt["mu_d"]) / scale,
        (sim_sig_u - tgt["sig_u"]) / scale,
        (sim_sig_d - tgt["sig_d"]) / scale,
        (float(r.mean()) - tgt["mean_r"]) / scale,
        (float(r.std(ddof=1)) - tgt["std_r"]) / scale,
        float(up.mean()) - tgt["p_u"],
    ]
    return float(np.dot(terms, terms))


def _fit_adam_numpy(init: dict, tgt: dict) -> dict:
    """Adam on simulated-moment MSE if PyTorch is not installed."""
    rng = np.random.default_rng(0)
    z = _pack(init)
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    last = None
    eps_fd = 1e-3
    p_u0 = float(init["p_u"])
    for t in range(1, N_ADAM + 1):
        last = _sim_loss(z, tgt, p_u0, rng)
        g = np.empty_like(z)
        for i in range(z.size):
            zp = z.copy()
            zp[i] += eps_fd
            g[i] = (_sim_loss(zp, tgt, p_u0, rng) - last) / eps_fd
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mhat = m / (1.0 - b1**t)
        vhat = v / (1.0 - b2**t)
        z = z - ADAM_LR * mhat / (np.sqrt(vhat) + eps)
    p = _unpack(z)
    return {
        "n_days": init["n_days"],
        "p_uu": p["p_uu"],
        "p_du": 1.0 - p["p_uu"],
        "p_ud": 1.0 - p["p_dd"],
        "p_dd": p["p_dd"],
        "mu_u": max(p["mu_u"], 1e-8),
        "sig_u": max(p["sig_u"], 1e-8),
        "mu_d": max(p["mu_d"], 1e-8),
        "sig_d": max(p["sig_d"], 1e-8),
        "last_up": init["last_up"],
        "p_u": init["p_u"],
        "ai_loss": last,
        "ai_adam": 1.0,
    }
