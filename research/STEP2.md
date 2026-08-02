# Research Step 2 — Status (Interactive Model Playgrounds)

**Status: ready to run**  
Small step only: Monte Carlo visualization with parameter sliders. **No market data** — synthetic paths only, to feel how each model behaves before fitting.

**Model guide:** **[`MODELS_EXPLAINED.ipynb`](MODELS_EXPLAINED.ipynb)** — open in Jupyter for rendered equations (also at [`notebooks/00_MODELS_EXPLAINED.ipynb`](notebooks/00_MODELS_EXPLAINED.ipynb)).

Checklist from [`../docs/ResearchProposal-v2.md`](../docs/ResearchProposal-v2.md) (model stack only):

| # | Item | Status |
|---|------|--------|
| 1 | GBM playground notebook (sliders + MC paths) | ✓ |
| 2 | Merton jump-diffusion playground | ✓ |
| 3 | Heston–Merton playground | ✓ |
| 4 | GARCH–Merton playground | ✓ |

Later steps (not this file): fit to real returns, American pricing / optimal stopping, regime comparisons.

## Scope

| In scope | Out of scope |
|----------|--------------|
| One `.ipynb` per model | Calibrating to SPY / AAPL / MSFT |
| `ipywidgets` sliders for parameters | American option pricing |
| Monte Carlo price paths + return histogram | RMSE vs history |
| Synthetic \(S_0\), \(T\), paths only | Regime / real data joins |

## Notebooks

Each of `01`–`04` now includes a **full beginner workflow** (collect prices → estimate parameters from history → which params stay constant vs update along a Monte Carlo path) **plus** the original interactive playground.


| Notebook | Model | Key sliders |
|----------|-------|-------------|
| [`notebooks/01_gbm.ipynb`](notebooks/01_gbm.ipynb) | Geometric Brownian Motion | \(\mu\), \(\sigma\), \(S_0\), \(T\), steps, paths |
| [`notebooks/02_merton.ipynb`](notebooks/02_merton.ipynb) | Merton jump diffusion | GBM + \(\lambda\), \(\mu_J\), \(\sigma_J\) |
| [`notebooks/03_heston_merton.ipynb`](notebooks/03_heston_merton.ipynb) | Heston–Merton | Heston \((\kappa,\theta,\xi,\rho,v_0)\) + jumps |
| [`notebooks/04_garch_merton.ipynb`](notebooks/04_garch_merton.ipynb) | GARCH–Merton | GARCH \((\omega,\alpha,\beta)\) + jumps |

```bash
cd research
../.venv/bin/pip install -r requirements-research.txt
../.venv/bin/jupyter notebook notebooks/01_gbm.ipynb
# or: ../.venv/bin/jupyter lab notebooks/
```

Run each notebook, move the sliders, watch paths and the return histogram update.

## Deliverables

```
research/notebooks/
  01_gbm.ipynb
  02_merton.ipynb
  03_heston_merton.ipynb
  04_garch_merton.ipynb
```

## Next (Step 3)

Fit **GBM** (then Merton) to prepared log returns by regime — still no American pricing yet. Keep each step small.
