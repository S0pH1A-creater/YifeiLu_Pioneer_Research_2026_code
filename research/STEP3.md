# Research Step 3 — American optimal stopping (LSM)

**Status: implemented in regime notebooks (§6)**

After §4 calibration and §5 Monte Carlo stock paths, each model/regime notebook prices **SPY American calls** with Longstaff–Schwartz optimal stopping.

## Where

| Model folder | Regime files | Section |
|--------------|--------------|---------|
| [`gbm notebook/`](gbm%20notebook/) | `2008-2009_gbm.ipynb` … `2019-2020_gbm.ipynb` | §6 |
| [`merton notebook/`](merton%20notebook/) | `*_merton.ipynb` | §6 |
| [`heston merton notebook/`](heston%20merton%20notebook/) | `*_heston_merton.ipynb` | §6 |
| [`garch merton notebook/`](garch%20merton%20notebook/) | `*_garch_merton.ipynb` | §6 |

**Not included:** `heston merton advanced notebook/` (Method B) — out of scope for this step.

Shared helper: [`scripts/american_lsm.py`](scripts/american_lsm.py).

## Method

1. Calibrate in §4 (**Reestimate**).
2. Optionally run §5 Start (path visualization).
3. §6 **Compute stopping**:
   - Sample SPY calls from `data/options/processed/SPY_calls_panel.csv` in the period
   - Simulate **risk-neutral** paths to each option’s expiry with the **same §5 simulator** (drift \(\mu \rightarrow r\); vol/jumps from §4)
   - Longstaff–Schwartz: exercise if \(\max(S_t-K,0) > \widehat{C}(S_t)\); continuation from polynomial regression on ITM paths
4. Results table + charts stored in notebook variable `stopping_results`

## How to run

Open a regime notebook → run all (or §0–§4, then §6) → click **Compute stopping**.
