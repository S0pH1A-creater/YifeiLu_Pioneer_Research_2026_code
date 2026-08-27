# GARCH–Merton Duan + Pan — monthly 10k re-run

Not in `Results_In_Short`. Sibling cache to `empirical_study_1p5y_monthly_10000/`.

## What changed

GARCH–Merton \(P\to Q\):

| Block | Method | Map |
|-------|--------|-----|
| GARCH | Duan (1995) LRNVR | \(\varepsilon\mapsto\xi=\varepsilon+\lambda\) |
| Jumps | Pan (2002) | \(\mu_J\mapsto\mu_J^*\); \(\lambda^*=\lambda\) |

Previously this model used only \(\mu\mapsto r_f\).

## Contents

- `payload.json` — seven-model tables (GARCH–Merton re-priced; other six copied from parent monthly cache)
- `partial/*GARCH-Merton.json` — new cells
- `contracts/**/ *garch_merton.csv` — contract-level LSM
- `pq_parameters.csv` — last-update P / premium / Q dumps
- `../empirical_study_1p5y_monthly_10000_garch_merton_duan_pan_run.log` — run log

## Reproduce

```bash
cd V4-Models_result
python scripts/patch_v4_garch_merton_pq.py   # once
python scripts/run_v4_1p5y_10k_monthly_garch_merton_duan_pan.py --recompute
```
