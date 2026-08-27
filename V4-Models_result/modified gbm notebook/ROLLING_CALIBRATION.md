# Modified GBM notebooks — true rolling-window calibration

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
