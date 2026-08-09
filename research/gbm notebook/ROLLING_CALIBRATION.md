# GBM notebooks — true rolling-window calibration

**Do not** lock Monte Carlo to the first-day parameters for the whole forecast period.

## Required behavior

1. At each update point (daily / monthly / or once if “none”), re-estimate **all** GBM parameters \((\hat\mu, \hat\sigma)\) using **only** returns inside the current lookback window ending at that update.
2. Use those updated parameters for the **subsequent** Monte Carlo segment until the next update.
3. Move the window forward and re-estimate at the chosen frequency.

## Modes

| Rolling | Update points | MC usage |
|---------|---------------|----------|
| `daily` | each trading day \(t\) | params from window ending at \(t\) drive the step \(t \to t+1\) |
| `monthly` | each month-end (plus period start) | params held until the next update |
| `none` | period start only | one window; params fixed for the whole period |

## Lookback choices

3 months · 6 months · 1 year · 2 years · 5 years (calendar lookback; uses available history if the series is shorter).

## Notebook section split

- **§4 Calibration only:** lookback / rolling sliders, **Reestimate**, parameter tables and rolling \(\hat\mu/\hat\sigma\) charts. No simulated-vs-history plots.
- **§5 Monte Carlo only:** **Start** / **Restart**; MC paths and expected path vs historical prices (one pair per ticker).

## Files

- `2008-2009_gbm.ipynb`
- `2013-2014_gbm.ipynb`
- `2018-2019_gbm.ipynb`
- `2019-2020_gbm.ipynb`


## Duplicate-graph root cause (fixed)

Graphs repeated because:

1. **Re-running a cell called `display(...)` again**, which **appends** another widget view instead of replacing the old one.
2. **`plt.show()` inside an `ipywidgets.Output`** with `%matplotlib inline` can emit the same figure twice.
3. **Stacked `on_click` / stacked UIs** after multiple cell executions left old Start buttons alive next to new ones.

Fix: persistent `display_id` hosts, replace `children` instead of stacking, and `display(fig); plt.close(fig)` (never `plt.show()` in Outputs). §4 shows rolling-parameter graphs only (no tables). §5 shows one MC pair per company.


## Reopen + Run All

Do **not** rely on `display_id` / `update=True` for §4/§5 UIs — after closing the notebook those display ids are gone, so a later `update=True` can show nothing.

Use plain `display(ui)`. Re-running a cell replaces that cell’s output. Workflow: open notebook → Restart kernel → Run All. Saving after runs is optional.


## Kernel memory after reopen (duplicate graphs)

Closing a notebook tab **without Restart Kernel** often keeps the same Jupyter kernel alive. Old figures/widgets stay in memory, so the next Run All can look like repeated graphs.

Also, `%matplotlib inline` can auto-echo a figure while it is also `display`ed inside an `Output`.

**After reopen:** Kernel → **Restart Kernel**, then **Run All**.

Code uses `plt.ioff()`, `plt.close('all')`, `display(fig); plt.close(fig)`, and §5 draws only when you click **Start**.

## Duplicate graph root cause (short)

`%matplotlib inline` was painting each figure **twice**: once from `display(fig)` inside the widget Output, and again from the inline backend flush. After close/reopen with the same kernel this is especially obvious.

**Fix:** save figure to PNG bytes, `plt.close(fig)`, then `display(Image(...))` once. Still restart kernel after reopen when possible.
