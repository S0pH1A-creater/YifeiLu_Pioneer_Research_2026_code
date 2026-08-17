"""One-shot: inject §6 Optimal stopping into all four model regime notebooks.

Skips heston merton advanced. Safe to re-run (replaces existing §6 Optimal stopping).
§6 prices American calls for SPY, AAPL, and MSFT with the same LSM harness.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared RN path builder body: uses row.underlying (or explicit ticker) + rolling[ticker]
_PATH_COMMON_HEAD = '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 simulator (μ → r)."""
    ticker = str(getattr(row, "underlying", "SPY")).upper()
    if ticker not in rolling or len(rolling[ticker]) == 0:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    p = params_asof(rolling[ticker], row.trading_date)
    if p is None:
        raise RuntimeError(f"No {ticker} calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(dte, r, dtype=float)
'''

MODELS = {
    "gbm": {
        "folder": ROOT / "gbm notebook",
        "glob": "*_gbm.ipynb",
        "label": "GBM",
        "path_builder": _PATH_COMMON_HEAD
        + '''
    sig_step = np.full(dte, float(p["sigma"]), dtype=float)
    return simulate_gbm_rolling(mu_step, sig_step, S0, n_paths, seed)
''',
    },
    "merton": {
        "folder": ROOT / "merton notebook",
        "glob": "*_merton.ipynb",
        "label": "Merton",
        "path_builder": _PATH_COMMON_HEAD
        + '''
    sig_step = np.full(dte, float(p["sigma"]), dtype=float)
    lam_step = np.full(dte, float(p["lam"]), dtype=float)
    muj_step = np.full(dte, float(p["mu_j"]), dtype=float)
    sj_step = np.full(dte, float(p["sigma_j"]), dtype=float)
    kap_step = np.full(dte, float(p["kappa"]), dtype=float)
    return simulate_merton_rolling(
        mu_step, sig_step, lam_step, muj_step, sj_step, kap_step, S0, n_paths, seed
    )
''',
    },
    "heston": {
        "folder": ROOT / "heston notebook",
        "glob": "*_heston.ipynb",
        "label": "Heston",
        "path_builder": _PATH_COMMON_HEAD
        + '''
    kappa_step = np.full(dte, float(p["kappa"]), dtype=float)
    theta_step = np.full(dte, float(p["theta"]), dtype=float)
    xi_step = np.full(dte, float(p["xi"]), dtype=float)
    rho_step = np.full(dte, float(p["rho"]), dtype=float)
    v0_step = np.full(dte, float(p["v0"]), dtype=float)
    return simulate_heston_rolling(
        mu_step, kappa_step, theta_step, xi_step, rho_step, v0_step,
        S0, n_paths, seed,
    )
''',
    },
    "heston_merton": {
        "folder": ROOT / "heston merton notebook",
        "glob": "*_heston_merton.ipynb",
        "label": "Heston–Merton",
        "path_builder": _PATH_COMMON_HEAD
        + '''
    kappa_step = np.full(dte, float(p["kappa"]), dtype=float)
    theta_step = np.full(dte, float(p["theta"]), dtype=float)
    xi_step = np.full(dte, float(p["xi"]), dtype=float)
    rho_step = np.full(dte, float(p["rho"]), dtype=float)
    v0_step = np.full(dte, float(p["v0"]), dtype=float)
    lam_step = np.full(dte, float(p["lam"]), dtype=float)
    muj_step = np.full(dte, float(p["mu_j"]), dtype=float)
    sj_step = np.full(dte, float(p["sigma_j"]), dtype=float)
    kapj_step = np.full(dte, float(p["kappa_j"]), dtype=float)
    return simulate_heston_merton_rolling(
        mu_step, kappa_step, theta_step, xi_step, rho_step, v0_step,
        lam_step, muj_step, sj_step, kapj_step, S0, n_paths, seed,
    )
''',
    },
    "garch_merton": {
        "folder": ROOT / "garch merton notebook",
        "glob": "*_garch_merton.ipynb",
        "label": "GARCH–Merton",
        "path_builder": _PATH_COMMON_HEAD
        + '''
    steps = {
        "mu": np.full(dte, r, dtype=float),
        "omega": np.full(dte, float(p["omega"]), dtype=float),
        "alpha": np.full(dte, float(p["alpha"]), dtype=float),
        "beta": np.full(dte, float(p["beta"]), dtype=float),
        "sigma0": np.full(dte, float(p["sigma0"]), dtype=float),
        "lam": np.full(dte, float(p["lam"]), dtype=float),
        "mu_j": np.full(dte, float(p["mu_j"]), dtype=float),
        "sigma_j": np.full(dte, float(p["sigma_j"]), dtype=float),
        "kappa": np.full(dte, float(p["kappa"]), dtype=float),
    }
    return simulate_garch_merton_rolling(steps, S0, n_paths, seed)
''',
    },
    "garch": {
        "folder": ROOT / "garch notebook",
        "glob": "*_garch.ipynb",
        "label": "GARCH",
        "path_builder": _PATH_COMMON_HEAD
        + '''
    steps = {
        "mu": np.full(dte, r, dtype=float),
        "omega": np.full(dte, float(p["omega"]), dtype=float),
        "alpha": np.full(dte, float(p["alpha"]), dtype=float),
        "beta": np.full(dte, float(p["beta"]), dtype=float),
        "sigma0": np.full(dte, float(p["sigma0"]), dtype=float),
    }
    return simulate_garch_rolling(steps, S0, n_paths, seed)
''',
    },
}


def fix_source_newlines(text: str) -> list:
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.split("\n")[:-1]] + (
        [text.split("\n")[-1] + "\n"] if text.split("\n")[-1] != "" or text.endswith("\n") else []
    )


def stopping_markdown(label: str) -> str:
    return f"""## 6. Optimal stopping (American calls — {label})

Continuous with §5: after Monte Carlo stock paths are available, use the **same {label} simulator** and §4 calibration on **SPY / AAPL / MSFT** to decide exercise vs wait for American calls.

At each day along each path:
1. Immediate payoff: $\\max(S_t - K, 0)$
2. Continuation value: Longstaff–Schwartz regression on the path cloud (basis $1, S, S^2$)
3. Exercise if payoff $>$ continuation

Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; vol/jumps from §4 for that ticker). Not the single expected path — the full Monte Carlo cloud.

**Workflow:** §4 **Reestimate** → §5 **Start** (optional viz) → §6 **Compute stopping** (all three underlyings)."""


def stopping_code(path_builder: str, label: str) -> str:
    return f'''import sys
_SCRIPTS = Path("..") / "scripts"
if str(_SCRIPTS.resolve()) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.resolve()))

from american_lsm import (
    STOP_TICKERS,
    lsm_american_call,
    load_calls,
    params_asof,
    sample_calls,
)

def _show_fig(fig):
    """Show figure once as PNG (same pattern as §5)."""
    import io
    from IPython.display import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    display(Image(data=buf.getvalue()))

{path_builder}

_STOP_TICKERS = list(STOP_TICKERS)
_contracts_by_ticker = {{}}
for _t in _STOP_TICKERS:
    _panel = load_calls(DATA, _t)
    _contracts_by_ticker[_t] = sample_calls(
        _panel, PERIOD_START, PERIOD_END, n_total=24, seed=42
    )

stopping_results = {{}}  # ticker -> DataFrame

for _t in _STOP_TICKERS:
    _n = len(_contracts_by_ticker[_t])
    display(Markdown(
        f"Sampled **{{_n}}** {{_t}} American calls in "
        f"{{PERIOD_START.date()}} → {{PERIOD_END.date()}} "
        f"(ATM band / DTE 7–60 as in the panel)."
    ))
    if _n:
        display(
            _contracts_by_ticker[_t][
                ["trading_date", "S_t", "K", "dte", "r", "moneyness", "option_price"]
            ].head(8)
        )

_stop_n_paths = widgets.IntSlider(
    value=2000, min=500, max=8000, step=500, description="n_paths",
    style={{"description_width": "90px"}},
    layout=widgets.Layout(width="360px"),
)
_stop_seed = widgets.IntText(value=42, description="seed", layout=widgets.Layout(width="200px"))
_btn_stop = widgets.Button(
    description="Compute stopping", button_style="primary", icon="calculator"
)
_stop_out = widgets.Output(layout=widgets.Layout(width="100%"))
_stop_busy = {{"on": False}}


def _run_optimal_stopping(_=None):
    global stopping_results
    if _stop_busy["on"]:
        return
    _stop_busy["on"] = True
    with _stop_out:
        clear_output(wait=True)
        try:
            missing = [t for t in _STOP_TICKERS if t not in rolling or len(rolling[t]) == 0]
            if missing:
                display(Markdown(
                    "Run **Reestimate** in §4 first "
                    f"(need calibration for: {{', '.join(missing)}})."
                ))
                return

            n_paths = int(_stop_n_paths.value)
            seed0 = int(_stop_seed.value)
            dt = 1.0 / N_DAYS
            stopping_results = {{}}

            for ticker in _STOP_TICKERS:
                contracts = _contracts_by_ticker[ticker]
                if contracts is None or len(contracts) == 0:
                    display(Markdown(f"No {{ticker}} call contracts in this period panel slice."))
                    continue

                rows = []
                example = None
                for i, row in enumerate(contracts.itertuples(index=False)):
                    paths = _rn_paths_for_contract(row, n_paths, seed0 + i)
                    res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
                    err = res.price - float(row.option_price)
                    rows.append({{
                        "ticker": ticker,
                        "trading_date": row.trading_date,
                        "S_t": float(row.S_t),
                        "K": float(row.K),
                        "dte": int(row.dte),
                        "r": float(row.r),
                        "market": float(row.option_price),
                        "model_price": res.price,
                        "error": err,
                        "early_ex_frac": res.early_exercise_frac,
                        "mean_ex_day": res.mean_exercise_step,
                    }})
                    if example is None:
                        example = (row, paths, res)

                df = pd.DataFrame(rows)
                stopping_results[ticker] = df
                rmse = float(np.sqrt(np.mean(df["error"] ** 2)))
                mae = float(np.mean(np.abs(df["error"])))
                color = COLORS.get(ticker, "#2ca02c")

                display(Markdown(
                    f"### {label} — LSM results ({{ticker}})\\n"
                    f"n_paths={{n_paths}} | contracts={{len(df)}} | "
                    f"RMSE={{rmse:.4f}} | MAE={{mae:.4f}} | "
                    f"mean early-exercise fraction="
                    f"{{df['early_ex_frac'].mean():.3f}}"
                ))
                display(
                    df[
                        ["trading_date", "S_t", "K", "dte", "market", "model_price",
                         "error", "early_ex_frac", "mean_ex_day"]
                    ].round(4)
                )

                with plt.ioff():
                    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
                    ax = axes[0]
                    ax.scatter(df["market"], df["model_price"], alpha=0.75, color=color)
                    lo = min(df["market"].min(), df["model_price"].min())
                    hi = max(df["market"].max(), df["model_price"].max())
                    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                    ax.set_xlabel("market option_price")
                    ax.set_ylabel("model LSM price")
                    ax.set_title("Price: model vs market")

                    axes[1].bar(
                        ["model", "market"],
                        [df["model_price"].mean(), df["market"].mean()],
                        color=[color, "#7f7f7f"],
                    )
                    axes[1].set_title("Mean option value")
                    axes[1].set_ylabel("price")

                    axes[2].hist(
                        df["mean_ex_day"], bins=12,
                        color=color, alpha=0.85, edgecolor="white",
                    )
                    axes[2].set_xlabel("mean exercise day (by contract)")
                    axes[2].set_title("Optimal exercise timing")
                    fig.suptitle(
                        f"{label} optimal stopping | {{ticker}} | "
                        f"{{cal_meta.get('rolling_mode')}} / {{cal_meta.get('window_label')}}",
                        fontsize=11, y=1.02,
                    )
                    fig.tight_layout()
                _show_fig(fig)

                if example is not None:
                    row, paths, res = example
                    j = int(np.argmin(np.abs(res.exercise_steps - res.mean_exercise_step)))
                    t_ex = int(res.exercise_steps[j])
                    with plt.ioff():
                        fig2, ax = plt.subplots(figsize=(10, 3.8))
                        ax.plot(paths[j], color=color, lw=1.5, label="one RN path")
                        ax.axhline(float(row.K), color="gray", ls="--", lw=1, label=f"K={{row.K:g}}")
                        ax.scatter(
                            [t_ex], [paths[j, t_ex]], color="crimson", zorder=5, s=50,
                            label=f"exercise day {{t_ex}}",
                        )
                        ax.set_xlabel("day")
                        ax.set_ylabel("S")
                        ax.set_title(
                            f"{{ticker}} example path | "
                            f"trade {{pd.Timestamp(row.trading_date).date()}} | "
                            f"dte={{int(row.dte)}} | model={{res.price:.3f}} vs "
                            f"mkt={{float(row.option_price):.3f}}"
                        )
                        ax.legend(frameon=False, loc="best")
                        fig2.tight_layout()
                    _show_fig(fig2)

            display(Markdown(
                "Results stored in `stopping_results` "
                "(dict keyed by ticker → model_price, error, early_ex_frac, mean_ex_day)."
            ))
        except Exception as exc:
            display(Markdown(f"**Error:** `{{type(exc).__name__}}: {{exc}}`"))
        finally:
            _stop_busy["on"] = False


_btn_stop.on_click(_run_optimal_stopping)
display(widgets.VBox([
    widgets.HTML("<b>§6 Optimal stopping — SPY / AAPL / MSFT American calls (LSM)</b>"),
    widgets.HBox([_stop_n_paths, _stop_seed, _btn_stop]),
    _stop_out,
]))
'''


def reminder_markdown() -> str:
    return """## 7. Reminder

1. **§4:** sliders → **Reestimate** → read parameter tables / rolling charts.
2. **§5:** **Start** → Monte Carlo stock paths + expected vs history (one pair per ticker).
3. **§6:** **Compute stopping** → LSM exercise decision + model vs market on SPY / AAPL / MSFT calls (needs §4).
4. **Restart** (§5) only changes the random seed for path plots.
"""


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def patch_notebook(path: Path, model_key: str) -> None:
    cfg = MODELS[model_key]
    nb = json.loads(path.read_text())
    cells = nb["cells"]

    # Remove prior §6 Optimal stopping + old reminder if re-injecting.
    # Also drop stray markdown cells that accidentally contain path-builder code.
    new_cells = []
    skip_next_code = False
    for c in cells:
        t = cell_text(c).lstrip()
        raw = cell_text(c)
        if c["cell_type"] == "markdown" and (
            "def _rn_paths_for_contract" in raw or "Compute stopping" in raw and "lsm_american_call" in raw
        ):
            # leftover code pasted into markdown
            continue
        if c["cell_type"] == "markdown" and t.startswith("## 6. Optimal stopping"):
            skip_next_code = True
            continue
        if skip_next_code and c["cell_type"] == "code":
            skip_next_code = False
            continue
        if c["cell_type"] == "markdown" and (
            t.startswith("## 6. Reminder") or t.startswith("## 7. Reminder")
        ):
            continue
        new_cells.append(c)

    # Append §6 + §7
    md = {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": fix_source_newlines(stopping_markdown(cfg["label"])),
    }
    code = {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": fix_source_newlines(stopping_code(cfg["path_builder"], cfg["label"])),
    }
    rem = {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": fix_source_newlines(reminder_markdown()),
    }
    new_cells.extend([md, code, rem])
    nb["cells"] = new_cells
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    print(f"patched {path.relative_to(ROOT.parent)}")


def main():
    for key, cfg in MODELS.items():
        files = sorted(cfg["folder"].glob(cfg["glob"]))
        for f in files:
            if "advanced" in f.name:
                continue
            patch_notebook(f, key)
    print("done")


if __name__ == "__main__":
    main()
