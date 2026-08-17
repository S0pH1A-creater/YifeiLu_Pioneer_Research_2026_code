"""One-shot: inject §6 Optimal stopping into all four model regime notebooks.

Skips heston merton advanced. Safe to re-run (replaces existing §6 Optimal stopping).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "gbm": {
        "folder": ROOT / "gbm notebook",
        "glob": "*_gbm.ipynb",
        "label": "GBM",
        "path_builder": '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 GBM simulator (μ → r)."""
    p = params_asof(rolling["SPY"], row.trading_date)
    if p is None:
        raise RuntimeError("No SPY calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(dte, r, dtype=float)
    sig_step = np.full(dte, float(p["sigma"]), dtype=float)
    return simulate_gbm_rolling(mu_step, sig_step, S0, n_paths, seed)
''',
    },
    "merton": {
        "folder": ROOT / "merton notebook",
        "glob": "*_merton.ipynb",
        "label": "Merton",
        "path_builder": '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 Merton simulator (μ → r)."""
    p = params_asof(rolling["SPY"], row.trading_date)
    if p is None:
        raise RuntimeError("No SPY calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(dte, r, dtype=float)
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
        "path_builder": '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 Heston simulator (μ → r)."""
    p = params_asof(rolling["SPY"], row.trading_date)
    if p is None:
        raise RuntimeError("No SPY calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(dte, r, dtype=float)
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
        "path_builder": '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 Heston–Merton simulator (μ → r)."""
    p = params_asof(rolling["SPY"], row.trading_date)
    if p is None:
        raise RuntimeError("No SPY calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
    mu_step = np.full(dte, r, dtype=float)
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
        "path_builder": '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 GARCH–Merton simulator (μ → r)."""
    p = params_asof(rolling["SPY"], row.trading_date)
    if p is None:
        raise RuntimeError("No SPY calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
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
        "path_builder": '''
def _rn_paths_for_contract(row, n_paths: int, seed: int):
    """Risk-neutral paths to expiry using §5 GARCH simulator (μ → r)."""
    p = params_asof(rolling["SPY"], row.trading_date)
    if p is None:
        raise RuntimeError("No SPY calibration — run Reestimate in §4 first.")
    dte = int(row.dte)
    if dte < 2:
        raise ValueError("dte must be >= 2")
    r = float(row.r)
    S0 = float(row.S_t)
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


def md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1] + ("\n" if text.endswith("\n") else "")],
    }


def code_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    # keep trailing newline style like other notebook cells
    source = [ln + "\n" for ln in lines[:-1]]
    if lines[-1] != "":
        source.append(lines[-1] + "\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def stopping_markdown(label: str) -> str:
    return f"""## 6. Optimal stopping (American calls — {label})

Continuous with §5: after Monte Carlo stock paths are available, use the **same {label} simulator** and §4 calibration on **SPY** to decide exercise vs wait for American calls.

At each day along each path:
1. Immediate payoff: $\\max(S_t - K, 0)$
2. Continuation value: Longstaff–Schwartz regression on the path cloud (basis $1, S, S^2$)
3. Exercise if payoff $>$ continuation

Paths for pricing are **risk-neutral** (drift $\\mu \\rightarrow r$ from the option panel; vol/jumps from §4). Not the single expected path — the full Monte Carlo cloud.

**Workflow:** §4 **Reestimate** → §5 **Start** (optional viz) → §6 **Compute stopping**."""


def stopping_code(path_builder: str, label: str) -> str:
    return f'''import sys
_SCRIPTS = Path("..") / "scripts"
if str(_SCRIPTS.resolve()) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.resolve()))

from american_lsm import (
    lsm_american_call,
    load_spy_calls,
    params_asof,
    sample_spy_calls,
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

_spy_calls_all = load_spy_calls(DATA)
_contracts = sample_spy_calls(
    _spy_calls_all, PERIOD_START, PERIOD_END, n_total=24, seed=42
)
stopping_results = None

display(Markdown(
    f"Sampled **{{len(_contracts)}}** SPY American calls in "
    f"{{PERIOD_START.date()}} → {{PERIOD_END.date()}} "
    f"(ATM band / DTE 7–60 as in the panel)."
))
if len(_contracts):
    display(
        _contracts[
            ["trading_date", "S_t", "K", "dte", "r", "moneyness", "option_price"]
        ].head(12)
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
            if "SPY" not in rolling or len(rolling["SPY"]) == 0:
                display(Markdown("Run **Reestimate** in §4 first (need SPY calibration)."))
                return
            if _contracts is None or len(_contracts) == 0:
                display(Markdown("No SPY call contracts in this period panel slice."))
                return

            n_paths = int(_stop_n_paths.value)
            seed0 = int(_stop_seed.value)
            rows = []
            example = None
            dt = 1.0 / N_DAYS

            for i, row in enumerate(_contracts.itertuples(index=False)):
                paths = _rn_paths_for_contract(row, n_paths, seed0 + i)
                res = lsm_american_call(paths, K=float(row.K), r=float(row.r), dt=dt)
                err = res.price - float(row.option_price)
                rows.append({{
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

            stopping_results = pd.DataFrame(rows)
            rmse = float(np.sqrt(np.mean(stopping_results["error"] ** 2)))
            mae = float(np.mean(np.abs(stopping_results["error"])))

            display(Markdown(
                f"### {label} — LSM results (SPY)\\n"
                f"n_paths={{n_paths}} | contracts={{len(stopping_results)}} | "
                f"RMSE={{rmse:.4f}} | MAE={{mae:.4f}} | "
                f"mean early-exercise fraction="
                f"{{stopping_results['early_ex_frac'].mean():.3f}}"
            ))
            display(
                stopping_results[
                    ["trading_date", "S_t", "K", "dte", "market", "model_price",
                     "error", "early_ex_frac", "mean_ex_day"]
                ].round(4)
            )

            with plt.ioff():
                fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
                ax = axes[0]
                ax.scatter(
                    stopping_results["market"], stopping_results["model_price"],
                    alpha=0.75, color=COLORS.get("SPY", "#2ca02c"),
                )
                lo = min(stopping_results["market"].min(), stopping_results["model_price"].min())
                hi = max(stopping_results["market"].max(), stopping_results["model_price"].max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set_xlabel("market option_price")
                ax.set_ylabel("model LSM price")
                ax.set_title("Price: model vs market")

                axes[1].bar(
                    ["model", "market"],
                    [stopping_results["model_price"].mean(), stopping_results["market"].mean()],
                    color=[COLORS.get("SPY", "#2ca02c"), "#7f7f7f"],
                )
                axes[1].set_title("Mean option value")
                axes[1].set_ylabel("price")

                axes[2].hist(
                    stopping_results["mean_ex_day"], bins=12,
                    color=COLORS.get("SPY", "#2ca02c"), alpha=0.85, edgecolor="white",
                )
                axes[2].set_xlabel("mean exercise day (by contract)")
                axes[2].set_title("Optimal exercise timing")
                fig.suptitle(
                    f"{label} optimal stopping | {{cal_meta.get('rolling_mode')}} / "
                    f"{{cal_meta.get('window_label')}}",
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
                    ax.plot(paths[j], color=COLORS.get("SPY", "#2ca02c"), lw=1.5, label="one RN path")
                    ax.axhline(float(row.K), color="gray", ls="--", lw=1, label=f"K={{row.K:g}}")
                    ax.scatter(
                        [t_ex], [paths[j, t_ex]], color="crimson", zorder=5, s=50,
                        label=f"exercise day {{t_ex}}",
                    )
                    ax.set_xlabel("day")
                    ax.set_ylabel("S")
                    ax.set_title(
                        f"Example path | trade {{pd.Timestamp(row.trading_date).date()}} | "
                        f"dte={{int(row.dte)}} | model={{res.price:.3f}} vs mkt={{float(row.option_price):.3f}}"
                    )
                    ax.legend(frameon=False, loc="best")
                    fig2.tight_layout()
                _show_fig(fig2)

            display(Markdown(
                "Results stored in `stopping_results` "
                "(model_price, error, early_ex_frac, mean_ex_day)."
            ))
        except Exception as exc:
            display(Markdown(f"**Error:** `{{type(exc).__name__}}: {{exc}}`"))
        finally:
            _stop_busy["on"] = False


_btn_stop.on_click(_run_optimal_stopping)
display(widgets.VBox([
    widgets.HTML("<b>§6 Optimal stopping — SPY American calls (LSM)</b>"),
    widgets.HBox([_stop_n_paths, _stop_seed, _btn_stop]),
    _stop_out,
]))
'''


def reminder_markdown() -> str:
    return """## 7. Reminder

1. **§4:** sliders → **Reestimate** → read parameter tables / rolling charts.
2. **§5:** **Start** → Monte Carlo stock paths + expected vs history (one pair per ticker).
3. **§6:** **Compute stopping** → LSM exercise decision + model vs market on SPY calls (needs §4).
4. **Restart** (§5) only changes the random seed for path plots.
"""


def fix_source_newlines(text: str) -> list:
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.split("\n")[:-1]] + (
        [text.split("\n")[-1] + "\n"] if text.split("\n")[-1] != "" or text.endswith("\n") else []
    )


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def patch_notebook(path: Path, model_key: str) -> None:
    cfg = MODELS[model_key]
    nb = json.loads(path.read_text())
    cells = nb["cells"]

    # Remove prior §6 Optimal stopping + old reminder if re-injecting
    new_cells = []
    skip_next_code = False
    for c in cells:
        t = cell_text(c).lstrip()
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
        # exclude accidental advanced matches (none in these folders)
        for f in files:
            if "advanced" in f.name:
                continue
            patch_notebook(f, key)
    print("done")


if __name__ == "__main__":
    main()
