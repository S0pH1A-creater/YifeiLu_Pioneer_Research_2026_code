#!/usr/bin/env python3
"""§5: pale 25–75% band, median as black dotted line, MAE / ICP / band width."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INSERT_AFTER = "        expected = paths.mean(axis=0)\n"
INSERT = '''        expected = paths.mean(axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        _hist = np.asarray(hist_now.values, dtype=float)
        _n = min(len(p50), len(_hist))
        p25, p50, p75, expected, _hist = p25[:_n], p50[:_n], p75[:_n], expected[:_n], _hist[:_n]
'''

# 2-year (calendar dates)
OLD_2Y_PLOTS = '''            axes[0].plot(dates_now, paths.T, color=COLORS[ticker], alpha=0.12, lw=0.7)
            axes[0].plot(dates_now, expected, color="black", lw=2.2, label="expected path (MC mean)")
            axes[0].set_title(f"{ticker}: Monte Carlo")
            axes[0].set_ylabel("price")
            axes[0].legend(loc="best", frameon=False)

            axes[1].plot(dates_now, hist_now.values, color=COLORS[ticker], lw=1.8, label="historical")
            axes[1].plot(dates_now, expected, color="black", lw=2.0, ls="--", label="expected path")
            axes[1].set_title(f"{ticker}: expected vs history")
'''

NEW_2Y_PLOTS = '''            x = dates_now[:_n] if hasattr(dates_now, "__getitem__") else dates_now
            axes[0].fill_between(x, p25, p75, color=COLORS[ticker], alpha=0.18, lw=0, zorder=1, label="25–75% range")
            axes[0].plot(x, paths.T[:_n] if getattr(paths.T, "ndim", 1) == 1 else paths[:, :_n].T, color=COLORS[ticker], alpha=0.10, lw=0.6, zorder=2)
            axes[0].plot(x, p50, color="black", lw=2.0, ls="--", label="median path (50th)", zorder=4)
            axes[0].set_title(f"{ticker}: Monte Carlo")
            axes[0].set_ylabel("price")
            axes[0].legend(loc="best", frameon=False)

            axes[1].fill_between(x, p25, p75, color=COLORS[ticker], alpha=0.18, lw=0, zorder=1, label="25–75% range")
            axes[1].plot(x, _hist, color=COLORS[ticker], lw=1.8, label="historical", zorder=3)
            axes[1].plot(x, p50, color="black", lw=2.0, ls="--", label="median path (50th)", zorder=4)
            axes[1].set_title(f"{ticker}: median vs history")
'''

# Heston titles differ slightly on left plot
OLD_2Y_HESTON = OLD_2Y_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: Heston–Merton Monte Carlo")',
)
NEW_2Y_HESTON = NEW_2Y_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: Heston–Merton Monte Carlo")',
)

OLD_2Y_GARCH = OLD_2Y_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: GARCH–Merton Monte Carlo")',
)
NEW_2Y_GARCH = NEW_2Y_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: GARCH–Merton Monte Carlo")',
)

OLD_INTRA_PLOTS = '''            tt = trading_x(dates_now)
            axes[0].plot(tt, paths.T, color=COLORS[ticker], alpha=0.12, lw=0.7)
            axes[0].plot(tt, expected, color="black", lw=2.2, label="expected path (MC mean)")
            axes[0].set_title(f"{ticker}: Monte Carlo")
            axes[0].set_ylabel("price")
            axes[0].legend(loc="best", frameon=False)

            axes[1].plot(tt, hist_now.values, color=COLORS[ticker], lw=1.8, label="historical")
            axes[1].plot(tt, expected, color="black", lw=2.0, ls="--", label="expected path")
            axes[1].set_title(f"{ticker}: expected vs history")
'''

NEW_INTRA_PLOTS = '''            tt = trading_x(dates_now)[:_n]
            axes[0].fill_between(tt, p25, p75, color=COLORS[ticker], alpha=0.18, lw=0, zorder=1, label="25–75% range")
            axes[0].plot(tt, paths[:, :_n].T, color=COLORS[ticker], alpha=0.10, lw=0.6, zorder=2)
            axes[0].plot(tt, p50, color="black", lw=2.0, ls="--", label="median path (50th)", zorder=4)
            axes[0].set_title(f"{ticker}: Monte Carlo")
            axes[0].set_ylabel("price")
            axes[0].legend(loc="best", frameon=False)

            axes[1].fill_between(tt, p25, p75, color=COLORS[ticker], alpha=0.18, lw=0, zorder=1, label="25–75% range")
            axes[1].plot(tt, _hist, color=COLORS[ticker], lw=1.8, label="historical", zorder=3)
            axes[1].plot(tt, p50, color="black", lw=2.0, ls="--", label="median path (50th)", zorder=4)
            axes[1].set_title(f"{ticker}: median vs history")
'''

OLD_INTRA_HESTON = OLD_INTRA_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: Heston–Merton Monte Carlo")',
)
NEW_INTRA_HESTON = NEW_INTRA_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: Heston–Merton Monte Carlo")',
)
OLD_INTRA_GARCH = OLD_INTRA_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: GARCH–Merton Monte Carlo")',
)
NEW_INTRA_GARCH = NEW_INTRA_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: GARCH–Merton Monte Carlo")',
)

# Simplify 2y x indexing — dates_now may be DatetimeIndex; slice it
NEW_2Y_PLOTS = '''            x = dates_now[:_n]
            axes[0].fill_between(x, p25, p75, color=COLORS[ticker], alpha=0.18, lw=0, zorder=1, label="25–75% range")
            axes[0].plot(x, paths[:, :_n].T, color=COLORS[ticker], alpha=0.10, lw=0.6, zorder=2)
            axes[0].plot(x, p50, color="black", lw=2.0, ls="--", label="median path (50th)", zorder=4)
            axes[0].set_title(f"{ticker}: Monte Carlo")
            axes[0].set_ylabel("price")
            axes[0].legend(loc="best", frameon=False)

            axes[1].fill_between(x, p25, p75, color=COLORS[ticker], alpha=0.18, lw=0, zorder=1, label="25–75% range")
            axes[1].plot(x, _hist, color=COLORS[ticker], lw=1.8, label="historical", zorder=3)
            axes[1].plot(x, p50, color="black", lw=2.0, ls="--", label="median path (50th)", zorder=4)
            axes[1].set_title(f"{ticker}: median vs history")
'''
NEW_2Y_HESTON = NEW_2Y_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: Heston–Merton Monte Carlo")',
)
NEW_2Y_GARCH = NEW_2Y_PLOTS.replace(
    'axes[0].set_title(f"{ticker}: Monte Carlo")',
    'axes[0].set_title(f"{ticker}: GARCH–Merton Monte Carlo")',
)

METRICS = '''        mae = float(np.mean(np.abs(p50 - _hist)))
        icp = float(np.mean((_hist >= p25) & (_hist <= p75)))
        abw = float(np.mean(p75 - p25))
        rmse = float(np.sqrt(np.mean((p50 - _hist) ** 2)))
'''

OLD_PRINT = '        rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))\n        print(f"RMSE(expected vs historical) = {rmse:.4f} | seed = {seed}")'
NEW_PRINT = METRICS + '        print(f"MAE(p50)={mae:.4f} | ICP(25–75)={100*icp:.1f}% | avg band width={abw:.4f} | RMSE(p50)={rmse:.4f} | seed={seed}")'

OLD_MD_RMSE = '''            rmse = float(np.sqrt(np.mean((expected - hist_now.values) ** 2)))
            fig.suptitle(
                f"{ticker} | RMSE(S_t)={rmse:.4f} | seed={seed} | "
                f"{cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')}",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        display(Markdown(
            f"**RMSE (expected vs historical \\(S_t\\))** = `{rmse:.4f}` "
            f"| seed = `{seed}`"
        ))'''

NEW_MD_RMSE = '''            mae = float(np.mean(np.abs(p50 - _hist)))
            icp = float(np.mean((_hist >= p25) & (_hist <= p75)))
            abw = float(np.mean(p75 - p25))
            rmse = float(np.sqrt(np.mean((p50 - _hist) ** 2)))
            fig.suptitle(
                f"{ticker} | MAE={mae:.4f} | ICP={100*icp:.1f}% | width={abw:.4f} | seed={seed} | "
                f"{cal_meta.get('rolling_mode')} / {cal_meta.get('window_label')}",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
        _show_fig(fig)
        display(Markdown(
            f"**MAE (50th vs \\(S_t\\))** = `{mae:.4f}` · "
            f"**ICP (25–75)** = `{100*icp:.1f}%` · "
            f"**avg band width** = `{abw:.4f}` · "
            f"RMSE(p50) = `{rmse:.4f}` | seed = `{seed}`"
        ))'''

PLOT_PAIRS = [
    (OLD_INTRA_GARCH, NEW_INTRA_GARCH),
    (OLD_INTRA_HESTON, NEW_INTRA_HESTON),
    (OLD_INTRA_PLOTS, NEW_INTRA_PLOTS),
    (OLD_2Y_GARCH, NEW_2Y_GARCH),
    (OLD_2Y_HESTON, NEW_2Y_HESTON),
    (OLD_2Y_PLOTS, NEW_2Y_PLOTS),
]


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [text]


def patch_nb(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for cell in nb["cells"]:
        raw = src(cell)
        new = raw
        if INSERT_AFTER in new and "p25 = np.percentile" not in new:
            new = new.replace(INSERT_AFTER, INSERT, 1)
            hits.append("percentiles")
        for old, repl in PLOT_PAIRS:
            if old in new:
                new = new.replace(old, repl)
                hits.append("plots")
                break
        if OLD_PRINT in new:
            new = new.replace(OLD_PRINT, NEW_PRINT)
            hits.append("print-metrics")
        if OLD_MD_RMSE in new:
            new = new.replace(OLD_MD_RMSE, NEW_MD_RMSE)
            hits.append("md-metrics")
        if "Expected path vs historical prices" in new:
            new = new.replace(
                "Expected path vs historical prices",
                "Median path + 25–75% band vs historical prices",
            )
            hits.append("md-caption")
        if "Monte Carlo paths + expected path" in new:
            new = new.replace(
                "Monte Carlo paths + expected path",
                "Monte Carlo paths + 25–75% band + median",
            )
            hits.append("md-left")
        if new != raw:
            set_src(cell, new)
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return hits


def main() -> int:
    nbs = sorted(p for p in ROOT.glob("*/*.ipynb") if ".ipynb_checkpoints" not in str(p))
    for p in nbs:
        text = p.read_text(encoding="utf-8")
        if "expected = paths.mean(axis=0)" not in text:
            continue
        hits = patch_nb(p)
        print(f"{p.relative_to(ROOT)}: {', '.join(hits) if hits else 'NO HITS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
