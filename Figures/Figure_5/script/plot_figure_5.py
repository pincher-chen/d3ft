"""Figure 5: plotting script.

Reads only ../data/ and writes ../Figure_5.pdf and ../Figure_5.png.
Run from anywhere:  python Figure_5/script/plot_figure_4.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent.parent   # Figure_N/
DATA = HERE / "data"

# Same ink as Figure 1.
BLUE = "#3b6fb6"
RED = "#c84f3d"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def panel(ax, s, x=-0.16, y=1.04, ha="left"):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha=ha)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def arm_of(name):
    return "qr" if "QR" in name else "method"



def fig5():
    base = DATA
    est = pd.read_csv(base / "GaAs_estimator.csv")
    sw = pd.read_csv(base / "GaAs_switch.csv")
    fig = plt.figure(figsize=(6.6, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.25, 1.25], wspace=0.55,
                          left=0.08, right=0.99, bottom=0.16, top=0.74)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(est))
    w = 0.36
    ax.bar(x - w / 2, est.maximum_old_bias_Ha, w, color=RED, label="floored")
    ax.bar(x + w / 2, est.maximum_positive_reconstruction_error_Ha, w, color=BLUE,
           label="exact")
    ax.set_yscale("log"); ax.set_ylim(1e-14, 1e3)
    ax.set_xticks(x, [f"seed {s}" for s in est.seed])
    ax.set_ylabel(r"max $|\hat\epsilon_i-h_{ii}|$ (Ha)")
    ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1, 1e3])
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.02),
              handlelength=1.2, columnspacing=0.6, borderaxespad=0.0)
    panel(ax, "a", x=-0.42, y=1.22)

    ax = fig.add_subplot(gs[1])
    rules = [("A_after_anneal", "after annealing\n(step 1000)"), ("B_fixed4000", "step 4000"),
             ("B_fixed8000", "step 8000"), ("C_current_gate", "residual gate")]
    for r, (rule, lab) in enumerate(rules):
        for j, (seed, mk) in enumerate([(201, "o"), (202, "s")]):
            row = sw[(sw.rule == rule) & (sw.seed == seed)].iloc[0]
            y = r + (j - 0.5) * 0.28
            ax.plot([row.switch_after_steps, row.first_qualified_audit_steps], [y, y],
                    color="0.55", lw=0.8)
            ax.plot(row.switch_after_steps, y, mk, mfc="white", mec="0.35", ms=3.5)
            ax.plot(row.first_qualified_audit_steps, y, mk, color=BLUE, ms=3.5)
    ax.set_yticks(range(len(rules)), [l for _, l in rules], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 17000)
    ax.set_xticks([0, 4000, 8000, 12000, 16000])
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    def mark(marker, face):
        return Line2D([0], [0], linestyle="none", marker=marker, markersize=5,
                      markerfacecolor=face, markeredgecolor="0.35" if face == "white" else BLUE,
                      markeredgewidth=0.8)
    ax.legend(
        [(mark("o", "white"), mark("s", "white")),
         (mark("o", BLUE), mark("s", BLUE))],
        ["activation", "converged"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.02),
        handlelength=2.4, columnspacing=0.8, borderaxespad=0.0, handletextpad=0.4,
    )
    panel(ax, "b", x=0.5, y=1.28, ha="center")

    ax = fig.add_subplot(gs[2])
    v = pd.read_csv(DATA / "GaAs_validation_final.csv").sort_values("seed")
    seeds = list(v.seed)
    y = np.arange(len(seeds))
    acc = v.this_work_first_accepted_audit.astype(float)
    ref_acc = v.PH_reference_accepted.astype(bool)
    ax.plot(acc, y, "o", color=BLUE, ms=4, label="This work")
    ax.plot(np.where(ref_acc, np.nan, v.PH_reference_budget.astype(float)), y, "x", color=RED, ms=4.5,
            mew=1.1, label="PH-only")
    for yi, a0 in zip(y, acc):
        ax.plot([a0, 20000], [yi, yi], color="0.85", lw=0.6, zorder=0)
    ax.set_yticks(y, [str(s) for s in seeds], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 22000)
    ax.set_xticks([0, 4000, 8000, 12000, 16000, 20000])
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_ylim(len(seeds) - 0.45, -0.55)
    ax.set_ylabel("independent seed")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.02),
              handlelength=1.2, columnspacing=0.6, borderaxespad=0.0,
              handletextpad=0.3)
    panel(ax, "c", x=-0.33, y=1.22)
    save(fig, "Figure_5")


if __name__ == "__main__":
    fig5()
