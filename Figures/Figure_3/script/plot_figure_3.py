"""Figure 3: plotting script.

Reads only ../data/ and writes ../Figure_3.pdf and ../Figure_3.png.
Run from anywhere:  python Figure_3/script/plot_figure_3.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
LAB = dict(method="This work", qr="QR/Adam")


def panel(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def arm_of(name):
    return "qr" if "QR" in name else "method"



def fig3():
    d = pd.read_csv(DATA / "fig3_Al_stopping_criteria.csv")
    d = d[d.mesh == "Gamma"]
    d["a"] = d.arm.map(arm_of)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5),
                            gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1.15],
                                             left=0.08, right=0.99, bottom=0.27, top=0.8))
    ax = axs[0]
    for a, col in [("method", BLUE), ("qr", RED)]:
        g = d[(d.a == a) & (d.kind == "variance")].sort_values("step")
        ax.semilogy(g.step, np.maximum(g.variance_F_Ha2, 1e-25), color=col, lw=0.9, label=LAB[a])
        hit = g[g.variance_F_Ha2 < 1e-8].step.min()
        ax.axvline(hit, color=col, ls="--", lw=0.6)
    ax.axhline(1e-8, color="k", ls=":", lw=0.8)
    ax.set(xlim=(1000, 6000), ylim=(3e-26, 1e2), xlabel="Optimization step",
           ylabel=r"Var$_{50}(\mathcal{F})$ (Ha$^2$)")
    ax.set_yticks([1e-24, 1e-18, 1e-12, 1e-6, 1e0])
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.02),
              handlelength=1.8, columnspacing=0.8, borderaxespad=0.0)
    panel(ax, "a", x=-0.22, y=1.2)
    ax = axs[1]
    tests = ["Charge", "Orthogonality", "External", "Commutator", "FD", "Ritz",
             "Density update", "F update", "QE F", "QE bands", "QE density"]
    ticklab = ["charge", "orth.", "proj. grad.", "commut.", "FD", "Ritz", r"$\Delta\rho$",
               r"$\Delta\mathcal{F}$", r"QE $\mathcal{F}$", "QE bands", r"QE $\rho$"]
    styles = {("method", "variance hit"): (BLUE, "s", "--", "This work, crossing"),
              ("method", "4k pass"): (BLUE, "o", "-", "This work, step 4000"),
              ("qr", "variance hit"): (RED, "s", "--", "QR/Adam, crossing"),
              ("qr", "6k censored"): (RED, "o", "-", "QR/Adam, step 6000")}
    for (a, k), (col, mk, ls, lab) in styles.items():
        g = d[(d.a == a) & (d.kind == k)].set_index("test").loc[tests]
        ax.semilogy(range(11), np.maximum(g.ratio.values, 1e-16), ls=ls, marker=mk,
                    color=col, ms=3, lw=0.8, label=lab, mfc="white" if ls == "--" else col)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.axvline(7.5, color="0.6", lw=0.5)
    ax.text(3.5, 1e7, "internal", ha="center", fontsize=6.5, color="0.35")
    ax.text(9, 1e7, "vs. QE", ha="center", fontsize=6.5, color="0.35")
    ax.set_xticks(range(11), ticklab, rotation=45, ha="right")
    ax.set(ylim=(1e-17, 1e9), ylabel="residual / tolerance")
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5])
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.02),
              handlelength=1.8, columnspacing=0.8, borderaxespad=0.0)
    panel(ax, "b", x=-0.17, y=1.2)
    save(fig, "Figure_3")


if __name__ == "__main__":
    fig3()
