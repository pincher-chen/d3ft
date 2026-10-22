"""Figure 6: plotting script.

Reads only ../data/ and writes ../Figure_solver_cost.pdf and ../Figure_solver_cost.png.
Run from anywhere:  python Figure_solver_cost/script/plot_figure_solver_cost.py
"""
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



def fig6():
    b1 = pd.read_csv(DATA / "B1_NC_main.csv")
    b2 = pd.read_csv(DATA / "timings.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5),
                            gridspec_kw=dict(wspace=0.32, left=0.08, right=0.99,
                                             bottom=0.18, top=0.92))
    ax = axs[0]
    for arm, col, mk, lab in [("Equivalent implementation", BLUE, "o", LAB["method"]),
                              ("QR adaptation", RED, "s", LAB["qr"])]:
        g = b1[b1.arm == arm].sort_values("bands")
        ax.loglog(g.bands, g.seconds_per1000, mk + "-", color=col, ms=3.5, lw=1, label=lab)
    m = b1[b1.arm == "Equivalent implementation"].set_index("bands").seconds_per1000
    q = b1[b1.arm == "QR adaptation"].set_index("bands").seconds_per1000
    for b in m.index:
        ax.text(b, m[b] * 1.18, f"{m[b] / q[b]:.2f}×", ha="center", fontsize=6, color=BLUE)
    ax.set_xticks([26, 52, 104, 208, 416], ["26", "52", "104", "208", "416"])
    ax.minorticks_off()
    ax.set(xlabel="Number of bands $b$", ylabel="Time per 1000 steps (s)", ylim=(10, 800))
    ax.legend(loc="upper left")
    panel(ax, "a", x=-0.2)
    ax = axs[1]
    for col_, col, mk, lab in [("QR_forward", RED, "s", "QR, forward"),
                               ("QR_forward_and_reverse", RED, "s", "QR, forward+reverse"),
                               ("FD_forward", BLUE, "o", "FD, forward"),
                               ("FD_forward_and_reverse", BLUE, "o", "FD, forward+reverse")]:
        rev = "reverse" in col_
        ax.loglog(b2.K, b2[col_] * 1e3, mk + ("--" if rev else "-"), color=col, ms=3.5, lw=1,
                  mfc="white" if rev else col, label=lab)
    ax.set_xticks([8, 27, 64, 216], ["8", "27", "64", "216"])
    ax.minorticks_off()
    ax.set(xlabel="Number of $k$ points $K$", ylabel="Latency (ms)", ylim=(0.8, 600))
    ax.legend(loc="upper left")
    panel(ax, "b", x=-0.2)
    save(fig, "Figure_solver_cost")


if __name__ == "__main__":
    fig6()
