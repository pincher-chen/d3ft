"""Figure S1: plotting script.

Reads only ../data/ and writes ../Figure_sodium_bands.pdf and ../Figure_sodium_bands.png.
Run from anywhere:  python Figure_sodium_bands/script/plot_figure_sodium_bands.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.special import expit

HERE = Path(__file__).resolve().parent.parent   # Figure_N/
DATA = HERE / "data"
HA2EV = 27.211386245988

BLUE = "#1f5fa8"     # this work
RED = "#c8442f"      # QR/idempotent-Adam
GREY = "#7a7a7a"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
LAB = dict(method="This work", qr="QR/idempotent–Adam")


def panel(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def arm_of(name):
    return "qr" if "QR" in name else "method"



def figS1():
    d = pd.read_csv(DATA / "band_comparison.csv")
    ref = d[(d.point == 0) & (d.band == 0)].QE_Ha.iloc[0]
    fig, axs = plt.subplots(2, 1, figsize=(3.3, 3.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08,
                                             left=0.18, right=0.97, top=0.97, bottom=0.1))
    ax = axs[0]
    for b, g in d.groupby("band"):
        g = g.sort_values("point")
        ax.plot(g.distance_inverse_A, (g.candidate_Ha - ref) * HA2EV, "-", color=BLUE, lw=1,
                label="this work" if b == 0 else None)
        ax.plot(g.distance_inverse_A, (g.QE_Ha - ref) * HA2EV, "o", mfc="none", mec=RED,
                ms=2.6, mew=0.6, label="QE" if b == 0 else None)
    ticks = sorted(d[d.point.isin([0, 12, 24, 36])].distance_inverse_A.unique())
    for t in ticks:
        for a in axs:
            a.axvline(t, color="0.7", lw=0.5)
    ax.set_ylabel("Energy (eV)")
    ax.legend(loc="upper right", fontsize=6.5)
    panel(ax, "a", x=-0.22, y=0.98)
    ax = axs[1]
    ax.plot(d.distance_inverse_A, d.aligned_delta_eV * 1e6, ".", color="k", ms=2)
    ax.set_ylabel(r"$\Delta$ ($\mu$eV)")
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "R", r"$\Gamma$"])
    ax.set_xlim(ticks[0], ticks[-1])
    panel(ax, "b", x=-0.22, y=0.9)
    save(fig, "Figure_sodium_bands")


if __name__ == "__main__":
    figS1()
