"""Figure 2: plotting script.

Reads only ../data/ and writes ../Figure_occupations.pdf and ../Figure_occupations.png.
Run from anywhere:  python Figure_occupations/script/plot_figure_occupations.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.special import expit

HERE = Path(__file__).resolve().parent.parent   # Figure_N/
DATA = HERE / "data"
HA2EV = 27.211386245988

# Same blue–white–red stops as the Figure 1 heatmap.
BLUE = "#164bff"
RED = "#ff2020"
# Panel (c) QR/Adam uses the quieter curve red from Figure 1.
QR_RED = "#c84f3d"
STEP1000 = "#8a8a8a"
MARKER = 12
OCC = LinearSegmentedColormap.from_list(
    "bright_blue_white_red",
    [
        (0.00, "#164bff"),
        (0.25, "#9ab4ff"),
        (0.50, "#ffffff"),
        (0.75, "#ffb3b3"),
        (1.00, "#ff2020"),
    ],
    N=256,
)
OCC_NORM = Normalize(vmin=0, vmax=1)
MARKER_LW = 0.7
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


def panel(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def arm_of(name):
    return "qr" if "QR" in name else "method"


def legend(ax, entries):
    """One marker size and one line length in every panel."""
    handles = []
    for entry in entries:
        if entry[0] == "fd":
            handles.append(Line2D([0], [0], linestyle=":", color="k", lw=0.9, label="FD"))
        else:
            marker, edge, face, label = entry
            handles.append(Line2D(
                [0], [0], linestyle="none", marker=marker, markersize=7,
                markeredgewidth=0.8, markeredgecolor=edge, markerfacecolor=face,
                label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              handletextpad=0.45, handlelength=1.8, borderaxespad=0.25)



def fig2():
    d = pd.read_csv(DATA / "fig2_Al_FD.csv")
    d["a"] = d.arm.map(arm_of)
    fin = d[(d.a == "method") & (d.step == 4000)]
    mu_ref = fin.mu_Ha.iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.05),
                            gridspec_kw=dict(wspace=0.38, left=0.07, right=0.99,
                                             bottom=0.18, top=0.90))
    xx = np.linspace(-12, 12, 400)
    # (a) one marker per represented state of the shared initial condition.
    g = d[(d.a == "method") & (d.step == 0)]
    ax = axs[0]
    ax.scatter(g.scaled_energy, g.normalized_occupation, s=MARKER,
               c=g.normalized_occupation, cmap=OCC, norm=OCC_NORM,
               lw=0, zorder=3)
    ax.plot(xx, expit(-xx), ":", color="k", lw=0.9, zorder=2)
    ax.set(xlim=(-12, 12), ylim=(-0.04, 1.04), xlabel=r"$(h_{ii}-\mu)/k_BT$",
           ylabel=r"$f_{i\mathbf{k}}/u$")
    legend(ax, [("fd",)])
    panel(ax, "a", x=-0.28)
    # (b) shape marks the step; color is the occupation, shared with (a).
    ax = axs[1]
    for s, mk, z in [(4000, "o", 3), (1000, "D", 4)]:
        g = d[(d.a == "method") & (d.step == s)]
        ax.scatter(g.scaled_energy, g.normalized_occupation, s=MARKER, marker=mk,
                   c=g.normalized_occupation, cmap=OCC, norm=OCC_NORM,
                   edgecolors="#4a4a4a" if mk == "D" else "none",
                   linewidths=0.45 if mk == "D" else 0, zorder=z)
    ax.plot(xx, expit(-xx), ":", color="k", lw=0.9, zorder=2)
    ax.set(xlim=(-12, 12), ylim=(-0.04, 1.04), xlabel=r"$(h_{ii}-\mu)/k_BT$")
    legend(ax, [
        ("D", STEP1000, "none", "step 1000"),
        ("o", BLUE, BLUE, "step 4000"),
        ("fd",),
    ])
    panel(ax, "b", x=-0.18)
    sm = plt.cm.ScalarMappable(norm=OCC_NORM, cmap=OCC)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axs[:2], fraction=0.046, pad=0.02)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=7, direction="out")
    # (c) both runs at step 4000. The dotted curve is the same FD function as in (a,b).
    ax = axs[2]
    for a, col, mk, face, alpha, z in [
            ("qr", QR_RED, "s", QR_RED, 0.45, 2),
            ("method", BLUE, "o", BLUE, 1.0, 4),
    ]:
        g = d[(d.a == a) & (d.step == 4000)]
        x = (g.diagonal_Ha - mu_ref) * HA2EV
        ax.scatter(x, g.normalized_occupation, s=MARKER, marker=mk, facecolors=face,
                   edgecolors="none", alpha=alpha, zorder=z)
    xe = np.linspace(-4, 4, 400)
    ax.plot(xe, expit(-xe / (0.01 * HA2EV)), ":", color="k", lw=0.9, zorder=2)
    ax.set(xlim=(-4, 4), ylim=(-0.04, 1.04), xlabel=r"$h_{ii}-\mu^\ast$ (eV)")
    legend(ax, [
        ("o", BLUE, BLUE, "This work"),
        ("s", QR_RED, "none", "QR/Adam"),
        ("fd",),
    ])
    panel(ax, "c", x=-0.18)
    save(fig, "Figure_occupations")
    q = d[(d.a == "qr") & (d.step == 4000)]
    return dict(mu_ref=mu_ref,
                qr_occ_max=float(q.diagonal_Ha[q.normalized_occupation > .5].max() - mu_ref) * HA2EV,
                qr_emp_min=float(q.diagonal_Ha[q.normalized_occupation < .5].min() - mu_ref) * HA2EV,
                n_frac_method=int(((fin.normalized_occupation > .02) & (fin.normalized_occupation < .98)).sum()),
                fd_err_method=float((fin.normalized_occupation - fin.FD_target).abs().max()))


if __name__ == "__main__":
    print(json.dumps(fig2(), indent=1))
