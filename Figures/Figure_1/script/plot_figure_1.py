"""Figure 1: Hamiltonian evolution and commutator convergence.

Layout revision v5:
- leave adequate clearance between colorbar ticks and panel (b) y-axis labels
- align (a) and (b) labels at the same absolute figure height
- preserve the short panel (b), original data, and heatmap colors
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
STEPS_RAW = DATA / "raw_hamiltonian"

STEPS = (0, 100, 1000, 4000)
ROWS = (("method", "This work"), ("qr", "QR/Adam"))
NAME = {"method": "ThisWork", "qr": "QRAdam"}

# ---- visual settings ----
HEATMAP_CMAP = "bright"
COLOR_LIMIT_HA = 1.0
COLORBAR_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)
COLORBAR_EXTEND = "neither"  # rectangular colorbar
B_HEIGHT_RATIO = 2 / 3         # of the heatmap block in panel (a)
B_WIDTH_IN = 1.00             # narrower than one fifth of the saved figure

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


def heatmap_colormap():
    if HEATMAP_CMAP == "bright":
        return LinearSegmentedColormap.from_list(
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
    return plt.get_cmap(HEATMAP_CMAP)


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def arm_of(name):
    return "qr" if "QR" in str(name).upper() else "method"


def load_h():
    files = sorted(STEPS_RAW.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {STEPS_RAW}")
    out = {}
    for file in files:
        with np.load(file) as z:
            out[file.stem] = np.asarray(z["hamiltonian_Ha"][0])
            out[file.stem + "_occ"] = np.asarray(z["occupations"][0, 0])
    for arm, _ in ROWS:
        for step in STEPS:
            if f"{arm}_{step}" not in out:
                raise KeyError(f"Missing Hamiltonian: {arm}_{step}.npz")
    return out


def sorted_h_and_occ(Z, arm, step):
    h = Z[f"{arm}_{step}"]
    occ = Z[f"{arm}_{step}_occ"]
    order = np.argsort(h.diagonal().real, kind="stable")
    return h[np.ix_(order, order)], occ[order], order


def plotted_matrix(h):
    return np.triu(h.real) + np.tril(h.imag, -1)


def export_origin_csv(Z):
    folder = DATA / "origin_csv"
    folder.mkdir(parents=True, exist_ok=True)
    for arm, _ in ROWS:
        for step in STEPS:
            h, occ, order = sorted_h_and_occ(Z, arm, step)
            tag = f"{NAME[arm]}_step{step}"
            np.savetxt(folder / f"a_{tag}_plotted.csv", plotted_matrix(h), delimiter=",", fmt="%.10e")
            np.savetxt(folder / f"a_{tag}_Re.csv", h.real, delimiter=",", fmt="%.10e")
            np.savetxt(folder / f"a_{tag}_Im.csv", h.imag, delimiter=",", fmt="%.10e")
            pd.DataFrame({
                "sorted_index": np.arange(len(order)),
                "original_band_index": order,
                "h_ii_Ha": h.diagonal().real,
                "occupation": occ,
            }).to_csv(folder / f"a_{tag}_states.csv", index=False)

    csv_file = DATA / "fig1_Al_H_commutator.csv"
    if csv_file.exists():
        c = pd.read_csv(csv_file)
        c = c[c.commutator_Frobenius_Ha.notna()].copy()
        c = c[["arm", "step", "commutator_Frobenius_Ha"]]
        c["arm"] = c["arm"].map(lambda x: "QRAdam_commutator_F_Ha" if arm_of(x) == "qr" else "ThisWork_commutator_F_Ha")
        wide = c.pivot_table(index="step", columns="arm", values="commutator_Frobenius_Ha").reset_index()
        wide.columns.name = None
        wide.to_csv(folder / "b_commutator_sparse.csv", index=False)


def diagnose_matrices(Z):
    print(f"Heatmap: {HEATMAP_CMAP}, fixed scale +/-{COLOR_LIMIT_HA:g} Ha")
    print("arm  step  plotted[min,max] Ha  offdiag abs P95/P99 Ha  plotted clipped %")
    for arm, _ in ROWS:
        for step in STEPS:
            h, _, _ = sorted_h_and_occ(Z, arm, step)
            m = plotted_matrix(h)
            off = np.abs(h[~np.eye(h.shape[0], dtype=bool)])
            p95, p99 = np.percentile(off, [95, 99])
            pct = 100 * np.mean(np.abs(m) > COLOR_LIMIT_HA)
            print(f"{arm:6s} {step:4d}  [{m.min():8.3g},{m.max():8.3g}]             {p95:8.3g}/{p99:8.3g}          {pct:6.2f}%")


def plot_convergence(ax):
    dense = DATA / "fig1_dense_trace.csv"
    if dense.exists():
        t = pd.read_csv(dense)
        for arm, label in ROWS:
            g = t[t.arm.map(arm_of) == arm].sort_values("step")
            ax.semilogy(g.step, np.maximum(g.commutator_F_Ha, 1e-16),
                        "-", color=BLUE if arm == "method" else RED,
                        lw=1.15, label=label)
    else:
        d = pd.read_csv(DATA / "fig1_Al_H_commutator.csv")
        d["a"] = d.arm.map(arm_of)
        comm = d[d.commutator_Frobenius_Ha.notna()]
        for arm, label in ROWS:
            g = comm[comm.a == arm].sort_values("step")
            ax.semilogy(g.step, np.maximum(g.commutator_Frobenius_Ha, 1e-16),
                        "o-" if arm == "method" else "s-",
                        color=BLUE if arm == "method" else RED,
                        ms=2.6, lw=1.05, label=label)

    ax.axvspan(0, 1000, color="0.93", lw=0, zorder=0)
    ax.set_xlim(0, 4000)
    ax.set_ylim(1e-11, 1e3)
    ax.set_xticks([0, 2000, 4000])
    ax.set_yticks([1e-10, 1e-7, 1e-4, 1e-1, 1e2])
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\|[h,F]\|_F$ (Ha)", labelpad=1)
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.55),
              fontsize=5.5, handlelength=0.7, borderaxespad=0.05,
              labelspacing=0.15, handletextpad=0.25)
    # Panel label is placed in figure coordinates, not relative to this axis.


def fig1(Z):
    norm = TwoSlopeNorm(vmin=-COLOR_LIMIT_HA, vcenter=0, vmax=COLOR_LIMIT_HA)
    cmap = heatmap_colormap()

    # Panel (b) is two thirds the heatmap-block height and centered on (a).
    FW, FH = 8.15, 3.16
    fig = plt.figure(figsize=(FW, FH), facecolor="white")

    def place(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    # --- panel (a) layout ---
    S = 1.08
    gx = 0.09
    gy = 0.40
    x0 = 1.05
    ybot = 0.28
    a_total_h = 2 * S + gy

    all_m = []
    im = None
    for r, (arm, label) in enumerate(ROWS):
        y = ybot + (1 - r) * (S + gy)
        for col, step in enumerate(STEPS):
            h, _, _ = sorted_h_and_occ(Z, arm, step)
            m = plotted_matrix(h)
            all_m.append(m)
            ax = place(x0 + col * (S + gx), y, S, S)
            im = ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            ax.set_xticks([0, 10, 20])
            ax.set_yticks([0, 10, 20])
            ax.tick_params(top=False, right=False, direction="out", length=2.5, pad=1.3)
            ax.set_title(f"Step: {step}", pad=2)
            if col == 0:
                ax.set_ylabel("orbital index", fontsize=8, labelpad=1)
                ax.text(-0.55, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8, clip_on=False)
            else:
                ax.set_yticklabels([])

    # --- colorbar ---
    cbar_x = x0 + 4 * S + 3 * gx + 0.12
    cbar_w = 0.075
    cax = place(cbar_x, ybot, cbar_w, a_total_h)
    cb = fig.colorbar(im, cax=cax, extend=COLORBAR_EXTEND)
    cb.set_ticks(COLORBAR_TICKS)
    cb.ax.set_title(r"$E$ (Ha)", pad=5, fontsize=8, loc="left")
    cb.ax.tick_params(direction="out", right=True)

    # --- panel (b): previous height, centered on the heatmaps ---
    b_w = B_WIDTH_IN
    b_h = a_total_h * B_HEIGHT_RATIO
    # Keep the clearance used before the heatmaps grew: colorbar labels, then (b).
    b_x = cbar_x + cbar_w + 0.98
    b_y = ybot + 0.5 * (a_total_h - b_h)
    axb = place(b_x, b_y, b_w, b_h)
    plot_convergence(axb)

    # (a) and (b) stay on one horizontal line, above the heatmap titles.
    label_y = (ybot + a_total_h + 0.20) / FH
    fig.text(0.055 / FW, label_y, "(a)", fontsize=9, fontweight="bold",
             va="top", ha="left")
    fig.text((b_x - 0.19) / FW, label_y, "(b)", fontsize=9, fontweight="bold",
             va="top", ha="left")

    save(fig, "Figure_1")


def main():
    Z = load_h()
    diagnose_matrices(Z)
    export_origin_csv(Z)
    fig1(Z)
    print(f"Saved {HERE / 'Figure_1.pdf'} and {HERE / 'Figure_1.png'}")


if __name__ == "__main__":
    main()
