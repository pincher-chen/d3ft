"""Boxed journal figure style used for the AD--DFPT-like manuscript plots.

Import this module. Do not retype the rcParams or the colour hex values.
"""
from pathlib import Path

import matplotlib.pyplot as plt

# Result series, baseline, ideal/reference line, single-series markers.
MAGENTA = "#D81B60"
INK = "#222222"
TEAL = "#00838F"
BLUE = "#3949AB"
BAND = "#F3F3F3"
BAND_EDGE = "#C8C8C8"

PALETTE = {
    "result": MAGENTA,
    "baseline": INK,
    "reference": TEAL,
    "series": BLUE,
    "band": BAND,
    "band_edge": BAND_EDGE,
}

# Full-width response / Nature-family text width. Font sizes below are for this width.
FULL_WIDTH_MM = 183.0

RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.labelweight": "bold",
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "lines.linewidth": 1.4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
}


def apply():
    """Set the matplotlib rcParams for this style. Call once before plotting."""
    plt.rcParams.update(RCPARAMS)


def style_ax(ax, minor="both"):
    """Close the box, point ticks inward, and add minor ticks on log axes.

    minor: "both", "y", "x", or None. Use "y" when x is categorical.
    """
    ax.tick_params(which="major", direction="in", top=True, right=True)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)
        if minor == "y":
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        elif minor == "x":
            ax.tick_params(axis="y", which="minor", left=False, right=False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def framed_legend(ax, *args, **kwargs):
    """Thin black legend frame. Keep labels short so the box misses the data."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fancybox", False)
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("framealpha", 1)
    kwargs.setdefault("borderpad", 0.35)
    kwargs.setdefault("handlelength", 1.7)
    leg = ax.legend(*args, **kwargs)
    leg.get_frame().set_linewidth(0.6)
    return leg


def panel_title(ax, letter, title):
    """Bold left-aligned title with the panel letter, e.g. 'a   Time ratio'."""
    ax.set_title(f"{letter}   {title}", loc="left", pad=5)


def mm(width_mm, height_mm):
    """Figure size in inches from millimetres."""
    return (width_mm / 25.4, height_mm / 25.4)


def assert_text_inside(fig):
    """Fail if a title, label, or tick string is outside the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for ax in fig.axes:
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            texts.extend(
                t.label1
                for t in axis.get_major_ticks()
                if lo - 1e-12 <= t.get_loc() <= hi + 1e-12
            )
        for text in texts:
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(renderer)
                if (
                    box.x0 < -0.5
                    or box.y0 < -0.5
                    or box.x1 > fig.bbox.width + 0.5
                    or box.y1 > fig.bbox.height + 0.5
                ):
                    outside.append(text.get_text())
    if outside:
        raise AssertionError(f"text outside canvas: {outside}")


def export(fig, stem):
    """Write pdf, svg, png, and tiff next to stem (a path without a suffix)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    assert_text_inside(fig)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=450)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
