"""Shared matplotlib style (same validated palette as notebooks/02_bolero_audit.ipynb)."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"

# Categorical slots in fixed order (validated adjacent-pair CVD-safe).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Sequential blue ramp (100 -> 700) and blue <-> red diverging with gray midpoint.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "gain_loss", ["#e34948", "#f0efec", "#2a78d6"]  # loss (red) <- 0 -> gain (blue)
)


def diverging_norm(values, floor: float = 1.0) -> TwoSlopeNorm:
    bound = max(floor, float(abs(values).max()))
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECOND,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECOND,
        "ytick.color": INK_SECOND,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def strip_axes(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
