"""Shared figure style.

All figures are saved at 300 dpi as both PNG and PDF, without a title or a
figure number baked into the image, since captions belong in the paper. The
palette is the Okabe Ito set, which stays distinguishable for the common forms
of colour vision deficiency and in greyscale print.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300

# Okabe Ito, colourblind safe.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

METHOD_COLORS = {
    "centralized": "#000000",
    "fedavg": "#0072B2",
    "fedprox": "#D55E00",
}


def apply_style() -> None:
    """Set the rcParams every figure in this repo uses."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "figure.autolayout": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def save_figure(fig, out_dir: str | Path, name: str) -> list[Path]:
    """Save a figure as PNG and PDF, returning both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", metadata={})
        written.append(path)
    plt.close(fig)
    return written
