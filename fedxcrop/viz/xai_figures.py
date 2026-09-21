"""Figures for the explainability results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fedxcrop.data.transforms import mask_transform
from fedxcrop.viz.style import PALETTE, apply_style, save_figure


def load_display_image(path: str | Path, image_size: int = 224) -> np.ndarray:
    """The image as the model sees it geometrically, in [0, 1] for display."""
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    return mask_transform(image_size)(image).numpy().transpose(1, 2, 0)


# Source images are 224 px. At 300 dpi a panel wider than about 0.75 inch is
# already upsampling, so panels are kept near native size: larger ones only
# grow the file without adding detail.
PANEL_INCHES = 1.15


def pseudo_mask_grid(
    rows: list[dict],
    out_dir: str | Path,
    name: str = "lesion_pseudo_masks",
    columns: int = 6,
    panel_inches: float = PANEL_INCHES,
) -> list[Path]:
    """Overlay of the derived lesion pseudo-masks, for visual quality control.

    These masks come from a colour rule, not from annotation. Showing a random
    sample of them is the only honest way to let a reader judge how much the
    lesion localization numbers are worth.
    """
    apply_style()
    n = len(rows)
    grid_rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(
        grid_rows, columns,
        figsize=(panel_inches * columns, (panel_inches + 0.15) * grid_rows),
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, row in zip(axes, rows):
        ax.imshow(row["image"])
        overlay = np.zeros((*row["mask"].shape, 4))
        overlay[row["mask"].astype(bool)] = [1.0, 0.0, 0.0, 0.45]
        ax.imshow(overlay)
        coverage = 100 * float(row["mask"].astype(bool).mean())
        ax.set_title(f"{row['label']}\n{coverage:.0f} percent", fontsize=5.5, pad=2)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    return save_figure(fig, out_dir, name)


def attribution_grid(
    entries: list[dict],
    out_dir: str | Path,
    name: str,
    model_labels: Optional[list[str]] = None,
    panel_inches: float = PANEL_INCHES,
) -> list[Path]:
    """Three rows (original, heatmap, overlay) by one column per class.

    The caption of the original figure promised three rows and the image
    contained one, so the rows are built explicitly here.
    """
    apply_style()
    n = len(entries)
    n_models = len(entries[0]["maps"])
    total_rows = 1 + 2 * n_models

    fig, axes = plt.subplots(
        total_rows, n, figsize=(panel_inches * n, (panel_inches + 0.07) * total_rows)
    )
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(-1, 1)

    for column, entry in enumerate(entries):
        axes[0, column].imshow(entry["image"])
        axes[0, column].set_title(entry["label"], fontsize=5.5, pad=3)

        for m in range(n_models):
            heatmap_row, overlay_row = 1 + 2 * m, 2 + 2 * m
            attribution = entry["maps"][m]
            axes[heatmap_row, column].imshow(attribution, cmap="inferno", vmin=0, vmax=1)
            axes[overlay_row, column].imshow(entry["image"])
            axes[overlay_row, column].imshow(attribution, cmap="inferno", alpha=0.5, vmin=0, vmax=1)

    labels = ["original"]
    for m in range(n_models):
        tag = model_labels[m] if model_labels else f"model {m}"
        labels += [f"{tag}\nheatmap", f"{tag}\noverlay"]

    for row in range(total_rows):
        axes[row, 0].set_ylabel(labels[row], fontsize=5.5, rotation=0, ha="right", va="center", labelpad=26)
        for column in range(n):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)

    return save_figure(fig, out_dir, name)


def metric_distribution(
    per_image: pd.DataFrame,
    metric: str,
    out_dir: str | Path,
    name: str,
    methods: Optional[list[str]] = None,
) -> list[Path]:
    """Box plots of a per image XAI metric across models, one panel per method."""
    apply_style()
    methods = methods or sorted(per_image["method"].unique())
    models = list(dict.fromkeys(per_image["model"]))

    fig, axes = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 3.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, methods):
        subset = per_image[per_image["method"] == method]
        data = [subset[subset["model"] == m][metric].dropna().to_numpy() for m in models]
        data = [d for d in data if len(d)]
        if not data:
            continue
        parts = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(parts["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in parts["medians"]:
            median.set_color("black")
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=6)
        ax.set_title(method)

    axes[0].set_ylabel(metric.replace("_", " "))
    return save_figure(fig, out_dir, name)
