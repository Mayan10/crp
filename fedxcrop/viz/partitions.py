"""Figures describing how the train split is spread across clients."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedxcrop.data.partition import client_frames, label_histogram
from fedxcrop.viz.style import apply_style, save_figure


def client_class_histogram(
    partition: pd.DataFrame,
    class_names: list[str],
    out_dir: str | Path,
    name: str,
) -> list[Path]:
    """Stacked bar of the label composition of every client.

    One bar per client, segments sized by how many images of each class that
    client holds. This is the figure the earlier version of this work never
    showed, which is why its claim of strong heterogeneity was unsupported.
    """
    apply_style()
    clients = client_frames(partition)
    num_classes = len(class_names)

    counts = np.stack([label_histogram(frame, num_classes) for frame in clients.values()])
    client_ids = list(clients.keys())

    fig, ax = plt.subplots(figsize=(1.1 * len(client_ids) + 2.2, 3.2))
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, num_classes))

    bottom = np.zeros(len(client_ids), dtype=float)
    for label in range(num_classes):
        heights = counts[:, label].astype(float)
        ax.bar(client_ids, heights, bottom=bottom, color=colors[label], width=0.75, linewidth=0)
        bottom += heights

    ax.set_xlabel("client")
    ax.set_ylabel("images")
    ax.set_xticks(client_ids)
    ax.margins(x=0.05)
    ax.grid(axis="x", visible=False)

    mappable = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=0, vmax=num_classes - 1)
    )
    bar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.04)
    bar.set_label("class label")

    return save_figure(fig, out_dir, name)


def alpha_comparison_grid(
    partitions: dict[float, pd.DataFrame],
    class_names: list[str],
    out_dir: str | Path,
    name: str = "fig1_client_class_histograms",
) -> list[Path]:
    """One panel per Dirichlet alpha, sharing a y axis for honest comparison."""
    apply_style()
    num_classes = len(class_names)
    alphas = sorted(partitions)

    fig, axes = plt.subplots(
        1, len(alphas), figsize=(3.1 * len(alphas), 3.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, num_classes))

    for ax, alpha in zip(axes, alphas):
        clients = client_frames(partitions[alpha])
        counts = np.stack([label_histogram(f, num_classes) for f in clients.values()])
        ids = list(clients.keys())

        bottom = np.zeros(len(ids), dtype=float)
        for label in range(num_classes):
            ax.bar(ids, counts[:, label], bottom=bottom, color=colors[label], width=0.75, linewidth=0)
            bottom += counts[:, label]

        ax.set_xlabel("client")
        ax.set_xticks(ids)
        ax.set_title(f"alpha = {alpha:g}")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("images")
    mappable = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=0, vmax=num_classes - 1)
    )
    bar = fig.colorbar(mappable, ax=axes, pad=0.015, fraction=0.03)
    bar.set_label("class label")

    return save_figure(fig, out_dir, name)
