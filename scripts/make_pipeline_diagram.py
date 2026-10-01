"""Draw the system overview figure.

Every number on the diagram is read from a file the pipeline wrote or from the
configuration it ran with: split sizes from the dataset report, training
settings from the base config, the alphas from the partition files and mu from
the validation selection. The diagram therefore cannot drift from the runs.

Usage:
    python scripts/make_pipeline_diagram.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fedxcrop.config import load_config
from fedxcrop.viz.style import PALETTE, apply_style, save_figure

FILL = {
    "data": "#E8F1F8",
    "client": "#FBEBDD",
    "server": "#F3E3EE",
    "model": "#E3F2EC",
    "xai": "#FFF3D6",
}
EDGE = {
    "data": PALETTE[0],
    "client": PALETTE[1],
    "server": PALETTE[3],
    "model": PALETTE[2],
    "xai": PALETTE[4],
}


def box(ax, x, y, w, h, kind, title, body="", title_size=7.5, body_size=6.2):
    """A rounded box centred on (x, y) with a bold title and an optional body."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=FILL[kind], edgecolor=EDGE[kind], linewidth=1.0,
    ))
    if not body:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
        return
    title_y = y + h / 2 - 0.02
    ax.text(x, title_y, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
    ax.text(x, (title_y - 0.012 + y - h / 2) / 2, body, ha="center", va="center",
            fontsize=body_size, linespacing=1.3)


def bus(ax, source, y, targets, label=""):
    """A trunk from source down to a horizontal bus at y, then one arrow per target."""
    sx, sy = source
    ax.plot([sx, sx], [sy, y], color="0.25", linewidth=0.8)
    xs = [t[0] for t in targets] + [sx]
    ax.plot([min(xs), max(xs)], [y, y], color="0.25", linewidth=0.8)
    for tx, ty in targets:
        arrow(ax, (tx, y), (tx, ty))
    if label:
        ax.text(sx + 0.012, (sy + y) / 2, label, ha="left", va="center",
                fontsize=5.8, color="0.2")


def arrow(ax, start, end, label="", both=False, label_offset=(0.01, 0.0), style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="<|-|>" if both else style,
        mutation_scale=7, linewidth=0.8, color="0.25", shrinkA=0, shrinkB=0,
    ))
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="left", va="center", fontsize=5.8, color="0.2")


def alphas_from_partitions(partitions_dir: Path) -> list[float]:
    alphas = set()
    for path in partitions_dir.glob("dirichlet_*.json"):
        alphas.add(float(json.loads(path.read_text())["alpha"]))
    return sorted(alphas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    results = Path(cfg.results_dir)
    report = json.loads((results / "data" / "dataset_report.json").read_text())
    sizes = report["split"]["sizes"]
    total = report["split"]["total"]
    n_classes = report["dataset"]["num_classes"]
    mu = json.loads((results / "tables" / "mu_selection.json").read_text())["selected_mu"]
    alphas = alphas_from_partitions(results / "data" / "partitions")
    k = int(cfg.partition.num_clients)
    alpha_text = ", ".join(f"{a:g}" for a in alphas)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Data and the fixed split.
    box(ax, 0.5, 0.95, 0.46, 0.065, "data", "PlantVillage, colour images",
        f"{total:,} images, {n_classes} classes")
    split_y, split_h = 0.82, 0.07
    split_xs = (0.2, 0.5, 0.8)
    box(ax, 0.2, split_y, 0.27, split_h, "data", f"Train split ({sizes['train']:,})",
        "partitioned across clients")
    box(ax, 0.5, split_y, 0.27, split_h, "data", f"Validation split ({sizes['val']:,})",
        "selects the round or epoch")
    box(ax, 0.8, split_y, 0.27, split_h, "data", f"Test split ({sizes['test']:,})",
        "evaluated once per run")
    bus(ax, (0.5, 0.95 - 0.0325), 0.885, [(x, split_y + split_h / 2) for x in split_xs],
        label=f"stratified 80/10/10 split, seed {cfg.data.split_seed}")

    # Clients.
    client_y, client_h = 0.645, 0.085
    width = 0.155
    xs = [0.1 + i * (0.8 / (k - 1)) for i in range(k)]
    for i, x in enumerate(xs):
        box(ax, x, client_y, width, client_h, "client", f"Client {i + 1}",
            f"local SGD, {cfg.federated.local_epochs} epoch\nper round")
    bus(ax, (0.2, split_y - split_h / 2), 0.73, [(x, client_y + client_h / 2) for x in xs],
        label=f"IID, or Dirichlet label skew with alpha in {{{alpha_text}}}")

    # Server.
    server_y, server_h = 0.49, 0.085
    box(ax, 0.5, server_y, 0.9, server_h, "server", "Aggregation server",
        f"sample size weighted averaging, {cfg.federated.rounds} rounds, "
        f"FedAvg or FedProx (mu = {mu:g})\n"
        "clients send model weights only, images never leave a client")
    for x in xs:
        arrow(ax, (x, client_y - client_h / 2), (x, server_y + server_h / 2), both=True)

    # Global model, and the centralized reference beside it.
    model_y = 0.355
    box(ax, 0.5, model_y, 0.40, 0.07, "model", "Global MobileNetV2",
        "ImageNet initialization, best round on validation")
    arrow(ax, (0.5, server_y - server_h / 2), (0.5, model_y + 0.035))
    box(ax, 0.86, model_y, 0.25, 0.07, "model", "Centralized reference",
        f"same train split, {cfg.centralized.epochs} epochs")

    # Explanations and their evaluation.
    xai_y = 0.24
    box(ax, 0.3, xai_y, 0.32, 0.07, "xai", "Grad-CAM",
        "last convolutional block, predicted class")
    box(ax, 0.7, xai_y, 0.32, 0.07, "xai", "SmoothGrad",
        f"{cfg.xai.smoothgrad_samples} noisy copies, predicted class")
    arrow(ax, (0.5, model_y - 0.035), (0.3, xai_y + 0.035))
    arrow(ax, (0.5, model_y - 0.035), (0.7, xai_y + 0.035))

    eval_y = 0.085
    n_images = int(cfg.xai.images_per_class) * n_classes
    box(ax, 0.5, eval_y, 0.84, 0.115, "xai",
        f"Quantitative evaluation on {n_images} test images ({cfg.xai.images_per_class} per class)",
        "faithfulness: deletion and insertion AUC\n"
        "localization: energy ratio and pointing game on leaf masks and lesion pseudo-masks\n"
        "agreement with the centralized model, and a model randomization check",
        body_size=6.0)
    arrow(ax, (0.3, xai_y - 0.035), (0.3, eval_y + 0.0575))
    arrow(ax, (0.7, xai_y - 0.035), (0.7, eval_y + 0.0575))

    paths = save_figure(fig, results / "figures", "fig0_pipeline")
    print(f"wrote {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
