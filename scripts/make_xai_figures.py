"""Build the attribution grids and the failure case figure.

The attribution grids need the trained checkpoints, so this runs after the
grid and after scripts/run_xai.py. The failure case figure is drawn from the
maps run_xai.py saved, so `--failure-only` needs no checkpoints.

Usage:
    python scripts/make_xai_figures.py --models centralized_seed0 fedprox_..._seed0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config, resolve_device
from fedxcrop.data.dataset import build_dataset
from fedxcrop.models.factory import build_model, load_checkpoint
from fedxcrop.viz.xai_figures import attribution_grid, load_display_image
from fedxcrop.xai.attribution import attribute, predicted_classes

# The classes the figure must show, per the reporting requirements, plus two
# more so a reader sees both a spotted and a chlorotic disease.
REQUIRED_CLASSES = [
    "Tomato___Early_blight",
    "Potato___Late_blight",
    "Apple___Apple_scab",
    "Tomato___healthy",
    "Grape___Black_rot",
    "Corn_(maize)___Common_rust_",
]


def resolve_checkpoint(name: str, runs_dir: Path) -> Path:
    for candidate in (
        runs_dir / "centralized" / name / "best.pt",
        runs_dir / "federated" / name / "best.pt",
        runs_dir / name / "best.pt",
        Path(name),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no checkpoint found for {name!r}")


def short_label(name: str) -> str:
    """Compact model name for a figure row label."""
    if name.startswith("centralized"):
        return "centralized"
    parts = name.split("_")
    tag = parts[0]
    for part in parts:
        if part.startswith("alpha"):
            tag += f" {part}"
        if part.startswith("mu"):
            tag += f" {part}"
    return tag


def short_labels(names: list[str]) -> list[str]:
    """Shortened row labels, falling back to full names if shortening collides.

    A figure whose two rows carry the same label is worse than one with long
    labels, since the reader cannot tell which model produced which row.
    """
    short = [short_label(name) for name in names]
    return short if len(set(short)) == len(short) else list(names)


def failure_case_figure(
    per_image: pd.DataFrame,
    maps_dir: Path,
    target_model: str,
    methods: list[str],
    root: Path,
    figures_dir: Path,
    image_size: int,
    n: int = 6,
) -> list[Path]:
    """The images where the target model's attribution falls least on the leaf.

    Images are ranked by the leaf energy ratio averaged over the methods, and
    each panel is labelled with that same value. The maps are the saved ones
    the metrics were computed on, read back by their position in the XAI
    sample, so the figure needs no checkpoint and shows exactly what was scored.
    """
    subset = per_image[(per_image["model"] == target_model) & per_image["leaf_energy_ratio"].notna()]
    if subset.empty:
        raise ValueError(f"no per image metrics for {target_model!r}")
    ranking = subset.groupby("path")["leaf_energy_ratio"].mean().nsmallest(n)

    maps_by_method = []
    for method in methods:
        rows = per_image[(per_image["model"] == target_model) & (per_image["method"] == method)]
        maps = np.load(maps_dir / f"{target_model}_{method}.npz")["maps"]
        if len(rows) != len(maps):
            raise ValueError(f"{len(rows)} metric rows but {len(maps)} saved maps for {method}")
        position = {path: i for i, path in enumerate(rows["path"])}
        maps_by_method.append([maps[position[path]].astype(np.float32) for path in ranking.index])

    class_names = subset.drop_duplicates("path").set_index("path")["class_name"]
    entries = []
    for i, (path, ratio) in enumerate(ranking.items()):
        entries.append(
            {
                "image": load_display_image(root / path, image_size),
                "label": f"{class_names[path].split('___')[-1].replace('_', ' ')[:20]}\n"
                         f"leaf {ratio:.2f}",
                "maps": [method_maps[i] for method_maps in maps_by_method],
            }
        )
    print(f"  lowest mean leaf energy ratios for {target_model}: "
          + ", ".join(f"{v:.3f}" for v in ranking))
    return attribution_grid(entries, figures_dir, "fig7_failure_cases", model_labels=list(methods))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--models", nargs="+", required=True,
                        help="run names, shown as rows in the order given")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "smoothgrad"])
    parser.add_argument("--classes", nargs="+", default=REQUIRED_CLASSES)
    parser.add_argument("--failure-only", action="store_true",
                        help="only the failure case figure, from saved maps, no checkpoints needed")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg.device)
    root = Path(cfg.data.root)
    runs_dir = Path(cfg.runs_dir)
    figures_dir = Path(cfg.results_dir) / "figures"

    sample = pd.read_csv(Path(cfg.data.splits_dir) / "xai_sample.csv")

    models = {}
    for name in [] if args.failure_only else args.models:
        model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False)
        load_checkpoint(model, resolve_checkpoint(name, runs_dir), map_location=device)
        models[name] = model.to(device).eval()

    # One representative image per requested class: the first in the fixed
    # sample, so the choice is not made by looking at the attributions.
    chosen = []
    for class_name in args.classes:
        matches = sample[sample["class_name"] == class_name]
        if matches.empty:
            print(f"  skipping {class_name}, not in the XAI sample")
            continue
        chosen.append(matches.iloc[0])
    chosen = pd.DataFrame(chosen).reset_index(drop=True)
    print(f"showing {len(chosen)} classes across {len(models)} models\n")

    dataset = build_dataset(cfg.data.root, chosen, train=False, image_size=cfg.data.image_size)
    images = torch.stack([dataset[i][0] for i in range(len(chosen))]).to(device)

    for method in [] if args.failure_only else args.methods:
        entries = []
        maps_per_model = []
        for name, model in models.items():
            targets = predicted_classes(model, images)
            maps_per_model.append(
                attribute(model, images, targets, method,
                          image_size=cfg.data.image_size,
                          n_samples=cfg.xai.smoothgrad_samples,
                          noise_fraction=cfg.xai.smoothgrad_noise_fraction)
            )

        for i, row in enumerate(chosen.itertuples()):
            entries.append(
                {
                    "image": load_display_image(root / row.path, cfg.data.image_size),
                    "label": row.class_name.replace("___", "\n").replace("_", " ")[:30],
                    "maps": [m[i] for m in maps_per_model],
                }
            )

        number = 4 if method == "gradcam" else 5
        paths = attribution_grid(
            entries, figures_dir, f"fig{number}_{method}_grid",
            model_labels=short_labels(list(models)),
        )
        print(f"wrote {paths[0]}")

    # Failure cases: the images where attribution falls least on the leaf.
    per_image_path = Path(cfg.results_dir) / "xai" / "per_image_metrics.csv"
    if per_image_path.is_file():
        paths = failure_case_figure(
            pd.read_csv(per_image_path), Path(cfg.results_dir) / "xai" / "maps",
            args.models[-1], args.methods, root, figures_dir, cfg.data.image_size,
        )
        print(f"wrote {paths[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
