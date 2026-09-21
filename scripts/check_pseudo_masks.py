"""Render a random sample of lesion pseudo-masks so their quality can be judged.

The lesion masks are derived by a colour rule rather than annotated, so every
lesion localization number depends on how good this rule actually is. This
script exists so that question is answered by looking, not by assuming.

Usage:
    python scripts/check_pseudo_masks.py --n 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config
from fedxcrop.data.splits import load_split
from fedxcrop.xai.masks import healthy_hue_range, lesion_pseudo_mask, load_mapping, save_hue_range
from fedxcrop.viz.xai_figures import load_display_image, pseudo_mask_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(cfg.data.root)
    splits_dir = Path(cfg.data.splits_dir)
    out_dir = Path(cfg.results_dir) / "xai"

    mapping = load_mapping(splits_dir)
    hue_path = out_dir / "healthy_hue_range.json"
    if hue_path.is_file():
        hue_range = json.loads(hue_path.read_text())
    else:
        train = load_split(splits_dir, "train")
        healthy = train[train["class_name"].str.contains("healthy", case=False)]
        hue_range = healthy_hue_range(cfg.data.root, healthy, mapping, cfg.data.image_size)
        save_hue_range(hue_range, hue_path)

    print(f"healthy hue range: {hue_range['hue_low']:.1f} to {hue_range['hue_high']:.1f}")

    sample = pd.read_csv(splits_dir / "xai_sample.csv")
    diseased = sample[~sample["class_name"].str.contains("healthy", case=False)]
    rng = np.random.default_rng(args.seed)
    chosen = diseased.iloc[rng.choice(len(diseased), size=min(args.n, len(diseased)), replace=False)]

    rows, empty, coverages = [], 0, []
    for row in chosen.itertuples():
        segmented = mapping.get(row.path, "")
        if not segmented:
            continue
        mask = lesion_pseudo_mask(
            root / row.path, root / segmented,
            hue_range["hue_low"], hue_range["hue_high"], cfg.data.image_size,
        )
        if not mask.any():
            empty += 1
        coverages.append(float(mask.mean()))
        rows.append(
            {
                "image": load_display_image(root / row.path, cfg.data.image_size),
                "mask": mask,
                "label": row.class_name.replace("___", " ").replace("_", " ")[:26],
            }
        )

    paths = pseudo_mask_grid(rows, Path(cfg.results_dir) / "figures", "lesion_pseudo_masks")
    print(f"rendered {len(rows)} pseudo-masks, {empty} of them empty")
    print(f"mask coverage: median {100 * np.median(coverages):.1f} percent, "
          f"min {100 * min(coverages):.1f}, max {100 * max(coverages):.1f}")
    print(f"wrote {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
