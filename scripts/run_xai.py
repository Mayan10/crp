"""Quantitative explainability evaluation across models.

Usage:
    python scripts/run_xai.py --build-sample
    python scripts/run_xai.py --models centralized fedavg_alpha0.1 --methods gradcam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config, resolve_device, save_run_metadata
from fedxcrop.data.dataset import build_dataset, build_loader
from fedxcrop.data.splits import load_classes, load_split
from fedxcrop.models.factory import build_model, load_checkpoint
from fedxcrop.xai.masks import healthy_hue_range, load_mapping, save_hue_range
from fedxcrop.xai.pipeline import (
    attribution_maps_for_model,
    build_xai_sample,
    dataset_channel_mean,
    evaluate_maps,
    load_maps,
    save_maps,
)
from fedxcrop.xai.sanity import randomization_curve


def resolve_checkpoint(name: str, runs_dir: Path) -> Path:
    """Locate the best checkpoint of a named run."""
    candidates = [
        runs_dir / "centralized" / name / "best.pt",
        runs_dir / "federated" / name / "best.pt",
        runs_dir / name / "best.pt",
        Path(name),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no checkpoint found for {name!r}. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--models", nargs="+", default=[],
                        help="run names, the first is the reference for agreement")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "smoothgrad"])
    parser.add_argument("--build-sample", action="store_true",
                        help="only build and save the fixed XAI image sample")
    parser.add_argument("--sanity", action="store_true", help="run the randomization check")
    parser.add_argument("--sanity-images", type=int, default=40)
    parser.add_argument("--no-faithfulness", action="store_true",
                        help="skip deletion and insertion, which dominate the runtime")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg.device)
    splits_dir = Path(cfg.data.splits_dir)
    out_dir = Path(cfg.results_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(splits_dir)
    sample_path = splits_dir / "xai_sample.csv"

    if args.build_sample or not sample_path.is_file():
        test = load_split(splits_dir, "test", with_class_names=False)
        sample = build_xai_sample(test, classes, cfg.xai.images_per_class, cfg.data.split_seed)
        sample[["path", "label", "class_name"]].to_csv(sample_path, index=False)
        print(f"wrote {sample_path}: {len(sample)} images, "
              f"{cfg.xai.images_per_class} per class over {sample['label'].nunique()} classes")
        if args.build_sample:
            return 0

    sample = pd.read_csv(sample_path)
    if args.limit:
        sample = sample.head(args.limit).reset_index(drop=True)
    print(f"XAI sample: {len(sample)} images\n")

    mapping = load_mapping(splits_dir)

    hue_path = out_dir / "healthy_hue_range.json"
    if hue_path.is_file():
        hue_range = json.loads(hue_path.read_text())
    else:
        print("deriving the healthy hue range from the healthy classes")
        train = load_split(splits_dir, "train")
        healthy = train[train["class_name"].str.contains("healthy", case=False)]
        hue_range = healthy_hue_range(cfg.data.root, healthy, mapping, cfg.data.image_size)
        save_hue_range(hue_range, hue_path)
        print(f"  hue {hue_range['hue_low']:.1f} to {hue_range['hue_high']:.1f} "
              f"(OpenCV 0 to 179 scale), from {hue_range['n_images_sampled']} images\n")

    if not args.models:
        print("no models given, nothing further to do")
        return 0

    runs_dir = Path(cfg.runs_dir)
    val_loader = build_loader(
        build_dataset(cfg.data.root, load_split(splits_dir, "val", with_class_names=False).head(2000),
                      train=False, image_size=cfg.data.image_size),
        batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
    )
    channel_mean = dataset_channel_mean(val_loader)
    print(f"deletion baseline, per channel mean of the normalized input: "
          f"{[round(float(v), 3) for v in channel_mean]}\n")

    reference_name = args.models[0]
    all_rows = []

    for method in args.methods:
        reference_maps = None
        for name in args.models:
            checkpoint = resolve_checkpoint(name, runs_dir)
            model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False)
            load_checkpoint(model, checkpoint, map_location=device)

            print(f"{method:<11} {name:<38} ", end="", flush=True)
            maps, predictions = attribution_maps_for_model(model, sample, cfg, device, method)
            save_maps(maps, out_dir / "maps" / f"{name}_{method}.npz")

            if name == reference_name:
                reference_maps = maps

            frame = evaluate_maps(
                model, sample, maps, predictions, cfg, device, mapping, hue_range,
                reference_maps=None if name == reference_name else reference_maps,
                channel_mean=channel_mean,
                compute_faithfulness=not args.no_faithfulness,
            )
            frame["model"] = name
            frame["method"] = method
            all_rows.append(frame)

            summary = []
            for column in ("deletion_auc", "insertion_auc", "leaf_energy_ratio",
                           "spearman_vs_centralized"):
                if column in frame and frame[column].notna().any():
                    summary.append(f"{column.split('_')[0]} {frame[column].mean():.3f}")
            print("  ".join(summary))

    per_image = pd.concat(all_rows, ignore_index=True)
    per_image.to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"\nwrote {out_dir / 'per_image_metrics.csv'} ({len(per_image)} rows)")

    if args.sanity:
        print("\nmodel randomization sanity check")
        subset = sample.head(args.sanity_images).reset_index(drop=True)
        dataset = build_dataset(cfg.data.root, subset, train=False, image_size=cfg.data.image_size)
        images = torch.stack([dataset[i][0] for i in range(len(subset))])

        model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False)
        load_checkpoint(model, resolve_checkpoint(reference_name, runs_dir), map_location=device)
        from fedxcrop.xai.attribution import predicted_classes
        targets = predicted_classes(model.to(device).eval(), images.to(device))

        records = []
        for method in args.methods:
            curve = randomization_curve(
                model, images, targets, method, device,
                image_size=cfg.data.image_size,
                n_samples=cfg.xai.smoothgrad_samples,
                noise_fraction=cfg.xai.smoothgrad_noise_fraction,
            )
            for record in curve:
                print(f"  {method:<11} {record['stage']:<22} "
                      f"spearman {record['mean_spearman']:+.3f}")
            records += curve
        pd.DataFrame(records).to_csv(out_dir / "sanity_randomization.csv", index=False)
        print(f"wrote {out_dir / 'sanity_randomization.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
