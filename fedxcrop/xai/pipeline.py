"""Running every attribution method and metric over a fixed image sample.

The same images, in the same order, for every model, so that differences
between models are differences in the model rather than in what was measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedxcrop.data.dataset import build_dataset, build_loader
from fedxcrop.xai.attribution import attribute, predicted_classes
from fedxcrop.xai.masks import leaf_mask_from_segmented, lesion_pseudo_mask
from fedxcrop.xai.metrics import (
    deletion_insertion_auc,
    energy_ratio,
    mask_area_fraction,
    pointing_game,
    spearman_agreement,
    topk_iou,
)


def build_xai_sample(
    test: pd.DataFrame,
    class_names: list[str],
    images_per_class: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """A fixed stratified sample of test images, the same for every model."""
    rng = np.random.default_rng(seed)
    parts = []
    for label in sorted(test["label"].unique()):
        pool = test[test["label"] == label].sort_values("path")
        take = min(images_per_class, len(pool))
        chosen = pool.iloc[rng.choice(len(pool), size=take, replace=False)]
        parts.append(chosen)
    sample = pd.concat(parts).sort_values(["label", "path"]).reset_index(drop=True)
    sample["class_name"] = [class_names[int(i)] for i in sample["label"]]
    return sample


def dataset_channel_mean(loader: DataLoader, max_batches: int = 20) -> torch.Tensor:
    """Per channel mean of the normalized inputs, the deletion baseline."""
    totals, count = None, 0
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        batch_sum = images.sum(dim=(0, 2, 3))
        totals = batch_sum if totals is None else totals + batch_sum
        count += images.shape[0] * images.shape[2] * images.shape[3]
    if totals is None:
        raise RuntimeError("no batches were available to compute the channel mean")
    return totals / count


def attribution_maps_for_model(
    model,
    sample: pd.DataFrame,
    cfg,
    device: torch.device,
    method: str,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Attribution maps and the predicted class for every sampled image."""
    dataset = build_dataset(cfg.data.root, sample, train=False, image_size=cfg.data.image_size)
    maps, predictions = [], []

    model = model.to(device).eval()
    for start in range(0, len(sample), batch_size):
        stop = min(start + batch_size, len(sample))
        images = torch.stack([dataset[i][0] for i in range(start, stop)]).to(device)
        targets = predicted_classes(model, images)
        maps.append(
            attribute(
                model, images, targets, method,
                image_size=cfg.data.image_size,
                n_samples=cfg.xai.smoothgrad_samples,
                noise_fraction=cfg.xai.smoothgrad_noise_fraction,
            )
        )
        predictions.append(targets.cpu().numpy())

    return np.concatenate(maps), np.concatenate(predictions)


def evaluate_maps(
    model,
    sample: pd.DataFrame,
    maps: np.ndarray,
    predictions: np.ndarray,
    cfg,
    device: torch.device,
    mapping: dict[str, str],
    hue_range: dict,
    reference_maps: Optional[np.ndarray] = None,
    channel_mean: Optional[torch.Tensor] = None,
    compute_faithfulness: bool = True,
) -> pd.DataFrame:
    """Per image metrics for one model and one attribution method.

    `reference_maps` are the centralized model's maps. When given, the
    agreement columns answer "is this explanation the same as the centralized
    one" as a number rather than as an impression from a figure.
    """
    root = Path(cfg.data.root)
    dataset = build_dataset(cfg.data.root, sample, train=False, image_size=cfg.data.image_size)
    model = model.to(device).eval()

    rows = []
    for i, row in enumerate(sample.itertuples()):
        attribution = maps[i]
        record = {
            "path": row.path,
            "label": int(row.label),
            "class_name": row.class_name,
            "predicted": int(predictions[i]),
            "correct": int(predictions[i]) == int(row.label),
        }

        if compute_faithfulness:
            image, _ = dataset[i]
            faithfulness = deletion_insertion_auc(
                model, image, attribution, int(predictions[i]), device,
                step=cfg.xai.occlusion_step, baseline_mean=channel_mean,
            )
            record["deletion_auc"] = faithfulness["deletion_auc"]
            record["insertion_auc"] = faithfulness["insertion_auc"]

        segmented = mapping.get(row.path, "")
        if segmented:
            leaf = leaf_mask_from_segmented(root / segmented, cfg.data.image_size)
            if leaf.any():
                record["leaf_energy_ratio"] = energy_ratio(attribution, leaf)
                record["leaf_area_fraction"] = mask_area_fraction(leaf)
                record["leaf_pointing_hit"] = pointing_game(attribution, leaf)

            is_healthy = "healthy" in row.class_name.lower()
            if not is_healthy:
                lesion = lesion_pseudo_mask(
                    root / row.path, root / segmented,
                    hue_range["hue_low"], hue_range["hue_high"], cfg.data.image_size,
                )
                if lesion.any():
                    record["lesion_energy_ratio"] = energy_ratio(attribution, lesion)
                    record["lesion_area_fraction"] = mask_area_fraction(lesion)
                    record["lesion_pointing_hit"] = pointing_game(attribution, lesion)

        if reference_maps is not None:
            record["spearman_vs_centralized"] = spearman_agreement(attribution, reference_maps[i])
            record["topk_iou_vs_centralized"] = topk_iou(
                attribution, reference_maps[i], cfg.xai.topk_fraction
            )

        rows.append(record)

    return pd.DataFrame(rows)


def save_maps(maps: np.ndarray, path: str | Path) -> Path:
    """Store maps compressed, as float16: they are only ever compared by rank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, maps=maps.astype(np.float16))
    return path


def load_maps(path: str | Path) -> np.ndarray:
    with np.load(path) as data:
        return data["maps"].astype(np.float32)
