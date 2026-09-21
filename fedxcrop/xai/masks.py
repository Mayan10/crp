"""Leaf masks and lesion pseudo-masks.

The leaf mask is free: the segmented variant of PlantVillage is the same
photograph with the background removed, so any non-black pixel is leaf.

The lesion mask is not free, and is therefore called a pseudo-mask throughout.
It is derived by colour, not annotated, so it is evidence about where diseased
looking tissue is, not ground truth. The healthy hue range it depends on is
measured from the healthy classes rather than assumed, and the resulting masks
are saved as an overlay grid so their quality can be judged by eye.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from fedxcrop.data.transforms import mask_transform

BACKGROUND_THRESHOLD = 10  # 0 to 255, pixels this dark in all channels are background
LEAF_EROSION_PIXELS = 3  # trims the blended leaf/background rim before lesion detection


def leaf_mask_from_segmented(
    segmented_path: str | Path,
    image_size: int = 224,
    threshold: int = BACKGROUND_THRESHOLD,
) -> np.ndarray:
    """Boolean leaf mask, aligned with the center cropped input image.

    The segmented image goes through the same resize and center crop as the
    model input, so mask and attribution share a pixel grid.
    """
    with Image.open(segmented_path) as handle:
        image = handle.convert("RGB")
    tensor = mask_transform(image_size)(image)
    as_uint8 = (tensor.numpy() * 255).astype(np.uint8)
    return (as_uint8.max(axis=0) > threshold)


def healthy_hue_range(
    root: str | Path,
    healthy_frame: pd.DataFrame,
    mapping: dict[str, str],
    image_size: int = 224,
    sample_size: int = 200,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    seed: int = 0,
) -> dict:
    """Measure the hue range of healthy leaf tissue.

    Sampled from images of the healthy classes only, taking the hue of leaf
    pixels and keeping the central 90 percent. Anything outside this range on a
    diseased leaf is what the pseudo-mask calls a lesion. The range is reported
    so a reader can see the threshold rather than trust it.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(healthy_frame))
    sample = healthy_frame.sample(n=n, random_state=int(rng.integers(0, 2**31)))

    hues, saturations, values = [], [], []
    for row in sample.itertuples():
        segmented = mapping.get(row.path)
        if not segmented:
            continue
        mask = leaf_mask_from_segmented(root / segmented, image_size)
        if not mask.any():
            continue

        with Image.open(root / row.path) as handle:
            image = handle.convert("RGB")
        rgb = (mask_transform(image_size)(image).numpy() * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb.transpose(1, 2, 0), cv2.COLOR_RGB2HSV)

        hues.append(hsv[:, :, 0][mask])
        saturations.append(hsv[:, :, 1][mask])
        values.append(hsv[:, :, 2][mask])

    if not hues:
        raise RuntimeError("no healthy leaf pixels were found, cannot derive a hue range")

    all_hues = np.concatenate(hues)
    return {
        "hue_low": float(np.percentile(all_hues, low_percentile)),
        "hue_high": float(np.percentile(all_hues, high_percentile)),
        "hue_median": float(np.median(all_hues)),
        "saturation_median": float(np.median(np.concatenate(saturations))),
        "value_median": float(np.median(np.concatenate(values))),
        "n_images_sampled": int(len(hues)),
        "n_pixels": int(all_hues.size),
        "percentiles": [low_percentile, high_percentile],
        "note": (
            "OpenCV hue is on a 0 to 179 scale. Leaf pixels whose hue falls "
            "outside this range are treated as lesion tissue. This is a colour "
            "heuristic, not an annotation."
        ),
    }


def lesion_pseudo_mask(
    image_path: str | Path,
    segmented_path: str | Path,
    hue_low: float,
    hue_high: float,
    image_size: int = 224,
    min_area: int = 20,
    erosion_pixels: int = LEAF_EROSION_PIXELS,
) -> np.ndarray:
    """Pixels inside the leaf whose hue falls outside the healthy green range.

    The leaf mask is eroded first. The outline of a segmented leaf is a band of
    pixels that blend leaf with the removed background, and those blended
    pixels are never green, so without erosion the rule traces the leaf
    boundary on almost every image and calls it disease. That artefact was
    visible in the first render of the mask grid.

    Small speckles are then removed with a morphological opening, since
    isolated pixels are usually sensor noise or compression artefacts rather
    than lesions.
    """
    leaf = leaf_mask_from_segmented(segmented_path, image_size)
    if not leaf.any():
        return np.zeros_like(leaf)

    if erosion_pixels > 0:
        size = 2 * erosion_pixels + 1
        leaf = cv2.erode(
            leaf.astype(np.uint8), np.ones((size, size), np.uint8), iterations=1
        ).astype(bool)
        if not leaf.any():
            return np.zeros_like(leaf)

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    rgb = (mask_transform(image_size)(image).numpy() * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb.transpose(1, 2, 0), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]

    outside_healthy = (hue < hue_low) | (hue > hue_high)
    lesion = (outside_healthy & leaf).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)

    if lesion.sum() < min_area:
        return np.zeros_like(leaf)
    return lesion.astype(bool)


def load_mapping(splits_dir: str | Path) -> dict[str, str]:
    """Color image path to segmented image path, as a lookup."""
    from fedxcrop.data.index import load_segmented_mapping

    frame = load_segmented_mapping(splits_dir)
    return dict(zip(frame["path"], frame["segmented_path"]))


def save_hue_range(hue_range: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(hue_range, handle, indent=2)
    return path
