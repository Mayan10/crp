"""Quantitative evaluation of attribution maps.

This is what the earlier version of this work was missing. It stated the gap
as the absence of quantitative post-aggregation XAI evaluation and then
presented a handful of qualitative images. Each function here turns one
statement a reader might otherwise have to take on trust into a number:

  faithfulness   does removing the highlighted pixels actually hurt the model
  localization   does the attribution fall on the leaf, and on the lesion
  agreement      is a federated model's explanation the same as the centralized one
  sanity         does the map change at all when the weights are destroyed
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


# numpy renamed trapz to trapezoid in 2.0. Bind once so the code runs on both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _ordered_pixels(attribution: np.ndarray, descending: bool = True) -> np.ndarray:
    """Flat pixel indices sorted by attribution."""
    flat = attribution.reshape(-1)
    order = np.argsort(flat)
    return order[::-1] if descending else order


@torch.no_grad()
def deletion_insertion_auc(
    model: nn.Module,
    image: torch.Tensor,
    attribution: np.ndarray,
    target: int,
    device: torch.device,
    step: float = 0.02,
    baseline_mean: Optional[torch.Tensor] = None,
    blur_sigma: float = 11.0,
    batch_size: int = 64,
) -> dict:
    """Deletion and insertion curves for one image.

    Deletion removes the most important pixels first, replacing them with the
    per channel dataset mean, and tracks the probability of the target class.
    A faithful attribution makes the probability fall quickly, so a **lower**
    deletion AUC is better.

    Insertion starts from a heavily blurred image and adds the most important
    pixels back first. A faithful attribution makes the probability rise
    quickly, so a **higher** insertion AUC is better.

    Both curves are sampled in `step` fractions of the pixels and the area is
    taken with the trapezoid rule over the fraction axis, so the AUC is in
    [0, 1] and comparable across images.
    """
    model.eval()
    image = image.to(device)
    channels, height, width = image.shape
    n_pixels = height * width

    if baseline_mean is None:
        baseline_mean = image.mean(dim=(1, 2))
    deletion_baseline = baseline_mean.to(device).view(channels, 1, 1).expand_as(image).clone()

    kernel = int(blur_sigma * 4) | 1
    blurred = _gaussian_blur(image.unsqueeze(0), kernel, blur_sigma).squeeze(0)

    order = _ordered_pixels(attribution, descending=True)
    fractions = np.arange(0.0, 1.0 + step / 2, step)
    counts = np.round(fractions * n_pixels).astype(int)

    def curve(start: torch.Tensor, finish: torch.Tensor) -> np.ndarray:
        """Probability of the target class as pixels move from start to finish."""
        probabilities = []
        for begin in range(0, len(counts), batch_size):
            chunk = counts[begin : begin + batch_size]
            batch = start.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
            for i, count in enumerate(chunk):
                if count > 0:
                    selected = order[:count]
                    rows, cols = np.unravel_index(selected, (height, width))
                    batch[i, :, rows, cols] = finish[:, rows, cols]
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1)[:, target].cpu().numpy())
        return np.concatenate(probabilities)

    deletion = curve(image, deletion_baseline)
    insertion = curve(blurred, image)

    return {
        "deletion_auc": float(_trapezoid(deletion, fractions)),
        "insertion_auc": float(_trapezoid(insertion, fractions)),
        "deletion_curve": deletion.tolist(),
        "insertion_curve": insertion.tolist(),
        "fractions": fractions.tolist(),
    }


def _gaussian_blur(batch: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, used as the insertion starting point."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=batch.device)
    coords -= (kernel_size - 1) / 2.0
    weights = torch.exp(-(coords**2) / (2 * sigma**2))
    weights = weights / weights.sum()

    channels = batch.shape[1]
    horizontal = weights.view(1, 1, 1, -1).expand(channels, 1, 1, kernel_size)
    vertical = weights.view(1, 1, -1, 1).expand(channels, 1, kernel_size, 1)
    padding = kernel_size // 2

    out = F.conv2d(batch, horizontal, padding=(0, padding), groups=channels)
    return F.conv2d(out, vertical, padding=(padding, 0), groups=channels)


def energy_ratio(attribution: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total attribution that falls inside a mask.

    With a leaf mask this answers "is the model looking at the leaf at all",
    which is the claim the original figures were used to support without ever
    being measured. A value near the mask's own area fraction means the
    attribution is no better placed than chance.
    """
    if attribution.shape != mask.shape:
        raise ValueError(f"shape mismatch: attribution {attribution.shape}, mask {mask.shape}")
    total = float(attribution.sum())
    if total <= 0:
        return 0.0
    return float(attribution[mask.astype(bool)].sum() / total)


def pointing_game(attribution: np.ndarray, mask: np.ndarray) -> bool:
    """Does the single highest attribution pixel land inside the mask.

    A deliberately strict, assumption free localization check: it needs only
    the peak of the map and does not depend on any threshold.
    """
    if attribution.shape != mask.shape:
        raise ValueError(f"shape mismatch: attribution {attribution.shape}, mask {mask.shape}")
    if not mask.any():
        raise ValueError("cannot play the pointing game against an empty mask")
    peak = np.unravel_index(np.argmax(attribution), attribution.shape)
    return bool(mask[peak])


def mask_area_fraction(mask: np.ndarray) -> float:
    """Share of the image the mask covers, the chance level for energy ratio."""
    return float(mask.astype(bool).mean())


def spearman_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two flattened attribution maps.

    Rank based, so it measures whether two models order pixels by importance
    the same way rather than whether they assign the same magnitudes. This is
    the number that makes "federated training preserves the explanation"
    testable instead of a claim about how two pictures look.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} and {b.shape}")
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    if np.ptp(flat_a) < 1e-12 or np.ptp(flat_b) < 1e-12:
        return float("nan")
    correlation, _ = spearmanr(flat_a, flat_b)
    return float(correlation)


def topk_iou(a: np.ndarray, b: np.ndarray, fraction: float = 0.2) -> float:
    """Intersection over union of the top `fraction` of pixels in two maps.

    Complements the rank correlation: two maps can correlate well overall and
    still disagree about where the most important region is.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} and {b.shape}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    k = max(1, int(round(fraction * a.size)))
    top_a = np.zeros(a.size, dtype=bool)
    top_b = np.zeros(b.size, dtype=bool)
    top_a[np.argsort(a.reshape(-1))[-k:]] = True
    top_b[np.argsort(b.reshape(-1))[-k:]] = True

    union = np.logical_or(top_a, top_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(top_a, top_b).sum() / union)
