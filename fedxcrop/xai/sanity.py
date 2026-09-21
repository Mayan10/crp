"""Model randomization sanity check (Adebayo et al., 2018).

An attribution method that produces the same map after the model's weights
have been destroyed is not explaining the model, it is describing the image
(edges, texture). Any localization or faithfulness result from such a method
says nothing about what the network learned.

The check randomizes the classifier first, then progressively deeper blocks of
the backbone, and reports the rank correlation between each randomized map and
the original. A correlation that stays high is a failure of the method, and is
reported either way rather than quietly dropped.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from fedxcrop.xai.attribution import attribute
from fedxcrop.xai.metrics import spearman_agreement


def randomize_module(module: nn.Module, generator: Optional[torch.Generator] = None) -> None:
    """Replace a module's learned parameters with fresh random values."""
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(child.weight, mode="fan_out", nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.BatchNorm2d):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)
            child.reset_running_stats()


def cascading_randomization_stages(model: nn.Module) -> list[tuple[str, list[nn.Module]]]:
    """Stages from the output backwards: classifier, then deeper feature blocks.

    Cascading (rather than independent) randomization, so each stage keeps the
    destruction from the previous one. By the last stage the whole network is
    random.
    """
    stages: list[tuple[str, list[nn.Module]]] = [("classifier", [model.classifier])]
    blocks = list(model.features)
    # Randomize the feature blocks from the deepest backwards, in chunks.
    checkpoints = [len(blocks) - 1, 14, 10, 6, 0]
    previous = len(blocks)
    for start in checkpoints:
        chunk = blocks[start:previous]
        if chunk:
            stages.append((f"features[{start}:{previous}]", chunk))
        previous = start
    return stages


def randomization_curve(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    device: torch.device,
    image_size: int = 224,
    n_samples: int = 20,
    noise_fraction: float = 0.1,
    seed: int = 0,
) -> list[dict]:
    """Agreement with the original map as more of the model is randomized.

    Returns one record per stage with the mean Spearman correlation over the
    given images. For a trustworthy method this falls towards zero.
    """
    torch.manual_seed(seed)
    model = model.to(device).eval()
    images = images.to(device)
    targets = targets.to(device)

    original = attribute(model, images, targets, method, image_size, n_samples, noise_fraction)

    randomized_model = copy.deepcopy(model)
    records = []
    for stage_name, modules in cascading_randomization_stages(randomized_model):
        for module in modules:
            randomize_module(module)
        randomized_model.eval()

        maps = attribute(
            randomized_model, images, targets, method, image_size, n_samples, noise_fraction
        )
        correlations = [
            spearman_agreement(original[i], maps[i]) for i in range(len(images))
        ]
        finite = [c for c in correlations if np.isfinite(c)]
        records.append(
            {
                "stage": stage_name,
                "method": method,
                "mean_spearman": float(np.mean(finite)) if finite else float("nan"),
                "std_spearman": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "n_images": int(len(images)),
                "n_finite": int(len(finite)),
            }
        )
    return records
