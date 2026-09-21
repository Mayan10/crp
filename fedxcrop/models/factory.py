"""Model construction.

One factory, used identically by the centralized baseline and by every
federated client, so the only difference between those experiments is how the
weights are trained rather than what is being trained.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torchvision

MOBILENET_V2_FEATURE_DIM = 1280


def build_model(
    name: str = "mobilenet_v2",
    num_classes: int = 38,
    pretrained: bool = True,
) -> nn.Module:
    """MobileNetV2 with the classifier replaced, whole backbone trainable.

    `pretrained` loads ImageNet weights. The corrected federated protocol starts
    every client from these weights, never from a model that has already seen
    PlantVillage centrally.
    """
    if name != "mobilenet_v2":
        raise ValueError(f"unsupported model {name!r}, expected 'mobilenet_v2'")

    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(MOBILENET_V2_FEATURE_DIM, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters, used for the communication cost accounting."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


def get_weights(model: nn.Module) -> list:
    """Model state as a list of numpy arrays, the form the FL server exchanges."""
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list) -> None:
    """Load a list of numpy arrays back into the model, in state_dict order."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"expected {len(keys)} arrays, got {len(weights)}")
    state = OrderedDict(
        (key, torch.as_tensor(value)) for key, value in zip(keys, weights)
    )
    model.load_state_dict(state, strict=True)


def save_checkpoint(
    model: nn.Module,
    path,
    extra: Optional[dict] = None,
) -> None:
    """Write model weights plus whatever run state the caller needs to resume."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path, map_location="cpu") -> dict:
    """Restore weights from a checkpoint, returning the rest of its contents."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return {k: v for k, v in payload.items() if k != "state_dict"}
