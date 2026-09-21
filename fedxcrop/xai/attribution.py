"""Attribution methods.

Two methods, chosen because they answer different questions. Grad-CAM is a
coarse, class discriminative map built from the last convolutional feature map.
SmoothGrad is a fine grained input gradient method that averages away the
noise a raw saliency map suffers from. Both are computed on the predicted
class, not the true class, because the question is what the model used to reach
the answer it actually gave.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerGradCam, NoiseTunnel, Saliency

GRADCAM_LAYER = "features.-1"
SMOOTHGRAD_SAMPLES = 20
SMOOTHGRAD_NOISE_FRACTION = 0.1


def target_layer(model: nn.Module) -> nn.Module:
    """The last convolutional block of MobileNetV2.

    This is the deepest layer that still has spatial extent (7x7 for a 224x224
    input), which is the standard choice for Grad-CAM: deep enough to be class
    specific, shallow enough to localize.
    """
    return model.features[-1]


def normalize_map(attribution: np.ndarray) -> np.ndarray:
    """Scale a map to [0, 1].

    A constant map (which happens when a method produces no signal at all)
    normalizes to zeros rather than dividing by zero.
    """
    low, high = float(attribution.min()), float(attribution.max())
    if high - low < 1e-12:
        return np.zeros_like(attribution, dtype=np.float32)
    return ((attribution - low) / (high - low)).astype(np.float32)


def gradcam(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    image_size: int = 224,
) -> np.ndarray:
    """Grad-CAM on the last convolutional block, upsampled to the input size.

    Captum's LayerGradCam returns a single channel map at feature resolution.
    It is upsampled bilinearly to the input size and normalized per image, so
    maps from different models are on a common scale and can be compared.

    ReLU is applied (relu_attributions), keeping only evidence for the target
    class rather than against it, which is what the original formulation
    specifies.
    """
    model.eval()
    explainer = LayerGradCam(model, target_layer(model))
    attribution = explainer.attribute(images, target=targets, relu_attributions=True)
    upsampled = F.interpolate(
        attribution, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    maps = upsampled.squeeze(1).detach().cpu().numpy()
    return np.stack([normalize_map(m) for m in maps])


def smoothgrad(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int = SMOOTHGRAD_SAMPLES,
    noise_fraction: float = SMOOTHGRAD_NOISE_FRACTION,
) -> np.ndarray:
    """SmoothGrad over input gradients.

    Noise convention, stated explicitly because the original write up did not:
    sigma is `noise_fraction` times the range (max minus min) of the
    **normalized input tensor**, computed per image, not in [0, 1] pixel space.
    With ImageNet normalization the tensor range is roughly 4.4, so the default
    0.1 corresponds to a sigma of about 0.44 in tensor units.

    Absolute gradients are averaged over the colour channels, since the
    question is which pixels matter rather than in which direction each channel
    would have to move.
    """
    model.eval()
    explainer = NoiseTunnel(Saliency(model))

    maps = []
    for i in range(images.shape[0]):
        single = images[i : i + 1]
        spread = float(single.max() - single.min())
        sigma = noise_fraction * spread
        attribution = explainer.attribute(
            single,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            stdevs=sigma,
            target=targets[i : i + 1],
            abs=True,
        )
        channel_mean = attribution.abs().mean(dim=1).squeeze(0)
        maps.append(normalize_map(channel_mean.detach().cpu().numpy()))
    return np.stack(maps)


def attribute(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    image_size: int = 224,
    n_samples: int = SMOOTHGRAD_SAMPLES,
    noise_fraction: float = SMOOTHGRAD_NOISE_FRACTION,
) -> np.ndarray:
    """Dispatch to an attribution method, returning maps normalized to [0, 1]."""
    if method == "gradcam":
        return gradcam(model, images, targets, image_size)
    if method == "smoothgrad":
        return smoothgrad(model, images, targets, n_samples, noise_fraction)
    raise ValueError(f"unknown attribution method {method!r}, expected 'gradcam' or 'smoothgrad'")


@torch.no_grad()
def predicted_classes(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The class each image is actually assigned, which is what gets explained."""
    model.eval()
    return model(images).argmax(dim=1)
