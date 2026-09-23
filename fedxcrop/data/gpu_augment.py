"""Colour augmentation on the GPU, with per image random factors.

Profiling the input pipeline showed that `ColorJitter` accounts for roughly
three quarters of its cost: decoding runs at about 3,750 images per second
while the jitter alone manages 577. On a runtime with two CPU cores that
starves the GPU, which then sits idle waiting for batches.

Moving the jitter to the GPU, where there is spare capacity, leaves the CPU
doing only decode and geometry. The geometric transforms stay on the CPU
because they are cheap and because doing them per image keeps their randomness
per image.

What this reproduces, and what it does not:

  * Factors are drawn per image, exactly as torchvision does, so two images in
    a batch never receive the same jitter.
  * The four operations are applied in a random order drawn per batch rather
    than per image. torchvision randomises the order per call to avoid a fixed
    ordering bias; drawing it per batch avoids the same bias while staying
    vectorised.
  * Each operation matches `torchvision.transforms.functional` numerically,
    which `tests/test_gpu_augment.py` checks against directly.
"""

from __future__ import annotations

from typing import Optional

import torch

GRAYSCALE_WEIGHTS = (0.2989, 0.587, 0.114)


def _grayscale(batch: torch.Tensor) -> torch.Tensor:
    """Luminance of each image, keeping the channel dimension."""
    weights = torch.tensor(GRAYSCALE_WEIGHTS, dtype=batch.dtype, device=batch.device)
    return (batch * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def _blend(image: torch.Tensor, other: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """factor * image + (1 - factor) * other, clamped, as torchvision blends."""
    return (factor * image + (1.0 - factor) * other).clamp(0.0, 1.0)


def adjust_brightness(batch: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return (batch * factor).clamp(0.0, 1.0)


def adjust_contrast(batch: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Blend towards each image's own mean luminance."""
    mean = _grayscale(batch).mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(batch, mean, factor)


def adjust_saturation(batch: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Blend towards each image's greyscale version."""
    return _blend(batch, _grayscale(batch), factor)


def rgb_to_hsv(batch: torch.Tensor) -> torch.Tensor:
    """RGB in [0, 1] to HSV, with hue in [0, 1)."""
    r, g, b = batch.unbind(dim=-3)
    # amax/amin, not max/min: the latter also compute argmax indices, which are
    # not needed here and cost two orders of magnitude more on an accelerator.
    maximum = batch.amax(dim=-3)
    minimum = batch.amin(dim=-3)
    spread = maximum - minimum

    # Guard the divisions; the values where the guard bites are replaced below.
    safe_spread = torch.where(spread == 0, torch.ones_like(spread), spread)
    safe_max = torch.where(maximum == 0, torch.ones_like(maximum), maximum)

    rc, gc, bc = (maximum - r) / safe_spread, (maximum - g) / safe_spread, (maximum - b) / safe_spread
    hue_r = bc - gc
    hue_g = 2.0 + rc - bc
    hue_b = 4.0 + gc - rc

    hue = torch.where(maximum == r, hue_r, torch.where(maximum == g, hue_g, hue_b))
    hue = (hue / 6.0) % 1.0
    hue = torch.where(spread == 0, torch.zeros_like(hue), hue)

    saturation = torch.where(maximum == 0, torch.zeros_like(spread), spread / safe_max)
    return torch.stack((hue, saturation, maximum), dim=-3)


def hsv_to_rgb(batch: torch.Tensor) -> torch.Tensor:
    """HSV with hue in [0, 1) back to RGB in [0, 1].

    Branchless form: each output channel is one evaluation of the standard
    piecewise function. Selecting the sector with a one hot mask instead would
    materialise a tensor six times the size of the batch, which on a batch of
    64 at 224 pixels is hundreds of megabytes and made this the slowest step in
    the pipeline rather than the cheapest.
    """
    hue, saturation, value = batch.unbind(dim=-3)
    scaled = hue * 6.0

    def channel(n: float) -> torch.Tensor:
        k = (n + scaled) % 6.0
        return value - value * saturation * torch.clamp(
            torch.minimum(k, 4.0 - k), min=0.0, max=1.0
        )

    return torch.stack((channel(5.0), channel(3.0), channel(1.0)), dim=-3)


def adjust_hue(batch: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Rotate the hue channel, wrapping around the colour wheel."""
    hsv = rgb_to_hsv(batch)
    hue, saturation, value = hsv.unbind(dim=-3)
    hue = (hue + factor.view(-1, 1, 1)) % 1.0
    return hsv_to_rgb(torch.stack((hue, saturation, value), dim=-3)).clamp(0.0, 1.0)


class BatchColorJitter:
    """torchvision's ColorJitter, vectorised over a batch on the GPU.

    Factors are drawn per image. The order of the four operations is drawn once
    per batch, which is the one deliberate difference from torchvision and is
    documented at the top of this module.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        for name, value in (("brightness", brightness), ("contrast", contrast),
                            ("saturation", saturation)):
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
        if not 0 <= hue <= 0.5:
            raise ValueError(f"hue must be in [0, 0.5], got {hue}")

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _factors(self, low: float, high: float, n: int, device, dtype, generator) -> torch.Tensor:
        """Per image factors, drawn on the CPU then moved to the device.

        Drawing on the CPU keeps the augmentation reproducible from a seed
        regardless of which accelerator the run happens to use, and n is a
        handful of floats so the transfer costs nothing.
        """
        values = torch.rand(n, generator=generator)
        factors = (low + values * (high - low)).to(device=device, dtype=dtype)
        return factors.view(-1, 1, 1, 1)

    def __call__(
        self, batch: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Jitter a float batch in [0, 1] of shape (B, 3, H, W)."""
        if batch.dim() != 4 or batch.shape[1] != 3:
            raise ValueError(f"expected a (B, 3, H, W) batch, got {tuple(batch.shape)}")

        n, device, dtype = batch.shape[0], batch.device, batch.dtype
        operations = []

        if self.brightness > 0:
            factor = self._factors(max(0.0, 1 - self.brightness), 1 + self.brightness,
                                   n, device, dtype, generator)
            operations.append((adjust_brightness, factor))
        if self.contrast > 0:
            factor = self._factors(max(0.0, 1 - self.contrast), 1 + self.contrast,
                                   n, device, dtype, generator)
            operations.append((adjust_contrast, factor))
        if self.saturation > 0:
            factor = self._factors(max(0.0, 1 - self.saturation), 1 + self.saturation,
                                   n, device, dtype, generator)
            operations.append((adjust_saturation, factor))
        if self.hue > 0:
            factor = self._factors(-self.hue, self.hue, n, device, dtype, generator)
            operations.append((adjust_hue, factor.view(-1)))

        if not operations:
            return batch

        order = torch.randperm(len(operations), generator=generator, device="cpu").tolist()
        for index in order:
            function, factor = operations[index]
            batch = function(batch, factor)
        return batch


class GpuAugment:
    """Turn a uint8 batch from the loader into a normalized float batch.

    The loader hands over uint8 images that have already been cropped, flipped
    and rotated on the CPU. This scales them to [0, 1], applies the colour
    jitter, and normalizes, all on the accelerator.
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.jitter = BatchColorJitter(brightness, contrast, saturation, hue)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.generator = None
        if seed is not None:
            # A CPU generator, because the factors are drawn on the CPU so the
            # augmentation is reproducible from a seed on any device.
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype == torch.uint8:
            batch = batch.to(torch.float32).div_(255.0)
        batch = self.jitter(batch, self.generator)
        mean = self.mean.to(batch.device, batch.dtype)
        std = self.std.to(batch.device, batch.dtype)
        return (batch - mean) / std


def make_train_augment(cfg, device, seed_offset: int = 0) -> Optional["GpuAugment"]:
    """Build the accelerator side augmenter for a training loader, or None.

    Returns None when `data.gpu_augment` is off, in which case the loader is
    already producing fully augmented, normalized float batches and the trainer
    has nothing further to do.
    """
    if not getattr(cfg.data, "gpu_augment", False):
        return None

    from fedxcrop.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    return GpuAugment(
        IMAGENET_MEAN,
        IMAGENET_STD,
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.1,
        seed=cfg.seed + seed_offset,
        device=device,
    )
