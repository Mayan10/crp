"""Tests for accelerator side colour augmentation.

The point of this module is to do exactly what torchvision's ColorJitter does,
only on whole batches and with per image factors, so most of these tests check
it numerically against torchvision rather than against expectations.
"""

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

from fedxcrop.data.gpu_augment import (
    BatchColorJitter,
    GpuAugment,
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    hsv_to_rgb,
    rgb_to_hsv,
)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(6, 3, 32, 32)


def per_image(function, batch, factor):
    """Apply a torchvision functional to each image, for comparison."""
    return torch.stack([function(batch[i], factor) for i in range(len(batch))])


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 1.5])
def test_brightness_matches_torchvision(batch, factor):
    got = adjust_brightness(batch, torch.full((len(batch), 1, 1, 1), factor))
    torch.testing.assert_close(got, per_image(TF.adjust_brightness, batch, factor))


@pytest.mark.parametrize("factor", [0.0, 0.6, 1.0, 1.4])
def test_contrast_matches_torchvision(batch, factor):
    got = adjust_contrast(batch, torch.full((len(batch), 1, 1, 1), factor))
    torch.testing.assert_close(got, per_image(TF.adjust_contrast, batch, factor), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("factor", [0.0, 0.7, 1.0, 1.3])
def test_saturation_matches_torchvision(batch, factor):
    got = adjust_saturation(batch, torch.full((len(batch), 1, 1, 1), factor))
    torch.testing.assert_close(got, per_image(TF.adjust_saturation, batch, factor), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("factor", [-0.5, -0.1, 0.0, 0.05, 0.1, 0.5])
def test_hue_matches_torchvision(batch, factor):
    got = adjust_hue(batch, torch.full((len(batch),), factor))
    torch.testing.assert_close(got, per_image(TF.adjust_hue, batch, factor), atol=2e-6, rtol=1e-4)


def test_hsv_round_trip(batch):
    torch.testing.assert_close(hsv_to_rgb(rgb_to_hsv(batch)), batch, atol=1e-5, rtol=1e-4)


def test_hsv_handles_grey_and_black(batch):
    """Zero saturation and zero value are the two places the maths divides by zero."""
    grey = torch.full((2, 3, 4, 4), 0.5)
    black = torch.zeros(2, 3, 4, 4)
    for image in (grey, black):
        hsv = rgb_to_hsv(image)
        assert torch.isfinite(hsv).all()
        torch.testing.assert_close(hsv_to_rgb(hsv), image, atol=1e-6, rtol=1e-5)


def test_factors_are_drawn_per_image():
    """Two identical images in one batch must not come out identical.

    This is the property that makes the batched jitter equivalent in spirit to
    applying torchvision per image, rather than a cheaper per batch jitter.
    """
    jitter = BatchColorJitter(0.4, 0.4, 0.3, 0.1)
    repeated = torch.rand(1, 3, 16, 16).repeat(8, 1, 1, 1)
    generator = torch.Generator()
    generator.manual_seed(0)

    out = jitter(repeated, generator)
    differences = [
        float((out[i] - out[j]).abs().max()) for i in range(8) for j in range(i + 1, 8)
    ]
    assert min(differences) > 1e-4


def test_zero_strength_is_a_no_op(batch):
    jitter = BatchColorJitter(0.0, 0.0, 0.0, 0.0)
    torch.testing.assert_close(jitter(batch), batch)


def test_output_stays_in_range():
    jitter = BatchColorJitter(0.9, 0.9, 0.9, 0.5)
    generator = torch.Generator()
    generator.manual_seed(3)
    out = jitter(torch.rand(16, 3, 16, 16), generator)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_same_seed_gives_the_same_augmentation():
    jitter = BatchColorJitter(0.4, 0.4, 0.3, 0.1)
    image = torch.rand(4, 3, 16, 16)

    def run(seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return jitter(image, generator)

    torch.testing.assert_close(run(11), run(11))
    assert float((run(11) - run(12)).abs().max()) > 1e-4


def test_distribution_matches_torchvision_color_jitter():
    """Per image statistics should land in the same place as torchvision's.

    Not an exact comparison: the two draw different random numbers. What must
    agree is the distribution the model actually trains on.
    """
    torch.manual_seed(1)
    images = torch.rand(400, 3, 24, 24)

    generator = torch.Generator()
    generator.manual_seed(0)
    mine = BatchColorJitter(0.4, 0.4, 0.3, 0.1)(images, generator)

    reference_transform = transforms.ColorJitter(0.4, 0.4, 0.3, 0.1)
    torch.manual_seed(2)
    theirs = torch.stack([reference_transform(images[i]) for i in range(len(images))])

    assert abs(float(mine.mean()) - float(theirs.mean())) < 0.02
    assert abs(float(mine.std()) - float(theirs.std())) < 0.02
    # The spread of per image contrast, which is what the jitter is varying.
    mine_spread = float(mine.std(dim=(1, 2, 3)).std())
    theirs_spread = float(theirs.std(dim=(1, 2, 3)).std())
    assert abs(mine_spread - theirs_spread) < 0.02


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        BatchColorJitter(brightness=-0.1)
    with pytest.raises(ValueError, match=r"hue must be in \[0, 0.5\]"):
        BatchColorJitter(hue=0.9)


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match=r"expected a \(B, 3, H, W\) batch"):
        BatchColorJitter(0.4)(torch.rand(3, 32, 32))
    with pytest.raises(ValueError, match=r"expected a \(B, 3, H, W\) batch"):
        BatchColorJitter(0.4)(torch.rand(2, 1, 32, 32))


def test_gpu_augment_converts_uint8_and_normalizes():
    """The full loader-to-model step: uint8 in, normalized float out."""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    augment = GpuAugment(mean, std, seed=0)  # no jitter, so this is just normalize

    raw = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    out = augment(raw)

    assert out.dtype == torch.float32
    expected = TF.normalize(raw.float() / 255.0, mean, std)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_gpu_augment_is_reproducible_from_a_seed():
    raw = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    a = GpuAugment((0.5,) * 3, (0.5,) * 3, 0.4, 0.4, 0.3, 0.1, seed=7)(raw)
    b = GpuAugment((0.5,) * 3, (0.5,) * 3, 0.4, 0.4, 0.3, 0.1, seed=7)(raw)
    torch.testing.assert_close(a, b)


def test_make_train_augment_respects_the_config_flag():
    from fedxcrop.config import load_config
    from fedxcrop.data.gpu_augment import make_train_augment

    on = load_config("configs/base.yaml", ["data.gpu_augment=true"])
    off = load_config("configs/base.yaml", ["data.gpu_augment=false"])
    assert make_train_augment(on, torch.device("cpu")) is not None
    assert make_train_augment(off, torch.device("cpu")) is None


def test_geometric_transform_yields_uint8_of_the_right_shape():
    """The loader side must hand over uint8 for the accelerator to finish."""
    from PIL import Image

    from fedxcrop.data.transforms import train_transform, train_transform_geometric

    image = Image.fromarray(
        (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    )

    geometric = train_transform_geometric(224)(image)
    assert geometric.dtype == torch.uint8
    assert geometric.shape == (3, 224, 224)

    # The CPU path still produces normalized floats.
    full = train_transform(224)(image)
    assert full.dtype == torch.float32
    assert full.shape == (3, 224, 224)


def test_both_paths_produce_the_same_distribution_end_to_end():
    """CPU augmentation and geometry-plus-GPU augmentation must agree.

    The two draw different random numbers, so this compares the statistics of
    what the model would actually be fed, which is what has to match for the
    results from the two paths to be comparable.
    """
    from PIL import Image

    from fedxcrop.data.transforms import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        train_transform,
        train_transform_geometric,
    )

    rng = np.random.default_rng(0)
    images = [
        Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8))
        for _ in range(64)
    ]

    torch.manual_seed(0)
    cpu_path = torch.stack([train_transform(224)(im) for im in images])

    torch.manual_seed(0)
    geometric = torch.stack([train_transform_geometric(224)(im) for im in images])
    gpu_path = GpuAugment(IMAGENET_MEAN, IMAGENET_STD, 0.4, 0.4, 0.3, 0.1, seed=0)(geometric)

    assert gpu_path.shape == cpu_path.shape
    assert abs(float(gpu_path.mean()) - float(cpu_path.mean())) < 0.1
    assert abs(float(gpu_path.std()) - float(cpu_path.std())) < 0.1
