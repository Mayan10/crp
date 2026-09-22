"""Tests for attribution methods and XAI metrics.

Mostly on synthetic maps with answers that can be worked out by hand, so a
failure points at the metric rather than at the model.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fedxcrop.xai.attribution import normalize_map
from fedxcrop.xai.metrics import (
    deletion_insertion_auc,
    energy_ratio,
    mask_area_fraction,
    pointing_game,
    spearman_agreement,
    topk_iou,
)
from fedxcrop.xai.sanity import cascading_randomization_stages, randomize_module


def corner_map(size: int = 8) -> np.ndarray:
    """A map whose mass sits entirely in the top left quadrant."""
    attribution = np.zeros((size, size), dtype=np.float32)
    attribution[: size // 2, : size // 2] = 1.0
    return attribution


def corner_mask(size: int = 8) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[: size // 2, : size // 2] = True
    return mask


def test_normalize_map_scales_to_unit_range():
    attribution = np.array([[2.0, 4.0], [6.0, 10.0]])
    normalized = normalize_map(attribution)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[0, 1] == pytest.approx((4 - 2) / (10 - 2))


def test_normalize_map_handles_a_constant_map():
    """A method that produced no signal must not divide by zero."""
    normalized = normalize_map(np.full((4, 4), 3.0))
    assert np.all(normalized == 0.0)


def test_energy_ratio_is_one_when_all_mass_is_inside_the_mask():
    assert energy_ratio(corner_map(), corner_mask()) == pytest.approx(1.0)


def test_energy_ratio_is_zero_when_no_mass_is_inside_the_mask():
    assert energy_ratio(corner_map(), ~corner_mask()) == pytest.approx(0.0)


def test_energy_ratio_matches_a_hand_computation():
    attribution = np.array([[1.0, 3.0], [0.0, 0.0]])
    mask = np.array([[True, False], [False, False]])
    assert energy_ratio(attribution, mask) == pytest.approx(0.25)


def test_uniform_attribution_scores_the_mask_area_fraction():
    """The chance level: a map that highlights nothing in particular."""
    attribution = np.ones((8, 8))
    mask = corner_mask()
    assert energy_ratio(attribution, mask) == pytest.approx(mask_area_fraction(mask))
    assert mask_area_fraction(mask) == pytest.approx(0.25)


def test_energy_ratio_of_an_empty_attribution_is_zero():
    assert energy_ratio(np.zeros((4, 4)), np.ones((4, 4), dtype=bool)) == 0.0


def test_pointing_game_hits_and_misses():
    attribution = np.zeros((8, 8))
    attribution[1, 1] = 1.0
    assert pointing_game(attribution, corner_mask()) is True

    attribution = np.zeros((8, 8))
    attribution[7, 7] = 1.0
    assert pointing_game(attribution, corner_mask()) is False


def test_pointing_game_rejects_an_empty_mask():
    with pytest.raises(ValueError, match="empty mask"):
        pointing_game(np.ones((4, 4)), np.zeros((4, 4), dtype=bool))


def test_spearman_agreement_of_a_map_with_itself_is_one():
    rng = np.random.default_rng(0)
    attribution = rng.random((16, 16))
    assert spearman_agreement(attribution, attribution) == pytest.approx(1.0)


def test_spearman_agreement_is_rank_based_not_scale_based():
    rng = np.random.default_rng(1)
    attribution = rng.random((16, 16))
    # A monotone rescaling keeps every pixel's rank, so agreement stays perfect.
    assert spearman_agreement(attribution, attribution * 5 + 2) == pytest.approx(1.0)


def test_spearman_agreement_of_a_reversed_map_is_minus_one():
    rng = np.random.default_rng(2)
    attribution = rng.random((16, 16))
    assert spearman_agreement(attribution, -attribution) == pytest.approx(-1.0)


def test_spearman_agreement_is_nan_for_a_constant_map():
    assert np.isnan(spearman_agreement(np.ones((8, 8)), np.random.random((8, 8))))


def test_topk_iou_of_identical_maps_is_one():
    rng = np.random.default_rng(3)
    attribution = rng.random((10, 10))
    assert topk_iou(attribution, attribution, 0.2) == pytest.approx(1.0)


def test_topk_iou_of_disjoint_regions_is_zero():
    a = np.zeros((10, 10))
    a[:2, :] = 1.0  # top 20 pixels
    b = np.zeros((10, 10))
    b[8:, :] = 1.0  # bottom 20 pixels
    assert topk_iou(a, b, 0.2) == pytest.approx(0.0)


def test_topk_iou_matches_a_hand_computation():
    a = np.array([4.0, 3.0, 2.0, 1.0])
    b = np.array([4.0, 1.0, 2.0, 3.0])
    # Top half of a is {0, 1}, of b is {0, 3}. Intersection 1, union 3.
    assert topk_iou(a, b, 0.5) == pytest.approx(1 / 3)


def test_topk_iou_rejects_an_invalid_fraction():
    a = np.ones((4, 4))
    with pytest.raises(ValueError, match="fraction must be"):
        topk_iou(a, a, 0.0)
    with pytest.raises(ValueError, match="fraction must be"):
        topk_iou(a, a, 1.5)


def test_metrics_reject_shape_mismatches():
    a, b = np.ones((4, 4)), np.ones((5, 5))
    for fn in (energy_ratio, pointing_game, spearman_agreement, topk_iou):
        with pytest.raises(ValueError, match="mismatch"):
            fn(a, b)


class CornerModel(nn.Module):
    """Scores class 1 by the average brightness of the top left corner only.

    The model reads a known set of pixels, so a map pointing at that corner is
    faithful by construction and a map pointing anywhere else is not. The
    scaling keeps the softmax away from saturation, where every curve would be
    flat at 1.0 and the two maps would be indistinguishable.
    """

    def forward(self, x):
        corner = x[:, :, :8, :8].mean(dim=(1, 2, 3))
        return torch.stack([torch.zeros_like(corner), corner * 4.0], dim=1)


def corner_image() -> torch.Tensor:
    """Bright only where the model looks, so the deletion baseline differs."""
    image = torch.zeros(3, 32, 32)
    image[:, :8, :8] = 1.0
    return image


def test_deletion_and_insertion_behave_in_the_expected_direction():
    """A map pointing at the pixels the model uses beats one pointing away."""
    model = CornerModel()
    device = torch.device("cpu")
    image = corner_image()

    informative = np.zeros((32, 32), dtype=np.float32)
    informative[:8, :8] = 1.0  # exactly the pixels the model reads
    misleading = 1.0 - informative

    good = deletion_insertion_auc(model, image, informative, 1, device, step=0.1, blur_sigma=2.0)
    bad = deletion_insertion_auc(model, image, misleading, 1, device, step=0.1, blur_sigma=2.0)

    # A faithful map destroys the prediction sooner: lower deletion AUC.
    assert good["deletion_auc"] < bad["deletion_auc"]
    # And restores it sooner: higher insertion AUC.
    assert good["insertion_auc"] > bad["insertion_auc"]


def test_deletion_curve_falls_when_the_important_pixels_go_first():
    model = CornerModel()
    result = deletion_insertion_auc(
        model, corner_image(),
        np.pad(np.ones((8, 8), dtype=np.float32), ((0, 24), (0, 24))),
        1, torch.device("cpu"), step=0.1, blur_sigma=2.0,
    )
    curve = result["deletion_curve"]
    assert curve[0] > curve[-1]


def test_deletion_curve_is_sampled_at_the_requested_step():
    model = CornerModel()
    result = deletion_insertion_auc(
        model, corner_image(), np.random.random((32, 32)), 1,
        torch.device("cpu"), step=0.1, blur_sigma=2.0,
    )
    assert len(result["fractions"]) == 11  # 0.0 to 1.0 inclusive
    assert result["fractions"][0] == 0.0
    assert result["fractions"][-1] == pytest.approx(1.0)
    assert len(result["deletion_curve"]) == len(result["fractions"])


def test_randomization_stages_start_at_the_classifier_and_reach_the_input():
    from fedxcrop.models.factory import build_model

    model = build_model(num_classes=5, pretrained=False)
    stages = cascading_randomization_stages(model)
    names = [name for name, _ in stages]
    assert names[0] == "classifier"
    assert len(names) > 2
    # The last stage reaches the earliest feature block.
    assert names[-1].startswith("features[0:")


def test_randomizing_a_module_changes_its_weights():
    layer = nn.Conv2d(3, 4, 3)
    before = layer.weight.detach().clone()
    randomize_module(layer)
    assert not torch.allclose(before, layer.weight)


def test_metric_distribution_drops_models_with_no_data_for_the_metric(tmp_path):
    """Labels and boxes must stay aligned when a metric is undefined somewhere.

    Agreement with the centralized model does not exist for the centralized
    model itself, so that model has to drop out of the axis labels as well as
    out of the data.
    """
    import pandas as pd

    from fedxcrop.viz.xai_figures import metric_distribution

    rows = []
    for model, rho in (("centralized", None), ("fedavg", 0.7), ("fedprox", 0.8)):
        for i in range(20):
            rows.append({
                "path": f"img_{i}.JPG", "model": model, "method": "gradcam",
                "spearman_vs_centralized": rho,
            })
    frame = pd.DataFrame(rows)

    paths = metric_distribution(frame, "spearman_vs_centralized", tmp_path, "demo")
    assert all(p.is_file() for p in paths)


def test_metric_distribution_survives_a_metric_with_no_data_at_all(tmp_path):
    import numpy as np
    import pandas as pd

    from fedxcrop.viz.xai_figures import metric_distribution

    frame = pd.DataFrame({
        "path": ["a.JPG"], "model": ["centralized"], "method": ["gradcam"],
        "spearman_vs_centralized": [np.nan],
    })
    paths = metric_distribution(frame, "spearman_vs_centralized", tmp_path, "empty")
    assert all(p.is_file() for p in paths)
