"""Tests for the model factory and weight exchange helpers."""

import numpy as np
import pytest
import torch

from fedxcrop.models.factory import (
    build_model,
    count_parameters,
    get_weights,
    load_checkpoint,
    save_checkpoint,
    set_weights,
)


@pytest.fixture(scope="module")
def model():
    return build_model(num_classes=38, pretrained=False)


def test_classifier_has_one_output_per_class(model):
    assert model.classifier[1].out_features == 38
    assert model.classifier[1].in_features == 1280


def test_forward_pass_shape(model):
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 38)


def test_whole_backbone_is_trainable(model):
    assert all(p.requires_grad for p in model.parameters())
    assert count_parameters(model) == count_parameters(model, trainable_only=True)


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError, match="unsupported model"):
        build_model("resnet50")


def test_weights_round_trip_exactly():
    a = build_model(num_classes=5, pretrained=False)
    b = build_model(num_classes=5, pretrained=False)

    weights = get_weights(a)
    set_weights(b, weights)
    for left, right in zip(get_weights(a), get_weights(b)):
        np.testing.assert_array_equal(left, right)


def test_set_weights_rejects_the_wrong_number_of_arrays():
    model = build_model(num_classes=5, pretrained=False)
    with pytest.raises(ValueError, match="expected"):
        set_weights(model, get_weights(model)[:-1])


def test_get_weights_covers_buffers_not_just_parameters():
    """Batch norm running statistics must travel with the weights.

    MobileNetV2 is full of batch norm layers. If only parameters were
    exchanged, the aggregated model would carry each client's local running
    statistics, which is a silent source of federated accuracy loss.
    """
    model = build_model(num_classes=5, pretrained=False)
    assert len(get_weights(model)) == len(model.state_dict())
    assert len(get_weights(model)) > len(list(model.parameters()))


def test_checkpoint_round_trip_preserves_weights_and_extras(tmp_path):
    a = build_model(num_classes=5, pretrained=False)
    path = tmp_path / "model.pt"
    save_checkpoint(a, path, extra={"round": 7, "val_accuracy": 0.99})

    b = build_model(num_classes=5, pretrained=False)
    extras = load_checkpoint(b, path)

    assert extras["round"] == 7
    assert extras["val_accuracy"] == pytest.approx(0.99)
    for left, right in zip(get_weights(a), get_weights(b)):
        np.testing.assert_array_equal(left, right)


def test_parameter_count_is_the_published_mobilenet_v2_size():
    model = build_model(num_classes=38, pretrained=False)
    # 2.22 M backbone parameters plus a 1280 to 38 classifier head.
    assert 2_200_000 < count_parameters(model) < 2_300_000
