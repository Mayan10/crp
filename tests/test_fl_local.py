"""Tests for local client training, the FedProx term, and aggregation."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedxcrop.fl.local import (
    build_local_optimizer,
    communication_bytes,
    proximal_term,
    snapshot_parameters,
    train_local,
    weighted_average,
)


def tiny_model(in_features: int = 4, out_features: int = 3) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(in_features, 8), nn.ReLU(), nn.Linear(8, out_features))


def tiny_loader(n: int = 32, in_features: int = 4, classes: int = 3) -> DataLoader:
    torch.manual_seed(1)
    x = torch.randn(n, in_features)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_proximal_term_is_zero_at_the_global_weights():
    model = tiny_model()
    anchor = snapshot_parameters(model)
    assert proximal_term(model, anchor, mu=0.1).item() == pytest.approx(0.0)


def test_proximal_term_matches_the_closed_form():
    """(mu/2) * ||w - w_global||^2, computed by hand on a one parameter model."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    anchor = [torch.tensor([[1.0]])]

    # ||3 - 1||^2 = 4, so with mu = 0.5 the term is 0.25 * 4 = 1.0.
    assert proximal_term(model, anchor, mu=0.5).item() == pytest.approx(1.0)


def test_proximal_term_grows_with_distance_and_with_mu():
    model = nn.Linear(1, 1, bias=False)
    anchor = [torch.tensor([[0.0]])]

    with torch.no_grad():
        model.weight.fill_(1.0)
    near = proximal_term(model, anchor, mu=0.1).item()
    with torch.no_grad():
        model.weight.fill_(2.0)
    far = proximal_term(model, anchor, mu=0.1).item()
    stronger = proximal_term(model, anchor, mu=0.2).item()

    assert far > near
    assert far == pytest.approx(4 * near)  # quadratic in the distance
    assert stronger == pytest.approx(2 * far)  # linear in mu


def test_proximal_term_is_differentiable():
    model = nn.Linear(2, 1, bias=False)
    anchor = [torch.zeros(1, 2)]
    penalty = proximal_term(model, anchor, mu=1.0)
    penalty.backward()
    # d/dw of (1/2)||w||^2 is w.
    np.testing.assert_allclose(
        model.weight.grad.detach().numpy(), model.weight.detach().numpy(), rtol=1e-6
    )


def test_non_positive_mu_is_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="mu must be positive"):
        proximal_term(model, snapshot_parameters(model), mu=0.0)


def test_fedprox_keeps_weights_closer_to_the_global_model_than_fedavg():
    """The whole point of the proximal term, stated as a test."""
    device = torch.device("cpu")
    loader = tiny_loader()

    def drift(strategy: str, mu: float) -> float:
        torch.manual_seed(7)
        model = tiny_model()
        start = snapshot_parameters(model)
        train_local(
            model, loader, device, epochs=3, lr=0.5, strategy=strategy, mu=mu
        )
        end = snapshot_parameters(model)
        return sum(float(((a - b) ** 2).sum()) for a, b in zip(start, end))

    fedavg_drift = drift("fedavg", 0.0)
    fedprox_drift = drift("fedprox", 1.0)
    assert fedprox_drift < fedavg_drift


def test_a_larger_mu_constrains_the_update_more():
    device = torch.device("cpu")
    loader = tiny_loader()

    def drift(mu: float) -> float:
        torch.manual_seed(7)
        model = tiny_model()
        start = snapshot_parameters(model)
        train_local(model, loader, device, epochs=3, lr=0.5, strategy="fedprox", mu=mu)
        return sum(
            float(((a - b) ** 2).sum()) for a, b in zip(start, snapshot_parameters(model))
        )

    assert drift(1.0) > drift(5.0)


def test_train_local_reports_counts_that_match_the_data():
    model = tiny_model()
    loader = tiny_loader(n=32)
    stats = train_local(model, loader, torch.device("cpu"), epochs=2)
    assert stats["n_images"] == 32
    assert stats["n_examples"] == 64  # two passes over 32 images
    assert 0.0 <= stats["accuracy"] <= 1.0


def test_train_local_actually_changes_the_weights():
    torch.manual_seed(3)
    model = tiny_model()
    before = snapshot_parameters(model)
    train_local(model, tiny_loader(), torch.device("cpu"), epochs=1, lr=0.1)
    after = snapshot_parameters(model)
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))


def test_fedavg_records_no_proximal_loss():
    model = tiny_model()
    stats = train_local(model, tiny_loader(), torch.device("cpu"), strategy="fedavg")
    assert stats["proximal_loss"] == 0.0


def test_unknown_strategy_and_optimizer_are_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="unknown strategy"):
        train_local(model, tiny_loader(), torch.device("cpu"), strategy="fedopt")
    with pytest.raises(ValueError, match="unsupported local optimizer"):
        build_local_optimizer(model, "rmsprop")


def test_optimizer_is_fresh_each_call():
    """Momentum must not leak across rounds."""
    model = tiny_model()
    first = build_local_optimizer(model, "sgd", lr=0.01)
    second = build_local_optimizer(model, "sgd", lr=0.01)
    assert first is not second
    assert all(not state for state in second.state.values()) or not second.state


def test_weighted_average_of_identical_clients_is_unchanged():
    weights = [np.ones((2, 2)), np.zeros(3)]
    result = weighted_average([weights, weights], [10, 10])
    for original, averaged in zip(weights, result):
        np.testing.assert_allclose(original, averaged)


def test_weighted_average_matches_a_hand_computation():
    a = [np.array([0.0, 0.0])]
    b = [np.array([1.0, 4.0])]
    # 25 percent of the data on client a, 75 percent on client b.
    result = weighted_average([a, b], [25, 75])
    np.testing.assert_allclose(result[0], [0.75, 3.0])


def test_weighted_average_favours_the_larger_client():
    a = [np.array([0.0])]
    b = [np.array([10.0])]
    small = weighted_average([a, b], [1, 99])[0][0]
    large = weighted_average([a, b], [99, 1])[0][0]
    assert small > 9.0
    assert large < 1.0


def test_weighted_average_input_validation():
    a = [np.ones(2)]
    with pytest.raises(ValueError, match="but 1 sizes"):
        weighted_average([a, a], [1])
    with pytest.raises(ValueError, match="zero clients"):
        weighted_average([], [])
    with pytest.raises(ValueError, match="must be positive"):
        weighted_average([a], [0])
    with pytest.raises(ValueError, match="same number of arrays"):
        weighted_average([a, [np.ones(2), np.ones(2)]], [1, 1])


def test_weighted_average_preserves_dtype():
    a = [np.ones(3, dtype=np.float32)]
    assert weighted_average([a, a], [1, 1])[0].dtype == np.float32


def test_communication_bytes_accounting():
    # 2.27 M parameters, 4 bytes each, 5 clients, down and up.
    assert communication_bytes(1_000, 5) == 1_000 * 4 * 5 * 2
    assert communication_bytes(2_272_550, 5) == 90_902_000
