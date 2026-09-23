"""Local client training, the part of federated learning that is not plumbing.

Kept free of any framework so it can be unit tested directly and reused
whichever way the rounds are orchestrated. The proximal term that separates
FedProx from FedAvg lives here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def proximal_term(model: nn.Module, global_weights: list[torch.Tensor], mu: float) -> torch.Tensor:
    """The FedProx penalty (mu/2) * ||w - w_global||^2.

    This pulls each client's update back toward the weights the server sent at
    the start of the round. Under label skew, a client whose data covers a few
    classes will otherwise drift toward a solution that is good locally and
    harmful once averaged, which is the failure FedProx is designed to limit.

    Only trainable parameters are penalized. Batch norm running statistics are
    buffers, not parameters, and carry no gradient.
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive for the proximal term, got {mu}")

    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter, reference in zip(model.parameters(), global_weights):
        total = total + torch.sum((parameter - reference.to(parameter.device)) ** 2)
    return (mu / 2.0) * total


def snapshot_parameters(model: nn.Module) -> list[torch.Tensor]:
    """Detached copy of the trainable parameters, the anchor for FedProx."""
    return [p.detach().clone() for p in model.parameters()]


def build_local_optimizer(
    model: nn.Module,
    name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
):
    """Fresh optimizer for one round.

    Re-created every round on purpose: carrying SGD momentum across rounds
    would mix in the direction of an update computed against a different global
    model, which is not what either FedAvg or FedProx specifies.
    """
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported local optimizer {name!r}, expected 'sgd' or 'adam'")


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    optimizer_name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    strategy: str = "fedavg",
    mu: float = 0.0,
    augment=None,
) -> dict:
    """Train one client for `epochs` local epochs on its own shard.

    For FedProx the anchor is taken once, before any local step, so every local
    update in the round is measured against the weights the server actually
    sent.
    """
    model.to(device)
    model.train()

    anchor: Optional[list[torch.Tensor]] = None
    if strategy == "fedprox":
        anchor = snapshot_parameters(model)
    elif strategy != "fedavg":
        raise ValueError(f"unknown strategy {strategy!r}, expected 'fedavg' or 'fedprox'")

    optimizer = build_local_optimizer(model, optimizer_name, lr, momentum, weight_decay)
    criterion = nn.CrossEntropyLoss()

    loss_total, proximal_total, n_correct, n_seen = 0.0, 0.0, 0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if augment is not None:
                images = augment(images)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            if anchor is not None:
                penalty = proximal_term(model, anchor, mu)
                proximal_total += float(penalty.detach().item()) * labels.numel()
                loss = loss + penalty

            loss.backward()
            optimizer.step()

            loss_total += float(loss.detach().item()) * labels.numel()
            n_correct += int((logits.argmax(dim=1) == labels).sum().item())
            n_seen += labels.numel()

    return {
        "loss": loss_total / max(n_seen, 1),
        "proximal_loss": proximal_total / max(n_seen, 1),
        "accuracy": n_correct / max(n_seen, 1),
        "n_examples": n_seen,
        "n_images": len(loader.dataset),
    }


def weighted_average(weight_lists: list[list], sizes: list[int]) -> list:
    """FedAvg aggregation: average client weights in proportion to shard size.

    Applied to the whole state dict, batch norm running statistics included, so
    the aggregated model does not silently keep one client's local statistics.
    """
    import numpy as np

    if len(weight_lists) != len(sizes):
        raise ValueError(f"got {len(weight_lists)} clients but {len(sizes)} sizes")
    if not weight_lists:
        raise ValueError("cannot aggregate zero clients")
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("total number of examples must be positive")

    n_arrays = len(weight_lists[0])
    if any(len(w) != n_arrays for w in weight_lists):
        raise ValueError("all clients must send the same number of arrays")

    aggregated = []
    for i in range(n_arrays):
        stacked = np.stack([np.asarray(w[i], dtype=np.float64) for w in weight_lists])
        weights = np.asarray(sizes, dtype=np.float64) / total
        averaged = np.tensordot(weights, stacked, axes=(0, 0))
        aggregated.append(averaged.astype(np.asarray(weight_lists[0][i]).dtype))
    return aggregated


def communication_bytes(num_parameters: int, num_clients: int, bytes_per_parameter: int = 4) -> int:
    """Bytes moved in one round: every client downloads and uploads the model."""
    return num_parameters * bytes_per_parameter * num_clients * 2
