"""Running a model over a split and collecting predictions.

Predictions are saved per image rather than only as summary rates. Paired tests
such as McNemar need to know which specific images each model got right, and
per image records are also what makes a reported accuracy auditable after the
fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_probabilities: bool = False,
) -> dict:
    """Predicted and true labels for every image in a loader, in loader order.

    The loader must not shuffle, otherwise the returned arrays cannot be lined
    up with the frame the dataset was built from.
    """
    model.eval()
    model.to(device)

    true_parts, pred_parts, prob_parts = [], [], []
    loss_total, n_seen = 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)

        loss_total += float(criterion(logits, labels).detach().item())
        n_seen += labels.numel()

        true_parts.append(labels.cpu().numpy())
        pred_parts.append(logits.argmax(dim=1).cpu().numpy())
        if return_probabilities:
            prob_parts.append(torch.softmax(logits, dim=1).cpu().numpy())

    out = {
        "y_true": np.concatenate(true_parts),
        "y_pred": np.concatenate(pred_parts),
        "loss": loss_total / max(n_seen, 1),
    }
    if return_probabilities:
        out["probabilities"] = np.concatenate(prob_parts)
    return out


def save_predictions(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
) -> Path:
    """Write one row per evaluated image: path, true label, predicted label.

    This file is what McNemar's test reads, and what lets anyone recount the
    reported accuracy without rerunning the model.
    """
    if not (len(frame) == len(y_true) == len(y_pred)):
        raise ValueError(
            f"length mismatch: frame {len(frame)}, true {len(y_true)}, predicted {len(y_pred)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "path": frame["path"].to_numpy(),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )
    out["correct"] = out["y_true"] == out["y_pred"]
    out.to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)
    return path


def load_predictions(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def accuracy_from_predictions(path: str | Path) -> dict:
    """Recount accuracy straight from a saved prediction file."""
    frame = load_predictions(path)
    n_correct = int((frame["y_true"] == frame["y_pred"]).sum())
    n_total = int(len(frame))
    return {"n_correct": n_correct, "n_total": n_total, "accuracy": n_correct / n_total}
