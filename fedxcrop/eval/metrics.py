"""Classification metrics.

Every metrics file this module writes carries `n_correct` and `n_total`. That
is deliberate: a reported accuracy has to be checkable against a whole number
of correct predictions on a stated number of images. Rates that cannot be
reproduced from integer counts are how the earlier version of this work ended
up with numbers no evaluation set could have produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _as_array(values: Sequence) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"expected a 1 dimensional array, got shape {array.shape}")
    return array.astype(int)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true, {len(y_pred)} predicted")
    if len(y_true) == 0:
        raise ValueError("cannot compute accuracy on an empty set")
    return float((y_true == y_pred).sum() / len(y_true))


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    num_classes: int = 38,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Full metric set for one model on one evaluation split.

    Macro averaging weights every class equally, which is the honest choice on
    PlantVillage: the largest class is 36 times the size of the smallest, so a
    weighted average is dominated by a handful of classes.

    Classes absent from `y_true` contribute zero to the macro average, matching
    scikit-learn's `zero_division=0`, and are listed in `classes_absent` so the
    reader can tell the difference between a class the model failed on and a
    class that was not evaluated.
    """
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true, {len(y_pred)} predicted")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty set")

    labels = list(range(num_classes))
    n_correct = int((y_true == y_pred).sum())
    n_total = int(len(y_true))

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    )

    names = class_names if class_names is not None else [str(i) for i in labels]
    if len(names) != num_classes:
        raise ValueError(f"expected {num_classes} class names, got {len(names)}")

    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            names[i]: {
                "label": i,
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "classes_absent": [names[i] for i in labels if support[i] == 0],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_metrics(metrics: dict, path: str | Path) -> Path:
    """Write a metrics dictionary as JSON, refusing to write one without counts."""
    for required in ("n_correct", "n_total"):
        if required not in metrics:
            raise ValueError(f"metrics must include {required!r} so the rate is auditable")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    return path


def load_metrics(path: str | Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def worst_classes(metrics: dict, n: int = 10, by: str = "f1") -> list[dict]:
    """The n weakest classes by a per class metric, for the error analysis table."""
    rows = [
        {"class_name": name, **stats}
        for name, stats in metrics["per_class"].items()
        if stats["support"] > 0
    ]
    return sorted(rows, key=lambda row: row[by])[:n]
