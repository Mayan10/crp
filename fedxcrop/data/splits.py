"""One fixed stratified train/val/test split, built once and committed.

Every experiment in this repo reads these CSVs. Nothing downstream is allowed
to resample them, so centralized and federated models are always compared on
exactly the same images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def stratified_split(
    index: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split per class so each class keeps its proportions in all three parts.

    Within a class the images are shuffled with a seeded generator and cut at
    rounded boundaries, which keeps every image in exactly one split and leaves
    each split within one image of its exact target size per class.
    """
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in SPLIT_NAMES}

    for _, group in index.groupby("label", sort=True):
        group = group.sort_values("path").reset_index(drop=True)
        order = rng.permutation(len(group))
        shuffled = group.iloc[order].reset_index(drop=True)

        n = len(shuffled)
        n_train = min(int(round(n * train_fraction)), n)
        n_val = min(int(round(n * val_fraction)), n - n_train)
        # The test split takes the remainder, so no image is lost to rounding.
        parts["train"].append(shuffled.iloc[:n_train])
        parts["val"].append(shuffled.iloc[n_train : n_train + n_val])
        parts["test"].append(shuffled.iloc[n_train + n_val :])

    splits = {}
    for name in SPLIT_NAMES:
        frame = pd.concat(parts[name], ignore_index=True)
        splits[name] = frame.sort_values(["label", "path"]).reset_index(drop=True)
    return splits


def save_splits(splits: dict[str, pd.DataFrame], splits_dir: str | Path) -> dict[str, Path]:
    """Write train.csv, val.csv and test.csv as (path, label).

    Class names are written once to classes.json rather than repeated on every
    row, which keeps the committed split files small and makes the label
    ordering explicit in a single place.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    names = (
        pd.concat(splits.values())[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
    )
    with open(splits_dir / "classes.json", "w") as handle:
        json.dump(list(names["class_name"]), handle, indent=2)

    written = {}
    for name, frame in splits.items():
        path = splits_dir / f"{name}.csv"
        frame[["path", "label"]].to_csv(path, index=False)
        written[name] = path
    return written


def load_classes(splits_dir: str | Path) -> list[str]:
    """Class names in label order."""
    path = Path(splits_dir) / "classes.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"class list not found: {path}. Run scripts/build_splits.py first."
        )
    with open(path) as handle:
        return json.load(handle)


def attach_class_names(frame: pd.DataFrame, splits_dir: str | Path) -> pd.DataFrame:
    """Add a class_name column derived from the label ordering in classes.json."""
    classes = load_classes(splits_dir)
    out = frame.copy()
    out["class_name"] = [classes[int(i)] for i in out["label"]]
    return out


def load_split(
    splits_dir: str | Path, name: str, with_class_names: bool = True
) -> pd.DataFrame:
    """Read one split CSV, attaching class names from classes.json by default."""
    path = Path(splits_dir) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"split not found: {path}. Run scripts/build_splits.py first.")
    frame = pd.read_csv(path)
    return attach_class_names(frame, splits_dir) if with_class_names else frame


def load_splits(splits_dir: str | Path, with_class_names: bool = True) -> dict[str, pd.DataFrame]:
    return {name: load_split(splits_dir, name, with_class_names) for name in SPLIT_NAMES}


def split_summary(splits: dict[str, pd.DataFrame]) -> dict:
    """Sizes and per class counts per split, for the data report."""
    sizes = {name: int(len(frame)) for name, frame in splits.items()}
    total = sum(sizes.values())
    per_class = {
        name: {
            str(row.class_name): int(row.count)
            for row in frame.groupby(["label", "class_name"], sort=True)
            .size()
            .reset_index(name="count")
            .itertuples()
        }
        for name, frame in splits.items()
    }
    return {
        "sizes": sizes,
        "total": total,
        "fractions": {name: size / total for name, size in sizes.items()},
        "per_class": per_class,
    }
