"""Torch datasets backed by the committed split CSVs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fedxcrop.data.splits import load_split
from fedxcrop.data.transforms import (
    eval_transform,
    train_transform,
    train_transform_geometric,
)


class PlantVillageDataset(Dataset):
    """Images listed by a split or partition CSV.

    The frame must have `path` (relative to the dataset root) and `label`
    columns. Keeping the file list in a frame rather than rescanning the disk is
    what lets a federated client hold an exact, reproducible subset.
    """

    def __init__(
        self,
        root: str | Path,
        frame: pd.DataFrame,
        transform=None,
        return_index: bool = False,
    ):
        self.root = Path(root)
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.return_index = return_index
        missing = {"path", "label"} - set(self.frame.columns)
        if missing:
            raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        image = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        if self.return_index:
            return image, label, i
        return image, label

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(self.frame["label"].to_numpy(), dtype=torch.long)


def build_dataset(
    root: str | Path,
    frame: pd.DataFrame,
    train: bool,
    image_size: int = 224,
    return_index: bool = False,
    gpu_augment: bool = False,
) -> PlantVillageDataset:
    """Dataset for one split.

    With `gpu_augment`, a training dataset yields uint8 images carrying only
    the geometric augmentation; the colour jitter and the normalization are
    then applied to whole batches on the accelerator. Evaluation datasets are
    unaffected, since they are never augmented.
    """
    if train:
        transform = train_transform_geometric(image_size) if gpu_augment else train_transform(image_size)
    else:
        transform = eval_transform(image_size)
    return PlantVillageDataset(root, frame, transform=transform, return_index=return_index)


def available_cpus() -> int:
    """CPUs this process may actually use, not the machine's total."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on macOS
        return max(1, os.cpu_count() or 1)


def effective_workers(requested: int) -> int:
    """Loader workers to actually use, never more than there are CPUs."""
    if requested <= 0:
        return 0
    return max(1, min(requested, available_cpus()))


def build_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    seed: Optional[int] = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """DataLoader with a seeded generator, so shuffling is reproducible.

    Workers are not persistent by default. A federated round builds one loader
    per client, so persistent workers would hold K times `num_workers`
    processes alive at once (20 for five clients with four workers each),
    which exhausts memory on a modest machine long before it speeds anything
    up. Only long lived loaders, such as validation, ask for persistence.

    `num_workers` is clamped to the number of CPUs actually available. More
    loader processes than cores does not increase throughput on a pipeline
    this CPU bound, it just adds context switching, and a two core runtime
    asked for four workers is a realistic way to lose a third of the speed.
    """
    num_workers = effective_workers(num_workers)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def split_loader(
    root: str | Path,
    splits_dir: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    seed: Optional[int] = None,
) -> DataLoader:
    """Convenience loader for a named split, augmented only when it is train."""
    frame = load_split(splits_dir, split)
    is_train = split == "train"
    dataset = build_dataset(root, frame, train=is_train, image_size=image_size)
    return build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        seed=seed,
    )
