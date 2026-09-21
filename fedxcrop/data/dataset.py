"""Torch datasets backed by the committed split CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fedxcrop.data.splits import load_split
from fedxcrop.data.transforms import eval_transform, train_transform


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
) -> PlantVillageDataset:
    transform = train_transform(image_size) if train else eval_transform(image_size)
    return PlantVillageDataset(root, frame, transform=transform, return_index=return_index)


def build_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with a seeded generator, so shuffling is reproducible."""
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
        persistent_workers=num_workers > 0,
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
