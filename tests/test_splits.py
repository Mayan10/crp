"""Tests for the dataset index and the fixed train/val/test split."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedxcrop.data.index import image_key, segmented_path_for
from fedxcrop.data.splits import (
    SPLIT_NAMES,
    load_classes,
    load_splits,
    split_summary,
    stratified_split,
)

SPLITS_DIR = Path("splits")
has_real_splits = (SPLITS_DIR / "train.csv").is_file()
needs_splits = pytest.mark.skipif(not has_real_splits, reason="run scripts/build_splits.py first")


def toy_index(per_class: dict[str, int]) -> pd.DataFrame:
    """Synthetic index with a known number of images per class."""
    rows = []
    for label, (name, count) in enumerate(sorted(per_class.items())):
        for i in range(count):
            rows.append({"path": f"color/{name}/img_{i:05d}.JPG", "label": label, "class_name": name})
    return pd.DataFrame(rows)


def test_split_is_a_partition_of_the_index():
    index = toy_index({"a": 100, "b": 250, "c": 33})
    splits = stratified_split(index, seed=42)

    total = sum(len(f) for f in splits.values())
    assert total == len(index)

    seen = pd.concat([f["path"] for f in splits.values()])
    assert seen.is_unique
    assert set(seen) == set(index["path"])


def test_splits_are_disjoint():
    index = toy_index({"a": 100, "b": 250, "c": 33})
    splits = stratified_split(index, seed=42)
    for left in SPLIT_NAMES:
        for right in SPLIT_NAMES:
            if left < right:
                assert not set(splits[left]["path"]) & set(splits[right]["path"])


def test_stratification_holds_within_one_image_per_class():
    per_class = {"a": 100, "b": 250, "c": 33, "d": 1000, "e": 7}
    index = toy_index(per_class)
    splits = stratified_split(index, 0.8, 0.1, 0.1, seed=42)

    for label, (name, count) in enumerate(sorted(per_class.items())):
        for split_name, fraction in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
            got = int((splits[split_name]["label"] == label).sum())
            assert abs(got - count * fraction) <= 1, (
                f"class {name} split {split_name}: got {got}, expected about {count * fraction}"
            )


def test_same_seed_gives_identical_splits_and_different_seed_does_not():
    index = toy_index({"a": 100, "b": 250})
    a = stratified_split(index, seed=42)
    b = stratified_split(index, seed=42)
    c = stratified_split(index, seed=7)
    for name in SPLIT_NAMES:
        pd.testing.assert_frame_equal(a[name], b[name])
    assert list(a["test"]["path"]) != list(c["test"]["path"])


def test_fractions_must_sum_to_one():
    index = toy_index({"a": 10})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        stratified_split(index, 0.8, 0.1, 0.2)


def test_every_class_appears_in_every_split():
    index = toy_index({"a": 100, "b": 250, "c": 33})
    splits = stratified_split(index, seed=42)
    for frame in splits.values():
        assert set(frame["label"]) == {0, 1, 2}


def test_segmented_path_follows_the_naming_rule():
    got = segmented_path_for("color/Apple___Apple_scab/abc___RS_1.JPG")
    assert got == "segmented/Apple___Apple_scab/abc___RS_1_final_masked.jpg"


def test_image_key_strips_uuid_prefix_and_mask_suffix():
    assert image_key("0012b9d2-2130___RS_Erly.B 8389") == "RS_Erly.B 8389"
    assert image_key("0012b9d2-2130___RS_Erly.B 8389_final_masked") == "RS_Erly.B 8389"
    assert image_key("RS_Rust 1563") == "RS_Rust 1563"


@needs_splits
def test_committed_splits_are_disjoint_and_complete():
    splits = load_splits(SPLITS_DIR)
    paths = pd.concat([f["path"] for f in splits.values()])
    assert paths.is_unique
    assert len(paths) == 54305


@needs_splits
def test_committed_splits_are_stratified():
    splits = load_splits(SPLITS_DIR)
    summary = split_summary(splits)
    per_class = summary["per_class"]
    for class_name, train_count in per_class["train"].items():
        total = train_count + per_class["val"][class_name] + per_class["test"][class_name]
        assert abs(train_count - 0.8 * total) <= 1
        assert abs(per_class["val"][class_name] - 0.1 * total) <= 1


@needs_splits
def test_committed_split_has_all_38_classes_and_matching_label_order():
    classes = load_classes(SPLITS_DIR)
    assert len(classes) == 38
    assert classes == sorted(classes)
    splits = load_splits(SPLITS_DIR)
    for frame in splits.values():
        assert set(frame["label"]) == set(range(38))
