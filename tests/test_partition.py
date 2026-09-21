"""Tests for federated partitioning of the train split."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedxcrop.data.partition import (
    build_partition,
    dirichlet_partition,
    iid_partition,
    jensen_shannon_divergence,
    label_histogram,
    load_partition,
    partition_filename,
    partition_summary,
)
from fedxcrop.data.splits import load_split

SPLITS_DIR = Path("splits")
has_real_splits = (SPLITS_DIR / "train.csv").is_file()
needs_splits = pytest.mark.skipif(not has_real_splits, reason="run scripts/build_splits.py first")


def toy_train(per_class: dict[str, int]) -> pd.DataFrame:
    rows = []
    for label, (name, count) in enumerate(sorted(per_class.items())):
        for i in range(count):
            rows.append({"path": f"color/{name}/img_{i:05d}.JPG", "label": label})
    return pd.DataFrame(rows)


PER_CLASS = {"a": 400, "b": 300, "c": 250, "d": 120, "e": 60}


def test_iid_partition_covers_train_exactly_once():
    train = toy_train(PER_CLASS)
    partition = iid_partition(train, num_clients=5, seed=0)
    assert len(partition) == len(train)
    assert partition["path"].is_unique
    assert set(partition["path"]) == set(train["path"])


def test_iid_client_sizes_are_balanced_within_one_image():
    train = toy_train(PER_CLASS)
    partition = iid_partition(train, num_clients=5, seed=0)
    sizes = partition.groupby("client_id").size()
    assert sizes.max() - sizes.min() <= 1


def test_iid_clients_match_the_global_label_distribution():
    train = toy_train(PER_CLASS)
    partition = iid_partition(train, num_clients=5, seed=0)
    global_histogram = label_histogram(partition, 5)
    for _, frame in partition.groupby("client_id"):
        divergence = jensen_shannon_divergence(label_histogram(frame, 5), global_histogram)
        assert divergence < 0.02


def test_dirichlet_partition_covers_train_exactly_once():
    train = toy_train(PER_CLASS)
    partition, _ = dirichlet_partition(train, num_clients=5, alpha=0.5, seed=0)
    assert len(partition) == len(train)
    assert partition["path"].is_unique
    assert set(partition["path"]) == set(train["path"])


def test_dirichlet_respects_the_minimum_client_size():
    train = toy_train(PER_CLASS)
    partition, _ = dirichlet_partition(
        train, num_clients=5, alpha=0.05, seed=3, min_samples_per_client=10
    )
    assert partition.groupby("client_id").size().min() >= 10


def test_every_client_id_is_present():
    train = toy_train(PER_CLASS)
    for scheme, kwargs in (("iid", {}), ("dirichlet", {"alpha": 0.5})):
        partition, _ = build_partition(train, scheme, num_clients=5, seed=0, **kwargs)
        assert set(partition["client_id"]) == set(range(5))


def test_smaller_alpha_gives_more_skew():
    train = toy_train(PER_CLASS)
    divergences = {}
    for alpha in (0.1, 1.0, 10.0):
        partition, _ = dirichlet_partition(train, num_clients=5, alpha=alpha, seed=0)
        summary = partition_summary(partition, 5, "dirichlet", 5, alpha, 0)
        divergences[alpha] = summary["jensen_shannon_divergence"]["mean"]
    assert divergences[0.1] > divergences[1.0] > divergences[10.0]


def test_same_seed_gives_identical_partitions():
    train = toy_train(PER_CLASS)
    for scheme, kwargs in (("iid", {}), ("dirichlet", {"alpha": 0.3})):
        a, _ = build_partition(train, scheme, num_clients=5, seed=11, **kwargs)
        b, _ = build_partition(train, scheme, num_clients=5, seed=11, **kwargs)
        pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_a_different_partition():
    train = toy_train(PER_CLASS)
    a, _ = dirichlet_partition(train, num_clients=5, alpha=0.3, seed=0)
    b, _ = dirichlet_partition(train, num_clients=5, alpha=0.3, seed=1)
    assert not a["client_id"].equals(b["client_id"])


def test_unknown_scheme_is_rejected():
    train = toy_train({"a": 50})
    with pytest.raises(ValueError, match="unknown partition scheme"):
        build_partition(train, "round_robin", num_clients=2)


def test_jensen_shannon_divergence_bounds():
    assert jensen_shannon_divergence([1, 1, 1], [1, 1, 1]) == pytest.approx(0.0)
    assert jensen_shannon_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    partial = jensen_shannon_divergence([2, 1], [1, 2])
    assert 0.0 < partial < 1.0


def test_partition_filename_is_canonical():
    assert partition_filename("iid", 5, 0.0, 0) == "iid_K5_alphana_seed0.csv.gz"
    assert partition_filename("dirichlet", 5, 0.1, 2) == "dirichlet_K5_alpha0.1_seed2.csv.gz"
    assert partition_filename("dirichlet", 10, 1.0, 0) == "dirichlet_K10_alpha1_seed0.csv.gz"


@needs_splits
@pytest.mark.parametrize("scheme,alpha", [
    ("iid", 0.0),
    ("dirichlet", 0.1),
    ("dirichlet", 0.5),
    ("dirichlet", 1.0),
])
def test_committed_partitions_cover_train_and_exclude_val_and_test(scheme, alpha):
    partition = load_partition(SPLITS_DIR, scheme, 5, alpha, 0)
    train = load_split(SPLITS_DIR, "train", with_class_names=False)
    val = load_split(SPLITS_DIR, "val", with_class_names=False)
    test = load_split(SPLITS_DIR, "test", with_class_names=False)

    assert partition["path"].is_unique
    assert set(partition["path"]) == set(train["path"])
    assert not set(partition["path"]) & set(val["path"])
    assert not set(partition["path"]) & set(test["path"])


@needs_splits
def test_committed_partitions_keep_labels_consistent_with_the_train_split():
    partition = load_partition(SPLITS_DIR, "dirichlet", 5, 0.1, 0)
    train = load_split(SPLITS_DIR, "train", with_class_names=False)
    merged = partition.merge(train, on="path", suffixes=("_partition", "_train"))
    assert len(merged) == len(partition)
    assert (merged["label_partition"] == merged["label_train"]).all()


@needs_splits
def test_saved_partition_round_trips_through_disk(tmp_path):
    """Saving the compact assignment and loading it back gives the same split."""
    import shutil

    from fedxcrop.data.partition import save_partition

    for name in ("train.csv", "classes.json"):
        shutil.copy(SPLITS_DIR / name, tmp_path / name)

    train = load_split(tmp_path, "train", with_class_names=False)
    original, _ = build_partition(train, "dirichlet", num_clients=5, alpha=0.3, seed=5)
    save_partition(original, tmp_path, "dirichlet", 5, 0.3, 5)
    restored = load_partition(tmp_path, "dirichlet", 5, 0.3, 5)

    pd.testing.assert_frame_equal(
        original[["path", "label", "client_id"]].sort_values("path").reset_index(drop=True),
        restored[["path", "label", "client_id"]].sort_values("path").reset_index(drop=True),
    )
