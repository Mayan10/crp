"""Dividing the train split across simulated clients.

Only the train split is partitioned. Validation and test stay global and
untouched, so every client configuration is measured on exactly the same
held out images. This is the property the earlier version of this work did
not have.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fedxcrop.data.splits import load_split

PARTITION_SCHEMES = ("iid", "dirichlet")


def iid_partition(
    train: pd.DataFrame,
    num_clients: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Uniform random assignment of images to clients.

    Every client sees roughly the global class distribution, which is the
    upper bound case for federated averaging.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    assigned = train.iloc[order].reset_index(drop=True)
    # Round robin over the shuffled order gives sizes within one image.
    assigned["client_id"] = np.arange(len(assigned)) % num_clients
    return assigned.sort_values(["client_id", "path"]).reset_index(drop=True)


def dirichlet_partition(
    train: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_samples_per_client: int = 10,
    max_attempts: int = 100,
) -> tuple[pd.DataFrame, int]:
    """Label skewed partition by per class Dirichlet sampling.

    For each class, a Dirichlet(alpha) vector over clients decides what share
    of that class each client receives. Small alpha concentrates each class on
    a few clients, which is the heterogeneity FedProx is meant to handle.

    A draw that leaves any client with fewer than `min_samples_per_client`
    images is rejected and redrawn. Returns the partition and the number of
    resamples it took, which is logged rather than hidden.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    labels = train["label"].to_numpy()
    classes = np.unique(labels)
    class_indices = {c: np.flatnonzero(labels == c) for c in classes}

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt * 1_000_003)
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]

        for c in classes:
            indices = class_indices[c].copy()
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Cut points along the shuffled class, one contiguous chunk per client.
            cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(indices, cuts)):
                buckets[client_id].append(chunk)

        sizes = [sum(len(chunk) for chunk in bucket) for bucket in buckets]
        if min(sizes) >= min_samples_per_client:
            rows = []
            for client_id, bucket in enumerate(buckets):
                merged = np.concatenate(bucket) if bucket else np.array([], dtype=int)
                part = train.iloc[merged].copy()
                part["client_id"] = client_id
                rows.append(part)
            partition = pd.concat(rows, ignore_index=True)
            return partition.sort_values(["client_id", "path"]).reset_index(drop=True), attempt

    raise RuntimeError(
        f"could not draw a Dirichlet partition with at least {min_samples_per_client} "
        f"images per client after {max_attempts} attempts (alpha={alpha}, K={num_clients})"
    )


def build_partition(
    train: pd.DataFrame,
    scheme: str,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
    min_samples_per_client: int = 10,
) -> tuple[pd.DataFrame, int]:
    """Dispatch to the requested partitioning scheme."""
    if scheme == "iid":
        return iid_partition(train, num_clients, seed), 0
    if scheme == "dirichlet":
        return dirichlet_partition(train, num_clients, alpha, seed, min_samples_per_client)
    raise ValueError(f"unknown partition scheme {scheme!r}, expected one of {PARTITION_SCHEMES}")


def partition_filename(scheme: str, num_clients: int, alpha: float, seed: int) -> str:
    """Canonical name, so a config and a file on disk always agree."""
    alpha_part = "na" if scheme == "iid" else f"{alpha:g}"
    return f"{scheme}_K{num_clients}_alpha{alpha_part}_seed{seed}.csv.gz"


def save_partition(
    partition: pd.DataFrame,
    splits_dir: str | Path,
    scheme: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> Path:
    """Write the client assignment for one partition.

    Stored as a single `client_id` column with one row per row of train.csv, in
    that file's order, rather than repeating all 43,447 image paths in every
    partition of the grid. train.csv is itself committed and fixed, so the two
    together pin down exactly which image each client holds, and
    `load_partition` rebuilds the full (path, label, client_id) frame.
    """
    train = load_split(splits_dir, "train", with_class_names=False)
    assignment = partition.set_index("path")["client_id"]
    if set(assignment.index) != set(train["path"]):
        raise ValueError("partition does not cover the train split exactly once")

    out_dir = Path(splits_dir) / "partitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / partition_filename(scheme, num_clients, alpha, seed)
    frame = pd.DataFrame({"client_id": assignment.reindex(train["path"]).to_numpy()})
    frame.to_csv(path, index=False, compression="gzip")
    return path


def load_partition(
    splits_dir: str | Path,
    scheme: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Rebuild the full (path, label, client_id) frame for one partition."""
    path = Path(splits_dir) / "partitions" / partition_filename(scheme, num_clients, alpha, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"partition not found: {path}. Run scripts/build_partitions.py first."
        )
    train = load_split(splits_dir, "train", with_class_names=False)
    assignment = pd.read_csv(path)
    if len(assignment) != len(train):
        raise ValueError(
            f"partition {path.name} has {len(assignment)} rows but the train split "
            f"has {len(train)}; rebuild the partitions"
        )
    out = train.copy()
    out["client_id"] = assignment["client_id"].to_numpy()
    return out.sort_values(["client_id", "path"]).reset_index(drop=True)


def client_frames(partition: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split a partition into one frame per client."""
    return {
        int(client_id): frame.reset_index(drop=True)
        for client_id, frame in partition.groupby("client_id", sort=True)
    }


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in bits between two label distributions.

    Bounded in [0, 1], zero when a client matches the global distribution and
    one when the two share no classes, which makes it a readable single number
    for how skewed a client is.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def label_histogram(frame: pd.DataFrame, num_classes: int) -> np.ndarray:
    """Count of each label, including classes with zero images."""
    counts = np.zeros(num_classes, dtype=int)
    values, occurrences = np.unique(frame["label"].to_numpy(), return_counts=True)
    counts[values] = occurrences
    return counts


def partition_summary(
    partition: pd.DataFrame,
    num_classes: int,
    scheme: str,
    num_clients: int,
    alpha: float,
    seed: int,
    resamples: int = 0,
) -> dict:
    """Client sizes, class coverage, and skew relative to the global distribution."""
    global_histogram = label_histogram(partition, num_classes)
    clients = client_frames(partition)

    per_client = {}
    for client_id, frame in clients.items():
        histogram = label_histogram(frame, num_classes)
        per_client[str(client_id)] = {
            "n_images": int(len(frame)),
            "n_classes_present": int((histogram > 0).sum()),
            "jensen_shannon_divergence": jensen_shannon_divergence(histogram, global_histogram),
            "largest_class_fraction": float(histogram.max() / max(histogram.sum(), 1)),
            "label_counts": histogram.tolist(),
        }

    sizes = [stats["n_images"] for stats in per_client.values()]
    divergences = [stats["jensen_shannon_divergence"] for stats in per_client.values()]
    classes_present = [stats["n_classes_present"] for stats in per_client.values()]

    return {
        "scheme": scheme,
        "num_clients": num_clients,
        "alpha": None if scheme == "iid" else alpha,
        "seed": seed,
        "resamples": resamples,
        "n_images": int(len(partition)),
        "num_classes": num_classes,
        "client_sizes": {"min": min(sizes), "max": max(sizes), "mean": float(np.mean(sizes))},
        "classes_present": {
            "min": min(classes_present),
            "max": max(classes_present),
            "mean": float(np.mean(classes_present)),
        },
        "jensen_shannon_divergence": {
            "min": float(np.min(divergences)),
            "max": float(np.max(divergences)),
            "mean": float(np.mean(divergences)),
        },
        "per_client": per_client,
    }


def save_summary(summary: dict, results_dir: str | Path, filename: str) -> Path:
    out_dir = Path(results_dir) / "data" / "partitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    return path
