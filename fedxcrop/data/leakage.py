"""Near duplicate detection between splits.

PlantVillage contains multiple photographs of the same physical leaf, taken in
one session under near identical conditions. When such images land on both
sides of a split, test accuracy is optimistic. This module measures how often
that happens. It reports the overlap and never changes the split, because
silently dropping images would make the results incomparable to prior work.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

HASH_SIZE = 8  # phash at size 8 gives a 64 bit fingerprint
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hash_one(args: tuple[str, str]) -> Optional[np.ndarray]:
    root, relpath = args
    try:
        with Image.open(Path(root) / relpath) as image:
            value = imagehash.phash(image.convert("RGB"), hash_size=HASH_SIZE)
    except OSError:
        return None
    return np.packbits(value.hash.flatten())


def phash_many(
    root: str | Path,
    relpaths: Iterable[str],
    workers: int = 4,
    chunksize: int = 64,
) -> np.ndarray:
    """Perceptual hashes as a (N, 8) uint8 array, one packed 64 bit row per image.

    A row of zeros means the image could not be read; those rows are kept so the
    array stays aligned with the input list.
    """
    relpaths = list(relpaths)
    root = str(root)
    hashes = np.zeros((len(relpaths), HASH_SIZE), dtype=np.uint8)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, packed in enumerate(
            pool.map(_hash_one, ((root, p) for p in relpaths), chunksize=chunksize)
        ):
            if packed is not None:
                hashes[i] = packed
    return hashes


def min_hamming_distance(
    query: np.ndarray,
    reference: np.ndarray,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query hash, the smallest Hamming distance to any reference hash.

    Returns the distances and the index of the nearest reference row. Chunked so
    the comparison of tens of thousands of images against each other stays
    within memory.
    """
    n_query = len(query)
    best_distance = np.full(n_query, 64, dtype=np.int16)
    best_index = np.zeros(n_query, dtype=np.int64)

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        block = query[start:stop]
        # XOR every pair, then count set bits with a byte lookup table.
        xor = np.bitwise_xor(block[:, None, :], reference[None, :, :])
        distances = _POPCOUNT[xor].sum(axis=2).astype(np.int16)
        best_index[start:stop] = distances.argmin(axis=1)
        best_distance[start:stop] = distances.min(axis=1)

    return best_distance, best_index


def leakage_report(
    root: str | Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: int = 4,
    workers: int = 4,
) -> tuple[dict, pd.DataFrame]:
    """How many test images have a near duplicate in train.

    A pair counts as a near duplicate when the Hamming distance between their
    64 bit perceptual hashes is at most `threshold`. Same label and cross label
    matches are counted separately: a cross label match is a labelling concern,
    a same label match inflates accuracy.
    """
    train_hashes = phash_many(root, train["path"], workers=workers)
    test_hashes = phash_many(root, test["path"], workers=workers)

    distances, indices = min_hamming_distance(test_hashes, train_hashes)
    is_duplicate = distances <= threshold

    matches = pd.DataFrame(
        {
            "test_path": test["path"].to_numpy(),
            "test_label": test["label"].to_numpy(),
            "test_class": test["class_name"].to_numpy(),
            "nearest_train_path": train["path"].to_numpy()[indices],
            "nearest_train_label": train["label"].to_numpy()[indices],
            "hamming_distance": distances,
            "is_near_duplicate": is_duplicate,
        }
    )

    duplicates = matches[matches["is_near_duplicate"]]
    same_label = duplicates[duplicates["test_label"] == duplicates["nearest_train_label"]]

    per_class = (
        matches.groupby("test_class")["is_near_duplicate"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "near_duplicates", "count": "test_images"})
    )

    report = {
        "threshold_hamming": threshold,
        "hash": f"phash_{HASH_SIZE}x{HASH_SIZE}_64bit",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_with_near_duplicate_in_train": int(is_duplicate.sum()),
        "fraction_test_with_near_duplicate": float(is_duplicate.mean()),
        "n_same_label_matches": int(len(same_label)),
        "n_cross_label_matches": int(len(duplicates) - len(same_label)),
        "n_exact_hash_matches": int((distances == 0).sum()),
        "distance_percentiles": {
            str(p): float(np.percentile(distances, p)) for p in (1, 5, 25, 50, 75)
        },
        "per_class": {
            name: {
                "near_duplicates": int(row.near_duplicates),
                "test_images": int(row.test_images),
                "fraction": float(row.near_duplicates / row.test_images),
            }
            for name, row in per_class.iterrows()
        },
        "note": (
            "Reported only. The split is not modified, so these results stay "
            "comparable with published PlantVillage numbers, which are subject "
            "to the same overlap."
        ),
    }
    return report, matches
