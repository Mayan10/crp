"""Scanning the PlantVillage directory tree into a class index and file list.

The dataset ships three variants of the same photographs: color (used for
training), grayscale (unused here), and segmented (a background removed version
that gives a free leaf mask). A segmented file is named after its color
counterpart with a "_final_masked" suffix and a lowercase .jpg extension.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEGMENTED_SUFFIX = "_final_masked"
EXPECTED_NUM_CLASSES = 38


def class_names(root: str | Path, variant: str = "color") -> list[str]:
    """Sorted class directory names, which fixes the label ordering everywhere."""
    variant_dir = Path(root) / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"dataset variant directory not found: {variant_dir}")
    names = sorted(
        d.name for d in variant_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not names:
        raise FileNotFoundError(f"no class directories under {variant_dir}")
    return names


def build_index(root: str | Path, variant: str = "color") -> pd.DataFrame:
    """List every image as (path, label, class_name), sorted for determinism.

    Paths are stored relative to the dataset root so the CSVs stay portable
    between a laptop and a cloud runtime.
    """
    root = Path(root)
    names = class_names(root, variant)
    name_to_label = {name: i for i, name in enumerate(names)}

    rows = []
    for name in names:
        class_dir = root / variant / name
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in files:
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "label": name_to_label[name],
                    "class_name": name,
                }
            )

    frame = pd.DataFrame(rows, columns=["path", "label", "class_name"])
    if frame.empty:
        raise FileNotFoundError(f"no images found under {root / variant}")
    return frame


def per_class_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Image count per class, ordered by label."""
    return (
        index.groupby(["label", "class_name"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )


def segmented_path_for(color_relpath: str, segmented_variant: str = "segmented") -> str:
    """Relative path of the segmented counterpart of a color image, by name.

    This is the common case: the segmented file carries the same stem with a
    "_final_masked" suffix. The file may not exist; `segmented_coverage` also
    tries a fallback for classes that use a different naming convention.
    """
    parts = Path(color_relpath).parts
    class_name, filename = parts[-2], parts[-1]
    stem = Path(filename).stem
    return str(Path(segmented_variant) / class_name / f"{stem}{SEGMENTED_SUFFIX}.jpg")


def image_key(stem: str) -> str:
    """Identifier shared by the color and segmented copies of one photograph.

    Most files are named "<uuid>___<source id>". A few classes store the color
    copy under the bare source id while the segmented copy keeps the uuid
    prefix, so the part after the last "___" is what the two reliably share.
    """
    if stem.endswith(SEGMENTED_SUFFIX):
        stem = stem[: -len(SEGMENTED_SUFFIX)]
    return stem.split("___")[-1]


def _segmented_lookup(root: Path, segmented_variant: str) -> dict[tuple[str, str], str]:
    """Map (class name, image key) to a segmented path, skipping ambiguous keys."""
    variant_dir = root / segmented_variant
    if not variant_dir.is_dir():
        return {}

    lookup: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for class_dir in variant_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for path in class_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = (class_dir.name, image_key(path.stem))
            if key in lookup:
                ambiguous.add(key)
                continue
            lookup[key] = str(path.relative_to(root))

    # A key that names more than one segmented file cannot identify a mask.
    for key in ambiguous:
        lookup.pop(key, None)
    return lookup


def segmented_coverage(
    root: str | Path,
    index: pd.DataFrame,
    segmented_variant: str = "segmented",
) -> tuple[pd.DataFrame, dict]:
    """Map every color image to its segmented counterpart and report coverage.

    Matching is by exact filename first, then by the shared image key for the
    classes whose color and segmented copies are named differently. Returns the
    index with a `segmented_path` column (empty string when there is no match)
    and a summary dictionary suitable for saving as JSON.
    """
    root = Path(root)
    mapped = index.copy()

    by_name = mapped["path"].map(lambda p: segmented_path_for(p, segmented_variant))
    name_hit = by_name.map(lambda p: (root / p).is_file())

    lookup = _segmented_lookup(root, segmented_variant)
    by_key = mapped["path"].map(
        lambda p: lookup.get((Path(p).parts[-2], image_key(Path(p).stem)), "")
    )

    resolved = by_name.where(name_hit, by_key)
    matched_mask = resolved.astype(bool)
    mapped["segmented_path"] = resolved.where(matched_mask, "")

    matched = int(matched_mask.sum())
    total = int(len(mapped))
    per_class = (
        mapped.assign(has_mask=matched_mask)
        .groupby("class_name")["has_mask"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "matched", "count": "total"})
    )
    summary = {
        "total_color_images": total,
        "matched_segmented": matched,
        "unmatched": total - matched,
        "coverage_fraction": matched / total if total else 0.0,
        "matched_by_filename": int(name_hit.sum()),
        "matched_by_image_key": int(matched - name_hit.sum()),
        "classes_with_full_coverage": int((per_class["matched"] == per_class["total"]).sum()),
        "num_classes": int(len(per_class)),
        "per_class": {
            name: {"matched": int(row.matched), "total": int(row.total)}
            for name, row in per_class.iterrows()
        },
    }
    return mapped, summary


def verify_dataset(index: pd.DataFrame) -> dict:
    """Facts about the scanned dataset, reported rather than enforced.

    The published PlantVillage color set is usually quoted as 54,305 or 54,306
    images over 38 classes. A different count is reported, not corrected.
    """
    counts = per_class_counts(index)
    num_classes = int(index["class_name"].nunique())
    total = int(len(index))
    return {
        "num_classes": num_classes,
        "num_classes_expected": EXPECTED_NUM_CLASSES,
        "num_classes_match": num_classes == EXPECTED_NUM_CLASSES,
        "total_images": total,
        "total_images_expected_range": [54305, 54306],
        "total_images_in_expected_range": total in (54305, 54306),
        "smallest_class": {
            "class_name": counts.loc[counts["count"].idxmin(), "class_name"],
            "count": int(counts["count"].min()),
        },
        "largest_class": {
            "class_name": counts.loc[counts["count"].idxmax(), "class_name"],
            "count": int(counts["count"].max()),
        },
        "per_class_counts": {row.class_name: int(row.count) for row in counts.itertuples()},
    }


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
    return path


def load_segmented_mapping(splits_dir: str | Path) -> pd.DataFrame:
    """Read the committed color to segmented path mapping."""
    path = Path(splits_dir) / "segmented_mapping.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(
            f"segmented mapping not found: {path}. Run scripts/build_splits.py first."
        )
    frame = pd.read_csv(path, keep_default_na=False)
    return frame
