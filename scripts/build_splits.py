"""Scan the dataset, verify it, and write the one fixed stratified split.

Usage:
    python scripts/build_splits.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedxcrop.config import load_config
from fedxcrop.data.index import (
    build_index,
    per_class_counts,
    save_json,
    segmented_coverage,
    verify_dataset,
)
from fedxcrop.data.splits import save_splits, split_summary, stratified_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, key.sub=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(cfg.data.root)

    print(f"scanning {root / cfg.data.variant}")
    index = build_index(root, cfg.data.variant)

    facts = verify_dataset(index)
    print(f"  classes: {facts['num_classes']} (expected {facts['num_classes_expected']})")
    print(f"  images:  {facts['total_images']} (published count is 54,305 or 54,306)")
    if not facts["total_images_in_expected_range"]:
        print(
            "  NOTE: the image count differs from the published count. "
            "Reporting it as found, not adjusting the data."
        )
    print(
        f"  smallest class: {facts['smallest_class']['class_name']} "
        f"({facts['smallest_class']['count']})"
    )
    print(
        f"  largest class:  {facts['largest_class']['class_name']} "
        f"({facts['largest_class']['count']})"
    )

    print("\nper class counts")
    counts = per_class_counts(index)
    for row in counts.itertuples():
        print(f"  {row.label:>2}  {row.class_name:<55} {row.count:>5}")

    print("\nmapping color images to their segmented counterparts")
    mapped, coverage = segmented_coverage(root, index, cfg.data.segmented_variant)
    print(
        f"  matched {coverage['matched_segmented']} of {coverage['total_color_images']} "
        f"({100 * coverage['coverage_fraction']:.2f} percent), "
        f"{coverage['classes_with_full_coverage']} of {coverage['num_classes']} classes complete"
    )

    splits = stratified_split(
        index,
        cfg.data.train_fraction,
        cfg.data.val_fraction,
        cfg.data.test_fraction,
        cfg.data.split_seed,
    )
    written = save_splits(splits, cfg.data.splits_dir)
    summary = split_summary(splits)
    print(f"\nsplit (seed {cfg.data.split_seed})")
    for name, path in written.items():
        print(f"  {name:<5} {summary['sizes'][name]:>6} images  ->  {path}")

    report = {
        "dataset": facts,
        "segmented_coverage": coverage,
        "split": {
            "seed": cfg.data.split_seed,
            "fractions": {
                "train": cfg.data.train_fraction,
                "val": cfg.data.val_fraction,
                "test": cfg.data.test_fraction,
            },
            **summary,
        },
    }
    report_path = save_json(report, Path(cfg.results_dir) / "data" / "dataset_report.json")

    # Compressed: the uncompressed mapping of all 54k image pairs is far
    # larger than anything that belongs in a repository, and pandas reads the
    # gzipped form transparently.
    mapping_path = Path(cfg.data.splits_dir) / "segmented_mapping.csv.gz"
    mapped[["path", "segmented_path"]].to_csv(mapping_path, index=False, compression="gzip")
    print(f"\nwrote {report_path}")
    print(f"wrote {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
