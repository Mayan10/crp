"""Report near duplicate overlap between the train and test splits.

Usage:
    python scripts/check_leakage.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedxcrop.config import load_config
from fedxcrop.data.index import save_json
from fedxcrop.data.leakage import leakage_report
from fedxcrop.data.splits import load_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--threshold", type=int, default=4, help="max Hamming distance")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    splits = load_splits(cfg.data.splits_dir)

    print(
        f"hashing {len(splits['train'])} train and {len(splits['test'])} test images "
        f"with {args.workers} workers"
    )
    started = time.time()
    report, matches = leakage_report(
        cfg.data.root,
        splits["train"],
        splits["test"],
        threshold=args.threshold,
        workers=args.workers,
    )
    print(f"done in {time.time() - started:.0f} s\n")

    n_dup = report["n_test_with_near_duplicate_in_train"]
    print(f"test images with a near duplicate in train (distance <= {args.threshold}):")
    print(f"  {n_dup} of {report['n_test']} ({100 * report['fraction_test_with_near_duplicate']:.2f} percent)")
    print(f"  exact hash matches:      {report['n_exact_hash_matches']}")
    print(f"  same label matches:      {report['n_same_label_matches']}")
    print(f"  cross label matches:     {report['n_cross_label_matches']}")

    worst = sorted(
        report["per_class"].items(), key=lambda kv: kv[1]["fraction"], reverse=True
    )[:10]
    print("\nmost affected classes:")
    for name, stats in worst:
        print(
            f"  {name:<55} {stats['near_duplicates']:>4}/{stats['test_images']:<5} "
            f"({100 * stats['fraction']:.1f} percent)"
        )

    out_dir = Path(cfg.results_dir) / "data"
    save_json(report, out_dir / "leakage_report.json")
    matches[matches["is_near_duplicate"]].to_csv(out_dir / "leakage_pairs.csv", index=False)
    print(f"\nwrote {out_dir / 'leakage_report.json'}")
    print(f"wrote {out_dir / 'leakage_pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
