"""Build every federated partition of the train split, with figures and summaries.

Usage:
    python scripts/build_partitions.py --config configs/base.yaml
    python scripts/build_partitions.py --clients 5 10 --alphas 0.1 0.5 1.0 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedxcrop.config import load_config
from fedxcrop.data.partition import (
    build_partition,
    partition_filename,
    partition_summary,
    save_partition,
    save_summary,
)
from fedxcrop.data.splits import load_classes, load_split
from fedxcrop.viz.partitions import alpha_comparison_grid, client_class_histogram


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--clients", type=int, nargs="+", default=[5])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    train = load_split(cfg.data.splits_dir, "train")
    classes = load_classes(cfg.data.splits_dir)
    num_classes = len(classes)

    print(f"partitioning {len(train)} train images over {num_classes} classes\n")
    print(f"{'partition':<38} {'min':>6} {'max':>6} {'classes':>9} {'JS mean':>8} {'resamples':>10}")
    print("-" * 82)

    jobs = [("iid", k, 0.0, s) for k in args.clients for s in args.seeds]
    jobs += [
        ("dirichlet", k, a, s)
        for k in args.clients
        for a in args.alphas
        for s in args.seeds
    ]

    built: dict[tuple[str, int, float, int], object] = {}
    for scheme, num_clients, alpha, seed in jobs:
        partition, resamples = build_partition(
            train,
            scheme,
            num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_client=cfg.partition.min_samples_per_client,
        )
        save_partition(partition, cfg.data.splits_dir, scheme, num_clients, alpha, seed)
        summary = partition_summary(
            partition, num_classes, scheme, num_clients, alpha, seed, resamples
        )
        name = partition_filename(scheme, num_clients, alpha, seed).split(".csv")[0]
        save_summary(summary, cfg.results_dir, f"{name}.json")
        built[(scheme, num_clients, alpha, seed)] = partition

        print(
            f"{name:<38} "
            f"{summary['client_sizes']['min']:>6} "
            f"{summary['client_sizes']['max']:>6} "
            f"{summary['classes_present']['min']:>3}-{summary['classes_present']['max']:<5} "
            f"{summary['jensen_shannon_divergence']['mean']:>8.3f} "
            f"{resamples:>10}"
        )

    figures_dir = Path(cfg.results_dir) / "figures"
    for num_clients in args.clients:
        for seed in args.seeds:
            grid = {
                a: built[("dirichlet", num_clients, a, seed)]
                for a in args.alphas
                if ("dirichlet", num_clients, a, seed) in built
            }
            if grid and seed == args.seeds[0]:
                paths = alpha_comparison_grid(
                    grid, classes, figures_dir, f"fig1_client_class_histograms_K{num_clients}"
                )
                print(f"\nwrote {paths[0]}")
        if ("iid", num_clients, 0.0, args.seeds[0]) in built:
            paths = client_class_histogram(
                built[("iid", num_clients, 0.0, args.seeds[0])],
                classes,
                figures_dir,
                f"fig1_client_class_histogram_iid_K{num_clients}",
            )
            print(f"wrote {paths[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
