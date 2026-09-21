"""Run one federated configuration and evaluate it once on the test split.

Usage:
    python scripts/run_federated.py --config configs/fedavg_iid.yaml --seed 0
    python scripts/run_federated.py --config configs/fedavg_iid.yaml --smoke
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedxcrop.config import load_config, resolve_device, save_run_metadata
from fedxcrop.data.dataset import build_dataset, build_loader
from fedxcrop.data.partition import build_partition, load_partition
from fedxcrop.data.splits import load_classes, load_split
from fedxcrop.eval.evaluate import save_predictions
from fedxcrop.eval.metrics import save_metrics
from fedxcrop.fl.simulation import evaluate_global, run_federated
from fedxcrop.models.factory import build_model, load_checkpoint


def run_name(cfg) -> str:
    """Stable directory name encoding what distinguishes this run."""
    parts = [cfg.federated.strategy, cfg.partition.scheme]
    if cfg.partition.scheme != "iid":
        parts.append(f"alpha{cfg.partition.alpha:g}")
    parts.append(f"K{cfg.partition.num_clients}")
    if cfg.federated.strategy == "fedprox":
        parts.append(f"mu{cfg.federated.mu:g}")
    if cfg.federated.init != "imagenet":
        parts.append("legacyinit")
    parts.append(f"seed{cfg.seed}")
    return "_".join(parts)


def smoke_subset(frame: pd.DataFrame, n_classes: int, per_class: int, seed: int) -> pd.DataFrame:
    keep = sorted(frame["label"].unique())[:n_classes]
    parts = [
        frame[frame["label"] == label].sample(
            n=min(per_class, int((frame["label"] == label).sum())), random_state=seed
        )
        for label in keep
    ]
    return pd.concat(parts).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fedavg_iid.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run of the whole path")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="exit immediately if the final test metrics already exist")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    # Smoke settings go in first so an explicit --set still wins over them.
    # Random init also keeps the smoke path hermetic: it exercises every stage
    # without needing the ImageNet weights download.
    smoke_overrides = [
        "federated.rounds=2", "federated.local_epochs=1", "partition.num_clients=2",
        "model.num_classes=2", "model.pretrained=false", "data.num_workers=0",
        "data.batch_size=16", "device=cpu",
    ]
    overrides = (smoke_overrides if args.smoke else []) + list(args.set)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    cfg = load_config(args.config, overrides)

    name = run_name(cfg)
    run_dir = Path(cfg.runs_dir) / "federated" / ("smoke" if args.smoke else name)
    results_dir = Path(cfg.results_dir) / "federated" / ("smoke" if args.smoke else name)

    if args.skip_existing and (results_dir / "test_metrics.json").is_file():
        print(f"{name}: results already exist, skipping")
        return 0

    device = resolve_device(cfg.device)
    save_run_metadata(run_dir, cfg)
    print(f"run       {name}\ndevice    {device}\nrun dir   {run_dir}\nresults   {results_dir}\n")

    classes = load_classes(cfg.data.splits_dir)
    val = load_split(cfg.data.splits_dir, "val", with_class_names=False)
    test = load_split(cfg.data.splits_dir, "test", with_class_names=False)

    if args.smoke:
        train = load_split(cfg.data.splits_dir, "train", with_class_names=False)
        train = smoke_subset(train, cfg.model.num_classes, 100, cfg.seed)
        val = smoke_subset(val, cfg.model.num_classes, 50, cfg.seed)
        test = smoke_subset(test, cfg.model.num_classes, 50, cfg.seed)
        classes = classes[: cfg.model.num_classes]
        partition, _ = build_partition(
            train, cfg.partition.scheme, cfg.partition.num_clients,
            alpha=cfg.partition.alpha, seed=cfg.seed,
            min_samples_per_client=cfg.partition.min_samples_per_client,
        )
    else:
        partition = load_partition(
            cfg.data.splits_dir, cfg.partition.scheme, cfg.partition.num_clients,
            cfg.partition.alpha, cfg.partition.seed,
        )

    val_loader = build_loader(
        build_dataset(cfg.data.root, val, train=False, image_size=cfg.data.image_size),
        batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
    )
    test_loader = build_loader(
        build_dataset(cfg.data.root, test, train=False, image_size=cfg.data.image_size),
        batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
    )

    outcome = run_federated(
        cfg, partition, val_loader, device, run_dir,
        class_names=classes, resume=not args.no_resume,
    )

    # Test is read once, with the round selected on validation accuracy alone.
    model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False)
    state = load_checkpoint(model, run_dir / "best.pt", map_location=device)
    metrics = evaluate_global(model, test_loader, device, cfg.model.num_classes, classes)
    metrics["selected_round"] = int(state.get("round", -1))
    metrics["selection"] = "best validation accuracy, test evaluated once"
    metrics["config"] = {
        "strategy": cfg.federated.strategy,
        "scheme": cfg.partition.scheme,
        "alpha": None if cfg.partition.scheme == "iid" else cfg.partition.alpha,
        "num_clients": cfg.partition.num_clients,
        "mu": cfg.federated.mu if cfg.federated.strategy == "fedprox" else None,
        "rounds": cfg.federated.rounds,
        "local_epochs": cfg.federated.local_epochs,
        "init": cfg.federated.init,
        "seed": cfg.seed,
    }

    save_metrics(metrics, results_dir / "test_metrics.json")

    from fedxcrop.eval.evaluate import predict
    output = predict(model, test_loader, device)
    save_predictions(test, output["y_true"], output["y_pred"], results_dir / "test_predictions.csv")
    shutil.copy(run_dir / "history.csv", results_dir / "history.csv")
    shutil.copy(run_dir / "summary.json", results_dir / "summary.json")

    print(
        f"\ntest: accuracy {metrics['accuracy']:.4f} "
        f"({metrics['n_correct']}/{metrics['n_total']}), "
        f"macro F1 {metrics['macro_f1']:.4f}, from round {metrics['selected_round']}"
    )
    print(f"wrote {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
