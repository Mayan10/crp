"""Train the centralized baseline and evaluate it once on the test split.

Usage:
    python scripts/train_centralized.py --config configs/centralized.yaml --seed 0
    python scripts/train_centralized.py --config configs/centralized.yaml --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedxcrop.config import load_config, resolve_device, save_run_metadata
from fedxcrop.data.dataset import build_dataset, build_loader
from fedxcrop.data.splits import load_classes, load_split
from fedxcrop.train.centralized import evaluate_on_test, train_centralized


def subset_for_smoke(frame, n_classes: int = 2, per_class: int = 100, seed: int = 0):
    """Tiny stratified subset, so the whole path can be exercised on a CPU."""
    keep = sorted(frame["label"].unique())[:n_classes]
    parts = [
        frame[frame["label"] == label].sample(
            n=min(per_class, int((frame["label"] == label).sum())), random_state=seed
        )
        for label in keep
    ]
    import pandas as pd

    return pd.concat(parts).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/centralized.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run of the whole path")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="exit immediately if the final test metrics already exist")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    # Smoke settings go in first so an explicit --set still wins over them.
    # Random init also keeps the smoke path hermetic: it exercises every stage
    # without needing the ImageNet weights download.
    smoke_overrides = [
        "centralized.epochs=2", "model.num_classes=2", "model.pretrained=false",
        "data.num_workers=0", "device=cpu",
    ]
    overrides = (smoke_overrides if args.smoke else []) + list(args.set)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    cfg = load_config(args.config, overrides)

    if args.smoke:
        cfg.name = f"{cfg.name}_smoke"

    cfg.name = f"{cfg.name}_seed{cfg.seed}" if not cfg.name.endswith(f"seed{cfg.seed}") else cfg.name
    run_dir = Path(cfg.runs_dir) / "centralized" / cfg.name
    results_dir = Path(cfg.results_dir) / "centralized" / f"seed{cfg.seed}"
    if args.smoke:
        results_dir = Path(cfg.results_dir) / "centralized" / "smoke"

    if args.skip_existing and (results_dir / "test_metrics.json").is_file():
        print(f"{cfg.name}: results already exist, skipping")
        return 0

    device = resolve_device(cfg.device)
    save_run_metadata(run_dir, cfg)
    print(f"run       {cfg.name}\ndevice    {device}\nrun dir   {run_dir}\nresults   {results_dir}\n")

    classes = load_classes(cfg.data.splits_dir)
    frames = {
        name: load_split(cfg.data.splits_dir, name, with_class_names=False)
        for name in ("train", "val", "test")
    }
    if args.smoke:
        frames = {name: subset_for_smoke(frame) for name, frame in frames.items()}
        classes = classes[:2]

    loaders = {
        "train": build_loader(
            build_dataset(cfg.data.root, frames["train"], train=True, image_size=cfg.data.image_size),
            batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, seed=cfg.seed,
        ),
    }
    for name in ("val", "test"):
        loaders[name] = build_loader(
            build_dataset(cfg.data.root, frames[name], train=False, image_size=cfg.data.image_size),
            batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
        )

    print(
        f"train {len(frames['train'])}  val {len(frames['val'])}  test {len(frames['test'])}  "
        f"classes {len(classes)}\n"
    )

    train_centralized(
        cfg, loaders["train"], loaders["val"], device, run_dir,
        class_names=classes, resume=not args.no_resume, log_every=args.log_every,
    )
    evaluate_on_test(
        cfg, loaders["test"], frames["test"], device, run_dir, results_dir, class_names=classes,
    )

    import shutil
    shutil.copy(run_dir / "history.csv", results_dir / "history.csv")
    print(f"\nwrote {results_dir}/history.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
