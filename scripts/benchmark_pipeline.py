"""Measure end to end training throughput with and without GPU augmentation.

Which side of the pipeline is the bottleneck depends entirely on the machine.
On a laptop with a modest GPU and many cores the GPU saturates first and
moving augmentation onto it changes nothing or costs a little. On a cloud
runtime with a fast GPU and two CPU cores the loader is the bottleneck and
moving the colour jitter off the CPU is worth several times the throughput.

So this measures rather than assumes. Run it before the grid and use whichever
setting wins on the machine the grid will run on.

Usage:
    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --batches 60 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config, resolve_device
from fedxcrop.data.dataset import available_cpus, build_dataset, build_loader, effective_workers
from fedxcrop.data.gpu_augment import make_train_augment
from fedxcrop.data.splits import load_split
from fedxcrop.models.factory import build_model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(cfg, gpu_augment: bool, device, frame, batches: int, warmup: int) -> float:
    """Images per second through loader, augmentation, and a real training step."""
    dataset = build_dataset(
        cfg.data.root, frame, train=True,
        image_size=cfg.data.image_size, gpu_augment=gpu_augment,
    )
    loader = build_loader(
        dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, seed=0, drop_last=True,
    )
    model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    augment = make_train_augment(cfg, device) if gpu_augment else None

    def step(images, labels):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if augment is not None:
            images = augment(images)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    iterator = iter(loader)
    for _ in range(warmup):
        images, labels = next(iterator)
        step(images, labels)

    synchronize(device)
    started, seen = time.time(), 0
    for _ in range(batches):
        images, labels = next(iterator)
        step(images, labels)
        seen += labels.numel()
    synchronize(device)
    return seen / (time.time() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample", type=int, default=6000, help="training images to draw from")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    overrides = list(args.set)
    if args.batch_size:
        overrides.append(f"data.batch_size={args.batch_size}")
    cfg = load_config(args.config, overrides)
    device = resolve_device(cfg.device)

    frame = load_split(cfg.data.splits_dir, "train", with_class_names=False)
    frame = frame.sample(n=min(args.sample, len(frame)), random_state=0).reset_index(drop=True)

    print(f"device        {device}")
    print(f"CPUs          {available_cpus()} (loader workers: "
          f"{effective_workers(cfg.data.num_workers)}, requested {cfg.data.num_workers})")
    print(f"batch size    {cfg.data.batch_size}")
    print(f"measuring     {args.batches} batches after {args.warmup} warmup\n")

    rates = {}
    for gpu_augment in (False, True):
        label = "GPU augmentation" if gpu_augment else "CPU augmentation"
        rate = measure(cfg, gpu_augment, device, frame, args.batches, args.warmup)
        rates[gpu_augment] = rate
        print(f"{label:<18} {rate:7.1f} img/s")

    faster = rates[True] / rates[False]
    epoch_images = 43447
    print(f"\nspeedup from moving augmentation to the GPU: {faster:.2f}x")
    print(f"projected time for one epoch or federated round over {epoch_images:,} images:")
    print(f"  data.gpu_augment=false   {epoch_images / rates[False] / 60:5.1f} min")
    print(f"  data.gpu_augment=true    {epoch_images / rates[True] / 60:5.1f} min")

    if faster < 1.05:
        print("\nThis machine is not bottlenecked on the loader, so GPU augmentation")
        print("wins nothing here. Set data.gpu_augment=false if it is slower.")
    else:
        print(f"\nThe loader was the bottleneck. Keep data.gpu_augment=true.")
        best = epoch_images / rates[True] / 60
        print(f"Core grid (24 federated runs x 30 rounds): {24 * 30 * best / 60:.1f} h")
        print(f"Centralized (3 seeds x 20 epochs):         {3 * 20 * best / 60:.1f} h")

    out = Path(cfg.results_dir) / "pipeline_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(
            {
                "device": str(device),
                "cpus": available_cpus(),
                "loader_workers": effective_workers(cfg.data.num_workers),
                "batch_size": cfg.data.batch_size,
                "images_per_second": {"cpu_augment": rates[False], "gpu_augment": rates[True]},
                "speedup": faster,
                "projected_minutes_per_epoch": {
                    "cpu_augment": epoch_images / rates[False] / 60,
                    "gpu_augment": epoch_images / rates[True] / 60,
                },
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
