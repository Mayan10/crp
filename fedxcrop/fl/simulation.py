"""Federated rounds over the committed client partitions.

Every client and the global model start from ImageNet weights. Nothing here is
ever initialized from a model that has already been trained on PlantVillage
centrally, which is the defect this rewrite exists to fix: starting from that
model makes federated accuracy and federated explanations largely properties of
the centralized model rather than of federated training.

Clients hold disjoint shards of the train split only. The server evaluates the
global model on the full validation split after every round, and the round that
is carried to test is chosen by validation accuracy alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedxcrop.data.dataset import build_dataset, build_loader
from fedxcrop.data.gpu_augment import make_train_augment
from fedxcrop.eval.evaluate import predict
from fedxcrop.eval.metrics import compute_metrics
from fedxcrop.fl.local import communication_bytes, train_local, weighted_average
from fedxcrop.models.factory import (
    build_model,
    count_parameters,
    get_weights,
    load_checkpoint,
    save_checkpoint,
    set_weights,
)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_client_loaders(
    cfg,
    partition: pd.DataFrame,
    seed: int = 0,
) -> dict[int, DataLoader]:
    """One training loader per client, over that client's shard of the train split."""
    loaders = {}
    for client_id, frame in partition.groupby("client_id", sort=True):
        dataset = build_dataset(
            cfg.data.root,
            frame.reset_index(drop=True),
            train=True,
            image_size=cfg.data.image_size,
            gpu_augment=cfg.data.gpu_augment,
        )
        loaders[int(client_id)] = build_loader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            seed=seed + int(client_id),
        )
    return loaders


def evaluate_global(
    model,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Score the aggregated model on a held out split."""
    output = predict(model, loader, device)
    metrics = compute_metrics(
        output["y_true"], output["y_pred"], num_classes=num_classes, class_names=class_names
    )
    metrics["loss"] = output["loss"]
    return metrics


def run_federated(
    cfg,
    partition: pd.DataFrame,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: str | Path,
    class_names: Optional[list[str]] = None,
    resume: bool = True,
    on_round_end: Optional[Callable[[int, dict], None]] = None,
) -> dict:
    """Run the configured number of federated rounds and return the history.

    Each round: the server sends the global weights to every selected client,
    each client trains locally on its own shard, and the server averages the
    returned weights in proportion to shard size. With fraction_fit at 1.0 all
    clients participate every round, so the clients are executed one after
    another on the single available accelerator. That ordering is an
    implementation detail: the aggregation is over the same per client updates
    either way, since no client sees another client's work within a round.

    State is checkpointed after every round, so a session that disconnects
    resumes from the last completed round instead of restarting.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history_path = run_dir / "history.csv"

    set_seed(cfg.seed)
    model = build_model(cfg.model.name, cfg.model.num_classes, cfg.model.pretrained)

    if cfg.federated.init == "checkpoint":
        if not cfg.federated.init_checkpoint:
            raise ValueError("federated.init is 'checkpoint' but no init_checkpoint was given")
        load_checkpoint(model, cfg.federated.init_checkpoint, map_location="cpu")
        print(
            f"initializing from {cfg.federated.init_checkpoint}. This reproduces the original "
            f"protocol and is not the corrected one.",
            flush=True,
        )
    elif cfg.federated.init != "imagenet":
        raise ValueError(
            f"unknown federated.init {cfg.federated.init!r}, expected 'imagenet' or 'checkpoint'"
        )

    model.to(device)
    global_weights = get_weights(model)

    client_loaders = build_client_loaders(cfg, partition, seed=cfg.seed)
    augmenters = {
        client_id: make_train_augment(cfg, device, seed_offset=client_id)
        for client_id in client_loaders
    }
    client_sizes = {cid: len(loader.dataset) for cid, loader in client_loaders.items()}
    n_parameters = count_parameters(model)
    bytes_per_round = communication_bytes(n_parameters, len(client_loaders))

    start_round = 1
    history: list[dict] = []
    best = {"round": -1, "val_accuracy": -1.0}

    if resume and last_path.is_file():
        state = load_checkpoint(model, last_path, map_location=device)
        global_weights = get_weights(model)
        start_round = int(state["round"]) + 1
        history = list(state.get("history", []))
        best = dict(state.get("best", best))
        print(
            f"resuming at round {start_round} (best so far: round {best['round']}, "
            f"val accuracy {best['val_accuracy']:.4f})",
            flush=True,
        )

    print(
        f"engine sequential  clients {len(client_loaders)}  "
        f"sizes {sorted(client_sizes.values())}  "
        f"strategy {cfg.federated.strategy}"
        + (f" mu {cfg.federated.mu}" if cfg.federated.strategy == "fedprox" else "")
        + f"\nrounds {cfg.federated.rounds}  local epochs {cfg.federated.local_epochs}  "
        f"{bytes_per_round / 1e6:.1f} MB per round\n",
        flush=True,
    )

    for round_number in range(start_round, cfg.federated.rounds + 1):
        started = time.time()
        client_weights, sizes, client_stats = [], [], []

        for client_id, loader in client_loaders.items():
            set_weights(model, global_weights)
            stats = train_local(
                model,
                loader,
                device,
                epochs=cfg.federated.local_epochs,
                optimizer_name=cfg.federated.optimizer,
                lr=cfg.federated.lr,
                momentum=cfg.federated.momentum,
                weight_decay=cfg.federated.weight_decay,
                strategy=cfg.federated.strategy,
                mu=cfg.federated.mu,
                augment=augmenters[client_id],
            )
            client_weights.append(get_weights(model))
            sizes.append(stats["n_images"])
            client_stats.append({"client_id": client_id, **stats})

        global_weights = weighted_average(client_weights, sizes)
        set_weights(model, global_weights)

        val_metrics = evaluate_global(
            model, val_loader, device, cfg.model.num_classes, class_names
        )

        record = {
            "round": round_number,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_n_correct": val_metrics["n_correct"],
            "val_n_total": val_metrics["n_total"],
            "mean_client_loss": float(np.mean([s["loss"] for s in client_stats])),
            "mean_client_accuracy": float(np.mean([s["accuracy"] for s in client_stats])),
            "mean_proximal_loss": float(np.mean([s["proximal_loss"] for s in client_stats])),
            "bytes_transmitted": bytes_per_round,
            "cumulative_bytes": bytes_per_round * round_number,
            "seconds": time.time() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        is_best = val_metrics["accuracy"] > best["val_accuracy"]
        if is_best:
            best = {"round": round_number, "val_accuracy": val_metrics["accuracy"]}
            save_checkpoint(
                model, best_path, extra={"round": round_number, "val_metrics": val_metrics}
            )

        save_checkpoint(
            model,
            last_path,
            extra={"round": round_number, "history": history, "best": best},
        )

        print(
            f"round {round_number:>3}/{cfg.federated.rounds}  "
            f"val loss {record['val_loss']:.4f} acc {record['val_accuracy']:.4f} "
            f"({val_metrics['n_correct']}/{val_metrics['n_total']})  "
            f"macro F1 {record['val_macro_f1']:.4f}  "
            f"client acc {record['mean_client_accuracy']:.4f}  "
            f"{record['seconds']:.0f} s"
            + ("  <- best" if is_best else ""),
            flush=True,
        )

        if on_round_end is not None:
            on_round_end(round_number, record)

    summary = {
        "engine": "sequential",
        "best": best,
        "rounds_completed": cfg.federated.rounds,
        "n_parameters": n_parameters,
        "bytes_per_round": bytes_per_round,
        "total_bytes": bytes_per_round * cfg.federated.rounds,
        "client_sizes": client_sizes,
        "strategy": cfg.federated.strategy,
        "mu": cfg.federated.mu if cfg.federated.strategy == "fedprox" else None,
        "init": cfg.federated.init,
    }
    with open(run_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return {"history": history, "best": best, "summary": summary}
