"""Centralized training, the baseline every federated run is measured against.

Trained from ImageNet initialization on the full train split. The epoch that
is carried forward is chosen by validation accuracy, and the test split is
touched exactly once, at the end, with that chosen checkpoint. Selecting on
validation and reporting on test is what keeps the reported number an estimate
of generalization rather than a number the selection procedure optimized.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedxcrop.data.gpu_augment import make_train_augment
from fedxcrop.eval.evaluate import predict, save_predictions
from fedxcrop.eval.metrics import compute_metrics, save_metrics
from fedxcrop.models.factory import build_model, load_checkpoint, save_checkpoint


def set_seed(seed: int) -> None:
    """Seed every generator that affects training."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, name: str, lr: float, momentum: float = 0.9):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    raise ValueError(f"unsupported optimizer {name!r}, expected 'adam' or 'sgd'")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    log_every: int = 0,
    augment=None,
) -> dict:
    """One pass over the training data, returning loss and accuracy on it.

    `augment` applies the colour jitter and normalization on the accelerator
    when the loader hands over uint8 batches. When it is None the loader has
    already done that work on the CPU.
    """
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    loss_total, n_correct, n_seen = 0.0, 0, 0
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if augment is not None:
            images = augment(images)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_total += float(loss.detach().item()) * labels.numel()
        n_correct += int((logits.argmax(dim=1) == labels).sum().item())
        n_seen += labels.numel()

        if log_every and step % log_every == 0:
            print(
                f"    step {step:>5}/{len(loader)}  "
                f"loss {loss_total / max(n_seen, 1):.4f}  "
                f"acc {n_correct / max(n_seen, 1):.4f}",
                flush=True,
            )

    return {
        "loss": loss_total / max(n_seen, 1),
        "accuracy": n_correct / max(n_seen, 1),
        "n_correct": n_correct,
        "n_total": n_seen,
    }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[list[str]] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Metrics for a model on one split, plus the raw predictions."""
    output = predict(model, loader, device)
    metrics = compute_metrics(
        output["y_true"], output["y_pred"], num_classes=num_classes, class_names=class_names
    )
    metrics["loss"] = output["loss"]
    return metrics, output["y_true"], output["y_pred"]


def train_centralized(
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: str | Path,
    class_names: Optional[list[str]] = None,
    resume: bool = True,
    log_every: int = 0,
) -> dict:
    """Train for the configured number of epochs, tracking the best val epoch.

    After every epoch the model is checkpointed to `last.pt` together with the
    optimizer and scheduler state and the history so far, so a run interrupted
    by a disconnected session resumes exactly where it stopped rather than
    restarting. A separate `best.pt` holds the best validation epoch.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history_path = run_dir / "history.csv"

    set_seed(cfg.seed)
    model = build_model(cfg.model.name, cfg.model.num_classes, cfg.model.pretrained)
    model.to(device)
    augment = make_train_augment(cfg, device)

    optimizer = build_optimizer(model, cfg.centralized.optimizer, cfg.centralized.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.centralized.lr_step_size, gamma=cfg.centralized.lr_gamma
    )

    start_epoch = 0
    history: list[dict] = []
    best = {"epoch": -1, "val_accuracy": -1.0}

    if resume and last_path.is_file():
        state = load_checkpoint(model, last_path, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        history = list(state.get("history", []))
        best = dict(state.get("best", best))
        print(f"resuming from epoch {start_epoch} (best so far: epoch {best['epoch']}, "
              f"val accuracy {best['val_accuracy']:.4f})", flush=True)

    for epoch in range(start_epoch, cfg.centralized.epochs):
        started = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, log_every, augment=augment
        )
        val_metrics, _, _ = evaluate_split(
            model, val_loader, device, cfg.model.num_classes, class_names
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_n_correct": val_metrics["n_correct"],
            "val_n_total": val_metrics["n_total"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        is_best = val_metrics["accuracy"] > best["val_accuracy"]
        if is_best:
            best = {"epoch": epoch, "val_accuracy": val_metrics["accuracy"]}
            save_checkpoint(model, best_path, extra={"epoch": epoch, "val_metrics": val_metrics})

        save_checkpoint(
            model,
            last_path,
            extra={
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best": best,
            },
        )

        print(
            f"epoch {epoch:>2}/{cfg.centralized.epochs - 1}  "
            f"train loss {record['train_loss']:.4f} acc {record['train_accuracy']:.4f}  "
            f"val loss {record['val_loss']:.4f} acc {record['val_accuracy']:.4f} "
            f"({val_metrics['n_correct']}/{val_metrics['n_total']})  "
            f"{record['seconds']:.0f} s"
            + ("  <- best" if is_best else ""),
            flush=True,
        )

    with open(run_dir / "best.json", "w") as handle:
        json.dump(best, handle, indent=2)
    return {"history": history, "best": best, "best_checkpoint": str(best_path)}


def evaluate_on_test(
    cfg,
    test_loader: DataLoader,
    test_frame: pd.DataFrame,
    device: torch.device,
    run_dir: str | Path,
    results_dir: str | Path,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Score the best validation checkpoint on the test split, once.

    This is the only place the test split is read during the centralized
    experiment.
    """
    run_dir, results_dir = Path(run_dir), Path(results_dir)
    model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False)
    state = load_checkpoint(model, run_dir / "best.pt", map_location=device)
    model.to(device)

    metrics, y_true, y_pred = evaluate_split(
        model, test_loader, device, cfg.model.num_classes, class_names
    )
    metrics["selected_epoch"] = int(state.get("epoch", -1))
    metrics["selection"] = "best validation accuracy, test evaluated once"

    save_metrics(metrics, results_dir / "test_metrics.json")
    save_predictions(test_frame, y_true, y_pred, results_dir / "test_predictions.csv")

    print(
        f"\ntest: accuracy {metrics['accuracy']:.4f} "
        f"({metrics['n_correct']}/{metrics['n_total']}), "
        f"macro F1 {metrics['macro_f1']:.4f}, "
        f"from epoch {metrics['selected_epoch']}",
        flush=True,
    )
    return metrics
