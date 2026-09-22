"""Flower strategy that evaluates, logs, and checkpoints every round.

Aggregation is plain FedAvg, weighted by shard size, over the whole state dict
including batch norm running statistics. FedProx differs from FedAvg only in
the client's local objective, so both strategies use this same aggregation:
that is what the FedProx paper specifies, and it is why `federated.strategy`
changes the client and not the server.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedxcrop.fl.simulation import evaluate_global
from fedxcrop.models.factory import save_checkpoint, set_weights


class TrackedFedAvg(FedAvg):
    """FedAvg that scores the aggregated model on the full validation split.

    After each aggregation the global model is evaluated on the entire
    validation split, the round is appended to `history.csv`, and the run is
    checkpointed. The best validation round is kept separately, and it is the
    only checkpoint the test split is ever applied to.

    `round_offset` supports resuming: Flower always counts rounds from one, so
    a resumed run adds the number of rounds already completed to every round it
    reports.
    """

    def __init__(
        self,
        cfg,
        val_loader,
        device: torch.device,
        run_dir: str | Path,
        model,
        class_names: Optional[list[str]] = None,
        history: Optional[list[dict]] = None,
        best: Optional[dict] = None,
        round_offset: int = 0,
        bytes_per_round: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.model = model
        self.class_names = class_names
        self.history: list[dict] = list(history or [])
        self.best = dict(best or {"round": -1, "val_accuracy": -1.0})
        self.round_offset = round_offset
        self.bytes_per_round = bytes_per_round
        self._round_started = time.time()

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if failures:
            # A dropped client silently changes what was averaged, so it is
            # raised rather than logged and skipped.
            raise RuntimeError(
                f"round {server_round + self.round_offset}: {len(failures)} client(s) failed: "
                f"{failures[0]}"
            )

        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            raise RuntimeError(f"round {server_round + self.round_offset}: aggregation returned nothing")

        absolute_round = server_round + self.round_offset
        weights = parameters_to_ndarrays(parameters)
        set_weights(self.model, weights)

        val_metrics = evaluate_global(
            self.model, self.val_loader, self.device, self.cfg.model.num_classes, self.class_names
        )

        client_metrics = [res.metrics for _, res in results]
        record = {
            "round": absolute_round,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_n_correct": val_metrics["n_correct"],
            "val_n_total": val_metrics["n_total"],
            "mean_client_loss": float(np.mean([m["loss"] for m in client_metrics])),
            "mean_client_accuracy": float(np.mean([m["accuracy"] for m in client_metrics])),
            "mean_proximal_loss": float(np.mean([m["proximal_loss"] for m in client_metrics])),
            "n_clients": len(results),
            "bytes_transmitted": self.bytes_per_round,
            "cumulative_bytes": self.bytes_per_round * absolute_round,
            "seconds": time.time() - self._round_started,
        }
        self._round_started = time.time()
        self.history.append(record)
        pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)

        is_best = val_metrics["accuracy"] > self.best["val_accuracy"]
        if is_best:
            self.best = {"round": absolute_round, "val_accuracy": val_metrics["accuracy"]}
            save_checkpoint(
                self.model, self.run_dir / "best.pt",
                extra={"round": absolute_round, "val_metrics": val_metrics},
            )

        save_checkpoint(
            self.model, self.run_dir / "last.pt",
            extra={"round": absolute_round, "history": self.history, "best": self.best},
        )

        print(
            f"round {absolute_round:>3}/{self.cfg.federated.rounds}  "
            f"val loss {record['val_loss']:.4f} acc {record['val_accuracy']:.4f} "
            f"({val_metrics['n_correct']}/{val_metrics['n_total']})  "
            f"macro F1 {record['val_macro_f1']:.4f}  "
            f"client acc {record['mean_client_accuracy']:.4f}  "
            f"{record['seconds']:.0f} s"
            + ("  <- best" if is_best else ""),
            flush=True,
        )
        return parameters, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        """Server side evaluation is done in aggregate_fit, so nothing here."""
        return None


def average_client_metrics(metrics: list[tuple[int, dict]]) -> dict:
    """Shard size weighted mean of the client training metrics."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        key: float(sum(n * m[key] for n, m in metrics) / total)
        for key in ("loss", "accuracy", "proximal_loss")
        if all(key in m for _, m in metrics)
    }
