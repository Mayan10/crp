"""Flower client.

A thin wrapper around `train_local`, so the client the Flower simulation runs
performs exactly the same local optimization as the sequential engine. Anything
about the federated method itself (the FedProx term, the optimizer policy)
lives in `fedxcrop.fl.local`, not here.
"""

from __future__ import annotations

from typing import Callable

import flwr as fl
import pandas as pd
import torch
from flwr.common import Context
from flwr.common.constant import PARTITION_ID_KEY

from fedxcrop.data.dataset import build_dataset, build_loader
from fedxcrop.data.gpu_augment import make_train_augment
from fedxcrop.fl.local import train_local
from fedxcrop.models.factory import build_model, get_weights, set_weights


class FedXCropClient(fl.client.NumPyClient):
    """One simulated farm, holding a fixed shard of the train split."""

    def __init__(self, client_id: int, frame: pd.DataFrame, cfg, device: torch.device):
        self.client_id = client_id
        self.frame = frame
        self.cfg = cfg
        self.device = device
        # Built without pretrained weights: the server sends the parameters to
        # start from at the beginning of every round, so downloading ImageNet
        # weights inside each client would be wasted work.
        self.model = build_model(cfg.model.name, cfg.model.num_classes, pretrained=False)
        self.augment = make_train_augment(cfg, device, seed_offset=client_id)
        self.loader = build_loader(
            build_dataset(cfg.data.root, frame, train=True, image_size=cfg.data.image_size,
                          gpu_augment=cfg.data.gpu_augment),
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            seed=cfg.seed + client_id,
        )

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        """Train locally for the configured number of epochs and return the weights.

        The number of examples returned is the shard size, which is what the
        server weights the average by.
        """
        set_weights(self.model, parameters)
        stats = train_local(
            self.model,
            self.loader,
            self.device,
            epochs=self.cfg.federated.local_epochs,
            optimizer_name=self.cfg.federated.optimizer,
            lr=self.cfg.federated.lr,
            momentum=self.cfg.federated.momentum,
            weight_decay=self.cfg.federated.weight_decay,
            strategy=self.cfg.federated.strategy,
            mu=self.cfg.federated.mu,
            augment=self.augment,
        )
        metrics = {
            "client_id": self.client_id,
            "loss": stats["loss"],
            "accuracy": stats["accuracy"],
            "proximal_loss": stats["proximal_loss"],
        }
        return get_weights(self.model), stats["n_images"], metrics

    def evaluate(self, parameters, config):
        """Not used: the server evaluates the global model on the full val split.

        Averaging per client validation scores would measure the model against
        each client's own label distribution, which is not the quantity of
        interest and is not comparable across partitions.
        """
        return 0.0, 0, {}


def make_client_fn(partition: pd.DataFrame, cfg, device: torch.device) -> Callable:
    """Build the client factory Flower calls, one call per participating client.

    Flower passes a Context whose node config carries the partition id, which
    indexes the committed client assignment.
    """
    shards = {
        int(client_id): frame.reset_index(drop=True)
        for client_id, frame in partition.groupby("client_id", sort=True)
    }

    def client_fn(context: Context) -> fl.client.Client:
        partition_id = int(context.node_config[PARTITION_ID_KEY])
        if partition_id not in shards:
            raise KeyError(
                f"partition id {partition_id} is not in the committed partition, "
                f"which has clients {sorted(shards)}"
            )
        client = FedXCropClient(partition_id, shards[partition_id], cfg, device)
        return client.to_client()

    return client_fn
