"""Running the federated rounds through the Flower simulation engine.

Flower drives the rounds and the client lifecycle; the local objective and the
aggregation weighting come from the same modules the sequential engine uses, so
the two engines implement one method rather than two.

Ray, which Flower's simulation engine requires, has no build for every Python
version. Where it is unavailable this module raises with an explanation and
`fedxcrop.fl.simulation` provides the equivalent sequential engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedxcrop.fl.local import communication_bytes
from fedxcrop.models.factory import (
    build_model,
    count_parameters,
    get_weights,
    load_checkpoint,
)


def simulation_available() -> tuple[bool, str]:
    """Whether Flower's simulation engine can run in this interpreter."""
    try:
        import ray  # noqa: F401
    except ImportError:
        import sys

        return False, (
            f"Flower's simulation engine needs Ray, which is not installed for "
            f"Python {sys.version_info.major}.{sys.version_info.minor}. "
            f"Install it with `pip install 'flwr[simulation]'`, or run with "
            f"`--engine sequential`."
        )
    try:
        from flwr.simulation import start_simulation  # noqa: F401
    except ImportError as error:
        return False, f"Flower simulation could not be imported: {error}"
    return True, ""


def run_federated_flower(
    cfg,
    partition: pd.DataFrame,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: str | Path,
    class_names: Optional[list[str]] = None,
    resume: bool = True,
) -> dict:
    """Run the configured rounds through Flower, returning the history.

    Client resources are set so that one client holds the whole GPU at a time.
    With full participation the clients of a round are independent, so running
    them one after another on a single accelerator changes the wall time, not
    the result.
    """
    available, reason = simulation_available()
    if not available:
        raise RuntimeError(reason)

    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerConfig
    from flwr.simulation import start_simulation

    from fedxcrop.fl.flower_client import make_client_fn
    from fedxcrop.fl.flower_strategy import TrackedFedAvg, average_client_metrics
    from fedxcrop.fl.simulation import set_seed

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

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

    history: list[dict] = []
    best = {"round": -1, "val_accuracy": -1.0}
    completed = 0

    if resume and last_path.is_file():
        state = load_checkpoint(model, last_path, map_location="cpu")
        completed = int(state["round"])
        history = list(state.get("history", []))
        best = dict(state.get("best", best))
        print(
            f"resuming after round {completed} (best so far: round {best['round']}, "
            f"val accuracy {best['val_accuracy']:.4f})",
            flush=True,
        )

    remaining = cfg.federated.rounds - completed
    num_clients = int(partition["client_id"].nunique())
    client_sizes = partition.groupby("client_id").size().to_dict()
    n_parameters = count_parameters(model)
    bytes_per_round = communication_bytes(n_parameters, num_clients)

    print(
        f"engine flower  clients {num_clients}  sizes {sorted(client_sizes.values())}  "
        f"strategy {cfg.federated.strategy}"
        + (f" mu {cfg.federated.mu}" if cfg.federated.strategy == "fedprox" else "")
        + f"\nrounds {cfg.federated.rounds} ({remaining} remaining)  "
        f"local epochs {cfg.federated.local_epochs}  "
        f"{bytes_per_round / 1e6:.1f} MB per round\n",
        flush=True,
    )

    if remaining <= 0:
        print("all rounds already completed")
        return {"history": history, "best": best, "summary": _summary(
            cfg, best, num_clients, client_sizes, n_parameters, bytes_per_round)}

    model.to(device)
    strategy = TrackedFedAvg(
        cfg=cfg,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        model=model,
        class_names=class_names,
        history=history,
        best=best,
        round_offset=completed,
        bytes_per_round=bytes_per_round,
        fraction_fit=cfg.federated.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(get_weights(model)),
        fit_metrics_aggregation_fn=average_client_metrics,
        accept_failures=False,
    )

    use_gpu = device.type == "cuda"
    client_resources = {"num_cpus": 1, "num_gpus": 1.0 if use_gpu else 0.0}

    start_simulation(
        client_fn=make_client_fn(partition, cfg, device),
        num_clients=num_clients,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=remaining),
        strategy=strategy,
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
    )

    summary = _summary(
        cfg, strategy.best, num_clients, client_sizes, n_parameters, bytes_per_round
    )
    with open(run_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return {"history": strategy.history, "best": strategy.best, "summary": summary}


def _summary(
    cfg, best: dict, num_clients: int, client_sizes: dict, n_parameters: int, bytes_per_round: int
) -> dict:
    return {
        "engine": "flower",
        "best": best,
        "rounds_completed": cfg.federated.rounds,
        "n_parameters": n_parameters,
        "bytes_per_round": bytes_per_round,
        "total_bytes": bytes_per_round * cfg.federated.rounds,
        "client_sizes": {str(k): int(v) for k, v in client_sizes.items()},
        "strategy": cfg.federated.strategy,
        "mu": cfg.federated.mu if cfg.federated.strategy == "fedprox" else None,
        "init": cfg.federated.init,
    }
