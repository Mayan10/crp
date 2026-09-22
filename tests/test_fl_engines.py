"""Tests for engine selection and the Flower wiring that can run without Ray.

The Flower engine itself needs Ray, so the parts exercised here are the ones
that do not: how the engine is chosen, how a missing Ray is reported, how the
partition reaches a client, and the metric aggregation the strategy hands to
Flower.
"""

import pandas as pd
import pytest

from fedxcrop.config import load_config
from fedxcrop.fl.flower_simulation import simulation_available


def toy_partition(num_clients: int = 3, per_client: int = 20) -> pd.DataFrame:
    rows = []
    for client_id in range(num_clients):
        for i in range(per_client):
            rows.append(
                {
                    "path": f"color/Apple___Apple_scab/img_{client_id}_{i}.JPG",
                    "label": client_id % 2,
                    "client_id": client_id,
                }
            )
    return pd.DataFrame(rows)


def test_engine_defaults_to_flower():
    cfg = load_config("configs/base.yaml")
    assert cfg.federated.engine == "flower"


def test_engine_can_be_overridden():
    cfg = load_config("configs/base.yaml", ["federated.engine=sequential"])
    assert cfg.federated.engine == "sequential"


def test_simulation_availability_reports_a_usable_reason():
    available, reason = simulation_available()
    assert isinstance(available, bool)
    if available:
        assert reason == ""
    else:
        # The message has to tell the reader what to do about it.
        assert "sequential" in reason or "flwr[simulation]" in reason


def test_flower_engine_refuses_to_run_when_ray_is_missing():
    """A missing dependency must fail loudly rather than silently switch engine."""
    available, _ = simulation_available()
    if available:
        pytest.skip("Ray is installed, so there is no missing dependency to report")

    from fedxcrop.fl.flower_simulation import run_federated_flower

    cfg = load_config("configs/base.yaml")
    with pytest.raises(RuntimeError, match="Ray"):
        run_federated_flower(cfg, toy_partition(), None, None, "runs/does_not_exist")


def test_client_factory_maps_partition_ids_to_shards():
    """Each Flower client must receive exactly its own shard, by partition id."""
    from fedxcrop.fl.flower_client import make_client_fn

    partition = toy_partition(num_clients=3, per_client=20)
    cfg = load_config("configs/base.yaml", ["model.num_classes=2", "data.num_workers=0"])

    # Inspect the closure's shard mapping without constructing a model or a
    # dataset, which would need the images on disk.
    import fedxcrop.fl.flower_client as module

    captured = {}

    class FakeClient:
        def __init__(self, client_id, frame, cfg, device):
            captured[client_id] = frame

        def to_client(self):
            return self

    original = module.FedXCropClient
    module.FedXCropClient = FakeClient
    try:
        client_fn = make_client_fn(partition, cfg, device=None)

        class FakeContext:
            node_config = {"partition-id": "1"}

        client_fn(FakeContext())
        assert set(captured) == {1}
        assert len(captured[1]) == 20
        assert set(captured[1]["client_id"]) == {1}

        class MissingContext:
            node_config = {"partition-id": "99"}

        with pytest.raises(KeyError, match="not in the committed partition"):
            client_fn(MissingContext())
    finally:
        module.FedXCropClient = original


def test_client_metric_aggregation_is_weighted_by_shard_size():
    from fedxcrop.fl.flower_strategy import average_client_metrics

    metrics = [
        (10, {"loss": 1.0, "accuracy": 0.5, "proximal_loss": 0.0}),
        (30, {"loss": 2.0, "accuracy": 0.9, "proximal_loss": 0.4}),
    ]
    result = average_client_metrics(metrics)
    # A client holding three quarters of the data carries three quarters of the weight.
    assert result["loss"] == pytest.approx((10 * 1.0 + 30 * 2.0) / 40)
    assert result["accuracy"] == pytest.approx((10 * 0.5 + 30 * 0.9) / 40)
    assert result["proximal_loss"] == pytest.approx((10 * 0.0 + 30 * 0.4) / 40)


def test_client_metric_aggregation_handles_no_examples():
    from fedxcrop.fl.flower_strategy import average_client_metrics

    assert average_client_metrics([(0, {"loss": 1.0, "accuracy": 1.0, "proximal_loss": 0.0})]) == {}


def test_unknown_engine_is_rejected_by_the_runner():
    """The runner must reject an unknown engine rather than pick one."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_federated.py", "--config", "configs/fedavg_iid.yaml",
         "--smoke", "--set", "federated.engine=nonsense"],
        cwd=repo, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode != 0
    assert "unknown federated.engine" in (result.stdout + result.stderr)
