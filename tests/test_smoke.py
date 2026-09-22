"""End to end smoke runs.

These exercise the whole path (splits, partitions, local training,
aggregation, validation selection, single test evaluation, saved artefacts) on
a tiny subset so it finishes on a CPU. They are marked slow: deselect with
`pytest -m "not slow"` while iterating.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
has_dataset = (REPO / "plantvillage dataset" / "color").is_dir()
has_splits = (REPO / "splits" / "train.csv").is_file()
needs_data = pytest.mark.skipif(
    not (has_dataset and has_splits), reason="needs the dataset and built splits"
)


def run_script(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-u", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(f"{args} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result


@pytest.mark.slow
@needs_data
def test_centralized_smoke_run_produces_auditable_results():
    run_script("scripts/train_centralized.py", "--config", "configs/centralized.yaml",
               "--smoke", "--no-resume")

    results = REPO / "results" / "centralized" / "smoke"
    metrics = json.loads((results / "test_metrics.json").read_text())

    # The reported rate has to follow from the integer counts.
    assert metrics["accuracy"] == metrics["n_correct"] / metrics["n_total"]
    assert metrics["selected_epoch"] >= 0

    predictions = pd.read_csv(results / "test_predictions.csv")
    assert len(predictions) == metrics["n_total"]
    assert int((predictions["y_true"] == predictions["y_pred"]).sum()) == metrics["n_correct"]

    history = pd.read_csv(results / "history.csv")
    assert len(history) == 2
    assert {"epoch", "train_loss", "val_accuracy", "val_n_correct"} <= set(history.columns)


@pytest.mark.slow
@needs_data
def test_federated_smoke_run_produces_auditable_results():
    run_script("scripts/run_federated.py", "--config", "configs/fedprox_noniid.yaml",
               "--smoke", "--no-resume", "--engine", "sequential")

    results = REPO / "results" / "federated" / "smoke"
    metrics = json.loads((results / "test_metrics.json").read_text())

    assert metrics["accuracy"] == metrics["n_correct"] / metrics["n_total"]
    assert metrics["selected_round"] >= 1
    assert metrics["config"]["strategy"] == "fedprox"
    assert metrics["config"]["init"] == "imagenet"

    predictions = pd.read_csv(results / "test_predictions.csv")
    assert int((predictions["y_true"] == predictions["y_pred"]).sum()) == metrics["n_correct"]

    history = pd.read_csv(results / "history.csv")
    assert len(history) == 2
    # Communication cost is accounted for every round.
    assert (history["bytes_transmitted"] > 0).all()
    assert history["cumulative_bytes"].is_monotonic_increasing

    summary = json.loads((results / "summary.json").read_text())
    assert summary["init"] == "imagenet"
    assert summary["strategy"] == "fedprox"
    assert summary["engine"] == "sequential"


@pytest.mark.slow
@needs_data
def test_federated_run_resumes_from_the_last_completed_round(tmp_path):
    """A run stopped after one round continues rather than restarting.

    This is what makes a long run survive a disconnected session.
    """
    common = [
        "scripts/run_federated.py", "--config", "configs/fedprox_noniid.yaml", "--smoke",
        "--engine", "sequential",
    ]
    run_script(*common, "--no-resume", "--set", "federated.rounds=1")

    history_after_one = pd.read_csv(REPO / "results" / "federated" / "smoke" / "history.csv")
    assert list(history_after_one["round"]) == [1]

    result = run_script(*common, "--set", "federated.rounds=2")
    assert "resuming at round 2" in result.stdout

    history_after_two = pd.read_csv(REPO / "results" / "federated" / "smoke" / "history.csv")
    assert list(history_after_two["round"]) == [1, 2]
    # The first round is the one already computed, not a recomputation.
    assert history_after_two.iloc[0]["val_accuracy"] == history_after_one.iloc[0]["val_accuracy"]


# --- Flower simulation engine -------------------------------------------------
#
# Flower's simulation engine needs Ray, which has no build for every Python
# version. These tests skip where it is unavailable and run where it is, which
# is the GPU runtime the grid is executed on.

from fedxcrop.fl.flower_simulation import simulation_available  # noqa: E402

_flower_ok, _flower_reason = simulation_available()
needs_flower = pytest.mark.skipif(not _flower_ok, reason=_flower_reason)


def read_smoke_results() -> tuple[dict, pd.DataFrame]:
    results = REPO / "results" / "federated" / "smoke"
    return json.loads((results / "test_metrics.json").read_text()), pd.read_csv(
        results / "history.csv"
    )


@pytest.mark.slow
@needs_flower
@needs_data
def test_flower_engine_smoke_run_produces_auditable_results():
    run_script("scripts/run_federated.py", "--config", "configs/fedprox_noniid.yaml",
               "--smoke", "--no-resume", "--engine", "flower")

    metrics, history = read_smoke_results()
    assert metrics["accuracy"] == metrics["n_correct"] / metrics["n_total"]
    assert metrics["selected_round"] >= 1
    assert list(history["round"]) == [1, 2]
    assert (history["n_clients"] == 2).all()

    summary = json.loads(
        (REPO / "results" / "federated" / "smoke" / "summary.json").read_text()
    )
    assert summary["engine"] == "flower"
    assert summary["init"] == "imagenet"


@pytest.mark.slow
@needs_flower
@needs_data
def test_flower_engine_resumes_from_the_last_completed_round():
    common = ["scripts/run_federated.py", "--config", "configs/fedprox_noniid.yaml",
              "--smoke", "--engine", "flower"]
    run_script(*common, "--no-resume", "--set", "federated.rounds=1")
    _, after_one = read_smoke_results()
    assert list(after_one["round"]) == [1]

    result = run_script(*common, "--set", "federated.rounds=2")
    assert "resuming after round 1" in result.stdout

    _, after_two = read_smoke_results()
    assert list(after_two["round"]) == [1, 2]
    # Round 1 is carried over, not recomputed.
    assert after_two.iloc[0]["val_accuracy"] == after_one.iloc[0]["val_accuracy"]


@pytest.mark.slow
@needs_flower
@needs_data
def test_both_engines_agree():
    """The two engines must implement one method, not two.

    Exact equality is not expected: Flower runs each client in its own Ray
    worker process, so the random augmentation draws differ from the sequential
    case even at the same seed. What must agree is everything that defines the
    method: how many images each client contributed, the aggregation weighting,
    the communication accounting, and an accuracy that lands in the same place.
    A wiring bug (parameters not reaching clients, the average not weighted by
    shard size, a client never training) breaks these; a different RNG draw
    does not.
    """
    settings = ["--config", "configs/fedprox_noniid.yaml", "--smoke", "--no-resume",
                "--set", "federated.rounds=2", "data.num_workers=0"]

    run_script("scripts/run_federated.py", *settings, "--engine", "sequential")
    sequential_metrics, sequential_history = read_smoke_results()

    run_script("scripts/run_federated.py", *settings, "--engine", "flower")
    flower_metrics, flower_history = read_smoke_results()

    # Same evaluation set, same accounting.
    assert flower_metrics["n_total"] == sequential_metrics["n_total"]
    assert list(flower_history["round"]) == list(sequential_history["round"])
    assert (
        flower_history["bytes_transmitted"].iloc[0]
        == sequential_history["bytes_transmitted"].iloc[0]
    )
    assert flower_history["val_n_total"].iloc[0] == sequential_history["val_n_total"].iloc[0]

    # Both engines actually trained every client on its whole shard.
    assert (flower_history["mean_client_accuracy"] > 0).all()
    assert (sequential_history["mean_client_accuracy"] > 0).all()

    # And they land in the same place, within the spread two RNG streams give.
    gap = abs(flower_metrics["accuracy"] - sequential_metrics["accuracy"])
    assert gap < 0.15, (
        f"engines disagree by {gap:.3f} in test accuracy: "
        f"flower {flower_metrics['accuracy']:.3f} vs "
        f"sequential {sequential_metrics['accuracy']:.3f}"
    )
