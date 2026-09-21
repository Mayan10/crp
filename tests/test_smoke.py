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
               "--smoke", "--no-resume")

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


@pytest.mark.slow
@needs_data
def test_federated_run_resumes_from_the_last_completed_round(tmp_path):
    """A run stopped after one round continues rather than restarting.

    This is what makes a long run survive a disconnected session.
    """
    common = [
        "scripts/run_federated.py", "--config", "configs/fedprox_noniid.yaml", "--smoke",
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
