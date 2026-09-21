"""Tests for assembling result tables from saved run files."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fedxcrop.eval.tables import (
    add_bootstrap_intervals,
    configuration_label,
    discover_runs,
    main_results_table,
    pairwise_significance,
    save_table,
)


def write_run(
    results_dir: Path,
    family: str,
    run: str,
    y_true: list[int],
    y_pred: list[int],
    config: dict | None = None,
) -> Path:
    """Create a results directory that looks like a finished run."""
    from fedxcrop.eval.metrics import compute_metrics, save_metrics

    run_dir = results_dir / family / run
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    if config is not None:
        metrics["config"] = config
    save_metrics(metrics, run_dir / "test_metrics.json")

    pd.DataFrame(
        {
            "path": [f"img_{i}.JPG" for i in range(len(y_true))],
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": [a == b for a, b in zip(y_true, y_pred)],
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)
    return run_dir


@pytest.fixture
def results_dir(tmp_path):
    base = [0, 1, 2] * 20  # 60 images, 3 classes
    for seed in (0, 1, 2):
        perfect = list(base)
        # A different number of mistakes per seed, so the spread is nonzero.
        wrong = list(base)
        for i in range(seed + 1):
            wrong[i] = (wrong[i] + 1) % 3
        write_run(
            tmp_path, "centralized", f"seed{seed}", perfect, wrong,
            config={"strategy": "centralized", "scheme": "centralized", "seed": seed},
        )
        write_run(
            tmp_path, "federated", f"fedavg_iid_K5_seed{seed}", perfect, wrong,
            config={"strategy": "fedavg", "scheme": "iid", "alpha": None,
                    "num_clients": 5, "mu": None, "rounds": 30, "init": "imagenet",
                    "seed": seed},
        )
    return tmp_path


def test_discover_runs_finds_every_finished_run(results_dir):
    runs = discover_runs(results_dir)
    assert len(runs) == 6
    assert set(runs["family"]) == {"centralized", "federated"}
    assert set(runs["seed"]) == {0, 1, 2}


def test_discover_runs_skips_smoke_runs(results_dir):
    write_run(results_dir, "federated", "smoke", [0, 1], [0, 1],
              config={"strategy": "fedavg", "seed": 0})
    # The smoke directory itself is named "smoke" and must not reach a table.
    runs = discover_runs(results_dir)
    assert "smoke" not in set(runs["run"])


def test_discover_runs_carries_the_auditable_counts(results_dir):
    runs = discover_runs(results_dir)
    for _, row in runs.iterrows():
        assert row["accuracy"] == pytest.approx(row["n_correct"] / row["n_total"])
        assert row["n_total"] == 60


def test_configuration_labels_are_readable():
    assert configuration_label({"strategy": "centralized"}) == "centralized"
    assert configuration_label({"strategy": "fedavg", "scheme": "iid"}) == "fedavg IID"
    assert configuration_label(
        {"strategy": "fedavg", "scheme": "dirichlet", "alpha": 0.1}
    ) == "fedavg alpha=0.1"
    assert configuration_label(
        {"strategy": "fedprox", "scheme": "dirichlet", "alpha": 0.5, "mu": 0.01}
    ) == "fedprox alpha=0.5 mu=0.01"
    assert "legacy init" in configuration_label(
        {"strategy": "fedavg", "scheme": "dirichlet", "alpha": 0.5, "init": "checkpoint"}
    )


def test_main_results_table_aggregates_over_seeds(results_dir):
    runs = discover_runs(results_dir)
    table = main_results_table(runs)

    assert len(table) == 2  # centralized and fedavg IID
    for _, row in table.iterrows():
        assert row["n_seeds"] == 3
        assert row["seeds"] == [0, 1, 2]
        assert row["accuracy_std"] > 0  # the seeds genuinely differ
        assert len(row["accuracy_values"]) == 3
        assert row["n_total"] == 60


def test_main_results_table_mean_matches_the_underlying_runs(results_dir):
    runs = discover_runs(results_dir)
    table = main_results_table(runs)
    centralized = runs[runs["strategy"] == "centralized"]
    row = table[table["label"] == "centralized"].iloc[0]
    assert row["accuracy_mean"] == pytest.approx(centralized["accuracy"].mean())


def test_bootstrap_intervals_bracket_the_point_estimate(results_dir):
    runs = discover_runs(results_dir)
    table = add_bootstrap_intervals(main_results_table(runs), runs, num_classes=3,
                                    n_resamples=100, seed=0)
    for _, row in table.iterrows():
        assert row["accuracy_ci_low"] <= row["accuracy_point"] <= row["accuracy_ci_high"]
        assert row["bootstrap_n_total"] == 60


def test_pairwise_significance_compares_the_same_images(results_dir):
    runs = discover_runs(results_dir)
    result = pairwise_significance(runs, [("fedavg IID", "centralized")])
    assert len(result) == 1
    row = result.iloc[0]
    # The two runs here make identical predictions, so there is nothing to separate.
    assert row["n_discordant"] == 0
    assert row["p_value"] == 1.0
    assert not row["significant_at_0.05"]


def test_pairwise_significance_detects_a_real_difference(tmp_path):
    y_true = [0] * 100
    strong = [0] * 100
    weak = [1] * 30 + [0] * 70
    write_run(tmp_path, "federated", "a_seed0", y_true, strong,
              config={"strategy": "fedavg", "scheme": "iid", "seed": 0})
    write_run(tmp_path, "federated", "b_seed0", y_true, weak,
              config={"strategy": "fedprox", "scheme": "dirichlet", "alpha": 0.1,
                      "mu": 0.01, "seed": 0})

    runs = discover_runs(tmp_path)
    result = pairwise_significance(runs, [("fedavg IID", "fedprox alpha=0.1 mu=0.01")])
    row = result.iloc[0]
    assert row["only_a_correct"] == 30
    assert row["p_value"] < 0.001
    assert row["significant_at_0.05"]


def test_save_table_writes_csv_and_markdown(tmp_path):
    table = pd.DataFrame({"configuration": ["centralized"], "accuracy": [0.9912]})
    csv_path, md_path = save_table(table, tmp_path, "demo")
    assert csv_path.is_file() and md_path.is_file()
    assert "centralized" in md_path.read_text()
    assert pd.read_csv(csv_path).iloc[0]["accuracy"] == pytest.approx(0.9912)


def test_empty_results_give_an_empty_table(tmp_path):
    assert discover_runs(tmp_path).empty
    assert main_results_table(pd.DataFrame()).empty
