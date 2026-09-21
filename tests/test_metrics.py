"""Tests for metrics and statistics, on inputs whose answers are known by hand."""

import json

import numpy as np
import pytest

from fedxcrop.eval.metrics import (
    accuracy,
    compute_metrics,
    load_metrics,
    save_metrics,
    worst_classes,
)
from fedxcrop.eval.stats import (
    aggregate_seeds,
    bootstrap_ci,
    bootstrap_metrics,
    format_ci,
    format_mean_std,
    mcnemar,
    wilcoxon,
)


def test_accuracy_on_a_hand_counted_example():
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 1]
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 5)


def test_perfect_prediction_gives_one_everywhere():
    y_true = [0, 1, 2, 0, 1, 2]
    metrics = compute_metrics(y_true, y_true, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["n_correct"] == 6
    assert metrics["n_total"] == 6


def test_counts_are_always_present_and_consistent_with_the_rate():
    y_true = [0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 1, 0, 1, 2, 1]
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    assert metrics["n_correct"] == 5
    assert metrics["n_total"] == 7
    assert metrics["accuracy"] == pytest.approx(5 / 7)
    # The rate has to be reproducible from the integer counts.
    assert metrics["accuracy"] == metrics["n_correct"] / metrics["n_total"]


def test_macro_and_weighted_differ_on_an_imbalanced_set():
    # Class 0 has 8 images and is perfect, class 1 has 2 and is entirely wrong.
    y_true = [0] * 8 + [1] * 2
    y_pred = [0] * 8 + [0] * 2
    metrics = compute_metrics(y_true, y_pred, num_classes=2)

    assert metrics["accuracy"] == pytest.approx(0.8)
    # Class 1 recall is 0, class 0 recall is 1, so macro recall is 0.5.
    assert metrics["macro_recall"] == pytest.approx(0.5)
    # Weighted recall equals accuracy by construction.
    assert metrics["weighted_recall"] == pytest.approx(0.8)
    assert metrics["macro_recall"] < metrics["weighted_recall"]


def test_per_class_values_match_a_hand_computation():
    # Class 0: 2 true, both predicted 0, plus one class 1 predicted as 0.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_metrics(y_true, y_pred, num_classes=2)

    class_0 = metrics["per_class"]["0"]
    assert class_0["recall"] == pytest.approx(1.0)
    assert class_0["precision"] == pytest.approx(2 / 3)
    assert class_0["f1"] == pytest.approx(2 * (2 / 3) / (1 + 2 / 3))
    assert class_0["support"] == 2

    class_1 = metrics["per_class"]["1"]
    assert class_1["recall"] == pytest.approx(0.5)
    assert class_1["precision"] == pytest.approx(1.0)
    assert class_1["support"] == 2


def test_confusion_matrix_is_the_hand_written_one():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 0]
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_absent_classes_are_listed_rather_than_silently_averaged():
    y_true = [0, 0, 1]
    y_pred = [0, 0, 1]
    metrics = compute_metrics(y_true, y_pred, num_classes=4, class_names=["a", "b", "c", "d"])
    assert metrics["classes_absent"] == ["c", "d"]
    assert metrics["per_class"]["c"]["support"] == 0


def test_class_names_are_used_when_given():
    metrics = compute_metrics([0, 1], [0, 1], num_classes=2, class_names=["apple", "grape"])
    assert set(metrics["per_class"]) == {"apple", "grape"}


def test_wrong_number_of_class_names_is_rejected():
    with pytest.raises(ValueError, match="expected 3 class names"):
        compute_metrics([0, 1], [0, 1], num_classes=3, class_names=["a", "b"])


def test_length_mismatch_and_empty_input_are_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics([0, 1], [0], num_classes=2)
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [], num_classes=2)


def test_saving_metrics_without_counts_is_refused(tmp_path):
    with pytest.raises(ValueError, match="n_correct"):
        save_metrics({"accuracy": 0.99}, tmp_path / "bad.json")


def test_metrics_round_trip_through_json(tmp_path):
    metrics = compute_metrics([0, 1, 1], [0, 1, 0], num_classes=2)
    path = save_metrics(metrics, tmp_path / "m.json")
    assert load_metrics(path) == metrics


def test_worst_classes_are_ordered_and_skip_absent_classes():
    y_true = [0] * 5 + [1] * 5 + [2] * 5
    y_pred = [0] * 5 + [0] * 5 + [2] * 5  # class 1 is entirely wrong
    metrics = compute_metrics(y_true, y_pred, num_classes=4)
    worst = worst_classes(metrics, n=2)
    assert worst[0]["class_name"] == "1"
    assert worst[0]["f1"] == 0.0
    assert all(row["support"] > 0 for row in worst)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=500)
    y_pred = y_true.copy()
    flip = rng.choice(500, size=50, replace=False)
    y_pred[flip] = (y_pred[flip] + 1) % 5

    result = bootstrap_ci(y_true, y_pred, accuracy, n_resamples=200, seed=1)
    assert result["point"] == pytest.approx(0.9)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    assert 0.85 < result["ci_low"] < result["ci_high"] < 0.95


def test_bootstrap_is_reproducible_and_seed_sensitive():
    y_true = np.array([0, 1] * 100)
    y_pred = np.array([0, 1] * 90 + [1, 0] * 10)
    a = bootstrap_ci(y_true, y_pred, accuracy, n_resamples=100, seed=7)
    b = bootstrap_ci(y_true, y_pred, accuracy, n_resamples=100, seed=7)
    c = bootstrap_ci(y_true, y_pred, accuracy, n_resamples=100, seed=8)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])


def test_a_perfect_model_has_a_degenerate_interval():
    y_true = np.arange(100) % 4
    result = bootstrap_ci(y_true, y_true, accuracy, n_resamples=100, seed=0)
    assert result["point"] == 1.0
    assert result["ci_low"] == 1.0 and result["ci_high"] == 1.0


def test_bootstrap_metrics_reports_counts_and_intervals():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 3, size=200)
    y_pred = y_true.copy()
    y_pred[:20] = (y_pred[:20] + 1) % 3

    out = bootstrap_metrics(y_true, y_pred, num_classes=3, n_resamples=50, seed=0)
    assert out["n_total"] == 200
    assert out["n_correct"] == 180
    for name in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        assert out[name]["ci_low"] <= out[name]["point"] <= out[name]["ci_high"]


def test_mcnemar_uses_only_discordant_pairs():
    # 6 images: A right and B wrong twice, B right and A wrong once, rest agree.
    y_true = [0, 0, 0, 0, 0, 0]
    y_pred_a = [0, 0, 0, 1, 1, 1]
    y_pred_b = [0, 1, 1, 0, 1, 1]
    result = mcnemar(y_true, y_pred_a, y_pred_b)
    assert result["only_a_correct"] == 2
    assert result["only_b_correct"] == 1
    assert result["n_discordant"] == 3
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 2


def test_mcnemar_is_exact_below_the_threshold_and_chi_square_above():
    y_true = np.zeros(400, dtype=int)

    few_a, few_b = np.zeros(400, dtype=int), np.zeros(400, dtype=int)
    few_b[:10] = 1  # 10 discordant pairs
    assert mcnemar(y_true, few_a, few_b)["test"] == "exact binomial"

    many_a, many_b = np.zeros(400, dtype=int), np.zeros(400, dtype=int)
    many_b[:60] = 1  # 60 discordant pairs
    assert mcnemar(y_true, many_a, many_b)["test"] == "chi square with continuity correction"


def test_mcnemar_on_identical_predictions_reports_nothing_to_test():
    y_true = [0, 1, 2, 0]
    result = mcnemar(y_true, y_true, y_true)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_detects_a_large_one_sided_difference():
    y_true = np.zeros(200, dtype=int)
    y_pred_a = np.zeros(200, dtype=int)
    y_pred_b = np.zeros(200, dtype=int)
    y_pred_b[:40] = 1  # B is wrong on 40 images A gets right
    result = mcnemar(y_true, y_pred_a, y_pred_b)
    assert result["p_value"] < 0.001


def test_mcnemar_on_a_balanced_difference_is_not_significant():
    y_true = np.zeros(200, dtype=int)
    y_pred_a = np.zeros(200, dtype=int)
    y_pred_b = np.zeros(200, dtype=int)
    y_pred_a[:20] = 1
    y_pred_b[20:40] = 1  # each model wrong on 20 different images
    assert mcnemar(y_true, y_pred_a, y_pred_b)["p_value"] > 0.5


def test_wilcoxon_detects_a_consistent_paired_shift():
    a = np.arange(30, dtype=float)
    b = a + 0.5  # b beats a on every pair
    result = wilcoxon(a, b)
    assert result["p_value"] < 0.001
    assert result["median_difference"] == pytest.approx(-0.5)
    assert result["n_pairs_nonzero"] == 30


def test_wilcoxon_on_identical_inputs_reports_nothing_to_test():
    a = np.arange(10, dtype=float)
    result = wilcoxon(a, a)
    assert result["p_value"] == 1.0
    assert result["n_pairs_nonzero"] == 0


def test_aggregate_seeds_matches_hand_computation():
    result = aggregate_seeds([0.90, 0.92, 0.94])
    assert result["mean"] == pytest.approx(0.92)
    assert result["std"] == pytest.approx(np.std([0.90, 0.92, 0.94], ddof=1))
    assert result["n_seeds"] == 3
    assert result["min"] == pytest.approx(0.90)
    assert result["max"] == pytest.approx(0.94)


def test_aggregate_of_a_single_seed_has_zero_spread():
    result = aggregate_seeds([0.99])
    assert result["std"] == 0.0
    assert result["n_seeds"] == 1


def test_aggregate_of_nothing_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        aggregate_seeds([])


def test_table_formatting():
    assert format_mean_std({"mean": 0.9912, "std": 0.0013, "n_seeds": 3}) == "99.12 +/- 0.13"
    assert format_mean_std({"mean": 0.9912, "std": 0.0, "n_seeds": 1}) == "99.12"
    entry = {"point": 0.9912, "ci_low": 0.9880, "ci_high": 0.9940}
    assert format_ci(entry) == "99.12 [98.80, 99.40]"
