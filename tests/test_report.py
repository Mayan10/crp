"""Tests for the claim audit rules in the results report.

The audit decides supported, partially supported, or not supported from the
numbers rather than from judgement, so the rules themselves are worth testing
against cases with known answers.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_report import build_claim_audit, lookup, significance_for


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Minimal main results table with the columns the audit reads."""
    defaults = {
        "label": "", "strategy": "", "scheme": "dirichlet", "alpha": None,
        "mu": None, "init": "imagenet", "accuracy_mean": 0.0, "accuracy_std": 0.0,
        "macro_precision_mean": 0.0, "macro_f1_mean": 0.0, "macro_recall_mean": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def significance_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def find(audit: list[dict], fragment: str) -> dict:
    matches = [row for row in audit if fragment.lower() in row["claim"].lower()]
    assert matches, f"no audit row mentioning {fragment!r}"
    return matches[0]


def test_iid_beating_centralized_is_supported():
    table = results_table([
        {"label": "centralized", "strategy": "centralized", "accuracy_mean": 0.990},
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid", "accuracy_mean": 0.993},
    ])
    significance = significance_table([
        {"model_a": "fedavg IID", "model_b": "centralized", "p_value": 0.001,
         "only_a_correct": 30, "only_b_correct": 5, "significant_at_0.05": True},
    ])
    audit = build_claim_audit(table, significance, pd.DataFrame())
    assert find(audit, "matches or exceeds centralized")["verdict"] == "supported"


def test_iid_losing_to_centralized_significantly_is_not_supported():
    table = results_table([
        {"label": "centralized", "strategy": "centralized", "accuracy_mean": 0.995},
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid", "accuracy_mean": 0.980},
    ])
    significance = significance_table([
        {"model_a": "fedavg IID", "model_b": "centralized", "p_value": 1e-8,
         "only_a_correct": 3, "only_b_correct": 90, "significant_at_0.05": True},
    ])
    audit = build_claim_audit(table, significance, pd.DataFrame())
    assert find(audit, "matches or exceeds centralized")["verdict"] == "not supported"


def test_an_indistinguishable_difference_is_only_partially_supported():
    """A gap the test cannot separate must not be reported as a win either way."""
    table = results_table([
        {"label": "centralized", "strategy": "centralized", "accuracy_mean": 0.9910},
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid", "accuracy_mean": 0.9914},
    ])
    significance = significance_table([
        {"model_a": "fedavg IID", "model_b": "centralized", "p_value": 0.62,
         "only_a_correct": 12, "only_b_correct": 10, "significant_at_0.05": False},
    ])
    audit = build_claim_audit(table, significance, pd.DataFrame())
    row = find(audit, "matches or exceeds centralized")
    assert row["verdict"] == "partially supported"
    assert "not significant" in row["new evidence"]


def test_non_iid_hurting_both_metrics_is_supported():
    table = results_table([
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid",
         "accuracy_mean": 0.99, "macro_precision_mean": 0.99},
        {"label": "fedavg alpha=0.1", "strategy": "fedavg", "alpha": 0.1,
         "accuracy_mean": 0.95, "macro_precision_mean": 0.94},
    ])
    audit = build_claim_audit(table, pd.DataFrame(), pd.DataFrame())
    assert find(audit, "Non-IID data reduces")["verdict"] == "supported"


def test_non_iid_not_hurting_is_not_supported():
    table = results_table([
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid",
         "accuracy_mean": 0.9894, "macro_precision_mean": 0.9892},
        {"label": "fedavg alpha=0.1", "strategy": "fedavg", "alpha": 0.1,
         "accuracy_mean": 0.9951, "macro_precision_mean": 0.9941},
    ])
    audit = build_claim_audit(table, pd.DataFrame(), pd.DataFrame())
    assert find(audit, "Non-IID data reduces")["verdict"] == "not supported"


def test_fedprox_recovering_significantly_at_every_alpha_is_supported():
    table = results_table([
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid", "accuracy_mean": 0.99,
         "macro_precision_mean": 0.99},
        {"label": "fedavg alpha=0.1", "strategy": "fedavg", "alpha": 0.1,
         "accuracy_mean": 0.950, "macro_precision_mean": 0.94},
        {"label": "fedprox alpha=0.1 mu=0.01", "strategy": "fedprox", "alpha": 0.1,
         "mu": 0.01, "accuracy_mean": 0.970, "macro_precision_mean": 0.96},
    ])
    significance = significance_table([
        {"model_a": "fedprox alpha=0.1 mu=0.01", "model_b": "fedavg alpha=0.1",
         "p_value": 0.0001, "only_a_correct": 80, "only_b_correct": 20,
         "significant_at_0.05": True},
    ])
    audit = build_claim_audit(table, significance, pd.DataFrame())
    assert find(audit, "FedProx recovers")["verdict"] == "supported"


def test_fedprox_not_separating_from_fedavg_is_not_supported():
    table = results_table([
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid", "accuracy_mean": 0.99,
         "macro_precision_mean": 0.99},
        {"label": "fedavg alpha=0.1", "strategy": "fedavg", "alpha": 0.1,
         "accuracy_mean": 0.9951, "macro_precision_mean": 0.9941},
        {"label": "fedprox alpha=0.1 mu=0.01", "strategy": "fedprox", "alpha": 0.1,
         "mu": 0.01, "accuracy_mean": 0.9951, "macro_precision_mean": 0.9907},
    ])
    significance = significance_table([
        {"model_a": "fedprox alpha=0.1 mu=0.01", "model_b": "fedavg alpha=0.1",
         "p_value": 0.83, "only_a_correct": 9, "only_b_correct": 9,
         "significant_at_0.05": False},
    ])
    audit = build_claim_audit(table, significance, pd.DataFrame())
    assert find(audit, "FedProx recovers")["verdict"] == "not supported"


def xai_frame(leaf_ratio: float, leaf_area: float, rho: float, n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [f"img_{i}.JPG" for i in range(n)],
            "model": ["fedprox_alpha0.1"] * n,
            "method": ["gradcam"] * n,
            "leaf_energy_ratio": [leaf_ratio] * n,
            "leaf_area_fraction": [leaf_area] * n,
            "spearman_vs_centralized": [rho] * n,
            "deletion_auc": [0.1] * n,
            "insertion_auc": [0.8] * n,
        }
    )


def test_attribution_well_above_chance_on_the_leaf_is_supported():
    audit = build_claim_audit(results_table([]), pd.DataFrame(),
                              xai_frame(leaf_ratio=0.90, leaf_area=0.60, rho=0.5))
    assert find(audit, "focuses on the leaf")["verdict"] == "supported"


def test_attribution_at_chance_on_the_leaf_is_not_supported():
    """A map that highlights nothing in particular scores the leaf area fraction."""
    audit = build_claim_audit(results_table([]), pd.DataFrame(),
                              xai_frame(leaf_ratio=0.60, leaf_area=0.62, rho=0.5))
    assert find(audit, "focuses on the leaf")["verdict"] == "not supported"


def test_explanation_preservation_scales_with_the_correlation():
    high = build_claim_audit(results_table([]), pd.DataFrame(),
                             xai_frame(0.9, 0.6, rho=0.85))
    mid = build_claim_audit(results_table([]), pd.DataFrame(),
                            xai_frame(0.9, 0.6, rho=0.55))
    low = build_claim_audit(results_table([]), pd.DataFrame(),
                            xai_frame(0.9, 0.6, rho=0.20))
    assert find(high, "preserves the explanations")["verdict"] == "supported"
    assert find(mid, "preserves the explanations")["verdict"] == "partially supported"
    assert find(low, "preserves the explanations")["verdict"] == "not supported"


def test_audit_is_empty_when_nothing_has_been_run():
    assert build_claim_audit(results_table([]), pd.DataFrame(), pd.DataFrame()) == []


def test_lookup_and_significance_helpers_return_none_when_absent():
    table = results_table([{"label": "centralized", "strategy": "centralized"}])
    assert lookup(table, "centralized") is not None
    assert lookup(table, "missing") is None
    assert significance_for(pd.DataFrame(), "a", "b") is None


def test_legacy_runs_are_excluded_from_corrected_protocol_comparisons():
    """The legacy reproduction keeps the original defects and must not be pooled in.

    It shares a label shape with a corrected run (both are fedavg at some
    alpha), so without filtering it lands inside the alpha sweep and corrupts
    the trend it is being compared against.
    """
    table = results_table([
        {"label": "fedavg IID", "strategy": "fedavg", "scheme": "iid",
         "accuracy_mean": 0.989, "macro_precision_mean": 0.985},
        {"label": "fedavg alpha=0.1", "strategy": "fedavg", "alpha": 0.1,
         "accuracy_mean": 0.969, "macro_precision_mean": 0.955},
        {"label": "fedavg alpha=0.5", "strategy": "fedavg", "alpha": 0.5,
         "accuracy_mean": 0.983, "macro_precision_mean": 0.974},
        {"label": "fedavg alpha=1", "strategy": "fedavg", "alpha": 1.0,
         "accuracy_mean": 0.987, "macro_precision_mean": 0.982},
        # Legacy: same strategy and alpha as a corrected run, much higher accuracy
        # because it started from the centralized model.
        {"label": "fedavg alpha=0.5 (legacy init)", "strategy": "fedavg", "alpha": 0.5,
         "init": "checkpoint", "accuracy_mean": 0.992, "macro_precision_mean": 0.988},
    ])
    audit = build_claim_audit(table, pd.DataFrame(), pd.DataFrame())
    row = find(audit, "Non-IID data reduces")
    # With the legacy row excluded, accuracy rises monotonically with alpha.
    assert "monotonically with alpha: True" in row["new evidence"]


def xai_pair_frame(prox_rho: float, avg_rho: float, n: int = 60) -> pd.DataFrame:
    """Paired per image agreement scores for FedProx and FedAvg at alpha 0.1."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        noise = rng.normal(0, 0.01)
        for model, rho in (("fedprox_alpha0.1", prox_rho), ("fedavg_alpha0.1", avg_rho)):
            rows.append({
                "path": f"img_{i}.JPG", "model": model, "method": "gradcam",
                "leaf_energy_ratio": 0.85, "leaf_area_fraction": 0.60,
                "spearman_vs_centralized": rho + noise,
                "deletion_auc": 0.1, "insertion_auc": 0.8,
            })
    return pd.DataFrame(rows)


def test_fedprox_explanation_verdict_reads_a_tiny_p_value_correctly():
    """A highly significant win must not be missed.

    The p value formats as scientific notation here, which an earlier
    string based rule failed to recognise as significant.
    """
    audit = build_claim_audit(results_table([]), pd.DataFrame(),
                              xai_pair_frame(prox_rho=0.80, avg_rho=0.70))
    row = find(audit, "FedProx preserves explanation")
    assert row["verdict"] == "supported"
    assert "higher" in row["new evidence"]


def test_fedprox_explanation_verdict_is_not_supported_when_fedprox_is_worse():
    audit = build_claim_audit(results_table([]), pd.DataFrame(),
                              xai_pair_frame(prox_rho=0.65, avg_rho=0.78))
    row = find(audit, "FedProx preserves explanation")
    assert row["verdict"] == "not supported"
    assert "lower" in row["new evidence"]
