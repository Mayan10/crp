"""Tests for choosing what goes on the result figures."""

import pandas as pd

from fedxcrop.viz.results_figures import heterogeneity_sweep


def test_heterogeneity_sweep_keeps_one_multi_seed_row_per_point():
    table = pd.DataFrame(
        [
            {"label": "centralized", "strategy": "centralized", "alpha": None, "mu": None,
             "init": "imagenet", "n_seeds": 3},
            {"label": "fedavg alpha=0.5", "strategy": "fedavg", "alpha": 0.5, "mu": None,
             "init": "imagenet", "n_seeds": 3},
            {"label": "fedavg alpha=0.5 (legacy init)", "strategy": "fedavg", "alpha": 0.5,
             "mu": None, "init": "checkpoint", "n_seeds": 1},
            {"label": "fedprox alpha=0.1 mu=0.001", "strategy": "fedprox", "alpha": 0.1,
             "mu": 0.001, "init": "imagenet", "n_seeds": 3},
            {"label": "fedprox alpha=0.1 mu=0.1", "strategy": "fedprox", "alpha": 0.1,
             "mu": 0.1, "init": "imagenet", "n_seeds": 1},
        ]
    )
    sweep = heterogeneity_sweep(table)
    assert list(sweep["label"]) == ["fedavg alpha=0.5", "fedprox alpha=0.1 mu=0.001"]
