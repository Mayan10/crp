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


def test_panel_labels_wrap_instead_of_truncating():
    from fedxcrop.viz.xai_figures import panel_label

    label = panel_label("Tomato___Spider_mites Two-spotted_spider_mite")
    lines = label.split("\n")
    assert lines[0] == "Tomato"
    assert all(len(line) <= 22 for line in lines[1:])
    assert "Spider mites" in label
    assert panel_label("Pepper,_bell___Bacterial_spot").split("\n") == ["Pepper bell", "Bacterial spot"]
