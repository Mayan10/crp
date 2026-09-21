"""Build every table and figure from the committed result files.

Reads only what earlier stages wrote under results/, so it can be rerun at any
time and never invents a number.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedxcrop.config import load_config
from fedxcrop.eval.stats import format_ci, format_mean_std
from fedxcrop.eval.tables import (
    add_bootstrap_intervals,
    configuration_label,
    discover_runs,
    main_results_table,
    pairwise_significance,
    save_table,
    worst_class_table,
)
from fedxcrop.viz.results_figures import (
    accuracy_vs_heterogeneity,
    test_metric_bars,
    training_curves,
    val_accuracy_per_round,
)
from fedxcrop.viz.xai_figures import metric_distribution


def collect_histories(results_dir: Path) -> tuple[dict, list[float]]:
    """Per round validation curves, grouped by strategy and alpha."""
    histories: dict[str, dict[float, list[pd.DataFrame]]] = {}
    alphas: set[float] = set()

    for metrics_path in sorted(results_dir.glob("federated/*/test_metrics.json")):
        if "smoke" in metrics_path.parts:
            continue
        config = json.loads(metrics_path.read_text()).get("config", {})
        history_path = metrics_path.parent / "history.csv"
        if not history_path.is_file() or config.get("alpha") is None:
            continue
        if config.get("init") != "imagenet":
            continue
        strategy, alpha = config["strategy"], float(config["alpha"])
        histories.setdefault(strategy, {}).setdefault(alpha, []).append(pd.read_csv(history_path))
        alphas.add(alpha)

    return histories, sorted(alphas)


def centralized_best_val(results_dir: Path) -> float | None:
    """Best validation accuracy of the centralized baseline, for the reference line."""
    values = []
    for history_path in sorted(results_dir.glob("centralized/seed*/history.csv")):
        values.append(float(pd.read_csv(history_path)["val_accuracy"].max()))
    return float(np.mean(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    results_dir = Path(cfg.results_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    runs = discover_runs(results_dir)
    if runs.empty:
        print("no finished runs found under results/. Run the grid first.")
        return 1
    print(f"found {len(runs)} finished runs\n")

    table = main_results_table(runs)
    table = add_bootstrap_intervals(table, runs, cfg.model.num_classes, args.bootstrap,
                                    cfg.eval.bootstrap_seed)

    readable = pd.DataFrame(
        {
            "configuration": table["label"],
            "seeds": table["n_seeds"],
            "accuracy": [format_mean_std({"mean": m, "std": s, "n_seeds": n})
                         for m, s, n in zip(table["accuracy_mean"], table["accuracy_std"],
                                            table["n_seeds"])],
            "accuracy 95 pct CI (seed 0)": [
                format_ci({"point": p, "ci_low": lo, "ci_high": hi})
                for p, lo, hi in zip(table["accuracy_point"], table["accuracy_ci_low"],
                                     table["accuracy_ci_high"])],
            "macro F1": [format_mean_std({"mean": m, "std": s, "n_seeds": n})
                         for m, s, n in zip(table["macro_f1_mean"], table["macro_f1_std"],
                                            table["n_seeds"])],
            "macro precision": [format_mean_std({"mean": m, "std": s, "n_seeds": n})
                                for m, s, n in zip(table["macro_precision_mean"],
                                                   table["macro_precision_std"], table["n_seeds"])],
            "macro recall": [format_mean_std({"mean": m, "std": s, "n_seeds": n})
                             for m, s, n in zip(table["macro_recall_mean"],
                                                table["macro_recall_std"], table["n_seeds"])],
            "correct / total (seed 0)": [f"{int(c)} / {int(t)}" for c, t in
                                         zip(table["n_correct_seed0"], table["n_total"])],
        }
    )
    save_table(readable, tables_dir, "main_results")
    save_table(table, tables_dir, "main_results_raw")
    print(readable.to_string(index=False))

    comparisons = []
    labels = set(table["label"])
    for alpha in (0.1, 0.5, 1.0):
        fedavg = f"fedavg alpha={alpha:g}"
        for label in labels:
            if label.startswith(f"fedprox alpha={alpha:g}"):
                comparisons.append((label, fedavg))
    if "fedavg IID" in labels:
        comparisons.append(("fedavg IID", "centralized"))
        for alpha in (0.1, 0.5, 1.0):
            if f"fedavg alpha={alpha:g}" in labels:
                comparisons.append((f"fedavg alpha={alpha:g}", "fedavg IID"))

    significance = pairwise_significance(runs, comparisons)
    if not significance.empty:
        save_table(significance, tables_dir, "significance_tests")
        print("\nMcNemar comparisons (seed 0, same test images):")
        print(significance[["model_a", "model_b", "only_a_correct", "only_b_correct",
                            "p_value", "significant_at_0.05"]].to_string(index=False))

    centralized_runs = runs[runs["strategy"] == "centralized"].sort_values("seed")
    if not centralized_runs.empty:
        worst = worst_class_table(centralized_runs.iloc[0]["metrics_path"], n=10)
        save_table(worst, tables_dir, "worst_classes_centralized")

    figure_notes = []

    histories, alphas = collect_histories(results_dir)
    if histories:
        paths = val_accuracy_per_round(histories, figures_dir,
                                       centralized_best_val=centralized_best_val(results_dir),
                                       alphas=alphas)
        figure_notes.append(("fig2_val_accuracy_per_round", paths[0],
                             "results/federated/*/history.csv and results/centralized/seed*/history.csv",
                             "Mean over seeds, band is plus or minus one standard deviation."))

    paths, notes = test_metric_bars(table, figures_dir, axis_starts_at_zero=True)
    figure_notes.append(("fig3_test_metrics", paths[0], "results/tables/main_results_raw.csv",
                         f"y axis starts at zero: {notes['axis_starts_at_zero']}. "
                         "Error bars are 95 percent bootstrap intervals over test images."))

    if table["alpha"].notna().any():
        paths = accuracy_vs_heterogeneity(table, figures_dir)
        figure_notes.append(("fig3b_accuracy_vs_alpha", paths[0],
                             "results/tables/main_results_raw.csv",
                             "Test accuracy against Dirichlet alpha, log x axis."))

    for history_path in sorted(results_dir.glob("centralized/seed*/history.csv")):
        seed = history_path.parent.name
        paths = training_curves(pd.read_csv(history_path), figures_dir, f"centralized_curves_{seed}")
        figure_notes.append((f"centralized_curves_{seed}", paths[0], str(history_path),
                             "Centralized training and validation curves."))

    per_image_path = results_dir / "xai" / "per_image_metrics.csv"
    if per_image_path.is_file():
        per_image = pd.read_csv(per_image_path)
        for metric in ("deletion_auc", "insertion_auc", "leaf_energy_ratio",
                       "spearman_vs_centralized"):
            if metric in per_image and per_image[metric].notna().any():
                paths = metric_distribution(per_image, metric, figures_dir, f"fig6_xai_{metric}")
                figure_notes.append((f"fig6_xai_{metric}", paths[0], str(per_image_path),
                                     "Box plots over the 380 image XAI sample, outliers hidden."))

    notes_path = figures_dir / "NOTES.md"
    with open(notes_path, "w") as handle:
        handle.write("# Figure notes\n\n")
        handle.write("Every figure below was produced by `scripts/make_figures.py` "
                     "from the result files listed.\n\n")
        for name, path, sources, note in figure_notes:
            handle.write(f"## {name}\n\n")
            handle.write(f"- file: `{path}`\n")
            handle.write(f"- script: `scripts/make_figures.py`\n")
            handle.write(f"- source data: `{sources}`\n")
            handle.write(f"- note: {note}\n\n")

    print(f"\nwrote {len(figure_notes)} figures and their notes to {notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
