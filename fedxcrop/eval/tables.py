"""Assembling result tables from the per run result files.

Every table is built by reading the JSON and CSV files that the runs wrote.
Nothing is typed in by hand, so a number in a table can always be traced back
to the file and the run that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from fedxcrop.eval.stats import aggregate_seeds, bootstrap_metrics, mcnemar


def discover_runs(results_dir: str | Path) -> pd.DataFrame:
    """Every finished run under results/, with the settings that define it."""
    results_dir = Path(results_dir)
    rows = []

    for metrics_path in sorted(results_dir.glob("*/*/test_metrics.json")):
        if "smoke" in metrics_path.parts:
            continue
        metrics = json.loads(metrics_path.read_text())
        family = metrics_path.parts[-3]
        config = metrics.get("config", {})

        rows.append(
            {
                "family": family,
                "run": metrics_path.parent.name,
                "strategy": config.get("strategy", "centralized"),
                "scheme": config.get("scheme", "centralized"),
                "alpha": config.get("alpha"),
                "num_clients": config.get("num_clients"),
                "mu": config.get("mu"),
                "rounds": config.get("rounds"),
                "init": config.get("init", "imagenet"),
                "seed": config.get("seed", _seed_from_name(metrics_path.parent.name)),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "weighted_f1": metrics["weighted_f1"],
                "n_correct": metrics["n_correct"],
                "n_total": metrics["n_total"],
                "selected_round": metrics.get("selected_round", metrics.get("selected_epoch")),
                "metrics_path": str(metrics_path),
                "predictions_path": str(metrics_path.parent / "test_predictions.csv"),
            }
        )

    return pd.DataFrame(rows)


def _seed_from_name(name: str) -> Optional[int]:
    for part in name.split("_"):
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                return None
    return None


def configuration_label(row) -> str:
    """Short, readable name for one configuration, used in tables and figures."""
    if row["strategy"] == "centralized":
        return "centralized"
    if row["scheme"] == "iid":
        return f"{row['strategy']} IID"
    label = f"{row['strategy']} alpha={row['alpha']:g}"
    if row["strategy"] == "fedprox" and row.get("mu") is not None:
        label += f" mu={row['mu']:g}"
    if row.get("init") == "checkpoint":
        label += " (legacy init)"
    return label


def main_results_table(runs: pd.DataFrame, group_seeds: bool = True) -> pd.DataFrame:
    """Mean plus or minus standard deviation over seeds, per configuration.

    The raw counts from the lowest seed are carried through so that at least
    one row per configuration can be recounted by hand.
    """
    if runs.empty:
        return pd.DataFrame()

    runs = runs.copy()
    runs["label"] = runs.apply(configuration_label, axis=1)
    keys = ["label", "strategy", "scheme", "alpha", "mu", "init"]

    rows = []
    for key, group in runs.groupby([k for k in keys], dropna=False, sort=False):
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        reference = group.sort_values("seed").iloc[0]

        for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1"):
            aggregate = aggregate_seeds(group[metric])
            record[f"{metric}_mean"] = aggregate["mean"]
            record[f"{metric}_std"] = aggregate["std"]
            record[f"{metric}_values"] = aggregate["values"]

        record["n_seeds"] = int(len(group))
        record["seeds"] = sorted(int(s) for s in group["seed"] if pd.notna(s))
        record["n_correct_seed0"] = int(reference["n_correct"])
        record["n_total"] = int(reference["n_total"])
        record["rounds"] = reference["rounds"]
        rows.append(record)

    return pd.DataFrame(rows)


def add_bootstrap_intervals(
    table: pd.DataFrame,
    runs: pd.DataFrame,
    num_classes: int = 38,
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Attach bootstrap intervals computed on the lowest seed of each row.

    Intervals come from one run rather than from pooled seeds: the bootstrap
    answers "how much would this number move on a different draw of test
    images", which is a separate question from the seed to seed spread already
    in the standard deviation column.
    """
    table = table.copy()
    runs = runs.copy()
    runs["label"] = runs.apply(configuration_label, axis=1)

    for metric in ("accuracy", "macro_f1"):
        for suffix in ("point", "ci_low", "ci_high"):
            table[f"{metric}_{suffix}"] = np.nan

    for i, row in table.iterrows():
        candidates = runs[runs["label"] == row["label"]].sort_values("seed")
        if candidates.empty:
            continue
        predictions_path = Path(candidates.iloc[0]["predictions_path"])
        if not predictions_path.is_file():
            continue

        predictions = pd.read_csv(predictions_path)
        result = bootstrap_metrics(
            predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy(),
            num_classes=num_classes, metrics=("accuracy", "macro_f1"),
            n_resamples=n_resamples, seed=seed,
        )
        for metric in ("accuracy", "macro_f1"):
            table.at[i, f"{metric}_point"] = result[metric]["point"]
            table.at[i, f"{metric}_ci_low"] = result[metric]["ci_low"]
            table.at[i, f"{metric}_ci_high"] = result[metric]["ci_high"]
        table.at[i, "bootstrap_n_correct"] = result["n_correct"]
        table.at[i, "bootstrap_n_total"] = result["n_total"]

    return table


def pairwise_significance(
    runs: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """McNemar's test for named pairs of configurations, on identical test sets."""
    runs = runs.copy()
    runs["label"] = runs.apply(configuration_label, axis=1)
    lookup = {row["label"]: row for _, row in runs.sort_values("seed").iterrows()}

    rows = []
    for left, right in comparisons:
        if left not in lookup or right not in lookup:
            continue
        a = pd.read_csv(lookup[left]["predictions_path"])
        b = pd.read_csv(lookup[right]["predictions_path"])
        if not a["path"].equals(b["path"]):
            merged = a.merge(b, on="path", suffixes=("_a", "_b"))
            y_true = merged["y_true_a"].to_numpy()
            pred_a, pred_b = merged["y_pred_a"].to_numpy(), merged["y_pred_b"].to_numpy()
        else:
            y_true = a["y_true"].to_numpy()
            pred_a, pred_b = a["y_pred"].to_numpy(), b["y_pred"].to_numpy()

        result = mcnemar(y_true, pred_a, pred_b)
        rows.append(
            {
                "model_a": left,
                "model_b": right,
                "accuracy_a": float((y_true == pred_a).mean()),
                "accuracy_b": float((y_true == pred_b).mean()),
                "only_a_correct": result["only_a_correct"],
                "only_b_correct": result["only_b_correct"],
                "n_discordant": result["n_discordant"],
                "test": result["test"],
                "p_value": result["p_value"],
                "significant_at_0.05": result["p_value"] < 0.05,
            }
        )

    return pd.DataFrame(rows)


def worst_class_table(metrics_path: str | Path, n: int = 10) -> pd.DataFrame:
    """The n weakest classes by F1 for one run."""
    from fedxcrop.eval.metrics import worst_classes

    metrics = json.loads(Path(metrics_path).read_text())
    return pd.DataFrame(worst_classes(metrics, n=n))


def save_table(table: pd.DataFrame, out_dir: str | Path, name: str) -> list[Path]:
    """Write a table as CSV and as Markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    table.to_csv(csv_path, index=False)
    with open(md_path, "w") as handle:
        handle.write(table.to_markdown(index=False, floatfmt=".4f"))
        handle.write("\n")
    return [csv_path, md_path]
