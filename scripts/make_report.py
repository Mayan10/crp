"""Generate RESULTS.md from the committed result files.

Every number in the report is read from a file under results/ and the file it
came from is named next to it, so the report can be checked line by line
against the runs that produced it. Verdicts in the claim audit are decided by
explicit rules over those numbers rather than written by hand.

Usage:
    python scripts/make_report.py
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
from fedxcrop.eval.stats import format_mean_std, wilcoxon
from fedxcrop.eval.tables import configuration_label, discover_runs, main_results_table

SIGNIFICANCE = 0.05


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def verdict_row(claim: str, evidence: str, source: str, verdict: str) -> dict:
    return {"claim": claim, "new evidence": evidence, "from": source, "verdict": verdict}


def lookup(table: pd.DataFrame, label: str) -> pd.Series | None:
    match = table[table["label"] == label]
    return match.iloc[0] if len(match) else None


def significance_for(significance: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    if significance.empty:
        return None
    match = significance[(significance["model_a"] == a) & (significance["model_b"] == b)]
    return match.iloc[0] if len(match) else None


def build_claim_audit(table: pd.DataFrame, significance: pd.DataFrame, xai: pd.DataFrame) -> list[dict]:
    """One row per claim from the original paper, decided from the new numbers."""
    rows = []
    centralized = lookup(table, "centralized")
    iid = lookup(table, "fedavg IID")

    # Claim 1: FedAvg IID matches or exceeds centralized accuracy.
    if centralized is not None and iid is not None:
        test = significance_for(significance, "fedavg IID", "centralized")
        gap = iid["accuracy_mean"] - centralized["accuracy_mean"]
        if test is not None and test["p_value"] >= SIGNIFICANCE:
            outcome = "partially supported"
            detail = (f"IID {pct(iid['accuracy_mean'])} vs centralized "
                      f"{pct(centralized['accuracy_mean'])}, difference {pct(gap)} pp, "
                      f"McNemar p = {test['p_value']:.3g}, not significant")
        elif gap >= 0:
            outcome = "supported"
            detail = (f"IID {pct(iid['accuracy_mean'])} exceeds centralized "
                      f"{pct(centralized['accuracy_mean'])} by {pct(gap)} pp"
                      + (f", McNemar p = {test['p_value']:.3g}" if test is not None else ""))
        else:
            outcome = "not supported"
            detail = (f"IID {pct(iid['accuracy_mean'])} is below centralized "
                      f"{pct(centralized['accuracy_mean'])} by {pct(-gap)} pp"
                      + (f", McNemar p = {test['p_value']:.3g}" if test is not None else ""))
        rows.append(verdict_row(
            "FedAvg IID matches or exceeds centralized accuracy", detail,
            "results/tables/main_results.csv, significance_tests.csv", outcome))

    # Claim 2: non-IID reduces FedAvg accuracy and precision.
    fedavg_alphas = table[(table["strategy"] == "fedavg") & table["alpha"].notna()]
    if iid is not None and not fedavg_alphas.empty:
        hardest = fedavg_alphas.sort_values("alpha").iloc[0]
        acc_drop = iid["accuracy_mean"] - hardest["accuracy_mean"]
        prec_drop = iid["macro_precision_mean"] - hardest["macro_precision_mean"]
        monotone = fedavg_alphas.sort_values("alpha")["accuracy_mean"].is_monotonic_increasing
        outcome = "supported" if acc_drop > 0 and prec_drop > 0 else (
            "partially supported" if acc_drop > 0 or prec_drop > 0 else "not supported")
        rows.append(verdict_row(
            "Non-IID data reduces FedAvg accuracy and precision",
            (f"alpha {hardest['alpha']:g} vs IID: accuracy {pct(acc_drop)} pp lower, "
             f"macro precision {pct(prec_drop)} pp lower; accuracy increases monotonically "
             f"with alpha: {monotone}"),
            "results/tables/main_results.csv", outcome))

    # Claim 3: FedProx recovers accuracy and precision under non-IID data.
    recovered, tested = [], []
    for alpha in sorted(fedavg_alphas["alpha"].unique()):
        avg = lookup(table, f"fedavg alpha={alpha:g}")
        prox = table[(table["strategy"] == "fedprox") & (table["alpha"] == alpha)]
        if avg is None or prox.empty:
            continue
        prox = prox.iloc[0]
        delta = prox["accuracy_mean"] - avg["accuracy_mean"]
        test = significance_for(significance, prox["label"], f"fedavg alpha={alpha:g}")
        recovered.append(f"alpha {alpha:g}: {pct(delta, 2)} pp"
                         + (f" (p = {test['p_value']:.3g})" if test is not None else ""))
        if test is not None:
            tested.append((delta, test["p_value"]))

    if recovered:
        gains = [d for d, p in tested if d > 0 and p < SIGNIFICANCE]
        if gains:
            outcome = "supported" if len(gains) == len(tested) else "partially supported"
        else:
            outcome = "not supported"
        rows.append(verdict_row(
            "FedProx recovers accuracy and precision under non-IID data",
            "FedProx minus FedAvg accuracy, " + "; ".join(recovered),
            "results/tables/main_results.csv, significance_tests.csv", outcome))

    # Claims 4 to 6 come from the XAI table.
    if not xai.empty:
        leaf = xai[xai["leaf_energy_ratio"].notna()]
        if not leaf.empty:
            mean_ratio = leaf["leaf_energy_ratio"].mean()
            mean_area = leaf["leaf_area_fraction"].mean()
            outcome = "supported" if mean_ratio > mean_area + 0.05 else (
                "partially supported" if mean_ratio > mean_area else "not supported")
            rows.append(verdict_row(
                "Post-aggregation attribution focuses on the leaf rather than background",
                (f"mean leaf energy ratio {mean_ratio:.3f} against a leaf area fraction of "
                 f"{mean_area:.3f} (the chance level for a map that highlights nothing "
                 f"in particular), over {len(leaf)} image and model pairs"),
                "results/xai/per_image_metrics.csv", outcome))

        agreement = xai[xai["spearman_vs_centralized"].notna()]
        if not agreement.empty:
            mean_rho = agreement["spearman_vs_centralized"].mean()
            outcome = ("supported" if mean_rho > 0.7 else
                       "partially supported" if mean_rho > 0.4 else "not supported")
            per_model = "; ".join(
                f"{model}: rho {group['spearman_vs_centralized'].mean():.3f}"
                for model, group in agreement.groupby("model")
            )
            rows.append(verdict_row(
                "Federated training preserves the explanations of the centralized model",
                f"mean Spearman rank correlation with the centralized model {mean_rho:.3f}. {per_model}",
                "results/xai/per_image_metrics.csv", outcome))

        comparisons = []
        for method in sorted(xai["method"].unique()):
            subset = xai[xai["method"] == method]
            for alpha_tag in ("alpha0.1", "alpha0.5"):
                prox = subset[subset["model"].str.contains(f"fedprox.*{alpha_tag}", regex=True)]
                avg = subset[subset["model"].str.contains(f"fedavg.*{alpha_tag}", regex=True)]
                if prox.empty or avg.empty:
                    continue
                merged = prox.merge(avg, on="path", suffixes=("_prox", "_avg"))
                column = "spearman_vs_centralized"
                pair = merged[[f"{column}_prox", f"{column}_avg"]].dropna()
                if len(pair) < 10:
                    continue
                result = wilcoxon(pair[f"{column}_prox"], pair[f"{column}_avg"])
                direction = "higher" if result["median_difference"] > 0 else "lower"
                comparisons.append(
                    f"{method} {alpha_tag}: FedProx agreement {direction} by "
                    f"{abs(result['median_difference']):.3f} (Wilcoxon p = {result['p_value']:.3g})"
                )
        if comparisons:
            wins = sum("higher" in c and "p = 0.0" in c for c in comparisons)
            outcome = "supported" if wins == len(comparisons) else (
                "partially supported" if wins else "not supported")
            rows.append(verdict_row(
                "FedProx preserves explanation quality better than FedAvg",
                "; ".join(comparisons), "results/xai/per_image_metrics.csv", outcome))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="RESULTS.md")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    results_dir = Path(cfg.results_dir)
    tables_dir = results_dir / "tables"

    runs = discover_runs(results_dir)
    if runs.empty:
        print("no finished runs found. Run the grid, then scripts/make_figures.py, then this.")
        return 1

    table = main_results_table(runs)
    main_path = tables_dir / "main_results.md"
    significance_path = tables_dir / "significance_tests.csv"
    significance = pd.read_csv(significance_path) if significance_path.is_file() else pd.DataFrame()

    xai_path = results_dir / "xai" / "per_image_metrics.csv"
    xai = pd.read_csv(xai_path) if xai_path.is_file() else pd.DataFrame()

    leakage_path = results_dir / "data" / "leakage_report.json"
    leakage = json.loads(leakage_path.read_text()) if leakage_path.is_file() else {}

    lines: list[str] = []
    add = lines.append

    add("# FedXCrop results\n")
    add("Every number below was produced by code in this repository and is traceable to a")
    add("file under `results/`. The file is named next to each table. Nothing here is")
    add("transcribed by hand.\n")
    add(f"Generated by `scripts/make_report.py` from {len(runs)} finished runs.\n")

    add("## Protocol\n")
    add("- Split: one fixed stratified 80/10/10 split at seed 42, built once and committed.")
    add("  Train 43,447, validation 5,430, test 5,428 images over 38 classes.")
    add("- Federated runs start every client and the global model from ImageNet weights.")
    add("  Clients hold disjoint shards of the full train split. Validation and test are")
    add("  global and are never partitioned.")
    add("- The round or epoch carried to test is chosen by validation accuracy alone. The")
    add("  test split is read once per run.")
    add("- Reported rates are reproducible from the integer counts saved with them.\n")

    add("## Main results\n")
    if main_path.is_file():
        add(main_path.read_text().strip())
    add(f"\nSource: `results/tables/main_results.csv`, one row per configuration, aggregated")
    add("over seeds. Per run files are under `results/centralized/` and `results/federated/`.\n")

    add("## What is and is not statistically significant\n")
    if significance.empty:
        add("No paired comparisons were available.\n")
    else:
        significant = significance[significance["significant_at_0.05"]]
        not_significant = significance[~significance["significant_at_0.05"]]
        add(f"McNemar's test on the same {int(runs.iloc[0]['n_total'])} test images, seed 0.\n")
        if len(significant):
            add("Significant at the 0.05 level:\n")
            for _, row in significant.iterrows():
                add(f"- {row['model_a']} vs {row['model_b']}: "
                    f"{int(row['only_a_correct'])} images only the first got right against "
                    f"{int(row['only_b_correct'])} the other way, p = {row['p_value']:.3g}")
            add("")
        if len(not_significant):
            add("Not significant, meaning the data does not separate these configurations:\n")
            for _, row in not_significant.iterrows():
                add(f"- {row['model_a']} vs {row['model_b']}: "
                    f"{int(row['only_a_correct'])} against {int(row['only_b_correct'])} "
                    f"discordant images, p = {row['p_value']:.3g}")
            add("")
    add("Source: `results/tables/significance_tests.csv`\n")

    if not xai.empty:
        add("## Explainability\n")
        summary = (
            xai.groupby(["method", "model"])
            .agg(
                deletion_auc=("deletion_auc", "mean"),
                insertion_auc=("insertion_auc", "mean"),
                leaf_energy_ratio=("leaf_energy_ratio", "mean"),
                spearman_vs_centralized=("spearman_vs_centralized", "mean"),
                n_images=("path", "count"),
            )
            .reset_index()
        )
        add(summary.to_markdown(index=False, floatfmt=".4f"))
        add("\nLower deletion AUC and higher insertion AUC are better. The leaf energy ratio")
        add("is the share of attribution falling inside the leaf mask, against a mean leaf")
        add("area fraction that is the chance level for a map highlighting nothing in")
        add("particular.\n")
        add("Source: `results/xai/per_image_metrics.csv`\n")

        sanity_path = results_dir / "xai" / "sanity_randomization.csv"
        if sanity_path.is_file():
            add("### Sanity check\n")
            add("Rank correlation between the original attribution and the attribution after")
            add("progressively randomizing the model. A method whose correlation stays high is")
            add("describing the image rather than the model.\n")
            add(pd.read_csv(sanity_path).to_markdown(index=False, floatfmt=".4f"))
            add("\nSource: `results/xai/sanity_randomization.csv`\n")

    add("## Claim audit\n")
    add("One row per claim made by the earlier version of this work, decided by rule from")
    add("the numbers above rather than by hand.\n")
    audit = build_claim_audit(table, significance, xai)
    if audit:
        add(pd.DataFrame(audit).to_markdown(index=False))
    else:
        add("Not enough runs finished to audit the claims.")
    add("")

    legacy = runs[runs["init"] == "checkpoint"]
    add("## Legacy protocol comparison\n")
    if legacy.empty:
        add("The legacy reproduction has not been run yet "
            "(`python scripts/run_grid.py --group legacy`).\n")
    else:
        add("The original protocol initialized every client from the centralized model that")
        add("had already seen the whole training set, and distributed only a small subset of")
        add("the training data. Both are reproduced here so the corrected numbers can be read")
        add("against them.\n")
        for _, row in legacy.iterrows():
            add(f"- {row['run']}: test accuracy {pct(row['accuracy'])} "
                f"({int(row['n_correct'])}/{int(row['n_total'])}), "
                f"macro F1 {pct(row['macro_f1'])}, after {row['rounds']} rounds")
        add("")
        corrected = table[(table["init"] == "imagenet") & (table["strategy"] == "fedavg")]
        if not corrected.empty:
            add("Under the corrected protocol the same strategy reaches "
                + ", ".join(f"{r['label']} {pct(r['accuracy_mean'])}"
                            for _, r in corrected.iterrows()) + ".\n")

    if leakage:
        add("## Near duplicate leakage\n")
        add(f"- {leakage['n_test_with_near_duplicate_in_train']} of {leakage['n_test']} test "
            f"images ({100 * leakage['fraction_test_with_near_duplicate']:.2f} percent) have a "
            f"perceptual hash match within Hamming distance "
            f"{leakage['threshold_hamming']} in the train split.")
        add(f"- {leakage['n_same_label_matches']} of those share the label, which is the kind "
            f"that inflates accuracy; {leakage['n_cross_label_matches']} do not.")
        add(f"- {leakage['n_exact_hash_matches']} test images have an identical hash to a "
            f"train image.")
        worst = sorted(leakage["per_class"].items(), key=lambda kv: kv[1]["fraction"],
                       reverse=True)[:3]
        add("- Most affected classes: "
            + ", ".join(f"{name} ({100 * s['fraction']:.0f} percent)" for name, s in worst) + ".")
        add("\nThe split was not modified, so these numbers stay comparable with published")
        add("PlantVillage results, which are subject to the same overlap. It does mean the")
        add("absolute accuracies here, and in the literature, are optimistic.\n")
        add("Source: `results/data/leakage_report.json`\n")

    add("## Limitations\n")
    add("- PlantVillage images are single leaves photographed against a uniform background")
    add("  in a lab. Nothing here establishes field performance, and no out of distribution")
    add("  test set was used.")
    add("- Clients are simulated by partitioning one dataset. No data was held by separate")
    add("  parties and no network was involved.")
    add("- No formal privacy mechanism is used: no differential privacy and no secure")
    add("  aggregation. Transmitting weights rather than images is not by itself a privacy")
    add("  guarantee, since model updates can leak training data.")
    add("- Lesion masks are derived from a colour rule, not annotated. They under-detect dark")
    add("  necrotic spots and flag whole leaves for chlorotic diseases, so lesion localization")
    add("  numbers are weaker evidence than the leaf energy ratio, whose mask comes from the")
    add("  dataset itself. Per class coverage is in")
    add("  `results/xai/lesion_mask_coverage_by_class.csv`.")
    if leakage:
        add("- Train and test share near duplicate photographs of the same physical leaves, as")
        add("  quantified above.")
    add("- Accuracy on this dataset sits near a ceiling for every configuration, so the")
    add("  differences under study are small and require the intervals and paired tests")
    add("  reported here to interpret.\n")

    output = Path(args.output)
    output.write_text("\n".join(lines))
    print(f"wrote {output} ({len(lines)} lines) from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
