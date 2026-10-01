# Figure notes

Every figure below was produced by `scripts/make_figures.py` from the result files listed.

## fig2_val_accuracy_per_round

- file: `results/figures/fig2_val_accuracy_per_round.png`
- script: `scripts/make_figures.py`
- source data: `results/federated/*/history.csv and results/centralized/seed*/history.csv`
- note: Mean over seeds, band is plus or minus one standard deviation.

## fig3_test_metrics

- file: `results/figures/fig3_test_metrics.png`
- script: `scripts/make_figures.py`
- source data: `results/tables/main_results_raw.csv`
- note: y axis starts at zero: True. Error bars are 95 percent bootstrap intervals over test images.

## fig3b_accuracy_vs_alpha

- file: `results/figures/fig3b_accuracy_vs_alpha.png`
- script: `scripts/make_figures.py`
- source data: `results/tables/main_results_raw.csv`
- note: Test accuracy against Dirichlet alpha, log x axis. Mean over seeds, bars are one standard deviation. Only the multi seed configuration at each alpha.

## centralized_curves_seed0

- file: `results/figures/centralized_curves_seed0.png`
- script: `scripts/make_figures.py`
- source data: `results/centralized/seed0/history.csv`
- note: Centralized training and validation curves.

## centralized_curves_seed1

- file: `results/figures/centralized_curves_seed1.png`
- script: `scripts/make_figures.py`
- source data: `results/centralized/seed1/history.csv`
- note: Centralized training and validation curves.

## centralized_curves_seed2

- file: `results/figures/centralized_curves_seed2.png`
- script: `scripts/make_figures.py`
- source data: `results/centralized/seed2/history.csv`
- note: Centralized training and validation curves.

## fig6_xai_deletion_auc

- file: `results/figures/fig6_xai_deletion_auc.png`
- script: `scripts/make_figures.py`
- source data: `results/xai/per_image_metrics.csv`
- note: Box plots over the 380 image XAI sample, outliers hidden.

## fig6_xai_insertion_auc

- file: `results/figures/fig6_xai_insertion_auc.png`
- script: `scripts/make_figures.py`
- source data: `results/xai/per_image_metrics.csv`
- note: Box plots over the 380 image XAI sample, outliers hidden.

## fig6_xai_leaf_energy_ratio

- file: `results/figures/fig6_xai_leaf_energy_ratio.png`
- script: `scripts/make_figures.py`
- source data: `results/xai/per_image_metrics.csv`
- note: Box plots over the 380 image XAI sample, outliers hidden.

## fig6_xai_spearman_vs_centralized

- file: `results/figures/fig6_xai_spearman_vs_centralized.png`
- script: `scripts/make_figures.py`
- source data: `results/xai/per_image_metrics.csv`
- note: Box plots over the 380 image XAI sample, outliers hidden.

