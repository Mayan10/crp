# Reading order

The codebase in the order that makes it easiest to follow. Each section says
what the file is responsible for and what to check when you read it.

## 1. Configuration and provenance

**`fedxcrop/config.py`** Nested dataclasses loaded from YAML, with inheritance
(`base:`) and dotted command line overrides (`--set federated.rounds=5`). The
part worth reading closely is `save_run_metadata`: every run writes its
resolved config, a config hash, the git commit, library versions and the device
into its run directory. That is what makes a result traceable months later.

**`configs/base.yaml`** Every default in one place. All other configs inherit
from it and override only what they change, so a diff between two configs is
the experimental difference.

## 2. Data

**`fedxcrop/data/index.py`** Scans the dataset into (path, label, class_name)
and maps every color image to its segmented counterpart. Read
`segmented_coverage`: matching by filename alone covered 97.8 percent, because
a few classes name their two variants differently, so there is a fallback on
the shared image key that brings it to 100 percent. `verify_dataset` reports
what it found (54,305 images, 38 classes) rather than asserting it.

**`fedxcrop/data/splits.py`** The one fixed stratified 80/10/10 split at seed
42. Built once, committed, and read by everything downstream. Nothing else in
the repository is allowed to resample it.

**`fedxcrop/data/partition.py`** IID and Dirichlet partitioning of the train
split only. `dirichlet_partition` does standard per class Dirichlet sampling
and logs how many resamples the minimum client size cost.
`partition_summary` is where heterogeneity becomes a number: Jensen-Shannon
divergence per client, classes present per client, client sizes.

**`fedxcrop/data/leakage.py`** Perceptual hash near duplicate detection between
train and test. Reports, never modifies: 157 of 5,428 test images have a match
within Hamming distance 4.

**`fedxcrop/data/transforms.py`, `dataset.py`** Augmentation on train only;
validation and test are deterministic. Two comments worth reading in
`build_loader`: `persistent_workers`, which is a memory trap with one loader
per client, and `effective_workers`, which stops the loader being asked for
more worker processes than there are cores.

**`fedxcrop/data/gpu_augment.py`** The colour jitter, vectorised onto the
accelerator because it was three quarters of the cost of the input pipeline
and was starving the GPU. Factors are drawn per image, and every operation is
checked against torchvision numerically. Two performance notes in here are
worth the read: `hsv_to_rgb` is branchless because the obvious one hot form
allocates six times the batch, and `rgb_to_hsv` uses `amax`/`amin` rather than
`max`/`min` because the latter also compute argmax indices, which cost two
orders of magnitude more.

## 3. Model and measurement

**`fedxcrop/models/factory.py`** MobileNetV2 with a new 38 class head. Read
`get_weights`: it exchanges the whole state dict, batch norm running statistics
included, which matters because averaging parameters while leaving each
client's local statistics behind silently costs accuracy.

**`fedxcrop/eval/metrics.py`** Every metric file carries `n_correct` and
`n_total`. `save_metrics` refuses to write a file without them. This is the
direct answer to the previous version reporting rates that no stated evaluation
set could produce.

**`fedxcrop/eval/stats.py`** Bootstrap intervals over test images, McNemar for
two models on the same images (exact below 25 discordant pairs), paired
Wilcoxon for per image XAI scores, mean and standard deviation across seeds.
On this dataset every configuration scores around 99 percent, so this file is
what separates a real difference from noise.

## 4. Training

**`fedxcrop/train/centralized.py`** The baseline. Best epoch chosen on
validation, test read once. Checkpoints after every epoch and resumes.

**`fedxcrop/fl/local.py`** The federated method itself, free of any framework:
the FedProx proximal term, the per round optimizer policy, and shard size
weighted aggregation. Both engines call this, so the method is defined once.
`proximal_term` and `weighted_average` are the two functions to read.

**`fedxcrop/fl/flower_simulation.py`** (default engine) Drives rounds through
Flower's simulation engine. **`fedxcrop/fl/flower_client.py`** and
**`flower_strategy.py`** are the client and the strategy;
`TrackedFedAvg.aggregate_fit` is where validation, logging and checkpointing
happen.

**`fedxcrop/fl/simulation.py`** The same rounds in one process
(`--engine sequential`). Exists because Ray has no build for every Python
version, and serves as the reference the Flower engine is tested against.

## 5. Explainability

**`fedxcrop/xai/attribution.py`** Grad-CAM on `features[-1]` and SmoothGrad on
input gradients, both on the predicted class. The SmoothGrad docstring states
the noise convention explicitly, which the original write up left ambiguous.

**`fedxcrop/xai/metrics.py`** The file that turns the old qualitative claims
into numbers: deletion and insertion AUC, energy ratio, pointing game, Spearman
agreement, top k IoU. Each docstring says which direction is better and what
the chance level is.

**`fedxcrop/xai/masks.py`** Leaf masks from the segmented variant, and lesion
pseudo-masks from a measured healthy hue range. Read the note on erosion in
`lesion_pseudo_mask`: without it the rule traced the leaf outline and scored it
as disease.

**`fedxcrop/xai/sanity.py`** Model randomization check. If an attribution map
survives having the weights destroyed, it is describing the image, not the
model.

## 6. Reporting

**`fedxcrop/eval/tables.py`** Builds tables by reading the files the runs wrote.
Nothing is transcribed by hand.

**`fedxcrop/viz/`** `style.py` (colourblind safe palette, 300 dpi PNG and PDF),
`results_figures.py`, `partitions.py`, `xai_figures.py`. Note
`_draw_axis_break`: if a y axis is ever truncated the break is drawn and
recorded, because the previous version's axis started at 97.5 percent.

**`scripts/make_report.py`** Generates `RESULTS.md`, including the claim audit.
`build_claim_audit` decides supported, partially supported or not supported by
rule from the measured numbers. The rule that matters: a difference the paired
test cannot separate is never reported as a win.

## 7. Entry points

```
scripts/build_splits.py        scan, verify, split          (run once)
scripts/check_leakage.py       near duplicate report        (run once)
scripts/build_partitions.py    client partitions, figure 1  (run once)
scripts/train_centralized.py   the baseline
scripts/run_federated.py       one federated configuration
scripts/run_grid.py            the grid, restartable
scripts/select_mu.py           picks mu on validation
scripts/run_xai.py             attribution metrics
scripts/check_pseudo_masks.py  visual check of lesion masks
scripts/make_figures.py        tables and figures
scripts/make_xai_figures.py    attribution grids, failure cases
scripts/make_report.py         RESULTS.md and the claim audit
```

## 8. Tests

`tests/test_config.py`, `test_splits.py`, `test_partition.py`,
`test_metrics.py`, `test_models.py`, `test_fl_local.py`, `test_fl_engines.py`,
`test_xai.py`, `test_tables.py`, `test_report.py` are unit tests on inputs with
known answers. `tests/test_smoke.py` runs the whole path end to end on a tiny
subset and is marked `slow`.

The three tests most worth reading, because they encode the claims this
rewrite exists to make checkable:

- `test_fedprox_keeps_weights_closer_to_the_global_model_than_fedavg`
- `test_deletion_and_insertion_behave_in_the_expected_direction`
- `test_both_engines_agree`
