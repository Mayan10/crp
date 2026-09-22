# FedXCrop

Federated crop disease classification on PlantVillage, with a quantitative
evaluation of post-aggregation explainability.

Every number this repository reports is produced by code in it and traceable to
a file under `results/`. Metric files carry `n_correct` and `n_total` alongside
every rate, so a reported accuracy can always be checked against a whole number
of correct predictions on a stated number of images.

## What the protocol does

- **One fixed split.** A stratified 80/10/10 split at seed 42, built once and
  committed as `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
  (43,447 / 5,430 / 5,428 images over 38 classes). Every experiment reads these
  files. Nothing downstream resamples them.
- **Federated runs start from ImageNet weights.** Every client and the global
  model start from ImageNet, never from a model already trained on PlantVillage.
- **Clients partition the train split only.** Validation and test stay global,
  so every configuration is scored on identical held out images.
- **Selection on validation, test read once.** The round or epoch carried
  forward is chosen by validation accuracy. The test split is evaluated once per
  run, at the end.
- **Heterogeneity is measured.** Dirichlet partitions are summarized by the
  Jensen-Shannon divergence of each client's label distribution from the global
  one, and shown as per client class histograms.

## Federated engines

Rounds can be driven two ways, selected by `federated.engine` or `--engine`:

- `flower` (default) uses Flower's simulation engine, which runs each client in
  its own Ray worker. This is what the reported results are produced with.
- `sequential` runs the same rounds in one process. It exists because Ray has
  no build for every Python version, so this is the path that works on a
  machine where Flower's simulation engine cannot start, and it is the
  reference the Flower engine is tested against.

Both call the same local training and the same aggregation
(`fedxcrop/fl/local.py`): the FedProx proximal term and the shard size
weighting are defined once and unit tested once. `test_both_engines_agree` in
`tests/test_smoke.py` checks the two against each other on the smoke
configuration and is the first thing to run on a new machine.

FedProx uses the same server side aggregation as FedAvg and differs only in the
client's local objective, which is what the FedProx paper specifies. That is
why `federated.strategy` changes the client, not the strategy class.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/Mayan10/crp.git
cd crp
pip install -r requirements.txt
```

Download PlantVillage so that one directory contains `color/`, `segmented/` and
`grayscale/` subdirectories, each holding 38 class folders. Pass its location as
`data.root` (the default is `plantvillage dataset`):

```bash
# Kaggle, needs ~/.kaggle/kaggle.json
kaggle datasets download -d amgadalameri/plantvillage-dataset-zip --unzip
```

Check the dataset matches the committed splits before running anything long:

```bash
python -m pytest tests/ -q -m "not slow"
```

## Reproducing everything

Commands are listed in the order they must run. Every one accepts
`--set key.sub=value` overrides, for example
`--set data.root='/path/to/plantvillage dataset' device=cuda`.

### 1. Data, splits, partitions

```bash
python scripts/build_splits.py                # splits/, results/data/dataset_report.json
python scripts/check_leakage.py               # results/data/leakage_report.json
python scripts/build_partitions.py            # splits/partitions/, figure 1
```

Runtime: about 2 minutes in total. These outputs are already committed, so this
step only needs rerunning if the dataset changes.

### 2. Smoke test

Exercises the whole path on a tiny subset, on CPU, in under five minutes:

```bash
python scripts/train_centralized.py --config configs/centralized.yaml --smoke
python scripts/run_federated.py --config configs/fedprox_noniid.yaml --smoke --engine sequential
python scripts/run_federated.py --config configs/fedprox_noniid.yaml --smoke --engine flower
python -m pytest tests/ -q                    # includes the slow end to end runs
```

### 3. Centralized baseline

```bash
python scripts/run_grid.py --group centralized --seeds 0 1 2
```

### 4. Federated grid

```bash
python scripts/run_grid.py --group fedavg_iid     --seeds 0 1 2
python scripts/run_grid.py --group fedavg_noniid  --alphas 0.1 0.5 1.0 --seeds 0 1 2
python scripts/run_grid.py --group mu_selection   --mus 0.001 0.01 0.1
python scripts/select_mu.py                       # reports the mu chosen on validation
python scripts/run_grid.py --group fedprox_noniid --mu <selected> --alphas 0.1 0.5 1.0 --seeds 0 1 2
python scripts/run_grid.py --group legacy         # reproduces the original protocol
```

The runner skips runs whose final results already exist and resumes an
interrupted run from its last completed round, so it can be stopped and
restarted at any point.

`scripts/run_grid.py --group core --dry-run` prints the full list without
running anything.

### 5. Explainability

```bash
python scripts/run_xai.py --build-sample          # splits/xai_sample.csv, 10 images per class
python scripts/check_pseudo_masks.py --n 30       # visual check of the lesion masks
python scripts/run_xai.py --models centralized_seed0 \
    fedavg_dirichlet_alpha0.1_K5_seed0 fedprox_dirichlet_alpha0.1_K5_mu<selected>_seed0 \
    fedavg_dirichlet_alpha0.5_K5_seed0 fedprox_dirichlet_alpha0.5_K5_mu<selected>_seed0 \
    --sanity
```

The first model listed is the reference the others are compared against.

### 6. Tables, figures, report

```bash
python scripts/make_figures.py                    # results/tables/, results/figures/, NOTES.md
python scripts/make_xai_figures.py --models centralized_seed0 \
    fedprox_dirichlet_alpha0.1_K5_mu<selected>_seed0    # figures 4, 5 and 7
python scripts/make_report.py                     # RESULTS.md
```

`results/figures/NOTES.md` records, for each figure, the script, the config and
the result files it came from.

## Running on Colab

`notebooks/run_grid_colab.ipynb` runs the grid on a GPU runtime with `runs/`
symlinked to Google Drive, so a disconnected session resumes rather than
restarts. It times one round first and prints an estimate for the whole grid
before committing to it.

## Expected runtimes

Measured on the hardware noted, for MobileNetV2 at 224x224, batch 32.

| Stage | Apple M2, 8 GB (MPS), measured |
|---|---|
| One federated round (K=5, E=1, one pass over 43,447 images plus validation) | 10.6 min |
| One federated run (30 rounds) | 5.3 h |
| Centralized, 3 seeds x 20 epochs | 11 h |
| Core grid, 24 federated runs | 128 h |
| Everything | about 138 h |

The measurement behind these is in `results/timing.json`. A centralized epoch
costs about the same as a federated round, since both are one pass over the
whole train split plus a validation pass.

The core grid is not practical on a laptop. Run it on a GPU: step 6 of
`notebooks/run_grid_colab.ipynb` times one round on the actual runtime and
prints the projected total for that machine, which is more reliable than
extrapolating from the table above.

## Understanding the code

`CODEMAP.md` walks the codebase in reading order, saying what each file is
responsible for and what to check when reading it.

## Repository layout

```
fedxcrop/
  config.py        configuration dataclasses, run provenance
  data/            dataset index, split, partitions, leakage, transforms
  models/          model factory and weight exchange
  train/           centralized training
  fl/              local client training, FedProx term, federated rounds
  eval/            metrics, bootstrap intervals, significance tests, tables
  xai/             attribution methods, XAI metrics, masks, sanity check
  viz/             figures
configs/           one YAML per experiment, all inheriting base.yaml
scripts/           command line entry points
splits/            the committed fixed splits and client partitions
results/           committed metrics, tables and figures
tests/             unit tests plus end to end smoke runs
```

## Data availability

PlantVillage was published by Hughes and Salathe (2015),
`https://arxiv.org/abs/1511.08060`, with the images at
`https://github.com/spMohanty/PlantVillage-Dataset`. The Kaggle mirror above is
a convenience copy. The `segmented` variant is used to obtain leaf masks.

## Code availability

All code needed to reproduce every table and figure is in this repository.
