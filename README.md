# BrainTumorSeg-3D

Volumetric brain tumour segmentation from multi-sequence MRI (FLAIR, T1w, T1gd, T2w) on the
Medical Segmentation Decathlon brain tumour task. The point of this repository is evaluation
discipline, not architectural novelty: strictly patient-level train/validation/test splits, a test
split that is evaluated exactly once, Dice and HD95 reported per tumour region as full per-case
distributions rather than single means, sliding-window inference, and an explicit analysis of where
the model fails, small lesions in particular.

> **Not a medical device.** This is a research artefact. It has not been clinically validated, it
> must not be used to inform diagnosis or treatment, and its outputs are not diagnoses.

## Results

**The test split was evaluated once, after the configuration was frozen.** The frozen
configuration is the baseline, [`configs/unet3d.yaml`](configs/unet3d.yaml): a MONAI U-Net trained
on 128³ patches with Dice + cross-entropy, AdamW, a cosine schedule and mixed precision for 300
epochs, using the checkpoint with the best mean validation Dice (epoch 275). It was chosen on the
validation split alone, and nothing was changed after its test scores were seen.

### Test split

Scores are for the 52 unique test scans: [`reports/test_results.csv`](reports/test_results.csv)
holds the distributions and [`reports/test_cases.csv`](reports/test_cases.csv) one row per scan.

| region | metric | n | mean ± std | median | IQR | min | max |
|---|---|---:|---|---:|---|---:|---:|
| WT | Dice | 52 | 0.870 ± 0.143 | 0.908 | 0.836–0.940 | 0.000 | 0.967 |
| TC | Dice | 52 | 0.824 ± 0.185 | 0.872 | 0.788–0.935 | 0.000 | 0.974 |
| ET | Dice | 51 | 0.744 ± 0.253 | 0.835 | 0.748–0.883 | 0.000 | 0.949 |
| WT | HD95 (mm) | 52 | 12.84 ± 19.96 | 4.47 | 3.00–10.81 | 1.41 | 93.60 |
| TC | HD95 (mm) | 52 | 15.06 ± 51.79 | 4.36 | 2.18–8.56 | 1.00 | 373.13 |
| ET | HD95 (mm) | 51 | 20.92 ± 72.47 | 2.45 | 1.87–6.49 | 1.00 | 373.13 |

ET is undefined for the one test scan with no enhancing tumour; the model correctly predicted
none there.

- **Test agrees with validation.** Median Dice is 0.908 / 0.872 / 0.835 (WT / TC / ET) on test
  and 0.917 / 0.866 / 0.800 on validation: lower for WT, higher for TC and ET. There is no sign
  that choosing the checkpoint on validation inflated the validation scores.
- **The median scan is within 0.05 Dice of the human reference.** Two annotations of the same
  scan agree at a median Dice of 0.933 / 0.915 / 0.883.
- **The mean is pulled down by a few scans.** Mean Dice is 0.870 / 0.824 / 0.744. One tumour is
  missed completely and scores 0 on all three regions, with an HD95 of 373.13 mm, the scan
  diagonal, for TC and ET.

![Per-case test Dice and HD95](reports/figures/test_boxplot.png)

### Where it fails

The full analysis is in [`reports/failure_analysis.md`](reports/failure_analysis.md).

- **Small lesions are the most consistent failure.** The quarter of test scans with the smallest
  lesions accounts for 44% (WT), 40% (TC) and 45% (ET) of all the Dice lost, where an even spread
  would give 25% ([`reports/test_dice_by_volume.csv`](reports/test_dice_by_volume.csv)). The
  Spearman correlation between Dice and lesion volume is 0.46, 0.43 and 0.49, each with a
  bootstrap interval excluding zero
  ([`reports/test_correlations.csv`](reports/test_correlations.csv)). The five enhancing tumours
  under 1.1 mL score ET Dice 0.00, 0.00, 0.39, 0.00 and 0.27.
- **For WT and TC, part of that is the metric.** Dice punishes a fixed boundary error more on a
  small object. Two annotations of the same scan show a correlation of 0.37 with volume for both
  regions, inside the model's intervals. For ET the annotators show 0.15 and the model 0.49.
- **The two worst scans are not small.** Their tumours are 26.8 mL and 76.8 mL
  ([`reports/test_failure_cases.csv`](reports/test_failure_cases.csv)). The first is the darkest
  tumour of the split on T2w and FLAIR and is missed completely. The second enhances weakly, and
  its core is labelled as oedema (TC Dice 0.20, ET Dice 0.10).
- **Faint lesions fail at any size.** ET Dice correlates with the T1gd intensity of the enhancing
  tumour at 0.67 on test and 0.50 on validation. Between annotators it is 0.08, so this is a
  weakness of the model and not an ambiguity in the labels. Intensity was looked at only after
  the test failures were seen, which is why the validation figure is given next to it.
- **HD95 catches what Dice forgives.** BRATS_392 scores a WT Dice of 0.94 and a WT HD95 of
  48.8 mm, from 2.4 mL of tumour predicted far from the real one.

![Dice against lesion volume on the test split](reports/figures/test_dice_vs_volume.png)

![The five worst test scans](reports/figures/test_worst_cases.png)

![The five best test scans](reports/figures/test_best_cases.png)

### Baseline on the validation split

Scores are for the 56 unique validation scans:
[`reports/baseline_results.csv`](reports/baseline_results.csv) holds the distributions and
[`reports/baseline_val_cases.csv`](reports/baseline_val_cases.csv) one row per scan.

| region | metric | n | mean ± std | median | IQR | min | max |
|---|---|---:|---|---:|---|---:|---:|
| WT | Dice | 56 | 0.889 ± 0.086 | 0.917 | 0.865–0.948 | 0.566 | 0.974 |
| TC | Dice | 56 | 0.806 ± 0.168 | 0.866 | 0.748–0.920 | 0.204 | 0.967 |
| ET | Dice | 55 | 0.745 ± 0.206 | 0.800 | 0.652–0.879 | 0.000 | 0.958 |
| WT | HD95 (mm) | 56 | 9.81 ± 14.32 | 4.63 | 2.24–10.23 | 1.41 | 62.61 |
| TC | HD95 (mm) | 56 | 10.12 ± 12.94 | 5.29 | 3.00–10.51 | 1.41 | 66.06 |
| ET | HD95 (mm) | 55 | 13.05 ± 50.85 | 3.00 | 1.73–5.82 | 1.00 | 373.13 |

ET is undefined for the one validation scan with no enhancing tumour; the model correctly predicted
none there.

- **The mean hides the failures.** One scan's enhancing tumour is missed completely: Dice 0, and an
  HD95 of 373.13 mm, the scan diagonal. That single case pulls the ET HD95 mean to 13.05 mm, while
  the median is 3.00 mm.
- **Small lesions fail.** The three lowest ET Dice scores (0.000, 0.018 and 0.259) belong to the
  three smallest enhancing lesions in the validation split (0.35, 0.32 and 0.03 mL). The test
  split shows the same.
- **Against the human reference** (two annotations of the same scan, measured on the 114
  duplicated scans rather than these), median Dice is 0.917 vs 0.933 for WT, 0.866 vs 0.915 for TC,
  and 0.800 vs 0.883 for ET. The gap is smallest for whole tumour.
- **Training converged.** Over the last eight validations (epochs 265–300), mean validation Dice
  stayed between 0.8115 and 0.8135
  ([`reports/baseline_history.csv`](reports/baseline_history.csv)).

![Per-case validation Dice and HD95 for the baseline](reports/figures/baseline_val_boxplot.png)

![Training loss and validation Dice per epoch](reports/figures/baseline_training_curve.png)

### Ablations

Each ablation changes one setting of the baseline and is scored on the same 56 validation scans
([`reports/ablations.csv`](reports/ablations.csv), full distributions in
[`reports/results.csv`](reports/results.csv)). The table gives mean Dice.

| change | WT | TC | ET | training | verdict |
|---|---:|---:|---:|---:|---|
| baseline | 0.889 | 0.806 | 0.745 | 1.09 h | |
| baseline, another training seed | 0.896 | 0.821 | 0.751 | 1.04 h | the seed-to-seed spread |
| loss: Dice only | 0.889 | 0.824 | 0.746 | 1.04 h | not robust to the baseline seed |
| loss: Dice + focal | 0.890 | 0.816 | 0.748 | 1.11 h | no detectable difference |
| patch 96³ | 0.887 | 0.808 | 0.743 | 0.70 h | not robust to the baseline seed |
| Attention U-Net | 0.899 | 0.824 | 0.755 | 4.82 h | not robust to the baseline seed |
| augmentation off | 0.885 | 0.802 | 0.744 | 1.01 h | not robust to the baseline seed |
| overlap 0 or 0.25 | 0.889 | 0.806 | 0.745 | none | identical to the baseline |
| overlap 0.75 | 0.890 | 0.806 | 0.746 | none | no detectable difference |

**No ablation is better or worse than the baseline.** A change counts only if its paired
bootstrap interval excludes zero against both baseline runs, on the same side, and none does.

- **Retraining the baseline moves the score as much as any ablation.** With only the training
  seed changed, mean Dice moves by +0.007 (WT) and +0.015 (TC), and a bootstrap over scans calls
  both differences real. The interval covers which scans were sampled, not how training went.
- **A single-seed study would have reported two wins.** Dice-only loss and Attention U-Net each
  gain 0.018 on TC against the baseline, with intervals excluding zero. Against the reseeded
  baseline the gains are 0.003, and Attention U-Net costs more than four times the training.
- **Overlap is inert here.** The brain crops are barely larger than the 128-voxel window, so
  overlap 0, 0.25 and 0.5 place the same windows and give identical scores.
- **The baseline is therefore the configuration that was frozen and scored on the test split.**

"Not robust to the baseline seed" means the change differs from one baseline run but not the
other on at least one region. The full account is in [`DECISIONS.md`](DECISIONS.md).

![Each ablation against both baseline runs](reports/figures/ablations.png)

## Data

484 labelled cases from MSD Task01_BrainTumour (BraTS-derived), each a co-registered 240 x 240 x 155
volume at 1 mm isotropic spacing with four sequences. Labels are oedema, non-enhancing tumour and
enhancing tumour, evaluated as the BraTS regions WT, TC and ET. Enhancing tumour is absent in 12
cases, so ET Dice is undefined for them. Per-case statistics are in
[`reports/dataset_stats.csv`](reports/dataset_stats.csv) and the summary in
[`reports/dataset_summary.md`](reports/dataset_summary.md).

**Case ids are not patients.** MSD has no patient identifiers, and the same patient appears under
several case ids. 228 cases are exact image duplicates of another case (in 108 of the 114 pairs
with a different label map), and further cases are repeat scans of one patient at different
timepoints. A case-level random split would leave 42 of 73 test cases with a scan of the same
patient in the training set. Cases are therefore linked by image similarity in the shared atlas
space, and the split is made over the 129 linked patient groups plus the unlinked cases:
342 / 71 / 71 cases for train / validation / test, committed in
[`configs/splits.json`](configs/splits.json). The reasoning and the threshold evidence are in
[`DECISIONS.md`](DECISIONS.md).

![Nearest-case similarity with the link threshold](reports/figures/repeat_scans.png)

![Tumour volume distribution per region](reports/figures/tumour_volumes.png)

## Evaluation protocol

- **Metrics:** Dice and HD95 (mm) per region, reported as distributions: n, mean, std, median,
  quartiles, min and max across cases. HD95 is the symmetric 95th-percentile surface distance.
  Both are verified against hand-computed cases and against MONAI on real label pairs
  (`make crosscheck`, [`reports/metric_crosscheck.csv`](reports/metric_crosscheck.csv)).
- **Empty ground truth:** when a region is absent, Dice and HD95 are undefined. Those cases are
  counted rather than averaged, and scored as detection (was tumour predicted where there is
  none?). When a region is present but missed, the case scores Dice 0 and an HD95 of the scan
  diagonal (373.13 mm), and is never dropped.
- **Duplicated scans** are scored once, so validation has 56 unique scans and test has 52.
- **Test split:** scored once, with one configuration. The config names the frozen configuration
  and the command line refuses the test split for any other. The model's predictions are stored
  by that run, so the tables, figures and failure analysis never run the model on test again.
- **Human reference:** MSD ships 114 scans twice, each copy with its own label map. Scoring the
  two label maps of each scan against each other shows how far annotations of an identical image
  disagree ([`reports/annotation_agreement_summary.csv`](reports/annotation_agreement_summary.csv)):

| region | Dice median (IQR) | Dice min | HD95 median (IQR), mm |
|---|---|---|---|
| WT | 0.933 (0.899–0.956) | 0.712 | 3.46 (2.24–5.79) |
| TC | 0.915 (0.833–0.949) | 0.361 | 5.15 (2.00–10.41) |
| ET | 0.883 (0.830–0.927) | 0.426 | 2.24 (1.41–3.46) |

The full rules and the reasoning behind them are in [`DECISIONS.md`](DECISIONS.md).

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), GNU Make, and for training an NVIDIA GPU with a
CUDA 12.x capable driver (developed on an RTX 3090, 24 GB).

```bash
make setup
```

This creates `.venv` with Python 3.11, installs the pinned dependencies with the CUDA 12.6 build of
PyTorch, and prints the torch and MONAI versions and the detected GPU. On a machine without an
NVIDIA GPU use `make setup TORCH_BACKEND=cpu`.

Data: download `Task01_BrainTumour` from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
and extract it so that `data/raw/Task01_BrainTumour/` contains `dataset.json`, `imagesTr/` and
`labelsTr/`. The data is licensed CC-BY-SA 4.0 and is not redistributed here.

## Usage

```bash
make data
```

This preprocesses every labelled case once into `data/processed/`: crop to the brain, per-case
per-channel z-score inside the brain mask, float16 cache. It then links repeat scans, checks that
`configs/splits.json` is exactly the split the config produces, and writes the dataset statistics,
figures and annotation-agreement reference to `reports/`. The first run decompresses about 7 GB of
NIfTI on a single CPU core and takes minutes; later runs reuse the cache.

```bash
make crosscheck
```

This rechecks `metrics.py` against MONAI's independent Dice and HD95 on real label pairs, and fails
if they disagree.

```bash
make train
make eval
make ablate
make eval SPLIT=test
make report
```

`make train` trains the config (default `configs/unet3d.yaml`, or pass `CONFIG=...`) into
`runs/<name>/`. It keeps the checkpoint with the best validation Dice, and rerunning it resumes an
interrupted run. On an RTX 3090 the baseline takes about an hour. `make eval` scores that
checkpoint on the validation split with the full metrics and stores the predicted label maps.
`make eval SPLIT=test` does the same on the test split, and only for the configuration named in
`data.frozen_config`; every other config is refused. `make ablate` trains and scores every config
in `configs/ablations/`, each of which changes one setting of the baseline; it takes about ten
hours in total and skips runs that are already finished. `make report` writes the tables and
figures in `reports/`, including the failure analysis, and needs the baseline to have been
evaluated on both splits and all ablations on validation. Every run is logged to a local MLflow
store in `mlruns/`, including the config, seed, per-epoch metrics, git commit and run directory.
To browse it:

```bash
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

(On Linux or macOS the executable is `.venv/bin/mlflow`.)

## License

Code is released under the MIT License, see [LICENSE](LICENSE).
