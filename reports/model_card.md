# Model card: 3D U-Net for brain tumour segmentation on MSD Task01

> **Not a medical device.** This model is a research artefact. It has not been clinically
> validated, it must not be used to inform diagnosis or treatment, and its outputs are not
> diagnoses.

## Model details

| | |
|---|---|
| Task | Voxel-wise segmentation of glioma sub-regions in multi-sequence brain MRI |
| Input | Four co-registered sequences (FLAIR, T1w, T1gd, T2w), skull-stripped, 1 mm isotropic |
| Output | One label per voxel: background, oedema, non-enhancing tumour, enhancing tumour |
| Reported regions | Whole tumour (WT), tumour core (TC), enhancing tumour (ET), formed from the labels |
| Architecture | MONAI `UNet`, 3D, channels 32-64-128-256-320, four stride-2 levels, 2 residual units per level |
| Training | 300 epochs on 128³ patches, batch size 2, Dice + cross-entropy, AdamW (lr 1e-4), cosine schedule, mixed precision, flips and intensity jitter, seed 42 |
| Checkpoint | Epoch 275, the best mean validation Dice |
| Inference | Sliding window of 128³ with 0.5 overlap and Gaussian weighting |
| Configuration | [`configs/unet3d.yaml`](../configs/unet3d.yaml) |
| Training time | 1.09 hours on one RTX 3090 (24 GB) |
| Weights | Not distributed. `make train` reproduces them, see the [README](../README.md) |
| Licence | Code MIT. Training data CC-BY-SA 4.0 |

## Intended use

- Research and teaching on how to evaluate a medical segmentation model: patient-level splits,
  per-case score distributions, an annotator reference, and failure analysis.
- A documented baseline for the MSD brain tumour task, for comparing other methods under the
  same split and the same metrics.

The intended users are researchers and students in medical image analysis. The intended input is
a scan prepared the way this dataset is prepared: pre-operative glioma, four sequences,
co-registered to a common template, skull-stripped and resampled to 1 mm.

## Out-of-scope use

- **Any clinical use.** Diagnosis, treatment or surgical planning, radiotherapy contouring,
  response assessment and triage are all out of scope. The model has no clinical validation.
- **Deciding whether a tumour is present.** Every one of the 484 cases contains a tumour. The
  model has never been trained or scored on a healthy brain.
- **Scans prepared differently.** Images with the skull, with a sequence missing, at another
  resolution, or not registered to the template were never seen and never tested.
- **Other conditions and populations.** Post-operative scans, metastases, meningioma, stroke and
  paediatric tumours are outside the training data.
- **Measuring small enhancing lesions.** Below about 1 mL the enhancing tumour is rarely found
  (see Known failure modes).

## Training data

[Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task01_BrainTumour (Antonelli et
al., 2022; Simpson et al., 2019), licensed CC-BY-SA 4.0. It is drawn from the BraTS 2016 and 2017
challenge data (Menze et al., 2015; Bakas et al., 2017): multi-site MRI of glioma with manual
labels for oedema, non-enhancing tumour and enhancing tumour.

- 484 labelled cases of 240 x 240 x 155 voxels. 342 are used for training, 71 for validation
  and 71 for testing ([`configs/splits.json`](../configs/splits.json)).
- The dataset has no patient identifiers, and case ids are not patients. 228 cases are exact
  image duplicates of another case, and further cases are repeat scans of one patient. Cases are
  linked by image similarity into 129 patient groups, and a group never crosses splits.
- Preprocessing: crop to the brain, then z-score each sequence within the brain mask, per case.
- The labels are noisy. Two annotations of the same scan, measured on the 114 duplicated scans,
  agree at a median Dice of 0.933 (WT), 0.915 (TC) and 0.883 (ET).

## Evaluation protocol

- **Split:** patient-level, fixed and committed before training. The test split was scored
  exactly once, with the configuration frozen beforehand, and nothing was changed afterwards.
- **Metrics:** Dice and HD95 (mm) per region and per scan, reported as distributions.
- **Absent regions:** Dice and HD95 are undefined when a region is absent from the ground truth.
  Those scans are counted, not averaged. A region that is present but missed scores Dice 0 and an
  HD95 of the scan diagonal, 373.13 mm.
- **Duplicated scans** are scored once, which leaves 52 unique test scans.

The reasoning behind each rule is in [`DECISIONS.md`](../DECISIONS.md).

## Results on the test split

| region | metric | n | mean ± std | median | IQR | min | max |
|---|---|---:|---|---:|---|---:|---:|
| WT | Dice | 52 | 0.870 ± 0.143 | 0.908 | 0.836–0.940 | 0.000 | 0.967 |
| TC | Dice | 52 | 0.824 ± 0.185 | 0.872 | 0.788–0.935 | 0.000 | 0.974 |
| ET | Dice | 51 | 0.744 ± 0.253 | 0.835 | 0.748–0.883 | 0.000 | 0.949 |
| WT | HD95 (mm) | 52 | 12.84 ± 19.96 | 4.47 | 3.00–10.81 | 1.41 | 93.60 |
| TC | HD95 (mm) | 52 | 15.06 ± 51.79 | 4.36 | 2.18–8.56 | 1.00 | 373.13 |
| ET | HD95 (mm) | 51 | 20.92 ± 72.47 | 2.45 | 1.87–6.49 | 1.00 | 373.13 |

Source: [`test_results.csv`](test_results.csv), with one row per scan in
[`test_cases.csv`](test_cases.csv). The median scan is within 0.05 Dice of the agreement between
two annotators. The means are lower because a few scans fail badly.

![Per-case test Dice and HD95](figures/test_boxplot.png)

## Known failure modes

All figures are from the test split. The evidence is in
[`failure_analysis.md`](failure_analysis.md).

- **A whole tumour can be missed.** One scan of 52 (BRATS_381, a 26.8 mL tumour) scores Dice 0
  on all three regions. Its tumour is the darkest of the split on T2w and FLAIR.
- **Small lesions.** The quarter of scans with the smallest lesions accounts for 44% (WT), 40%
  (TC) and 45% (ET) of all the Dice lost. The five enhancing tumours under 1.1 mL score ET Dice
  0.00, 0.00, 0.39, 0.00 and 0.27.
- **Faint lesions, at any size.** ET Dice correlates with the T1gd intensity of the enhancing
  tumour at 0.67 on test and 0.50 on validation. Two annotators show no such dependence (0.08).
  A weakly enhancing core can be labelled as oedema (BRATS_368: TC Dice 0.20, ET Dice 0.10).
- **Non-enhancing tumour is the least reliable label.** The model recovers 66.2% of its voxels.
  Two annotators agree on 75.7%.
- **Detached false positives.** Seven scans contain at least 1 mL of predicted tumour that
  touches no real tumour, up to 38.0 mL. These barely lower Dice and produce HD95 values of tens
  of millimetres. No post-processing is applied to remove them.

## Limitations

- **The data does not represent clinical practice.** MSD is a curated research dataset. Every
  scan was selected for a challenge, has all four sequences, and was co-registered, resampled
  and skull-stripped by the dataset's authors. It does not cover the range of scanners,
  protocols, field strengths, artefacts and incomplete studies that a hospital produces. How the
  model behaves on such scans is unknown.
- **No external validation.** The test split comes from the same dataset as the training split.
- **No subgroup analysis.** The dataset carries no age, sex, site, scanner or tumour grade, so
  performance across any of these cannot be checked.
- **Patient identity is inferred.** Repeat scans are linked by image similarity. A missed link
  would leave a patient on both sides of a split, which cannot be fully excluded.
- **Small test set, one training run.** 52 scans give wide intervals. Retraining with another
  seed moves mean validation Dice by up to 0.015, so differences of that size are not meaningful.
- **Label noise bounds the scores.** A score near the annotator agreement cannot be improved in
  any way this ground truth can measure.

## References

- Antonelli, M. et al. The Medical Segmentation Decathlon. *Nature Communications* 13, 4128
  (2022). https://doi.org/10.1038/s41467-022-30695-9
- Simpson, A. L. et al. A large annotated medical image dataset for the development and
  evaluation of segmentation algorithms. arXiv:1902.09063 (2019).
- Menze, B. H. et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE
  Transactions on Medical Imaging* 34(10), 1993–2024 (2015).
  https://doi.org/10.1109/TMI.2014.2377694
- Bakas, S. et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert
  segmentation labels and radiomic features. *Scientific Data* 4, 170117 (2017).
  https://doi.org/10.1038/sdata.2017.117
