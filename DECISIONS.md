# Decisions

A short log of every non-obvious choice and the reason for it. Newest entries at the bottom.

## Dataset: MSD Task01_BrainTumour rather than raw BraTS

The Medical Segmentation Decathlon brain task is BraTS-derived (484 labelled cases, the same four
MRI sequences) but downloads directly, without the Synapse account and data-use agreement that BraTS
requires. Anyone can reproduce the results end to end.

## Evaluation regions: WT, TC, ET

MSD labels are 1 oedema, 2 non-enhancing tumour, 3 enhancing tumour. Results are reported on the
standard BraTS composite regions, whole tumour (1+2+3), tumour core (2+3) and enhancing tumour (3),
so they are comparable with the literature. The mapping lives in the config, not in code.

## Environment: uv and the CUDA 12.6 PyTorch build

uv provisions Python 3.11 itself, so a clean checkout needs only uv and make. Versions are pinned to
the minor release in `pyproject.toml`; `uv pip install --torch-backend cu126` pulls the CUDA build of
the same pinned torch, which keeps the version in one place. cu126 is the CUDA 12.x build of torch
2.14 and runs on older drivers than the 13.x builds.

## Strict config loading

Unknown keys, missing keys and wrong types raise at load time instead of being defaulted. PyYAML
follows YAML 1.1, where `1e-4` without a decimal point is a string. The loader rejects that rather
than letting it surface as an obscure error inside the optimiser.

## CPU budget

The reference machine pairs an RTX 3090 with a 4-core CPU, so the CPU, not the GPU, is the likely
bottleneck. The config runs no data-loader worker processes and caps CPU threads. Decompressing the
NIfTI files is the one unavoidable CPU cost, so it happens once (next entry). After that the only
per-sample CPU work is reading a cached array; the copy to the GPU comes first in the MONAI
pipeline, and cropping, flipping and intensity jitter all run there.

## Preprocessing cache

`make data` loads each case once, crops it to the bounding box of the brain mask plus any tumour
voxels, z-scores each channel with the mean and standard deviation of the brain voxels, and caches
the result under `data/processed/` as float16 (z-scores do not need more precision) with the label
as uint8. The brain mask is "non-zero in any channel": the MSD volumes are skull-stripped with an
exact-zero background. Background stays zero after normalisation. Per-case provenance (crop box,
raw intensity ranges, normalisation statistics, label volumes) is stored next to the arrays. The
run resumes where it stopped, because a case is marked complete only after all its files exist.

## Patient identity: repeat scans are linked before splitting

MSD Task01 ships no patient identifiers, and a case id is not a patient. Every pair of cases is
compared in the shared atlas space (cosine similarity of 4x downsampled, z-scored volumes). This
finds two kinds of repeats (`reports/dataset_summary.md`, `reports/figures/repeat_scans.png`):

- 228 cases whose image appears verbatim under a second case id (similarity 1.0), each pairing a
  case id in 1–274 with one in 275–484: the same scan released twice. In 108 of the 114 pairs the
  two label maps differ.
- Runs of consecutive case ids whose scans are similar but not identical, consistent with one
  patient imaged at several timepoints. BRATS_175–180 is one: enhancing tumour is absent in the
  first four scans and present in the last two.

A case-level split with the same seed would put a scan of the same patient in training for 42 of
73 test cases. Cases with similarity >= 0.55 are therefore linked, transitively, into one patient
group, and a group never straddles splits (a test enforces this on the committed split). A missed
link leaks a patient into evaluation, while a spurious one only makes the split a little lumpier,
so the threshold errs low. `reports/link_threshold_sweep.csv` shows the largest group holding at 9
cases from 0.52 to 0.81, then growing quickly below 0.51 as linking chains unrelated patients
together. 0.55 sits at the inclusive end of that stable range, with margin above where chaining
starts.

## Split strategy

Whole patient groups are assigned to train/validation/test at 70/15/15 from the config seed. The
assignment is stratified by which evaluation regions are empty. Only ET is ever empty (12 cases),
and Dice is undefined on an empty ground truth, so a split holding many such cases would report a
different ET mean for reasons unrelated to the model. Held-out quotas are never overshot; a group
that does not fit goes to training. The result is 342/71/71. The patient constraint wins over
stratification: BRATS_175–180 holds 4 of the 12 empty-ET cases and has to stay together, which
leaves one empty-ET case each in validation and test. `make data` recomputes the split on every
run and stops if it differs from the committed `configs/splits.json`, so the test split cannot
move silently.

## Unlabelled MSD test images are not used

`imagesTs/` has no labels. All three splits are carved from the 484 labelled cases.
