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

## Metric definitions

Every metric is computed per BraTS region from the merged labels, so a voxel labelled with the
wrong tumour class still counts as found for whole tumour.

- Dice = 2|P ∩ T| / (|P| + |T|).
- HD95 is computed between surfaces, where a surface is the set of mask voxels with at least one
  face neighbour outside the mask. For every surface voxel of one mask, take the distance in mm
  (using the voxel spacing) to the nearest surface voxel of the other. HD95 is the larger of the
  two directed 95th percentiles, with linear interpolation.

This is the definition MONAI uses. Two checks back the implementation in `metrics.py`. First,
hand-computed unit tests, which also fail on each classic mistake injected into the code: IoU in
place of Dice, a one-sided denominator, one-directional HD95, whole masks in place of surfaces,
ignoring spacing, and the maximum in place of the 95th percentile. Second, `make crosscheck`
compares against MONAI on 36 real label pairs: Dice agrees to 2.6e-8 and HD95 to 6.0e-6 mm
(`reports/metric_crosscheck.csv`). The implementation runs on the GPU, because MONAI's distance
transform needs cuCIM for that, and cuCIM has no Windows build.

## Empty ground truth

Dice is 0/0 when the truth is empty, so it is undefined; this project does not assign it a value.

| truth | prediction | Dice | HD95 |
|---|---|---|---|
| empty | empty | undefined | undefined |
| empty | non-empty | undefined | undefined |
| non-empty | empty | 0 | field-of-view diagonal (373.13 mm here) |
| non-empty | non-empty | computed | computed |

Undefined cases are stored as NaN in per-case tables. They are left out of the distribution and
counted (`n_undefined`). Empty-truth cases are then scored as detection instead: every summary
row carries `empty_truth` and `empty_truth_predicted`, the number of false positives. A miss on
a non-empty truth is never excluded. It scores Dice 0 and the largest distance the scan allows,
because leaving it out would hide exactly the catastrophic failures this project is meant to
expose.

The BraTS challenge instead scores both-empty as Dice 1 and a false positive as Dice 0. That folds
a detection outcome into a segmentation average: with one empty-ET scan in each held-out split, a
single 0-or-1 value would move the test ET mean by about 1/52 for reasons unrelated to how well
tumour is outlined. Per-case tables keep the truth and predicted volumes, so BraTS-convention
figures can still be recomputed for comparison.

## Duplicated scans count once in evaluation

The 114 exact-duplicate pairs are one scan with two annotations. Scoring both copies would count
that scan twice, against two different truths. Evaluation therefore scores each scan once, against
the label of the higher case id: 56 of the 71 validation cases and 52 of the 71 test cases
(`reports/dataset_summary.md`). Every pair joins an id in 1–274 with one in 275–484, which
suggests two releases concatenated, so the higher id is treated as the later annotation. Without a
third reader there is no way to know which label is right. What matters is that the rule was fixed
before any prediction existed, so it cannot drift toward a better score. Training keeps both
labels: both copies are in the same split, and the model sees the annotation variability.

## Annotation agreement as a reference

The two labels of each duplicated scan are scored against each other with the same metrics
(`reports/annotation_agreement_summary.csv`). The median agreement between two annotations of an
identical image is Dice 0.933 (WT), 0.915 (TC) and 0.883 (ET), with HD95 of 3.46, 5.15 and
2.24 mm. The worst pairs fall to Dice 0.36 (TC) and 0.43 (ET). Model scores near these medians are
at the label-noise floor. A per-case difference smaller than this spread is not evidence that one
model is better than another.

## Training data lives on the GPU

MONAI's U-Net downsamples at its first layer, so a 128³ training step is cheap. Streaming cases from
disk would then make the CPU the bottleneck. The cropped float16 training and validation volumes fit
in the RTX 3090's 24 GB next to the model, so a `CacheDataset` loads them once and keeps them on the
GPU. Every random crop, flip and intensity jitter runs there, and the CPU only launches kernels.
The cast to float32 happens on the cropped patch, not the cached volume, which keeps the cache at
half size. On a smaller GPU, set `runtime.cache_on_device: false` to load each case per sample
instead.

## Output, loss and optimisation

- The network predicts the four MSD labels with a softmax, and the evaluation regions are formed
  from the predicted label map. Predicting the overlapping regions directly with sigmoids is a
  common alternative; it is left for the ablations to test, not assumed.
- The loss is Dice + cross-entropy. The Dice term ignores the background class, which fills most
  of every patch. It is computed over the batch rather than per patch, so a class missing from one
  patch does not produce an unstable 0/0 term.
- Half of the training patches are centred on tumour, so small lesions are sampled often enough to
  be learned.
- AdamW with a cosine schedule to zero. Mixed precision is float16 with a gradient scaler, which
  works on any CUDA GPU, where bfloat16 would need Ampere or newer.

## Model selection and what "best" means

The checkpoint kept is the one with the highest mean over regions of the per-case mean validation
Dice, checked every 5 epochs on the 56 unique validation scans. HD95 is too slow to recompute every
5 epochs, so it is measured once, on the selected checkpoint. The test split is not touched: the
command line refuses any evaluation split other than `val` until the configuration is frozen.

## Seeded, but not bitwise reproducible

Python, NumPy, PyTorch, the MONAI transform chain and the data loader order are all seeded from
the config. cuDNN autotuning stays on, and some GPU kernels are non-deterministic, so two runs of
the same config agree closely but not bit for bit. Forcing deterministic kernels would slow
training for a guarantee that no reported number depends on. Each MLflow run records the config,
seed, git commit, which paths were uncommitted, the GPU and the run directory. A run interrupted
mid-way resumes from its last checkpoint when `make train` is rerun.

## Training length: 300 epochs, and why that counts as converged

One epoch passes every training case once: 342 cases at batch size 2, so 171 steps. 300 epochs
with a cosine schedule to zero is in line with MONAI's BraTS recipes in samples seen. What decides
it is the curve (`reports/baseline_history.csv`). Over the last eight validations, epochs 265–300,
mean validation Dice stays between 0.8115 and 0.8135 while the learning rate anneals to zero.
Longer training would buy at most a change of that order, well inside the case-to-case spread.
The best checkpoint is from epoch 275.

The baseline run was interrupted once after epoch 9 and resumed from its epoch-5 checkpoint, so
epochs 6–9 were retrained. Its MLflow record names the one uncommitted path at launch,
`DECISIONS.md`; the code was exactly the recorded commit.

## Ablations: how they were run and judged

Each ablation is a config in `configs/ablations/` that extends the baseline and states one change.
Nothing in the source differs between runs, and a test checks that every config loads and keeps
the same split. All are trained for the same 300 epochs and scored on the same 56 validation scans
with their best-validation checkpoint. `reports/ablations.csv` holds the comparison,
`reports/results.csv` the full distributions and `reports/figures/ablations.png` the figure.

Differences are judged per scan, with a paired bootstrap (10,000 resamples, 95% interval) on mean
Dice and median HD95. That interval covers which scans happened to be in the validation split. It
does not cover training randomness, so the baseline was also retrained with another training seed
(the split has its own seed and does not move). The two baseline runs differ from each other by
+0.0065 mean Dice on WT [+0.0011, +0.0131] and +0.0147 on TC [+0.0012, +0.0307], both intervals
excluding zero. Judged by the bootstrap alone, the baseline "beats itself".

The rule is therefore: an ablation is better or worse only if its interval excludes zero against
both baseline runs, on the same side. The first version of the rule compared against one baseline
run and required the change to exceed the seed-to-seed difference. It flagged three cells as
better: Dice-only loss on TC (+0.0178) and Attention U-Net on WT (+0.0099) and TC (+0.0182).
Against the reseeded baseline the same three are +0.0031, +0.0034 and +0.0035, with intervals
spanning zero. The rule was tightened after seeing that, in the direction of claiming less.

## Ablations: what mattered and what did not

Nothing mattered. No ablation is better or worse than the baseline on any region, for Dice or
HD95. Of the 24 Dice cells, 7 differ from one baseline run but not the other and 17 differ from
neither. Median HD95 shows no detectable difference in any cell.

- **Loss (Dice, Dice + focal, against Dice + cross-entropy): no difference.** Dice-only is the
  case above: +0.018 on TC against one baseline run, +0.003 against the other. On WT it points
  the other way, -0.0004 and -0.007, the second with an interval excluding zero. Dice + focal
  differs from neither run on any region.
- **Patch size 96³ against 128³: no difference.** Mean Dice moves by -0.002, +0.002 and -0.003
  (WT, TC, ET) against the baseline, all intervals spanning zero. Against the reseeded run it is
  lower on all three, and the WT interval (-0.009) excludes zero. It trains in 0.70 hours against
  1.09.
- **Attention U-Net against the plain U-Net: no difference that survives the seed.** It has the
  highest mean Dice on every region (0.899, 0.824, 0.755), but against the reseeded baseline the
  gains are +0.003, +0.003 and +0.004. It takes 4.82 hours to train against 1.09.
- **Augmentation off: no difference that survives the seed.** Mean Dice is lower in all six
  comparisons, by -0.004, -0.004 and -0.001 against the baseline and by -0.011, -0.019 and -0.006
  against the reseeded one. Only the WT and TC intervals against the reseeded run exclude zero.
  That is consistent with augmentation helping a little, and not enough to say so.
- **Sliding-window overlap: inert at this patch size.** Overlap 0, 0.25 and 0.5 give identical
  scores, not just similar ones. The validation crops are 119–187 voxels per axis against a
  128-voxel window, so any overlap up to 0.5 places the same two windows per axis, one at each
  end. Overlap 0.75 adds windows and changes mean Dice by +0.0003 to +0.0005, intervals spanning
  zero.

These negative results are the finding. A single-seed comparison would have reported that
Dice-only loss and Attention U-Net improve tumour core by 0.018, with a bootstrap interval to
back it. Both claims disappear when the baseline is retrained. With one run per ablation, two of
the baseline and 56 scans, this design cannot resolve effects below about 0.01–0.02 mean Dice;
resolving them needs several seeds per config.

The configuration carried forward to the test split is therefore the baseline. No ablation earned
a change, and choosing the numerically highest one would be selecting on noise at more than four
times the training cost.
