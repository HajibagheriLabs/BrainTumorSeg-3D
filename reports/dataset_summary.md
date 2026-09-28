# Dataset summary

Written by `scripts/inspect_data.py` from the preprocessed cache; do not edit.

- Labelled cases: 484
- Volume shapes: 240 x 240 x 155
- Voxel spacing (mm): 1 x 1 x 1
- Tumour voxels outside the brain mask: 96
- Cases whose image reappears under another case id (similarity >= 0.999): 228
- Duplicate pairs whose two label maps differ: 108 of 114
- Repeat-scan groups linked at similarity >= 0.55: 129, covering 320 cases
- Lowest nearest-case similarity among linked cases: 0.5526
- Highest nearest-case similarity among unlinked cases: 0.5446
- A case-level split with the same seed would leave 38 of 73 val, 42 of 73 test cases with a scan of the same patient in train

## Cases per split and empty labels

Scored cases count each duplicated scan once, see DECISIONS.md.

| split | cases | scored | empty WT | empty TC | empty ET |
|---|---:|---:|---:|---:|---:|
| train | 342 | 262 | 0 | 0 | 10 |
| val | 71 | 56 | 0 | 0 | 1 |
| test | 71 | 52 | 0 | 0 | 1 |
| all | 484 | 370 | 0 | 0 | 12 |

## Tumour volume per region (mL, non-empty cases)

| region | cases | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|
| WT | 484 | 7.29 | 57.44 | 94.22 | 147.55 | 318.35 |
| TC | 484 | 0.09 | 13.81 | 31.98 | 56.58 | 133.75 |
| ET | 472 | 0.01 | 6.16 | 15.35 | 30.07 | 116.73 |
