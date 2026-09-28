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
bottleneck. The config runs no data-loader worker processes and caps CPU threads. The data pipeline
is to keep random augmentation on the GPU.
