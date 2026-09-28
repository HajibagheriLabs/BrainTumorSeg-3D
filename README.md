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

TBD. No model has been trained yet. Every number that appears here will be produced by a script in
this repository and written to `reports/`.

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

TBD. `make lint` and `make test` work now; `make data`, `make train`, `make eval` and `make report`
are wired to the command-line entrypoint and are filled in as the pipeline is built.

## License

Code is released under the MIT License, see [LICENSE](LICENSE).
