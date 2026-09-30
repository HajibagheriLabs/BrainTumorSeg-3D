CONFIG ?= configs/unet3d.yaml
# cu126 is the cuda 12.x build of the pinned torch; use cpu for a machine without an nvidia gpu
TORCH_BACKEND ?= cu126
# inference-only runs first and the slowest architecture last, so partial results come early
ABLATIONS = \
  configs/ablations/overlap_000.yaml \
  configs/ablations/overlap_025.yaml \
  configs/ablations/overlap_075.yaml \
  configs/ablations/patch_96.yaml \
  configs/ablations/augment_off.yaml \
  configs/ablations/loss_dice.yaml \
  configs/ablations/loss_dice_focal.yaml \
  configs/ablations/seed_43.yaml \
  configs/ablations/attention_unet.yaml

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: setup env test lint format data crosscheck train eval ablate report

setup:
	uv venv --allow-existing --python 3.11 .venv
	uv pip install --python $(PY) --torch-backend $(TORCH_BACKEND) -e ".[dev]"
	$(PY) scripts/check_env.py

env:
	$(PY) scripts/check_env.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

data:
	$(PY) -m braintumorseg.cli data --config $(CONFIG)
	$(PY) scripts/inspect_data.py --config $(CONFIG)
	$(PY) scripts/annotation_agreement.py --config $(CONFIG)

crosscheck:
	$(PY) scripts/crosscheck_metrics.py --config $(CONFIG)

train:
	$(PY) -m braintumorseg.cli train --config $(CONFIG)

eval:
	$(PY) -m braintumorseg.cli eval --config $(CONFIG)

ablate:
	for config in $(ABLATIONS); do \
	    $(PY) -m braintumorseg.cli train --config $$config || exit 1; \
	    $(PY) -m braintumorseg.cli eval --config $$config || exit 1; \
	done

report:
	$(PY) scripts/report.py --config $(CONFIG)
