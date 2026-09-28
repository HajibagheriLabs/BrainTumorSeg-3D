CONFIG ?= configs/unet3d.yaml
# cu126 is the cuda 12.x build of the pinned torch; use cpu for a machine without an nvidia gpu
TORCH_BACKEND ?= cu126

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: setup env test lint format data train eval report

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

train:
	$(PY) -m braintumorseg.cli train --config $(CONFIG)

eval:
	$(PY) -m braintumorseg.cli eval --config $(CONFIG)

report:
	$(PY) -m braintumorseg.cli report --config $(CONFIG)
