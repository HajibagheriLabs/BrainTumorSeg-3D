import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from braintumorseg.config import load_config
from braintumorseg.inference import evaluate_cases
from braintumorseg.train import load_best_model, train

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "unet3d.yaml"
SHAPE = (20, 20, 20)


def _write_case(processed: Path, case_id: str, rng: np.random.Generator) -> None:
    label = np.zeros((1, *SHAPE), dtype=np.uint8)
    label[0, 6:14, 6:14, 6:14] = 1
    label[0, 8:12, 8:12, 8:12] = 2
    label[0, 9:11, 9:11, 9:11] = 3
    image = rng.standard_normal((4, *SHAPE)).astype(np.float16)
    for kind, array in (("images", image), ("labels", label)):
        (processed / kind).mkdir(parents=True, exist_ok=True)
        np.save(processed / kind / f"{case_id}.npy", array)
    meta = {"case_id": case_id, "shape": list(SHAPE), "spacing": [1.0, 1.0, 1.0]}
    (processed / "meta").mkdir(exist_ok=True)
    (processed / "meta" / f"{case_id}.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def test_two_training_steps_produce_a_checkpoint_that_evaluates(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    processed = tmp_path / "processed"
    for case_id in ("case_0", "case_1", "case_2"):
        _write_case(processed, case_id, rng)
    splits = {"train": ["case_0", "case_1"], "val": ["case_2"], "test": []}
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps(splits | {"exact_duplicates": []}), encoding="utf-8"
    )

    raw = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    raw["data"] |= {"processed_dir": str(processed), "splits_file": str(splits_file)}
    raw["model"] |= {"channels": [4, 8, 16], "strides": [2, 2], "num_res_units": 1}
    # two training cases at batch size 1 for one epoch: exactly two optimiser steps
    raw["train"] |= {
        "patch_size": [16, 16, 16],
        "batch_size": 1,
        "epochs": 1,
        "amp": False,
    }
    raw["runtime"] |= {
        "device": "cpu",
        "output_dir": str(tmp_path / "runs"),
        "tracking_dir": str(tmp_path / "mlruns"),
    }
    config_file = tmp_path / "smoke.yaml"
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(config_file)
    cpu = torch.device("cpu")

    out = train(cfg, cpu)

    assert (out / "best.pt").exists() and (out / "config.yaml").exists()
    assert pd.read_csv(out / "history.csv")["steps"].tolist() == [2]
    rows = evaluate_cases(load_best_model(cfg, cpu), cfg, ["case_2"], cpu)
    assert rows[0]["case_id"] == "case_2"
    scored = {f"{region}_{m}" for region in cfg.data.regions for m in ("dice", "hd95")}
    assert scored <= set(rows[0])
