from pathlib import Path

import pytest
import yaml

from braintumorseg.config import load_config, save_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASELINE = CONFIGS / "unet3d.yaml"

INVALID = [
    ("train.momentum", 0.9, "unknown keys"),
    ("data.split.test", 0.3, "sum to 1"),
    # pyyaml reads an unquoted 1e-4 as a string, which is what this simulates
    ("train.learning_rate", "1e-4", "learning_rate must be a number"),
    ("train.patch_size", [100, 128, 128], "divisible"),
    ("inference.overlap", 1.0, "overlap"),
    ("data.regions.ET", [4], "foreground classes"),
]


def test_every_config_loads_and_round_trips(tmp_path: Path) -> None:
    baseline = load_config(BASELINE)
    assert isinstance(baseline.data.root, Path)
    assert isinstance(baseline.train.learning_rate, float)
    assert len(baseline.train.patch_size) == 3

    configs = sorted(CONFIGS.rglob("*.yaml"))
    assert len(configs) > 1
    for path in configs:
        cfg = load_config(path)
        resolved = tmp_path / "resolved.yaml"
        save_config(cfg, resolved)
        assert load_config(resolved) == cfg, path
        # an ablation inherits the split, so no config can quietly move the test set
        assert cfg.data.split == baseline.data.split, path


def test_config_rejects_invalid_values(tmp_path: Path) -> None:
    for dotted, value, message in INVALID:
        raw = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        *parents, key = dotted.split(".")
        node = raw
        for parent in parents:
            node = node[parent]
        node[key] = value
        variant = tmp_path / "variant.yaml"
        variant.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_config(variant)
