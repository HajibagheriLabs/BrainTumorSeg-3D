import dataclasses
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from braintumorseg.config import load_config
from braintumorseg.data import SPLITS, make_splits, read_splits, zscore_normalise
from braintumorseg.transforms import train_transforms

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "configs" / "unet3d.yaml"


def test_committed_splits_share_no_patient() -> None:
    splits = read_splits(REPO / "configs" / "splits.json")
    members = {name: set(splits[name]) for name in SPLITS}
    assert all(len(members[name]) == len(splits[name]) for name in SPLITS)
    for first, second in combinations(SPLITS, 2):
        assert not members[first] & members[second], f"{first} and {second} overlap"

    # repeat scans of one patient carry different case ids, so they must travel together
    split_of = {case_id: name for name in SPLITS for case_id in members[name]}
    for group in splits["linked_groups"]:
        assert len({split_of[case_id] for case_id in group}) == 1, group


def test_split_is_deterministic_under_fixed_seed() -> None:
    fractions = load_config(BASELINE).data.split
    ids = [f"case_{i:03d}" for i in range(92)]
    groups = [(case_id,) for case_id in ids[:90]] + [tuple(ids[90:])]
    strata = {case_id: "ET" if i % 10 == 0 else "" for i, case_id in enumerate(ids)}

    first = make_splits(groups, strata, fractions, seed=7)
    assert make_splits(groups, strata, fractions, seed=7) == first
    assert make_splits(groups, strata, fractions, seed=8) != first
    assert sorted(case_id for split in first.values() for case_id in split) == ids


def test_training_patch_shape_and_dtype(tmp_path: Path) -> None:
    cfg = load_config(BASELINE)
    patch = (32, 32, 32)
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, patch_size=patch)
    )
    rng = np.random.default_rng(0)
    # the second axis is shorter than the patch, which exercises the padding path
    shape = (40, 24, 36)
    label = np.zeros((1, *shape), dtype=np.uint8)
    label[0, 10:20, 5:15, 10:20] = rng.integers(1, 4, size=(10, 10, 10))
    np.save(tmp_path / "image.npy", rng.standard_normal((4, *shape)).astype(np.float16))
    np.save(tmp_path / "label.npy", label)
    record = {
        "case_id": "synthetic",
        "image": str(tmp_path / "image.npy"),
        "label": str(tmp_path / "label.npy"),
    }

    samples = train_transforms(cfg, torch.device("cpu"))(record)

    assert len(samples) == 1
    image, target = samples[0]["image"], samples[0]["label"]
    assert image.shape == (4, *patch) and image.dtype == torch.float32
    assert target.shape == (1, *patch) and target.dtype == torch.uint8
    assert set(torch.unique(target).tolist()) <= {0, 1, 2, 3}


def test_normalisation_is_standardised_inside_mask_per_channel() -> None:
    generator = torch.Generator().manual_seed(0)
    # channels on wildly different scales, as raw mri sequences are
    scales = torch.tensor([1.0, 40.0, 900.0, 5000.0]).view(-1, 1, 1, 1)
    image = torch.rand((4, 20, 24, 16), generator=generator) * scales + 50.0
    mask = torch.zeros((20, 24, 16), dtype=torch.bool)
    mask[4:16, 5:20, 3:13] = True
    image[:, ~mask] = 0.0

    normalised = zscore_normalise(image, mask)

    inside = normalised[:, mask].double()
    assert torch.allclose(
        inside.mean(dim=1), torch.zeros(4, dtype=torch.float64), atol=1e-5
    )
    assert torch.allclose(
        inside.std(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-5
    )
    assert (normalised[:, ~mask] == 0).all()
