"""Dataset parsing, patient-level splits and per-case normalisation."""

import dataclasses
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from braintumorseg.config import Config, SplitConfig

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    label: Path


@dataclass(frozen=True)
class DatasetIndex:
    cases: tuple[Case, ...]
    modalities: tuple[str, ...]
    labels: dict[int, str]


def read_index(root: Path) -> DatasetIndex:
    """Labelled cases listed in the MSD dataset.json, sorted by case id."""
    manifest_path = root / "dataset.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found: extract Task01_BrainTumour from "
            f"http://medicaldecathlon.com into {root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["training"]:
        image, label = root / entry["image"], root / entry["label"]
        if image.name != label.name:
            raise ValueError(f"image and label differ: {image.name} vs {label.name}")
        cases.append(Case(image.name.removesuffix(".nii.gz"), image, label))
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids) or len(ids) != manifest["numTraining"]:
        raise ValueError(
            "dataset.json has duplicate cases or disagrees with numTraining"
        )
    modality = manifest["modality"]
    return DatasetIndex(
        cases=tuple(sorted(cases, key=lambda case: case.case_id)),
        modalities=tuple(modality[str(i)] for i in range(len(modality))),
        labels={int(value): name for value, name in manifest["labels"].items()},
    )


def load_case(case: Case) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Image as (channel, x, y, z) float32, label as (x, y, z) uint8, spacing in mm."""
    image_nii, label_nii = nib.load(case.image), nib.load(case.label)
    if not np.allclose(image_nii.affine, label_nii.affine):
        raise ValueError(f"{case.case_id}: image and label affines differ")
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    label = np.asarray(label_nii.dataobj, dtype=np.uint8)
    if image.ndim != 4 or image.shape[:3] != label.shape:
        raise ValueError(f"{case.case_id}: image {image.shape} vs label {label.shape}")
    spacing = tuple(float(zoom) for zoom in image_nii.header.get_zooms()[:3])
    return np.moveaxis(image, -1, 0), label, spacing


def brain_mask(image: torch.Tensor) -> torch.Tensor:
    # msd volumes are skull-stripped with an exact zero background
    return (image != 0).any(dim=0)


def channel_stats(
    image: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    voxels = image[:, mask].double()
    if voxels.shape[1] < 2:
        raise ValueError("mask selects fewer than two voxels")
    mean, std = voxels.mean(dim=1), voxels.std(dim=1)
    if (std == 0).any():
        raise ValueError(f"channel is constant inside the mask, std={std.tolist()}")
    return mean, std


def zscore_normalise(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Z-score each channel over the masked voxels and zero everything outside."""
    mean, std = channel_stats(image, mask)
    shape = (-1,) + (1,) * (image.ndim - 1)
    normalised = (image.double() - mean.view(shape)) / std.view(shape)
    return (normalised * mask).to(image.dtype)


def bounding_box(mask: torch.Tensor) -> tuple[slice, ...]:
    if not mask.any():
        raise ValueError("an empty mask has no bounding box")
    box = []
    for axis in range(mask.ndim):
        others = tuple(a for a in range(mask.ndim) if a != axis)
        hits = torch.nonzero(mask.any(dim=others)).flatten()
        box.append(slice(int(hits[0]), int(hits[-1]) + 1))
    return tuple(box)


def _intensity_summary(image: torch.Tensor, mask: torch.Tensor) -> list[dict]:
    summary = []
    for channel in image:
        voxels = channel[mask].double()
        low, high = torch.quantile(voxels, voxels.new_tensor([0.01, 0.99])).tolist()
        summary.append(
            {
                "min": voxels.min().item(),
                "p01": low,
                "p99": high,
                "max": voxels.max().item(),
                "mean": voxels.mean().item(),
                "std": voxels.std().item(),
            }
        )
    return summary


def _write_atomic(path: Path, write: Any) -> None:
    # write-then-rename so an interrupted run never leaves a truncated file behind
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        write(handle)
    os.replace(partial, path)


def _save_array(path: Path, tensor: torch.Tensor) -> None:
    _write_atomic(path, lambda handle: np.save(handle, tensor.cpu().numpy()))


def preprocess_case(
    case: Case,
    index: DatasetIndex,
    out_dir: Path,
    thumbnail_factor: int,
    device: torch.device,
) -> None:
    """Crop to the brain, z-score, and cache arrays plus provenance for one case."""
    image_np, label_np, spacing = load_case(case)
    image = torch.from_numpy(image_np).to(device).contiguous()
    label = torch.from_numpy(label_np).to(device)
    if int(label.max()) >= len(index.labels):
        raise ValueError(
            f"{case.case_id}: label {int(label.max())} is not in dataset.json"
        )

    brain = brain_mask(image)
    tumour = label > 0
    box = bounding_box(brain | tumour)
    normalised = zscore_normalise(image, brain)
    # thumbnails stay in the shared atlas space so repeat scans can be matched later
    thumbnail = F.avg_pool3d(normalised[None], kernel_size=thumbnail_factor)[0]

    _save_array(out_dir / "images" / f"{case.case_id}.npy", normalised[:, *box].half())
    _save_array(out_dir / "labels" / f"{case.case_id}.npy", label[box][None])
    _save_array(out_dir / "thumbnails" / f"{case.case_id}.npy", thumbnail.half())

    counts = torch.bincount(label.flatten().long(), minlength=len(index.labels))
    intensity = _intensity_summary(image, brain)
    meta = {
        "case_id": case.case_id,
        "shape": list(label_np.shape),
        "spacing": list(spacing),
        "crop_start": [s.start for s in box],
        "crop_stop": [s.stop for s in box],
        "brain_voxels": int(brain.sum()),
        "tumour_voxels_outside_brain": int((tumour & ~brain).sum()),
        "label_voxels": {index.labels[v]: int(n) for v, n in enumerate(counts)},
        "intensity": dict(zip(index.modalities, intensity, strict=True)),
        "thumbnail_factor": thumbnail_factor,
    }
    # the metadata file is written last and marks the case as complete
    text = json.dumps(meta, indent=1).encode("utf-8")
    _write_atomic(out_dir / "meta" / f"{case.case_id}.json", lambda h: h.write(text))


def preprocess_dataset(cfg: Config, device: torch.device) -> DatasetIndex:
    index = read_index(cfg.data.root)
    out_dir = cfg.data.processed_dir
    todo = [
        c for c in index.cases if not (out_dir / "meta" / f"{c.case_id}.json").exists()
    ]
    print(f"preprocessing {len(todo)} of {len(index.cases)} cases into {out_dir}")
    for done, case in enumerate(todo, start=1):
        preprocess_case(case, index, out_dir, cfg.data.link.thumbnail_factor, device)
        if done % 25 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)}", flush=True)
    return index


def full_label(processed_dir: Path, meta: dict) -> np.ndarray:
    """A cached label pasted back into its scan's original grid."""
    label = np.zeros(meta["shape"], dtype=np.uint8)
    box = tuple(
        slice(a, b) for a, b in zip(meta["crop_start"], meta["crop_stop"], strict=True)
    )
    label[box] = np.load(processed_dir / "labels" / f"{meta['case_id']}.npy")[0]
    return label


def read_meta(processed_dir: Path, case_ids: list[str]) -> list[dict]:
    return [
        json.loads((processed_dir / "meta" / f"{case_id}.json").read_text("utf-8"))
        for case_id in case_ids
    ]


def region_voxels(
    meta: dict, index: DatasetIndex, regions: dict[str, tuple[int, ...]]
) -> dict[str, int]:
    return {
        region: sum(meta["label_voxels"][index.labels[value]] for value in values)
        for region, values in regions.items()
    }


def thumbnail_similarity(
    processed_dir: Path, case_ids: list[str], device: torch.device
) -> torch.Tensor:
    """Pairwise cosine similarity of the atlas-space thumbnails, shape (n, n)."""
    vectors = torch.stack(
        [
            torch.from_numpy(np.load(processed_dir / "thumbnails" / f"{case_id}.npy"))
            for case_id in case_ids
        ]
    )
    vectors = F.normalize(vectors.flatten(1).to(device, torch.float32), dim=1)
    return vectors @ vectors.T


def link_repeat_scans(
    case_ids: list[str], similarity: torch.Tensor, threshold: float
) -> list[tuple[str, ...]]:
    """Group cases whose scans are similar enough to belong to the same patient."""
    parent = list(range(len(case_ids)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    linked = torch.triu(similarity >= threshold, diagonal=1)
    for i, j in torch.nonzero(linked).tolist():
        parent[root(i)] = root(j)
    groups: dict[int, list[str]] = {}
    for i, case_id in enumerate(case_ids):
        groups.setdefault(root(i), []).append(case_id)
    return sorted(tuple(sorted(group)) for group in groups.values())


def make_splits(
    groups: list[tuple[str, ...]],
    strata: dict[str, str],
    fractions: SplitConfig,
    seed: int,
) -> dict[str, list[str]]:
    """Assign whole groups to splits so each split gets its share of every stratum."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[tuple[str, ...]]] = {}
    for group in sorted(groups):
        key = "|".join(sorted({strata[case_id] for case_id in group}))
        by_stratum.setdefault(key, []).append(group)

    splits: dict[str, list[str]] = {name: [] for name in SPLITS}
    for key in sorted(by_stratum):
        members = by_stratum[key]
        rng.shuffle(members)
        total = sum(len(group) for group in members)
        quota = {
            "val": round(total * fractions.val),
            "test": round(total * fractions.test),
        }
        filled = dict.fromkeys(quota, 0)
        for group in members:
            # held-out quotas are never overshot; a group that does not fit trains instead
            fits = (n for n in ("test", "val") if filled[n] + len(group) <= quota[n])
            name = next(fits, "train")
            if name != "train":
                filled[name] += len(group)
            splits[name].extend(group)
    return {name: sorted(ids) for name, ids in splits.items()}


def build_splits(cfg: Config, index: DatasetIndex, device: torch.device) -> dict:
    case_ids = [case.case_id for case in index.cases]
    processed = cfg.data.processed_dir
    metas = read_meta(processed, case_ids)
    factors = {meta["thumbnail_factor"] for meta in metas}
    if factors != {cfg.data.link.thumbnail_factor}:
        raise ValueError(
            f"thumbnails were built with factor {factors}, config says "
            f"{cfg.data.link.thumbnail_factor}: delete {processed} and rerun make data"
        )
    similarity = thumbnail_similarity(processed, case_ids, device)
    groups = link_repeat_scans(case_ids, similarity, cfg.data.link.threshold)
    # an empty region makes its dice undefined, so every split gets its share of those cases
    strata = {}
    for meta in metas:
        volumes = region_voxels(meta, index, cfg.data.regions)
        strata[meta["case_id"]] = ",".join(r for r, n in volumes.items() if n == 0)
    splits = make_splits(groups, strata, cfg.data.split, cfg.seed)
    identical = torch.triu(similarity >= cfg.data.link.duplicate_threshold, diagonal=1)
    return {
        "seed": cfg.seed,
        "fractions": dataclasses.asdict(cfg.data.split),
        "link_threshold": cfg.data.link.threshold,
        "linked_groups": [list(group) for group in groups if len(group) > 1],
        "exact_duplicates": [
            [case_ids[i], case_ids[j]] for i, j in torch.nonzero(identical).tolist()
        ],
        **splits,
    }


def ensure_splits(cfg: Config, index: DatasetIndex, device: torch.device) -> dict:
    """Write the split file, or check that the committed one is what the config yields."""
    splits = build_splits(cfg, index, device)
    path = cfg.data.splits_file
    if path.exists():
        if read_splits(path) != splits:
            raise ValueError(
                f"{path} differs from the split this config produces; the test split "
                "must not move silently, so delete the file only if the change is intended"
            )
        return splits
    path.write_text(json.dumps(splits, indent=1) + "\n", "utf-8", newline="\n")
    return splits


def read_splits(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation_cases(splits: dict, split: str) -> list[str]:
    """Cases scored for a split: a scan stored twice counts once, under its higher id."""
    superseded = {lower for lower, _ in splits["exact_duplicates"]}
    return [case_id for case_id in splits[split] if case_id not in superseded]


def case_records(cfg: Config, case_ids: list[str]) -> list[dict[str, str]]:
    processed = cfg.data.processed_dir
    return [
        {
            "case_id": case_id,
            "image": str(processed / "images" / f"{case_id}.npy"),
            "label": str(processed / "labels" / f"{case_id}.npy"),
        }
        for case_id in case_ids
    ]
