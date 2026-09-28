"""Dataset parsing, patient-level splits and per-case normalisation."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from braintumorseg.config import Config


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


def read_meta(processed_dir: Path, case_ids: list[str]) -> list[dict]:
    return [
        json.loads((processed_dir / "meta" / f"{case_id}.json").read_text("utf-8"))
        for case_id in case_ids
    ]
