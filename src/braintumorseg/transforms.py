"""MONAI transform pipelines for training and validation."""

import torch
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)

from braintumorseg.config import Config

KEYS = ("image", "label")


def _load(device: torch.device) -> list:
    # cropping and normalisation happened once in preprocessing; these steps are
    # deterministic, so a CacheDataset keeps their float16 output resident on the device
    return [
        LoadImaged(keys=KEYS, reader="NumpyReader", dtype=None, image_only=True),
        EnsureTyped(keys=KEYS, device=device, track_meta=False),
    ]


def train_transforms(cfg: Config, device: torch.device) -> Compose:
    patch = cfg.train.patch_size
    steps = _load(device) + [
        SpatialPadd(keys=KEYS, spatial_size=patch),
        RandCropByPosNegLabeld(
            keys=KEYS,
            label_key="label",
            spatial_size=patch,
            pos=cfg.train.foreground_crop_prob,
            neg=1.0 - cfg.train.foreground_crop_prob,
            num_samples=1,
        ),
        # cast the patch, not the cached volume, so the cache stays float16
        CastToTyped(keys=KEYS, dtype=(torch.float32, torch.uint8)),
    ]
    aug = cfg.augment
    if aug.enabled:
        steps += [
            RandFlipd(keys=KEYS, prob=aug.flip_prob, spatial_axis=axis)
            for axis in range(3)
        ]
        steps += [
            RandScaleIntensityd(
                keys="image",
                factors=aug.intensity_scale,
                prob=aug.intensity_prob,
                channel_wise=True,
            ),
            RandShiftIntensityd(
                keys="image",
                offsets=aug.intensity_shift,
                prob=aug.intensity_prob,
                channel_wise=True,
            ),
        ]
    return Compose(steps)


def val_transforms(device: torch.device) -> Compose:
    return Compose(_load(device))
