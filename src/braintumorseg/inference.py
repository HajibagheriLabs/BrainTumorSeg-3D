"""Sliding-window inference with Gaussian-weighted patch blending."""

from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from torch import nn

from braintumorseg.config import Config
from braintumorseg.data import case_records, read_meta
from braintumorseg.metrics import field_of_view_diagonal, score_case
from braintumorseg.transforms import val_transforms


def predict_labels(model: nn.Module, image: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Label map (x, y, z) for one (channel, x, y, z) volume from a model in eval mode."""
    device = image.device.type
    amp = cfg.train.amp and device == "cuda"
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=amp):
        logits = sliding_window_inference(
            image[None].float(),
            roi_size=cfg.train.patch_size,
            sw_batch_size=cfg.inference.sw_batch_size,
            predictor=model,
            overlap=cfg.inference.overlap,
            mode=cfg.inference.blend_mode,
        )
    return logits.argmax(dim=1)[0].to(torch.uint8)


def evaluate_cases(
    model: nn.Module,
    cfg: Config,
    case_ids: list[str],
    device: torch.device,
    prediction_dir: Path,
) -> list[dict]:
    """Per-case Dice, HD95 and volumes per region; the label maps go to prediction_dir."""
    prediction_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(case_records(cfg, case_ids), val_transforms(device))
    metas = read_meta(cfg.data.processed_dir, case_ids)
    model.eval()
    rows = []
    for sample, meta in zip(dataset, metas, strict=True):
        pred = predict_labels(model, sample["image"], cfg)
        # stored like the cached labels, so later analysis never needs a second forward pass
        np.save(prediction_dir / f"{meta['case_id']}.npy", pred[None].cpu().numpy())
        penalty = field_of_view_diagonal(meta["shape"], meta["spacing"])
        truth = sample["label"][0]
        scores = score_case(pred, truth, cfg.data.regions, meta["spacing"], penalty)
        rows.append({"case_id": meta["case_id"]} | scores)
    return rows
