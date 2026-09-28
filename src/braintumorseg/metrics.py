"""Per-region Dice and HD95 and their across-case distributions."""

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

STATISTICS = ("mean", "std", "median", "q25", "q75", "min", "max")
# caps each block of the surface distance matrix at 256 MB of float64
_PAIR_BUDGET = 2**25


def dice(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Dice of two boolean masks; nan when the truth is empty (see DECISIONS.md)."""
    truth_voxels = int(truth.sum())
    if truth_voxels == 0:
        return math.nan
    overlap = int((pred & truth).sum())
    return 2.0 * overlap / (int(pred.sum()) + truth_voxels)


def surface(mask: torch.Tensor) -> torch.Tensor:
    """Voxels of a 3D mask with at least one face neighbour outside it."""
    padded = F.pad(mask.to(torch.uint8), (1, 1, 1, 1, 1, 1)).bool()
    interior = mask.clone()
    for axis in range(3):
        for start in (0, 2):
            window = [slice(1, -1)] * 3
            window[axis] = slice(start, start + mask.shape[axis])
            interior &= padded[tuple(window)]
    return mask & ~interior


def _nearest_distances(
    source: torch.Tensor, target: torch.Tensor, spacing: torch.Tensor
) -> torch.Tensor:
    # float64 keeps the matrix-product form of cdist exact to well under a micrometre
    points = torch.nonzero(source).double() * spacing
    targets = torch.nonzero(target).double() * spacing
    rows = max(1, _PAIR_BUDGET // len(targets))
    return torch.cat(
        [torch.cdist(block, targets).amin(dim=1) for block in points.split(rows)]
    )


def hd95(
    pred: torch.Tensor,
    truth: torch.Tensor,
    spacing: Sequence[float],
    miss_penalty: float,
) -> float:
    """Symmetric 95th-percentile surface distance in mm; empty masks per DECISIONS.md."""
    if not truth.any():
        return math.nan
    if not pred.any():
        return miss_penalty
    step = torch.tensor(spacing, dtype=torch.float64, device=pred.device)
    pred_surface, truth_surface = surface(pred), surface(truth)
    forward = _nearest_distances(pred_surface, truth_surface, step)
    backward = _nearest_distances(truth_surface, pred_surface, step)
    return max(torch.quantile(d, 0.95).item() for d in (forward, backward))


def region_mask(label_map: torch.Tensor, labels: Sequence[int]) -> torch.Tensor:
    values = torch.tensor(labels, dtype=label_map.dtype, device=label_map.device)
    return torch.isin(label_map, values)


def field_of_view_diagonal(shape: Sequence[int], spacing: Sequence[float]) -> float:
    """Longest distance inside a scan, the HD95 charged for a missed region."""
    return math.dist(
        [0.0] * len(shape), [n * s for n, s in zip(shape, spacing, strict=True)]
    )


def score_case(
    pred: torch.Tensor,
    truth: torch.Tensor,
    regions: dict[str, tuple[int, ...]],
    spacing: Sequence[float],
    miss_penalty: float,
) -> dict[str, float]:
    """Dice, HD95 and volumes per region for one pair of 3D label maps."""
    voxel_ml = math.prod(spacing) / 1000.0
    row: dict[str, float] = {}
    for region, labels in regions.items():
        pred_mask, truth_mask = region_mask(pred, labels), region_mask(truth, labels)
        row[f"{region}_dice"] = dice(pred_mask, truth_mask)
        row[f"{region}_hd95"] = hd95(pred_mask, truth_mask, spacing, miss_penalty)
        row[f"{region}_truth_ml"] = int(truth_mask.sum()) * voxel_ml
        row[f"{region}_pred_ml"] = int(pred_mask.sum()) * voxel_ml
    return row


def summarise(values: Sequence[float]) -> dict[str, float]:
    """Distribution of a per-case metric; undefined (nan) cases are counted, not averaged."""
    scores = np.asarray(values, dtype=np.float64)
    defined = scores[~np.isnan(scores)]
    summary: dict[str, float] = {
        "n": len(defined),
        "n_undefined": len(scores) - len(defined),
    }
    if len(defined) == 0:
        return summary | dict.fromkeys(STATISTICS, math.nan)
    q25, median, q75 = np.percentile(defined, [25, 50, 75])
    return summary | {
        "mean": float(defined.mean()),
        "std": float(defined.std(ddof=1)) if len(defined) > 1 else math.nan,
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(defined.min()),
        "max": float(defined.max()),
    }


def summarise_cases(
    rows: Sequence[dict[str, float]], regions: Sequence[str]
) -> list[dict]:
    """A distribution row per region and metric, with the empty-truth cases counted."""
    table = []
    for region in regions:
        empty_truth = [row for row in rows if row[f"{region}_truth_ml"] == 0]
        false_positives = sum(row[f"{region}_pred_ml"] > 0 for row in empty_truth)
        for metric in ("dice", "hd95"):
            summary = summarise([row[f"{region}_{metric}"] for row in rows])
            table.append(
                {"region": region, "metric": metric}
                | summary
                | {
                    "empty_truth": len(empty_truth),
                    "empty_truth_predicted": false_positives,
                }
            )
    return table
