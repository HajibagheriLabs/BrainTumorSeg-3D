"""Compare metrics.py with MONAI's independent Dice and HD95 on real label pairs."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance

from braintumorseg.cli import runtime_device
from braintumorseg.config import load_config
from braintumorseg.data import full_label, read_meta, read_splits
from braintumorseg.metrics import dice, field_of_view_diagonal, hd95, region_mask

# float64 here against monai's float32 distance transform; anything larger is a real bug
TOLERANCE = {"dice": 1e-6, "hd95": 1e-3}


def _pairs(splits: dict, count: int) -> list[tuple[str, str]]:
    # half re-annotated scans (small, realistic disagreements), half different patients
    # (large distances); both lists are deterministic so the report is reproducible
    duplicates = [tuple(pair) for pair in splits["exact_duplicates"][-(count // 2) :]]
    others = count - len(duplicates)
    cases = sorted(splits["train"] + splits["val"] + splits["test"])
    step = len(cases) // others
    strangers = [(cases[k * step], cases[k * step + step // 2]) for k in range(others)]
    return duplicates + strangers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = runtime_device(cfg)
    processed = cfg.data.processed_dir

    rows = []
    for first, second in _pairs(read_splits(cfg.data.splits_file), args.pairs):
        meta_a, meta_b = read_meta(processed, [first, second])
        labels_a = torch.from_numpy(full_label(processed, meta_a))
        labels_b = torch.from_numpy(full_label(processed, meta_b))
        penalty = field_of_view_diagonal(meta_b["shape"], meta_b["spacing"])
        for region, labels in cfg.data.regions.items():
            a, b = region_mask(labels_a, labels), region_mask(labels_b, labels)
            if not (a.any() and b.any()):
                continue
            # monai scores (batch, channel, ...) float masks on the cpu
            ref_a, ref_b = a[None, None].float(), b[None, None].float()
            row = {"case_a": first, "case_b": second, "region": region}
            row["dice"] = dice(a.to(device), b.to(device))
            row["monai_dice"] = compute_dice(
                ref_a, ref_b, include_background=True
            ).item()
            row["hd95"] = hd95(a.to(device), b.to(device), meta_b["spacing"], penalty)
            row["monai_hd95"] = compute_hausdorff_distance(
                ref_a,
                ref_b,
                include_background=True,
                percentile=95,
                spacing=meta_b["spacing"],
            ).item()
            rows.append(row)

    table = pd.DataFrame(rows)
    for metric, tolerance in TOLERANCE.items():
        table[f"{metric}_abs_diff"] = (table[metric] - table[f"monai_{metric}"]).abs()
        worst = table[f"{metric}_abs_diff"].max()
        print(f"{metric}: {len(table)} comparisons, max abs difference {worst:.3g}")
        if worst > tolerance:
            raise SystemExit(
                f"{metric} disagrees with monai by {worst:.3g} > {tolerance}"
            )
    table.to_csv(
        args.out / "metric_crosscheck.csv",
        index=False,
        lineterminator="\n",
        float_format="%.6g",
    )


if __name__ == "__main__":
    main()
