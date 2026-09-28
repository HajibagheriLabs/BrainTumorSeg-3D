"""Score the two label maps of every scan MSD ships twice: a human agreement reference."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from braintumorseg.cli import runtime_device
from braintumorseg.config import load_config
from braintumorseg.data import SPLITS, full_label, read_meta, read_splits
from braintumorseg.metrics import field_of_view_diagonal, score_case, summarise_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = runtime_device(cfg)
    splits = read_splits(cfg.data.splits_file)
    split_of = {case_id: name for name in SPLITS for case_id in splits[name]}
    processed = cfg.data.processed_dir

    rows = []
    for superseded, kept in splits["exact_duplicates"]:
        old, new = read_meta(processed, [superseded, kept])
        # the kept label plays the truth, as it does when models are evaluated
        truth = torch.from_numpy(full_label(processed, new)).to(device)
        other = torch.from_numpy(full_label(processed, old)).to(device)
        penalty = field_of_view_diagonal(new["shape"], new["spacing"])
        scores = score_case(other, truth, cfg.data.regions, new["spacing"], penalty)
        rows.append(
            {"kept": kept, "superseded": superseded, "split": split_of[kept]} | scores
        )

    summary = pd.DataFrame(summarise_cases(rows, list(cfg.data.regions)))
    # fixed precision and lf keep the committed reports byte-stable across machines
    options = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    pd.DataFrame(rows).to_csv(args.out / "annotation_agreement.csv", **options)
    summary.to_csv(args.out / "annotation_agreement_summary.csv", **options)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
