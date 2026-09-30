"""Command-line entrypoint behind the make targets."""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import torch

from braintumorseg.config import Config, load_config
from braintumorseg.data import (
    SPLITS,
    ensure_splits,
    evaluation_cases,
    preprocess_dataset,
    read_splits,
)
from braintumorseg.inference import evaluate_cases
from braintumorseg.metrics import STATISTICS, summarise_cases
from braintumorseg.train import (
    load_best_model,
    record_identity,
    run_dir,
    tracked_run,
    train,
)

COMMANDS = ("data", "train", "eval")
# the test split is scored once, after the configuration is frozen; until then only val
EVAL_SPLITS = ("val",)


def runtime_device(cfg: Config) -> torch.device:
    torch.set_num_threads(cfg.runtime.num_threads)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("runtime.device is cuda but no cuda device is available")
    return torch.device(cfg.runtime.device)


def run_data(cfg: Config) -> None:
    device = runtime_device(cfg)
    index = preprocess_dataset(cfg, device)
    splits = ensure_splits(cfg, index, device)
    sizes = ", ".join(f"{name} {len(splits[name])}" for name in SPLITS)
    linked = sum(len(group) for group in splits["linked_groups"])
    print(f"splits in {cfg.data.splits_file}: {sizes}")
    print(f"{linked} cases in {len(splits['linked_groups'])} linked repeat-scan groups")


def run_eval(cfg: Config, split: str) -> None:
    device = runtime_device(cfg)
    model = load_best_model(cfg, device)
    case_ids = evaluation_cases(read_splits(cfg.data.splits_file), split)
    out = run_dir(cfg)
    rows = evaluate_cases(model, cfg, case_ids, device, out / f"{split}_predictions")
    summary = pd.DataFrame(summarise_cases(rows, list(cfg.data.regions)))
    options = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    pd.DataFrame(rows).to_csv(out / f"{split}_cases.csv", **options)
    summary.to_csv(out / f"{split}_summary.csv", **options)
    with tracked_run(cfg) as run:
        # an inference-only ablation has no training run to describe it, so describe it here
        if not run.data.params:
            record_identity(cfg, device)
        for row in summary.to_dict("records"):
            prefix = f"{split}_{row['region']}_{row['metric']}"
            mlflow.log_metrics({f"{prefix}_{stat}": row[stat] for stat in STATISTICS})
        mlflow.log_artifact(str(out / f"{split}_cases.csv"))
        mlflow.log_artifact(str(out / f"{split}_summary.csv"))
    print(summary.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="braintumorseg")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=EVAL_SPLITS, default="val")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "data":
        run_data(cfg)
    elif args.command == "train":
        train(cfg, runtime_device(cfg))
    else:
        run_eval(cfg, args.split)


if __name__ == "__main__":
    main()
