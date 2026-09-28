"""Command-line entrypoint behind the make targets."""

import argparse
from pathlib import Path

import torch

from braintumorseg.config import Config, load_config
from braintumorseg.data import SPLITS, ensure_splits, preprocess_dataset

COMMANDS = ("data", "train", "eval", "report")


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="braintumorseg")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "data":
        run_data(cfg)
        return
    raise SystemExit(f"braintumorseg {args.command}: not implemented yet")


if __name__ == "__main__":
    main()
