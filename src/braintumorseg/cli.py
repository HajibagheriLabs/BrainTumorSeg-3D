"""Command-line entrypoint behind the make targets."""

import argparse
from pathlib import Path

from braintumorseg.config import load_config

COMMANDS = ("data", "train", "eval", "report")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="braintumorseg")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    load_config(args.config)
    raise SystemExit(f"braintumorseg {args.command}: not implemented yet")


if __name__ == "__main__":
    main()
