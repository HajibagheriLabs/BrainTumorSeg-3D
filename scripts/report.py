"""Tables and figures for one scored split of a trained config, written to reports/."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from braintumorseg.cli import EVAL_SPLITS
from braintumorseg.config import load_config
from braintumorseg.train import run_dir

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
# categorical slots 1-3, which stay distinguishable for colour-blind readers
REGION_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
METRIC_LABELS = {
    "dice": "Dice (higher is better)",
    "hd95": "HD95 in mm (lower is better)",
}
HD95_TICKS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 400)
SPLIT_NAMES = {"val": "validation", "test": "test"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def boxplot_figure(
    cases: pd.DataFrame,
    reference: pd.DataFrame,
    regions: list[str],
    split_name: str,
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)
    jitter = np.random.default_rng(0)
    for ax, metric in zip(axes, METRIC_LABELS, strict=True):
        values = [cases[f"{region}_{metric}"].dropna().to_numpy() for region in regions]
        boxes = ax.boxplot(
            values,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 2},
            whiskerprops={"color": MUTED},
            capprops={"color": MUTED},
            boxprops={"linewidth": 1},
        )
        for position, (box, colour, points) in enumerate(
            zip(boxes["boxes"], REGION_COLOURS, values, strict=True), start=1
        ):
            box.set(facecolor=colour + "33", edgecolor=colour)
            spread = jitter.uniform(-0.16, 0.16, len(points))
            ax.scatter(
                position + spread, points, s=12, color=colour, alpha=0.75, zorder=3
            )
            agreed = reference.query(
                "region == @regions[@position - 1] and metric == @metric"
            )
            ax.hlines(
                agreed["median"],
                position - 0.32,
                position + 0.32,
                colors=INK,
                linestyles=":",
            )
        ax.set_xticks(range(1, len(regions) + 1), regions)
        ax.set_ylabel(METRIC_LABELS[metric], color=INK_SECONDARY, fontsize=9)
        if metric == "hd95":
            # log for the 373 mm misses, symlog because a perfect boundary scores 0
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(bottom=0)
            ax.set_yticks(HD95_TICKS, [f"{tick:g}" for tick in HD95_TICKS])
            ax.set_ylabel(
                f"{METRIC_LABELS[metric]}, log scale", color=INK_SECONDARY, fontsize=9
            )
        _style(ax)
    counts = ", ".join(f"{r} n={len(cases[f'{r}_dice'].dropna())}" for r in regions)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        f"One point per {split_name} scan ({counts}). Dotted line: median agreement between "
        "two annotations of the same scan.\nA region the model misses entirely scores the "
        "scan diagonal (373 mm here) as its HD95.",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def curve_figure(
    history: pd.DataFrame, regions: list[str], title: str, path: Path
) -> None:
    fig, (loss_ax, dice_ax) = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.patch.set_facecolor(SURFACE)
    loss_ax.plot(
        history["epoch"], history["train_loss"], color=REGION_COLOURS[0], linewidth=2
    )
    loss_ax.set_ylabel("training loss", color=INK_SECONDARY, fontsize=9)
    validated = history.dropna(subset=["val_dice_mean"])
    for region, colour in zip(regions, REGION_COLOURS, strict=True):
        dice_ax.plot(
            validated["epoch"],
            validated[f"val_dice_{region}"],
            color=colour,
            linewidth=2,
        )
        dice_ax.annotate(
            f" {region}",
            (validated["epoch"].iloc[-1], validated[f"val_dice_{region}"].iloc[-1]),
            color=colour,
            fontsize=9,
            va="center",
        )
    best = validated.loc[validated["val_dice_mean"].idxmax()]
    dice_ax.axvline(best["epoch"], color=MUTED, linestyle=":", linewidth=1)
    dice_ax.text(
        best["epoch"],
        0.02,
        f"best mean Dice, epoch {int(best['epoch'])} ",
        transform=dice_ax.get_xaxis_transform(),
        ha="right",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    dice_ax.set_ylabel(
        "validation Dice (mean per case)", color=INK_SECONDARY, fontsize=9
    )
    for ax in (loss_ax, dice_ax):
        ax.set_xlabel("epoch", color=INK_SECONDARY, fontsize=9)
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=EVAL_SPLITS, default="val")
    parser.add_argument("--prefix", default="baseline")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    run, split = run_dir(cfg), args.split
    regions = list(cfg.data.regions)
    cases = pd.read_csv(run / f"{split}_cases.csv")
    summary = pd.read_csv(run / f"{split}_summary.csv")
    reference = pd.read_csv(args.out / "annotation_agreement_summary.csv")
    # only one configuration is ever scored on test, so its files need no run prefix
    validating = split == "val"
    results = f"{args.prefix}_results.csv" if validating else "test_results.csv"
    stem = f"{args.prefix}_val" if validating else "test"

    options = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    summary.to_csv(args.out / results, **options)
    cases.to_csv(args.out / f"{stem}_cases.csv", **options)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    boxplot_figure(
        cases,
        reference,
        regions,
        SPLIT_NAMES[split],
        f"{cfg.name}: per-case {SPLIT_NAMES[split]} scores",
        figures / f"{stem}_boxplot.png",
    )
    # the training curve belongs to the run, not to a split, so it is drawn once
    if validating:
        history = pd.read_csv(run / "history.csv")
        history.to_csv(args.out / f"{args.prefix}_history.csv", **options)
        curve_figure(
            history,
            regions,
            f"{cfg.name}: training loss and validation Dice",
            figures / f"{args.prefix}_training_curve.png",
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
