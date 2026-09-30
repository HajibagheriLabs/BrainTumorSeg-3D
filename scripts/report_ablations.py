"""Compare each ablation with the baseline on the same validation scans, paired per case."""

import argparse
import dataclasses
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from braintumorseg.config import Config, load_config
from braintumorseg.train import flatten_config, run_dir

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
NOISE_BAND = "#e1e0d9"
REGION_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
# settings that name a run rather than change what it does
IDENTITY_KEYS = {"name", "inference.checkpoint_run"}
# (column prefix, statistic, direction in which a change is an improvement)
COMPARISONS = (("dice_mean", np.mean, 1.0), ("hd95_median", np.median, -1.0))


def changed_settings(cfg: Config, baseline: Config) -> str:
    flat = flatten_config(dataclasses.asdict(cfg))
    base = flatten_config(dataclasses.asdict(baseline))
    changes = [
        f"{k}={v}" for k, v in flat.items() if v != base[k] and k not in IDENTITY_KEYS
    ]
    return ", ".join(changes) or "baseline"


def paired_bootstrap(
    base: np.ndarray,
    other: np.ndarray,
    statistic: Callable[..., np.ndarray],
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    """n, and the change in a statistic from base to other with its bootstrap interval."""
    # a case with an undefined score on either side has nothing to pair
    keep = ~(np.isnan(base) | np.isnan(other))
    base, other = base[keep], other[keep]
    delta = float(statistic(other) - statistic(base))
    draws = rng.integers(0, len(base), size=(BOOTSTRAP_SAMPLES, len(base)))
    samples = statistic(other[draws], axis=1) - statistic(base[draws], axis=1)
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return len(base), delta, float(low), float(high)


def verdict(delta: float, low: float, high: float, noise: float, better: float) -> str:
    # a change counts only if the interval excludes zero and it outgrows seed-to-seed noise
    if low <= 0.0 <= high or abs(delta) <= abs(noise):
        return "no detectable difference"
    return "better" if delta * better > 0 else "worse"


def compare(
    baseline: Config, ablations: list[Config], reference: Config
) -> pd.DataFrame:
    rng = np.random.default_rng(baseline.seed)
    base_cases = pd.read_csv(run_dir(baseline) / "val_cases.csv").set_index("case_id")
    rows = []
    for cfg in [baseline, *ablations]:
        cases = pd.read_csv(run_dir(cfg) / "val_cases.csv").set_index("case_id")
        if set(cases.index) != set(base_cases.index):
            raise ValueError(
                f"{cfg.name} was scored on different cases than the baseline"
            )
        cases = cases.loc[base_cases.index]
        for region in baseline.data.regions:
            row = {"config": cfg.name, "change": changed_settings(cfg, baseline)}
            row["region"] = region
            for metric in ("dice", "hd95"):
                values = cases[f"{region}_{metric}"]
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_median"] = values.median()
            for column, statistic, _ in COMPARISONS:
                metric = column.split("_")[0]
                n, delta, low, high = paired_bootstrap(
                    base_cases[f"{region}_{metric}"].to_numpy(),
                    cases[f"{region}_{metric}"].to_numpy(),
                    statistic,
                    rng,
                )
                row["n"] = n
                row |= {
                    f"{column}_delta": delta,
                    f"{column}_low": low,
                    f"{column}_high": high,
                }
            rows.append(row)
    table = pd.DataFrame(rows)

    noise = table[table["config"] == reference.name].set_index("region")
    for column, _, better in COMPARISONS:
        table[f"{column}_verdict"] = [
            "noise reference"
            if row["config"] == reference.name
            else "baseline"
            if row["config"] == baseline.name
            else verdict(
                row[f"{column}_delta"],
                row[f"{column}_low"],
                row[f"{column}_high"],
                noise.loc[row["region"], f"{column}_delta"],
                better,
            )
            for _, row in table.iterrows()
        ]
    ordered = ["config", "change", "region", "n", "dice_mean", "dice_median"]
    ordered += ["hd95_mean", "hd95_median"]
    return table[ordered + [c for c in table.columns if c not in ordered]]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, left=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def forest_figure(
    table: pd.DataFrame, reference: str, regions: list[str], path: Path
) -> None:
    shown = table[table["change"] != "baseline"]
    labels = list(dict.fromkeys(shown["change"]))
    fig, axes = plt.subplots(
        2, len(regions), figsize=(12, 1.2 + 0.42 * len(labels) * 2), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    titles = {
        "dice_mean": "change in mean Dice vs baseline",
        "hd95_median": "change in median HD95 (mm) vs baseline",
    }
    for row_axes, (column, _, _) in zip(axes, COMPARISONS, strict=True):
        for ax, region, colour in zip(row_axes, regions, REGION_COLOURS, strict=True):
            part = shown[shown["region"] == region].set_index("change").loc[labels]
            noise = abs(part[part["config"] == reference][f"{column}_delta"].iloc[0])
            ax.axvspan(-noise, noise, color=NOISE_BAND, alpha=0.6, linewidth=0)
            ax.axvline(0.0, color=MUTED, linewidth=1)
            for y, (label, item) in enumerate(part.iterrows()):
                decided = item[f"{column}_verdict"] in ("better", "worse")
                ax.plot(
                    [item[f"{column}_low"], item[f"{column}_high"]],
                    [y, y],
                    color=colour,
                    linewidth=2,
                )
                ax.plot(
                    item[f"{column}_delta"],
                    y,
                    "o",
                    markersize=8,
                    color=colour,
                    markerfacecolor=colour if decided else SURFACE,
                    markeredgewidth=2,
                )
            ax.set_yticks(range(len(labels)), labels)
            # explicit limits, because inverting shared axes once per panel cancels out
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(
                f"{region}: {titles[column]}", color=INK, fontsize=9.5, loc="left"
            )
            _style(ax)
    fig.suptitle(
        "Ablations against the baseline on the same 56 validation scans",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        f"Lines: paired bootstrap {CONFIDENCE:.0%} intervals. Grey band: the difference "
        "between two seeds of the baseline. Filled markers: interval excludes 0 and the change "
        "exceeds that band.",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--noise-reference", type=Path, required=True)
    parser.add_argument("ablations", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    baseline = load_config(args.baseline)
    ablations = [load_config(path) for path in args.ablations]
    reference = load_config(args.noise_reference)
    if reference not in ablations:
        raise SystemExit("the noise reference must be one of the ablations")

    table = compare(baseline, ablations, reference)
    options = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    table.to_csv(args.out / "ablations.csv", **options)
    summaries = []
    for cfg in [baseline, *ablations]:
        summary = pd.read_csv(run_dir(cfg) / "val_summary.csv")
        summary.insert(0, "config", cfg.name)
        summary.insert(1, "change", changed_settings(cfg, baseline))
        summaries.append(summary)
    pd.concat(summaries).to_csv(args.out / "results.csv", **options)
    regions = list(baseline.data.regions)
    forest_figure(
        table, reference.name, regions, args.out / "figures" / "ablations.png"
    )
    columns = [
        "change",
        "region",
        "n",
        "dice_mean",
        "dice_mean_delta",
        "dice_mean_verdict",
    ]
    print(table[columns].to_string(index=False))


if __name__ == "__main__":
    main()
