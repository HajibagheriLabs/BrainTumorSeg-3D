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
REGION_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
# settings that name a run or say how it executes, rather than change what it computes
IDENTITY_KEYS = {"name", "inference.checkpoint_run"}
EXECUTION_PREFIX = "runtime."
# (column prefix, statistic, direction in which a change is an improvement)
COMPARISONS = (("dice_mean", np.mean, 1.0), ("hd95_median", np.median, -1.0))


def changed_settings(cfg: Config, baseline: Config) -> str:
    flat = flatten_config(dataclasses.asdict(cfg))
    base = flatten_config(dataclasses.asdict(baseline))
    changes = [
        f"{key}={value}"
        for key, value in flat.items()
        if value != base[key]
        and key not in IDENTITY_KEYS
        and not key.startswith(EXECUTION_PREFIX)
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


def excluded_side(low: float, high: float) -> int:
    """+1 or -1 when an interval lies wholly on one side of zero, otherwise 0."""
    return 1 if low > 0.0 else -1 if high < 0.0 else 0


def verdict(sides: list[int], better: float) -> str:
    # one pairing per baseline run; a real effect has to hold against every one of them
    if sides[0] != 0 and all(side == sides[0] for side in sides):
        return "better" if sides[0] * better > 0 else "worse"
    if any(sides):
        return "not robust to the baseline seed"
    return "no detectable difference"


def compare(
    baseline: Config, ablations: list[Config], reseeded: Config
) -> pd.DataFrame:
    rng = np.random.default_rng(baseline.seed)
    base_cases = pd.read_csv(run_dir(baseline) / "val_cases.csv").set_index("case_id")
    loaded = {}
    for cfg in [baseline, *ablations]:
        cases = pd.read_csv(run_dir(cfg) / "val_cases.csv").set_index("case_id")
        if set(cases.index) != set(base_cases.index):
            raise ValueError(
                f"{cfg.name} was scored on different cases than the baseline"
            )
        loaded[cfg.name] = cases.loc[base_cases.index]
    reseeded_cases = loaded[reseeded.name]

    rows = []
    for cfg in [baseline, *ablations]:
        cases = loaded[cfg.name]
        # a run that reuses the baseline weights shares its seed, so only that pairing applies
        trained = cfg.inference.checkpoint_run is None
        retrained = trained and cfg not in (baseline, reseeded)
        history = run_dir(cfg) / "history.csv"
        hours = (
            pd.read_csv(history)["epoch_seconds"].sum() / 3600.0 if trained else np.nan
        )
        for region in baseline.data.regions:
            row = {"config": cfg.name, "change": changed_settings(cfg, baseline)}
            row |= {"region": region, "train_hours": hours}
            for metric in ("dice", "hd95"):
                values = cases[f"{region}_{metric}"]
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_median"] = values.median()
            for column, statistic, better in COMPARISONS:
                key = f"{region}_{column.split('_')[0]}"
                other = cases[key].to_numpy()
                n, delta, low, high = paired_bootstrap(
                    base_cases[key].to_numpy(), other, statistic, rng
                )
                row["n"] = n
                row |= {f"{column}_delta": delta, f"{column}_low": low}
                row[f"{column}_high"] = high
                sides = [excluded_side(low, high)]
                again = (np.nan, np.nan, np.nan)
                if retrained:
                    again = paired_bootstrap(
                        reseeded_cases[key].to_numpy(), other, statistic, rng
                    )[1:]
                    sides.append(excluded_side(again[1], again[2]))
                for part, value in zip(("delta", "low", "high"), again, strict=True):
                    row[f"{column}_{part}_reseeded"] = value
                if cfg == baseline:
                    row[f"{column}_verdict"] = "baseline"
                elif cfg == reseeded:
                    row[f"{column}_verdict"] = "baseline with another seed"
                else:
                    row[f"{column}_verdict"] = verdict(sides, better)
            rows.append(row)
    table = pd.DataFrame(rows)
    ordered = ["config", "change", "region", "n", "dice_mean", "dice_median"]
    ordered += ["hd95_mean", "hd95_median", "train_hours"]
    return table[ordered + [c for c in table.columns if c not in ordered]]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, left=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def forest_figure(table: pd.DataFrame, regions: list[str], path: Path) -> None:
    shown = table[table["change"] != "baseline"]
    labels = list(dict.fromkeys(shown["change"]))
    fig, axes = plt.subplots(
        2, len(regions), figsize=(12, 1.6 + 0.5 * len(labels) * 2), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    titles = {
        "dice_mean": "change in mean Dice",
        "hd95_median": "change in median HD95 (mm)",
    }
    # (column suffix, marker, vertical offset, filled): the baseline, then its reseeded twin
    pairings = (("", "o", -0.17, True), ("_reseeded", "D", 0.17, False))
    for row_axes, (column, _, _) in zip(axes, COMPARISONS, strict=True):
        for ax, region, colour in zip(row_axes, regions, REGION_COLOURS, strict=True):
            part = shown[shown["region"] == region].set_index("change").loc[labels]
            ax.axvline(0.0, color=MUTED, linewidth=1)
            for y, (_, item) in enumerate(part.iterrows()):
                for suffix, marker, offset, filled in pairings:
                    delta = item[f"{column}_delta{suffix}"]
                    if np.isnan(delta):
                        continue
                    span = [
                        item[f"{column}_low{suffix}"],
                        item[f"{column}_high{suffix}"],
                    ]
                    ax.plot(span, [y + offset] * 2, color=colour, linewidth=2)
                    ax.plot(
                        delta,
                        y + offset,
                        marker,
                        markersize=7,
                        color=colour,
                        markerfacecolor=colour if filled else SURFACE,
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
    marker_style = {"color": INK_SECONDARY, "linewidth": 2, "markersize": 7}
    handles = [
        plt.Line2D([], [], marker="o", **marker_style),
        plt.Line2D(
            [],
            [],
            marker="D",
            markerfacecolor=SURFACE,
            markeredgewidth=2,
            **marker_style,
        ),
    ]
    names = ["against the baseline", "against the baseline retrained with another seed"]
    fig.legend(
        handles,
        names,
        loc="lower left",
        bbox_to_anchor=(0.005, 0.035),
        ncols=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        f"Ablations on the same {int(table['n'].max())} validation scans, paired per scan",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.008,
        f"Lines are paired bootstrap {CONFIDENCE:.0%} intervals. A change counts only if "
        "both intervals lie on the same side of zero. Rows that reuse the baseline weights "
        "have one pairing.",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reseeded-baseline", type=Path, required=True)
    parser.add_argument("ablations", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    baseline = load_config(args.baseline)
    ablations = [load_config(path) for path in args.ablations]
    reseeded = load_config(args.reseeded_baseline)
    if reseeded not in ablations:
        raise SystemExit("the reseeded baseline must be one of the ablations")

    table = compare(baseline, ablations, reseeded)
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
    forest_figure(table, regions, args.out / "figures" / "ablations.png")
    columns = ["change", "region", "dice_mean", "dice_mean_delta"]
    columns += ["dice_mean_delta_reseeded", "dice_mean_verdict", "hd95_median_verdict"]
    print(table[columns].to_string(index=False))


if __name__ == "__main__":
    main()
