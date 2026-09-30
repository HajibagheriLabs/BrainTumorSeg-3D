"""Per-case dataset statistics, repeat-scan evidence and their figures, from make data."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from braintumorseg.cli import runtime_device
from braintumorseg.config import Config, load_config
from braintumorseg.data import (
    SPLITS,
    DatasetIndex,
    link_repeat_scans,
    make_splits,
    read_index,
    read_meta,
    read_splits,
    region_voxels,
    thumbnail_similarity,
)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
MARKER = "#eb6834"
SWEEP_RANGE = (0.40, 0.96, 0.01)


def _slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def threshold_sweep(similarity: torch.Tensor, case_ids: list[str]) -> pd.DataFrame:
    """Group structure across link thresholds, the evidence behind the configured one."""
    rows = []
    for threshold in np.round(np.arange(*SWEEP_RANGE), 2):
        groups = [
            g for g in link_repeat_scans(case_ids, similarity, threshold) if len(g) > 1
        ]
        rows.append(
            {
                "threshold": threshold,
                "linked_groups": len(groups),
                "linked_cases": sum(len(g) for g in groups),
                "largest_group": max((len(g) for g in groups), default=1),
            }
        )
    return pd.DataFrame(rows)


def naive_leakage(
    cfg: Config, splits: dict, case_ids: list[str]
) -> dict[str, tuple[int, int]]:
    """Per held-out split of a case-level split: (cases with a same-patient scan in train, size)."""
    singletons = [(case_id,) for case_id in case_ids]
    naive = make_splits(
        singletons, dict.fromkeys(case_ids, ""), cfg.data.split, cfg.data.split.seed
    )
    train = set(naive["train"])
    group_of = {
        case_id: group for group in splits["linked_groups"] for case_id in group
    }
    return {
        name: (
            sum(
                any(other in train for other in group_of.get(c, ()))
                for c in naive[name]
            ),
            len(naive[name]),
        )
        for name in ("val", "test")
    }


def case_table(
    cfg: Config, index: DatasetIndex, splits: dict, similarity: torch.Tensor
) -> pd.DataFrame:
    case_ids = [case.case_id for case in index.cases]
    split_of = {case_id: name for name in SPLITS for case_id in splits[name]}
    group_of = {
        case_id: group for group in splits["linked_groups"] for case_id in group
    }
    others = similarity.clone().fill_diagonal_(-math.inf)
    nearest_value, nearest_index = (t.tolist() for t in others.max(dim=1))

    rows = []
    for k, meta in enumerate(read_meta(cfg.data.processed_dir, case_ids)):
        case_id = meta["case_id"]
        voxel_ml = math.prod(meta["spacing"]) / 1000.0
        crop = np.subtract(meta["crop_stop"], meta["crop_start"])
        linked = [other for other in group_of.get(case_id, ()) if other != case_id]
        row = {
            "case_id": case_id,
            "split": split_of[case_id],
            "linked_with": ";".join(linked),
            "nearest_case": case_ids[nearest_index[k]],
            "nearest_similarity": round(nearest_value[k], 4),
        }
        row |= {f"shape_{a}": n for a, n in zip("xyz", meta["shape"], strict=True)}
        row |= {f"spacing_{a}": s for a, s in zip("xyz", meta["spacing"], strict=True)}
        row |= {f"crop_{a}": int(n) for a, n in zip("xyz", crop, strict=True)}
        row["brain_ml"] = round(meta["brain_voxels"] * voxel_ml, 2)
        row["tumour_voxels_outside_brain"] = meta["tumour_voxels_outside_brain"]
        for modality, stats in meta["intensity"].items():
            for key in ("min", "p01", "p99", "max"):
                row[f"{_slug(modality)}_{key}"] = round(stats[key], 2)
        for name, count in meta["label_voxels"].items():
            if name != index.labels[0]:
                row[f"{_slug(name)}_ml"] = round(count * voxel_ml, 3)
        for region, count in region_voxels(meta, index, cfg.data.regions).items():
            row[f"{region.lower()}_ml"] = round(count * voxel_ml, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _unique(frame: pd.DataFrame, prefix: str) -> str:
    columns = [f"{prefix}_{axis}" for axis in "xyz"]
    values = frame[columns].drop_duplicates().itertuples(index=False)
    return ", ".join(" x ".join(f"{v:g}" for v in row) for row in values)


def summary_markdown(
    cfg: Config, frame: pd.DataFrame, splits: dict, leakage: dict[str, tuple[int, int]]
) -> str:
    regions = list(cfg.data.regions)
    linked = frame["linked_with"] != ""
    nearest = frame["nearest_similarity"]
    groups = len(splits["linked_groups"])
    outside = frame["tumour_voxels_outside_brain"].sum()
    threshold = cfg.data.link.threshold
    pairs = splits["exact_duplicates"]
    duplicated = len({case_id for pair in pairs for case_id in pair})
    volumes = frame.set_index("case_id")[[f"{r.lower()}_ml" for r in regions]]
    relabelled = sum((volumes.loc[a] != volumes.loc[b]).any() for a, b in pairs)
    leaked = ", ".join(f"{n} of {total} {name}" for name, (n, total) in leakage.items())
    lines = [
        "# Dataset summary",
        "",
        "Written by `scripts/inspect_data.py` from the preprocessed cache; do not edit.",
        "",
        f"- Labelled cases: {len(frame)}",
        f"- Volume shapes: {_unique(frame, 'shape')}",
        f"- Voxel spacing (mm): {_unique(frame, 'spacing')}",
        f"- Tumour voxels outside the brain mask: {outside}",
        "- Cases whose image reappears under another case id "
        + f"(similarity >= {cfg.data.link.duplicate_threshold}): {duplicated}",
        f"- Duplicate pairs whose two label maps differ: {relabelled} of {len(pairs)}",
        f"- Repeat-scan groups linked at similarity >= {threshold}: {groups}, "
        + f"covering {int(linked.sum())} cases",
        f"- Lowest nearest-case similarity among linked cases: {nearest[linked].min():.4f}",
        f"- Highest nearest-case similarity among unlinked cases: {nearest[~linked].max():.4f}",
        f"- A case-level split with the same seed would leave {leaked} cases "
        + "with a scan of the same patient in train",
        "",
        "## Cases per split and empty labels",
        "",
        "Scored cases count each duplicated scan once, see DECISIONS.md.",
        "",
        "| split | cases | scored | "
        + " | ".join(f"empty {r}" for r in regions)
        + " |",
        "|---|---:|---:|" + "---:|" * len(regions),
    ]
    superseded = {lower for lower, _ in pairs}
    parts = [(name, frame[frame["split"] == name]) for name in SPLITS] + [
        ("all", frame)
    ]
    for name, part in parts:
        scored = int((~part["case_id"].isin(superseded)).sum())
        empty = [str(int((part[f"{r.lower()}_ml"] == 0).sum())) for r in regions]
        lines.append(f"| {name} | {len(part)} | {scored} | " + " | ".join(empty) + " |")
    lines += ["", "## Tumour volume per region (mL, non-empty cases)", ""]
    lines += [
        "| region | cases | min | p25 | median | p75 | max |",
        "|---|---:|" + "---:|" * 5,
    ]
    for region in cfg.data.regions:
        volume = frame[f"{region.lower()}_ml"]
        volume = volume[volume > 0]
        q = volume.quantile([0.25, 0.5, 0.75]).tolist()
        lines.append(
            f"| {region} | {len(volume)} | {volume.min():.2f} | {q[0]:.2f} | {q[1]:.2f} "
            f"| {q[2]:.2f} | {volume.max():.2f} |"
        )
    return "\n".join(lines) + "\n"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def volume_figure(cfg: Config, frame: pd.DataFrame, path: Path) -> None:
    regions = list(cfg.data.regions)
    fig, axes = plt.subplots(
        1, len(regions), figsize=(4 * len(regions), 3.4), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    positive = frame[[f"{r.lower()}_ml" for r in regions]]
    low, high = positive[positive > 0].min().min(), positive.max().max()
    bins = np.logspace(np.log10(low), np.log10(high), 30)
    for ax, region in zip(np.atleast_1d(axes), regions, strict=True):
        volume = frame[f"{region.lower()}_ml"]
        ax.hist(
            volume[volume > 0], bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=1
        )
        ax.set_xscale("log")
        _style(ax)
        ax.set_title(f"{region}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("tumour volume (mL, log scale)", color=INK_SECONDARY, fontsize=9)
        empty = int((volume == 0).sum())
        if empty:
            note = f"{empty} case{'s' if empty > 1 else ''} with an empty\n{region} label not shown"
            ax.text(
                0.02,
                0.97,
                note,
                transform=ax.transAxes,
                va="top",
                color=INK_SECONDARY,
                fontsize=8.5,
            )
    np.atleast_1d(axes)[0].set_ylabel("cases", color=INK_SECONDARY, fontsize=9)
    fig.suptitle(
        f"Tumour volume per evaluation region, {len(frame)} labelled cases",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def linkage_figure(cfg: Config, frame: pd.DataFrame, path: Path) -> None:
    threshold = cfg.data.link.threshold
    fig, ax = plt.subplots(figsize=(8, 3.4))
    fig.patch.set_facecolor(SURFACE)
    bins = np.linspace(frame["nearest_similarity"].min(), 1.0, 60)
    ax.hist(
        frame["nearest_similarity"],
        bins=bins,
        color=SERIES,
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.axvline(threshold, color=MARKER, linewidth=2)
    ax.text(
        threshold,
        0.97,
        f"  link threshold {threshold}",
        transform=ax.get_xaxis_transform(),
        va="top",
        color=INK_SECONDARY,
        fontsize=9,
    )
    _style(ax)
    ax.set_xlabel(
        "cosine similarity to the most similar other case (atlas-space thumbnails)",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_ylabel("cases", color=INK_SECONDARY, fontsize=9)
    duplicated = int(
        (frame["nearest_similarity"] >= cfg.data.link.duplicate_threshold).sum()
    )
    ax.annotate(
        f"{duplicated} exact duplicates ",
        xy=(bins[-2], duplicated),
        ha="right",
        va="center",
        color=INK_SECONDARY,
        fontsize=9,
    )
    linked = int((frame["linked_with"] != "").sum())
    ax.set_title(
        f"Repeat scans: {linked} of {len(frame)} cases linked to another scan "
        + "of the same patient",
        color=INK,
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = runtime_device(cfg)
    index = read_index(cfg.data.root)
    splits = read_splits(cfg.data.splits_file)
    similarity = thumbnail_similarity(
        cfg.data.processed_dir, [case.case_id for case in index.cases], device
    ).cpu()

    frame = case_table(cfg, index, splits, similarity)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # explicit lf so regenerated reports are byte-identical on every platform
    frame.to_csv(args.out / "dataset_stats.csv", index=False, lineterminator="\n")
    sweep = threshold_sweep(similarity, frame["case_id"].tolist())
    sweep.to_csv(
        args.out / "link_threshold_sweep.csv", index=False, lineterminator="\n"
    )
    leakage = naive_leakage(cfg, splits, frame["case_id"].tolist())
    summary = summary_markdown(cfg, frame, splits, leakage)
    (args.out / "dataset_summary.md").write_text(summary, "utf-8", newline="\n")
    volume_figure(cfg, frame, figures / "tumour_volumes.png")
    linkage_figure(cfg, frame, figures / "repeat_scans.png")
    print(summary)


if __name__ == "__main__":
    main()
