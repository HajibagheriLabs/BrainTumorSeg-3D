"""Where a scored split fails: ranked cases, overlays, and what the errors go with."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import ndimage, stats

from braintumorseg.cli import EVAL_SPLITS, runtime_device
from braintumorseg.config import Config, load_config
from braintumorseg.data import full_label, read_index, read_meta, read_splits
from braintumorseg.metrics import region_mask
from braintumorseg.train import run_dir

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
# one colour per evaluation region, reused for the label that region adds to the one inside it
REGION_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
CASES_SHOWN = 5
# the sequences a reader needs to judge a contour: oedema shows on flair, enhancement on t1gd
BACKDROPS = {"FLAIR": "FLAIR", "t1gd": "T1 with contrast"}
DISPLAY_PERCENTILES = (1.0, 99.0)
OVERLAY_ALPHA = 0.6
VOLUME_BINS = 4
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
LABELLED_POINTS = 3
# below this many decades of volume a log axis needs ticks between the powers of ten
SPARSE_DECADES = 2.0


def rank_cases(cases: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Cases ordered worst first by mean Dice over the regions present in the truth."""
    ranked = cases.copy()
    ranked["mean_dice"] = ranked[[f"{region}_dice" for region in regions]].mean(axis=1)
    ranked = ranked.sort_values(["mean_dice", "case_id"], ignore_index=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked


def label_maps(
    cfg: Config, predictions: Path, case_id: str
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.load(cfg.data.processed_dir / "labels" / f"{case_id}.npy")[0]
    pred = np.load(predictions / f"{case_id}.npy")[0]
    if pred.shape != truth.shape:
        raise ValueError(f"{case_id}: prediction {pred.shape} vs truth {truth.shape}")
    return truth, pred


def _untouched(
    mask: np.ndarray, other: np.ndarray, voxel_ml: float
) -> tuple[float, int]:
    components, count = ndimage.label(mask)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    apart = np.setdiff1d(np.arange(1, count + 1), np.unique(components[other]))
    return float(sizes[apart].sum() * voxel_ml), len(apart)


def _confusion(truth: torch.Tensor, other: torch.Tensor, classes: int) -> torch.Tensor:
    pairs = truth.long() * classes + other.long()
    counts = torch.bincount(pairs.flatten(), minlength=classes * classes)
    return counts.reshape(classes, classes)


def case_errors(
    cfg: Config, predictions: Path, case_ids: list[str], device: torch.device
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-case error volumes, and the label confusion matrix pooled over the cases."""
    classes = cfg.model.out_channels
    confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
    rows = []
    for meta in read_meta(cfg.data.processed_dir, case_ids):
        voxel_ml = math.prod(meta["spacing"]) / 1000.0
        truth, pred = label_maps(cfg, predictions, meta["case_id"])
        truth_gpu = torch.from_numpy(truth).to(device)
        pred_gpu = torch.from_numpy(pred).to(device)
        confusion += _confusion(truth_gpu, pred_gpu, classes)
        row: dict[str, float | str] = {"case_id": meta["case_id"]}
        for region, labels in cfg.data.regions.items():
            wanted = region_mask(truth_gpu, labels)
            found = region_mask(pred_gpu, labels)
            row[f"{region}_missed_ml"] = int((wanted & ~found).sum()) * voxel_ml
            row[f"{region}_extra_ml"] = int((found & ~wanted).sum()) * voxel_ml
        # a component counts as found if the other mask touches it: dice already scores overlap
        tumour, predicted = truth > 0, pred > 0
        row["detached_pred_ml"], row["detached_pred_count"] = _untouched(
            predicted, tumour, voxel_ml
        )
        row["unfound_truth_ml"], row["unfound_truth_count"] = _untouched(
            tumour, predicted, voxel_ml
        )
        rows.append(row)
    return pd.DataFrame(rows), confusion.cpu().numpy()


def annotator_confusion(cfg: Config, device: torch.device) -> np.ndarray:
    """Label confusion between the two annotations of every scan the dataset holds twice."""
    classes = cfg.model.out_channels
    processed = cfg.data.processed_dir
    confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
    for superseded, kept in read_splits(cfg.data.splits_file)["exact_duplicates"]:
        old, new = read_meta(processed, [superseded, kept])
        # the kept label plays the truth, as it does when the model is scored
        truth = torch.from_numpy(full_label(processed, new)).to(device)
        other = torch.from_numpy(full_label(processed, old)).to(device)
        confusion += _confusion(truth, other, classes)
    return confusion.cpu().numpy()


def lesion_contrast(
    cfg: Config, case_ids: list[str], modalities: tuple[str, ...], device: torch.device
) -> pd.DataFrame:
    """Mean normalised intensity of every sequence inside each true region, per case."""
    processed = cfg.data.processed_dir
    rows = []
    for case_id in case_ids:
        image = torch.from_numpy(np.load(processed / "images" / f"{case_id}.npy"))
        truth = torch.from_numpy(np.load(processed / "labels" / f"{case_id}.npy")[0])
        image, truth = image.to(device), truth.to(device)
        row: dict[str, float | str] = {"case_id": case_id}
        for region, labels in cfg.data.regions.items():
            inside = region_mask(truth, labels)
            for channel, modality in enumerate(modalities):
                # the mean of no voxels is nan, which leaves an absent region out later
                values = image[channel][inside].float()
                row[f"{region}_{modality}_z"] = values.mean().item()
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_table(
    confusion: np.ndarray, names: dict[int, str], source: str
) -> pd.DataFrame:
    rows = []
    for truth, counts in enumerate(confusion):
        for predicted, voxels in enumerate(counts):
            rows.append(
                {
                    "source": source,
                    "truth": names[truth],
                    "predicted": names[predicted],
                    "voxels": int(voxels),
                    "share_of_truth": voxels / counts.sum(),
                }
            )
    return pd.DataFrame(rows)


def spearman_interval(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Spearman correlation with a bootstrap interval over cases; nan pairs are left out."""
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    draws = rng.integers(0, len(x), size=(BOOTSTRAP_SAMPLES, len(x)))
    ranks_x = stats.rankdata(x[draws], axis=1)
    ranks_y = stats.rankdata(y[draws], axis=1)
    ranks_x -= ranks_x.mean(axis=1, keepdims=True)
    ranks_y -= ranks_y.mean(axis=1, keepdims=True)
    samples = (ranks_x * ranks_y).sum(axis=1) / np.sqrt(
        (ranks_x**2).sum(axis=1) * (ranks_y**2).sum(axis=1)
    )
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "n": len(x),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "low": float(low),
        "high": float(high),
    }


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman correlation of x and y with the rank of a third variable held fixed."""
    keep = ~(np.isnan(x) | np.isnan(y) | np.isnan(control))
    xy, xc, yc = (
        stats.spearmanr(a[keep], b[keep]).statistic
        for a, b in ((x, y), (x, control), (y, control))
    )
    return float((xy - xc * yc) / math.sqrt((1.0 - xc**2) * (1.0 - yc**2)))


def _volume(table: pd.DataFrame, region: str) -> np.ndarray:
    # an absent region has no score and no volume to rank
    volume = table[f"{region}_truth_ml"].to_numpy()
    return np.where(volume > 0, volume, np.nan)


def covariate_correlations(
    sources: dict[str, pd.DataFrame],
    regions: list[str],
    modalities: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    """How each score tracks lesion volume and lesion contrast, for every source."""
    rng = np.random.default_rng(seed)
    rows = []
    # volume first and in one pass, so its intervals do not depend on the covariates after it
    for source, table in sources.items():
        for region in regions:
            for metric in ("dice", "hd95"):
                scores = table[f"{region}_{metric}"].to_numpy()
                row = {"source": source, "region": region, "covariate": "volume"}
                row |= {"metric": metric}
                row |= spearman_interval(_volume(table, region), scores, rng)
                rows.append(row | {"partial_given_volume": np.nan})
    for source, table in sources.items():
        for region in regions:
            volume = _volume(table, region)
            for modality in modalities:
                contrast = table[f"{region}_{modality}_z"].to_numpy()
                for metric in ("dice", "hd95"):
                    scores = table[f"{region}_{metric}"].to_numpy()
                    row = {"source": source, "region": region}
                    row |= {"covariate": f"{modality} intensity", "metric": metric}
                    row |= spearman_interval(contrast, scores, rng)
                    row["partial_given_volume"] = partial_spearman(
                        contrast, scores, volume
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def dice_by_volume(cases: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Dice and HD95 within equal-count bins of true volume, smallest lesions first."""
    rows = []
    for region in regions:
        present = cases[cases[f"{region}_truth_ml"] > 0]
        volume = present[f"{region}_truth_ml"]
        bins = pd.qcut(volume, VOLUME_BINS, labels=False) + 1
        lost = (1.0 - present[f"{region}_dice"]).sum()
        for number, part in present.groupby(bins):
            dice, hd95 = part[f"{region}_dice"], part[f"{region}_hd95"]
            rows.append(
                {
                    "region": region,
                    "volume_bin": int(number),
                    "n": len(part),
                    "volume_min_ml": part[f"{region}_truth_ml"].min(),
                    "volume_max_ml": part[f"{region}_truth_ml"].max(),
                    "dice_mean": dice.mean(),
                    "dice_median": dice.median(),
                    "dice_min": dice.min(),
                    "deficit_share": (1.0 - dice).sum() / lost,
                    "hd95_median": hd95.median(),
                    "hd95_max": hd95.max(),
                }
            )
    return pd.DataFrame(rows)


def _shade(channel: np.ndarray, brain: np.ndarray) -> np.ndarray:
    low, high = np.percentile(channel[brain], DISPLAY_PERCENTILES)
    shade = np.clip((channel - low) / (high - low), 0.0, 1.0)
    # the normalised background is zero, which is mid-grey for a z-score
    shade[~brain] = 0.0
    return np.repeat(shade[..., None], 3, axis=2)


def _painted(
    shade: np.ndarray, labels: np.ndarray, colours: dict[int, str]
) -> np.ndarray:
    painted = shade.copy()
    for value, colour in colours.items():
        where = labels == value
        painted[where] = (1.0 - OVERLAY_ALPHA) * painted[where] + OVERLAY_ALPHA * (
            np.array(to_rgb(colour))
        )
    return painted


def _tile(panel: np.ndarray, side: int) -> np.ndarray:
    # one canvas size for every scan keeps the millimetre scale equal across rows
    tile = np.zeros((side, side, 3), dtype=panel.dtype)
    x, y = ((side - size) // 2 for size in panel.shape[:2])
    tile[x : x + panel.shape[0], y : y + panel.shape[1]] = panel
    # rotated so anterior is up, as axial slices are conventionally shown
    return np.rot90(tile)


def case_panels(
    cfg: Config,
    predictions: Path,
    case_id: str,
    modalities: tuple[str, ...],
    colours: dict[int, str],
) -> tuple[int, list[np.ndarray]]:
    """The axial slice shown for a case and its image, truth and prediction panels."""
    image = np.load(cfg.data.processed_dir / "images" / f"{case_id}.npy")
    truth, pred = label_maps(cfg, predictions, case_id)
    # the slice where prediction and truth disagree most is where the case is decided
    z = int((truth != pred).sum(axis=(0, 1)).argmax())
    brain = (image[:, :, :, z] != 0).any(axis=0)
    shades = [
        _shade(image[modalities.index(name), :, :, z].astype(np.float32), brain)
        for name in BACKDROPS
    ]
    overlays = [
        _painted(shades[0], labels[:, :, z], colours) for labels in (truth, pred)
    ]
    return z, shades + overlays


def overlay_figure(
    cfg: Config,
    predictions: Path,
    shown: pd.DataFrame,
    modalities: tuple[str, ...],
    names: dict[int, str],
    title: str,
    path: Path,
) -> None:
    regions = list(cfg.data.regions)
    colours = dict(zip(sorted(names)[1:], REGION_COLOURS, strict=True))
    headers = [*BACKDROPS.values(), "ground truth", "prediction"]
    drawn = [
        case_panels(cfg, predictions, case_id, modalities, colours)
        for case_id in shown["case_id"]
    ]
    side = max(max(panels[0].shape[:2]) for _, panels in drawn)
    fig, axes = plt.subplots(
        len(shown), len(headers), figsize=(10.4, 2.1 * len(shown) + 1.1), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for row_axes, (_, panels) in zip(axes, drawn, strict=True):
        for ax, panel in zip(row_axes, panels, strict=True):
            ax.imshow(_tile(panel, side), interpolation="nearest")
            ax.set_axis_off()
    for ax, header in zip(axes[0], headers, strict=True):
        ax.set_title(header, color=INK, fontsize=10)
    fig.tight_layout(rect=(0.2, 0.055, 1, 0.975), h_pad=0.5, w_pad=0.3)
    for row_axes, (_, case), (z, _) in zip(axes, shown.iterrows(), drawn, strict=True):
        scores = "  ".join(
            f"{region} absent"
            if np.isnan(case[f"{region}_dice"])
            else f"{region} {case[f'{region}_dice']:.2f}"
            for region in regions
        )
        volumes = " / ".join(f"{case[f'{region}_truth_ml']:.1f}" for region in regions)
        box = row_axes[0].get_position()
        fig.text(
            0.012,
            (box.y0 + box.y1) / 2.0,
            f"{case['case_id']}, rank {case['rank']}\nmean Dice {case['mean_dice']:.3f}\n"
            f"{scores}\ntrue {'/'.join(regions)} volume\n{volumes} mL\naxial slice {z}",
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
            linespacing=1.5,
        )
    handles = [
        Patch(color=colour, label=names[value]) for value, colour in colours.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.005, 0.026),
        ncols=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(title, color=INK, fontsize=12, x=0.012, ha="left")
    fig.text(
        0.012,
        0.008,
        "Each row is one scan at the axial slice where prediction and ground truth "
        "disagree on the most voxels. Both overlays are drawn on FLAIR.",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def _volume_axis(ax: plt.Axes) -> None:
    low, high = ax.get_xlim()
    sparse = math.log10(high / low) < SPARSE_DECADES
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0, 2.0, 5.0) if sparse else (1.0,))
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def volume_figure(
    ranked: pd.DataFrame,
    agreement: pd.DataFrame,
    correlations: pd.DataFrame,
    regions: list[str],
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(regions), figsize=(12, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    of_volume = correlations.query("metric == 'dice' and covariate == 'volume'")
    found = of_volume.set_index(["source", "region"])
    sources = list(found.index.get_level_values("source").unique())
    for ax, region, colour in zip(axes, regions, REGION_COLOURS, strict=True):
        volume, dice = f"{region}_truth_ml", f"{region}_dice"
        pairs = agreement[agreement[volume] > 0]
        ax.scatter(
            pairs[volume],
            pairs[dice],
            s=18,
            facecolors="none",
            edgecolors=MUTED,
            linewidths=0.8,
        )
        scored = ranked[ranked[volume] > 0]
        ax.scatter(
            scored[volume],
            scored[dice],
            s=42,
            color=colour,
            edgecolors=SURFACE,
            linewidths=1.0,
            zorder=3,
        )
        lowest = scored.nsmallest(LABELLED_POINTS, dice)
        for place, (_, case) in enumerate(lowest.iterrows()):
            # alternate sides so two misses at nearly the same point stay readable
            right = place % 2 == 0
            ax.annotate(
                case["case_id"].split("_")[-1],
                (case[volume], case[dice]),
                xytext=(6 if right else -6, 3),
                textcoords="offset points",
                ha="left" if right else "right",
                color=INK_SECONDARY,
                fontsize=8,
            )
        lines = []
        for source in sources:
            item = found.loc[(source, region)]
            lines.append(
                f"{source} ρ = {item['spearman']:+.2f} "
                f"[{item['low']:+.2f}, {item['high']:+.2f}]"
            )
        ax.set_xscale("log")
        _volume_axis(ax)
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(region, color=INK, fontsize=10, loc="left")
        ax.set_title("\n".join(lines), color=INK_SECONDARY, fontsize=8.5, loc="right")
        ax.set_xlabel("true volume (mL), log scale", color=INK_SECONDARY, fontsize=9)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Dice", color=INK_SECONDARY, fontsize=9)
    marker = {"linestyle": "none", "marker": "o"}
    handles = [
        plt.Line2D([], [], color=INK_SECONDARY, markersize=7, **marker),
        plt.Line2D(
            [],
            [],
            markerfacecolor="none",
            markeredgecolor=MUTED,
            markersize=5,
            **marker,
        ),
    ]
    fig.legend(
        handles,
        [
            f"{sources[0]}: prediction against ground truth, one point per scan",
            f"{sources[1]}: two annotations of one scan, one point per duplicated scan",
        ],
        loc="lower left",
        bbox_to_anchor=(0.005, 0.035),
        ncols=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(
        0.01,
        0.008,
        f"ρ is the Spearman correlation between Dice and true volume, with a bootstrap "
        f"{CONFIDENCE:.0%} interval. Numbers beside points are the case ids of the "
        f"{LABELLED_POINTS} lowest scores.",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=EVAL_SPLITS, default="test")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = runtime_device(cfg)
    run, split = run_dir(cfg), args.split
    predictions = run / f"{split}_predictions"
    regions = list(cfg.data.regions)
    index = read_index(cfg.data.root)

    ranked = rank_cases(pd.read_csv(run / f"{split}_cases.csv"), regions)
    case_ids = list(ranked["case_id"])
    errors, confusion = case_errors(cfg, predictions, case_ids, device)
    ranked = ranked.merge(errors, on="case_id", validate="one_to_one")
    contrast = lesion_contrast(cfg, case_ids, index.modalities, device)
    ranked = ranked.merge(contrast, on="case_id", validate="one_to_one")
    agreement = pd.read_csv(args.out / "annotation_agreement.csv")
    # the kept annotation is the truth of a pair, so its regions give the pair's contrast
    kept = lesion_contrast(cfg, list(agreement["kept"]), index.modalities, device)
    agreement = agreement.merge(kept.rename(columns={"case_id": "kept"}), on="kept")
    correlations = covariate_correlations(
        {"model": ranked, "annotators": agreement}, regions, index.modalities, cfg.seed
    )
    by_volume = dice_by_volume(ranked, regions)
    between_annotators = annotator_confusion(cfg, device)
    confused = pd.concat(
        [
            confusion_table(confusion, index.labels, "model"),
            confusion_table(between_annotators, index.labels, "annotators"),
        ]
    )

    options = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    ranked.to_csv(args.out / f"{split}_failure_cases.csv", **options)
    correlations.to_csv(args.out / f"{split}_correlations.csv", **options)
    by_volume.to_csv(args.out / f"{split}_dice_by_volume.csv", **options)
    confused.to_csv(args.out / f"{split}_confusion.csv", **options)
    print(correlations.to_string(index=False))
    print(by_volume.to_string(index=False))
    print(confused.to_string(index=False))
    if args.tables_only:
        return

    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    groups = {
        "worst": ranked.head(CASES_SHOWN),
        "best": ranked.tail(CASES_SHOWN).iloc[::-1],
    }
    for group, shown in groups.items():
        overlay_figure(
            cfg,
            predictions,
            shown,
            index.modalities,
            index.labels,
            f"{cfg.name}: the {len(shown)} {group} {split} scans by mean Dice",
            figures / f"{split}_{group}_cases.png",
        )
    volume_figure(
        ranked,
        agreement,
        correlations,
        regions,
        f"{cfg.name}: Dice against tumour volume on the {split} split",
        figures / f"{split}_dice_vs_volume.png",
    )


if __name__ == "__main__":
    main()
