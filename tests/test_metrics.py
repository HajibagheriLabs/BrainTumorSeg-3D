import math

import pytest
import torch

from braintumorseg.metrics import (
    dice,
    field_of_view_diagonal,
    hd95,
    score_case,
    summarise,
    summarise_cases,
    surface,
)

SHAPE = (6, 6, 6)
UNIT = (1.0, 1.0, 1.0)
PENALTY = 100.0


def voxels(*points: tuple[int, int, int]) -> torch.Tensor:
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    for point in points:
        mask[point] = True
    return mask


def line(start: int, stop: int) -> torch.Tensor:
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0, start:stop] = True
    return mask


def test_perfect_overlap_scores_one_and_disjoint_masks_score_zero() -> None:
    block = torch.zeros(SHAPE, dtype=torch.bool)
    block[1:3, 1:3, 1:3] = True
    far = torch.zeros(SHAPE, dtype=torch.bool)
    far[4:6, 4:6, 4:6] = True

    assert dice(block, block) == 1.0
    assert hd95(block, block, UNIT, PENALTY) == 0.0
    assert dice(far, block) == 0.0


def test_half_overlap_uses_both_mask_sizes_in_the_denominator() -> None:
    # 4 voxels each, sharing 2: 2 * 2 / (4 + 4)
    assert dice(line(0, 4), line(2, 6)) == 0.5
    # prediction is 3 of the truth's 6 voxels: 2 * 3 / (3 + 6) = 2/3, where iou would give
    # 0.5, dividing by twice the truth 0.5, and dividing by twice the prediction 1.0
    assert dice(line(0, 3), line(0, 6)) == pytest.approx(2 / 3)


def test_empty_prediction_with_nonempty_truth_is_a_full_miss() -> None:
    empty = torch.zeros(SHAPE, dtype=torch.bool)
    assert dice(empty, line(1, 4)) == 0.0
    assert hd95(empty, line(1, 4), UNIT, PENALTY) == PENALTY
    # the miss penalty is the scan diagonal, the 373.13 mm brats uses for a 240x240x155 grid
    assert field_of_view_diagonal((240, 240, 155), UNIT) == pytest.approx(
        373.13, abs=0.005
    )


def test_empty_truth_leaves_dice_and_hd95_undefined() -> None:
    empty = torch.zeros(SHAPE, dtype=torch.bool)
    # empty on both sides: the ratio is 0 / 0
    assert math.isnan(dice(empty, empty))
    assert math.isnan(hd95(empty, empty, UNIT, PENALTY))
    # a false positive on empty truth is also undefined here; it is counted separately
    assert math.isnan(dice(line(0, 2), empty))
    assert math.isnan(hd95(line(0, 2), empty, UNIT, PENALTY))


def test_hd95_matches_hand_computed_distances() -> None:
    corner, far = voxels((0, 0, 0)), voxels((3, 4, 0))
    # two single voxels forming a 3-4-5 triangle
    assert hd95(corner, far, UNIT, PENALTY) == pytest.approx(5.0)
    # spacing scales each axis: sqrt((3 * 2)^2 + (4 * 1)^2)
    assert hd95(corner, far, (2.0, 1.0, 1.0), PENALTY) == pytest.approx(math.sqrt(52))
    # prediction voxels lie 0 and 5 from the truth, the truth voxel lies 0 from the
    # prediction; the 95th percentile of [0, 5] interpolates to 4.75 and the larger side wins
    pred = voxels((0, 0, 0), (0, 0, 5))
    assert hd95(pred, corner, UNIT, PENALTY) == pytest.approx(4.75)
    # a 3x3x3 cube has 26 surface voxels, all but its centre
    cube = torch.zeros(SHAPE, dtype=torch.bool)
    cube[1:4, 1:4, 1:4] = True
    assert int(surface(cube).sum()) == 26 and not surface(cube)[2, 2, 2]
    # distances run between surfaces: a solid cube and its hollow shell share one surface,
    # so hd95 is 0, where counting the 27 interior voxels would give 1
    solid = torch.zeros(SHAPE, dtype=torch.bool)
    solid[:5, :5, :5] = True
    shell = solid.clone()
    shell[1:4, 1:4, 1:4] = False
    assert hd95(solid, shell, UNIT, PENALTY) == 0.0


def test_regions_are_scored_from_merged_labels() -> None:
    truth = torch.zeros(SHAPE, dtype=torch.uint8)
    pred = torch.zeros(SHAPE, dtype=torch.uint8)
    truth[0, 0, :4] = torch.tensor([1, 2, 3, 3])
    pred[0, 0, :4] = torch.tensor([2, 2, 3, 1])
    regions = {"WT": (1, 2, 3), "TC": (2, 3), "ET": (3,)}

    row = score_case(pred, truth, regions, UNIT, PENALTY)

    # every tumour voxel is found, so whole tumour is perfect despite the class confusion
    assert row["WT_dice"] == 1.0
    # core: truth at 1, 2, 3 and prediction at 0, 1, 2 share 2 voxels: 2 * 2 / (3 + 3)
    assert row["TC_dice"] == pytest.approx(2 / 3)
    # enhancing: truth at 2, 3 and prediction at 2 share 1 voxel: 2 * 1 / (1 + 2)
    assert row["ET_dice"] == pytest.approx(2 / 3)
    assert row["ET_truth_ml"] == pytest.approx(0.002)


def test_summary_reports_the_distribution_and_counts_undefined_cases() -> None:
    stats = summarise([0.2, 0.4, 0.6, 0.8, math.nan])
    assert (stats["n"], stats["n_undefined"]) == (4, 1)
    assert stats["mean"] == pytest.approx(0.5) and stats["median"] == pytest.approx(0.5)
    # linear interpolation puts the quartiles a quarter of a step inside 0.2..0.8
    assert stats["q25"] == pytest.approx(0.35) and stats["q75"] == pytest.approx(0.65)
    assert (stats["min"], stats["max"]) == (0.2, 0.8)
    # sample standard deviation: sqrt((0.09 + 0.01 + 0.01 + 0.09) / 3)
    assert stats["std"] == pytest.approx(math.sqrt(0.2 / 3))

    empty = {"ET_dice": math.nan, "ET_hd95": math.nan, "ET_truth_ml": 0.0}
    rows = [
        empty | {"ET_pred_ml": 0.0},
        empty | {"ET_pred_ml": 0.5},
        {"ET_dice": 0.9, "ET_hd95": 2.0, "ET_truth_ml": 3.0, "ET_pred_ml": 3.1},
    ]
    dice_row = summarise_cases(rows, ["ET"])[0]
    assert (dice_row["n"], dice_row["n_undefined"]) == (1, 2)
    assert (dice_row["empty_truth"], dice_row["empty_truth_predicted"]) == (2, 1)
