"""Protect the declared validation selection rule without running a model."""

import pytest

from ferramentas.scripts.calibrate_water import choose_candidate, presence_metrics


def test_presence_metrics_count_negative_images_in_specificity():
    metrics = presence_metrics(
        [True, True, True, False, False, False, False],
        [True, True, False, True, False, False, False],
    )
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 1, 1, 3)
    assert (
        metrics["precision"]
        == metrics["recall"]
        == metrics["f1"]
        == pytest.approx(2 / 3)
    )
    assert metrics["specificity"] == 0.75


def test_no_profile_is_chosen_if_specificity_floor_is_not_met():
    assert (
        choose_candidate(
            [
                {
                    "metrics": {"specificity": 0.89, "f1": 0.99},
                }
            ]
        )
        is None
    )


def test_selection_prefers_higher_presence_f1_inside_specificity_constraint():
    candidates = []
    for f1, specificity in ((0.8, 1.0), (0.9, 0.91), (0.99, 0.8)):
        candidates.append(
            {
                "metrics": {"f1": f1, "specificity": specificity, "precision": 0.95},
                "fusion_policy": "replace_yolo_water",
                "pixel_confidence": 0.5,
                "min_area_fraction": 0.003,
            }
        )
    assert choose_candidate(candidates) is candidates[1]


def test_exact_score_ties_prefer_replacement_then_stricter_pixel_and_area_gates():
    candidates = []
    for policy, pixel, area in (
        ("augment_yolo_water_union", 0.7, 0.01),
        ("replace_yolo_water", 0.5, 0.001),
        ("replace_yolo_water", 0.7, 0.003),
        ("replace_yolo_water", 0.7, 0.01),
    ):
        candidates.append(
            {
                "metrics": {"f1": 0.9, "specificity": 1.0, "precision": 1.0},
                "fusion_policy": policy,
                "pixel_confidence": pixel,
                "min_area_fraction": area,
            }
        )
    assert choose_candidate(candidates) is candidates[-1]
