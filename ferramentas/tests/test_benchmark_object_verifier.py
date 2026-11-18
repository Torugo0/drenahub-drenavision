from __future__ import annotations

import copy

import pytest

from ferramentas.scripts.benchmark_object_verifier import (
    eligible_candidates,
    evaluate_grid,
    litter_bbox_metrics,
    presence_metrics,
    select_grid_row,
    validate_verifier_result,
)


def candidate(box=(0, 0, 20, 20), *, cls=0, score=0.4, margin=0.1, index=0):
    return {
        "candidate_index": index,
        "class_id": cls,
        "confidence": score,
        "bounding_box": list(box),
        "clipped_bounding_box": list(box),
        "verification": {
            "margin": margin,
            "positive_similarity": 0.4,
            "negative_similarity": 0.4 - margin,
            "best_positive": "target object",
            "best_negative": "background",
        },
    }


def record(key="image", *, residue=True, animal=False, candidates=None):
    return {
        "sha256": key,
        "presence": {"0": residue, "2": animal},
        "verified_candidates": candidates or [],
    }


def master(*, source="click_trash", split="val", truth=None, predictions=None):
    return {
        "source_key": source,
        "split": split,
        "truth": truth or [],
        "predictions": predictions or [],
    }


def raw_prediction(box=(0, 0, 20, 20), *, cls=0, score=0.8):
    return {"class_id": cls, "confidence": score, "bounding_box": list(box)}


def test_unknown_presence_is_excluded_even_with_a_verified_detection():
    rows = [
        record("unknown", residue=None, candidates=[candidate()]),
        record("positive"),
        record("negative", residue=False),
    ]
    masters = {row["sha256"]: master() for row in rows}
    result = presence_metrics(rows, masters, 0)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (0, 0, 1, 1)
    assert result["unknown_excluded"] == 1
    assert result["known_images"] == 2


def test_legacy_litter_is_not_crop_verified_but_master_animals_are_not_augmented():
    rows = [record(candidates=[candidate(margin=-0.2)])]
    masters = {
        "image": master(
            predictions=[raw_prediction(), raw_prediction(cls=2, score=0.9)]
        )
    }
    assert presence_metrics(rows, masters, 0, mode="augment")["tp"] == 1
    assert presence_metrics(rows, masters, 0, mode="replace")["fn"] == 1
    assert presence_metrics(rows, masters, 2, mode="replace")["fp"] == 0
    assert presence_metrics(rows, masters, 2, mode="master_only")["fp"] == 1
    with pytest.raises(ValueError, match="always replace"):
        presence_metrics(rows, masters, 2, mode="augment")


def test_threshold_and_margin_gates_are_inclusive():
    rows = [record(candidates=[candidate(score=0.2, margin=0.02)])]
    masters = {"image": master()}
    assert presence_metrics(rows, masters, 0, threshold=0.2, margin=0.02)["tp"] == 1
    assert presence_metrics(rows, masters, 0, threshold=0.25, margin=0.02)["fn"] == 1
    assert presence_metrics(rows, masters, 0, threshold=0.2, margin=0.04)["fn"] == 1


def test_bbox_metrics_deduplicate_dino_and_score_only_click_validation():
    rows = [
        record(
            "click",
            candidates=[
                candidate(score=0.4),
                candidate((1, 1, 19, 19), score=0.5, index=1),
            ],
        ),
        record("rodent", candidates=[candidate()]),
        record("water", candidates=[candidate()]),
    ]
    masters = {
        "click": master(truth=[raw_prediction()]),
        "rodent": master(source="irodent", split="test", truth=[raw_prediction(cls=2)]),
        "water": master(source="stagnant_water"),
    }
    result = litter_bbox_metrics(rows, masters)
    assert result["images"] == result["reference_boxes"] == 1
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    assert result["f1"] == 1


def test_bbox_selection_captures_completeness_when_baseline_presence_is_already_perfect():
    second = (40, 40, 60, 60)
    rows = [
        record("positive", candidates=[candidate(second, score=0.4, margin=0.1)]),
        record(
            "negative", residue=False, candidates=[candidate(score=0.25, margin=0.0)]
        ),
    ]
    masters = {
        "positive": master(
            truth=[raw_prediction(), raw_prediction(second)],
            predictions=[raw_prediction()],
        ),
        "negative": master(source="stagnant_water"),
    }
    report = evaluate_grid(rows, masters)
    assert report["baseline"]["0"]["presence"]["f1"] == 1
    assert report["baseline"]["0"]["bbox"]["fn"] == 1
    selected = report["selected"]["0"]
    assert selected["mode"] == "augment"
    assert selected["bbox"]["f1"] == 1
    assert selected["bbox"]["tp"] == 2
    assert selected["presence"]["fp"] == 0
    assert selected["margin"] == 0.1
    assert selected["threshold"] == 0.4


def test_bbox_selection_enforces_presence_guard_then_fewer_fp_and_larger_margin():
    def row(bbox_f1, bbox_fp, presence_fp, margin):
        return {
            "bbox": {"f1": bbox_f1, "fp": bbox_fp},
            "presence": {"f1": 0.8, "fp": presence_fp},
            "margin": margin,
            "threshold": 0.3,
            "mode": "augment",
        }

    rejected = row(1.0, 0, 3, 0.16)
    low_margin = row(0.8, 2, 1, 0.02)
    high_margin = row(0.8, 2, 1, 0.08)
    more_box_fp = row(0.8, 3, 0, 0.12)
    selected = select_grid_row(
        [rejected, low_margin, high_margin, more_box_fp],
        0,
        residue_selection="bbox",
        max_presence_fp=2,
    )
    assert selected is high_margin


def test_candidate_floor_keeps_order_and_does_not_reinterpret_unknown_classes():
    raw = [
        {"class_id": None, "confidence": 0.9},
        {"class_id": "0", "confidence": 0.9},
        {"class_id": True, "confidence": 0.9},
        {"class_id": 0, "confidence": 0.199},
        {"class_id": 2, "confidence": 0.2, "candidate_index": 9},
        {"class_id": 0, "confidence": 0.4, "candidate_index": 11},
    ]
    original = copy.deepcopy(raw)
    selected = eligible_candidates(raw)
    assert [item["candidate_index"] for item in selected] == [9, 11]
    selected[0]["class_id"] = 0
    assert raw == original


def test_verifier_contract_rejects_filtering_reordering_or_modified_boxes():
    scored = [candidate(index=0), candidate((40, 40, 60, 60), index=1)]
    raw = [
        {key: value for key, value in item.items() if key != "verification"}
        for item in scored
    ]
    validate_verifier_result(raw, scored)
    with pytest.raises(ValueError, match="every candidate"):
        validate_verifier_result(raw, scored[:1])
    with pytest.raises(ValueError, match="changed a candidate"):
        validate_verifier_result(raw, list(reversed(scored)))
    changed = copy.deepcopy(scored)
    changed[0]["bounding_box"][0] = 1
    with pytest.raises(ValueError, match="changed a candidate"):
        validate_verifier_result(raw, changed)


def test_test_split_click_boxes_cannot_drive_parameter_selection():
    with pytest.raises(ValueError, match="validation only"):
        litter_bbox_metrics([record()], {"image": master(split="test")})


def test_f2_formula_prioritizes_recall_with_an_explicit_precision_guard():
    rows = [record(candidates=[candidate()])]
    masters = {
        "image": master(truth=[raw_prediction(), raw_prediction((40, 40, 60, 60))])
    }
    metrics = litter_bbox_metrics(rows, masters)
    assert metrics["f2"] == pytest.approx(5 / 9)
    good = {
        "bbox": {"f2": 0.81, "fp": 14, "precision": 0.71},
        "presence": {"fp": 0},
        "margin": 0.04,
        "threshold": 0.3,
        "mode": "augment",
    }
    rejected = {
        "bbox": {"f2": 0.9, "fp": 19, "precision": 0.69},
        "presence": {"fp": 0},
        "margin": 0.02,
        "threshold": 0.2,
        "mode": "augment",
    }
    assert (
        select_grid_row(
            [good, rejected],
            0,
            residue_selection="bbox-f2",
            max_presence_fp=2,
            min_residue_box_precision=0.7,
        )
        is good
    )
