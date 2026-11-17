import csv
from pathlib import Path

import pytest
from PIL import Image

from drenavision.data.converter import SourceCandidate, sha256_file
from drenavision.data.yolo import YoloBox
from ferramentas.scripts import prepare_water_v3
from ferramentas.scripts.prepare_water_v3 import merge_source_water_boxes


def test_unreviewed_wet_surface_negatives_are_not_automatically_imported(tmp_path):
    with pytest.raises(ValueError, match="not reliable visible-water negatives"):
        prepare_water_v3.select_extra_water_training(
            tmp_path,
            tmp_path / "not_read.csv",
            tmp_path,
            negative_count=1,
        )


def _box(left, top, right, bottom, class_id=1):
    return YoloBox(
        class_id, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
    )


def test_water_components_merge_transitive_original_intersections_without_padding():
    boxes = [
        _box(0.1, 0.2, 0.3, 0.4),
        _box(0.2, 0.3, 0.4, 0.5),
        _box(0.3, 0.4, 0.5, 0.6),
    ]
    merged = merge_source_water_boxes(boxes)
    assert len(merged) == 1
    assert merged[0].to_line() == _box(0.1, 0.2, 0.5, 0.6).to_line()
    assert len(boxes) == 3


def test_water_component_envelope_cannot_absorb_disconnected_water_inside_gap():
    # Two original boxes form an L; its empty interior encloses a separate pool.
    boxes = [_box(0, 0, 0.25, 0.8), _box(0, 0, 0.8, 0.25), _box(0.5, 0.5, 0.7, 0.7)]
    merged = merge_source_water_boxes(boxes)
    assert len(merged) == 2
    assert merged[0].to_line() == _box(0, 0, 0.8, 0.8).to_line()
    assert merged[1] == boxes[2]


def test_water_components_touching_exactly_merge_but_positive_gap_does_not():
    boxes = [
        _box(0, 0, 0.25, 0.25),
        _box(0.25, 0.25, 0.5, 0.5),
        _box(0.51, 0.51, 0.6, 0.6),
    ]
    merged = merge_source_water_boxes(boxes)
    assert len(merged) == 2
    assert merged[0].to_line() == _box(0, 0, 0.5, 0.5).to_line()


def test_water_merge_keeps_other_classes_and_empty_labels():
    litter = _box(0, 0, 0.5, 0.5, 0)
    animal = _box(0.1, 0.1, 0.6, 0.6, 2)
    water = _box(0, 0, 0.8, 0.8)
    assert merge_source_water_boxes([litter, animal, water]) == (litter, animal, water)
    assert merge_source_water_boxes([]) == ()


def test_extra_water_training_excludes_held_out_group_and_perceptual_neighbor(
    tmp_path: Path, monkeypatch
):
    held_image = tmp_path / "held.png"
    Image.new("RGB", (12, 12), "red").save(held_image)
    baseline_manifest = tmp_path / "manifest.csv"
    with baseline_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["split", "source_group", "sha256", "output_image"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "test",
                "source_group": "00_held",
                "sha256": sha256_file(held_image),
                "output_image": held_image.name,
            }
        )
    candidates = []
    for name, group, color in (
        ("same_group", "00_held", "blue"),
        ("near_duplicate", "01_near", "green"),
        ("safe", "02_safe", "yellow"),
    ):
        image_path = tmp_path / (name + ".png")
        Image.new("RGB", (12, 12), color).save(image_path)
        candidates.append(
            SourceCandidate(
                image_path,
                (_box(0.1, 0.2, 0.3, 0.4),),
                "stagnant_water",
                "folder/" + image_path.name,
                group,
                "fixture",
            )
        )

    class PreserveOrder:
        def shuffle(self, values):
            pass

    monkeypatch.setattr(prepare_water_v3.random, "Random", lambda seed: PreserveOrder())
    monkeypatch.setattr(
        prepare_water_v3, "collect_water_candidates", lambda root: (candidates, [])
    )
    monkeypatch.setattr(
        prepare_water_v3,
        "_phash",
        lambda path: (
            0
            if path.name == "held.png"
            else 1
            if path.name == "near_duplicate.png"
            else 65535
        ),
    )
    selected, report = prepare_water_v3.select_extra_water_training(
        tmp_path, baseline_manifest, tmp_path, positive_count=1, negative_count=0
    )
    assert len(selected) == 1
    assert selected[0].source_group == "02_safe"
    assert report["counts"]["positive_held_group_excluded"] == 1
    assert report["counts"]["positive_held_phash_excluded"] == 1
