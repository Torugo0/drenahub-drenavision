"""Exercise v3 assembly isolation without source downloads or model inference."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

from drenavision.data.converter import (
    SourceCandidate,
    _write_candidate,
    write_manifest,
)
from drenavision.data.yolo import YoloBox, read_yolo
from ferramentas.scripts import prepare_water_v3
from ferramentas.scripts.prepare_v3_dataset import prepare


def _fixture(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (baseline / kind / split).mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    rows, seen = [], {}
    specs = (
        ("train", "residue", "click_trash", 0),
        ("train", "rat", "irodent", 2),
        ("train", "water", "stagnant_water", 1),
        ("train", "rato_residuos", "irodent", 2),
        ("val", "water_val", "stagnant_water", 1),
        ("val", "rat_val", "irodent", 2),
        ("test", "water_test", "stagnant_water", 1),
        ("test", "residue_test", "click_trash", 0),
    )
    for index, (split, group, source, class_id) in enumerate(specs):
        path = raw / f"{group}.png"
        pixels = np.random.default_rng(index + 70).integers(
            0, 256, (64, 80, 3), dtype=np.uint8
        )
        Image.fromarray(pixels).save(path)
        boxes = (
            (YoloBox(1, 0.4, 0.4, 0.3, 0.3), YoloBox(1, 0.6, 0.6, 0.3, 0.3))
            if class_id == 1
            else (YoloBox(class_id, 0.5, 0.5, 0.3, 0.3),)
        )
        candidate = SourceCandidate(
            path, boxes, source, path.name, group, "fixture source annotations"
        )
        rows.append(_write_candidate(candidate, split, baseline, seen))
    write_manifest(rows, baseline / "data_manifest.csv")
    monkeypatch.setattr(
        prepare_water_v3,
        "select_extra_water_training",
        lambda *_args, **_kwargs: ([], {}),
    )
    return baseline, rows


def _domain(tmp_path, image, split="train"):
    domain = tmp_path / "domain"
    (domain / "images").mkdir(parents=True)
    (domain / "labels").mkdir()
    image.save(domain / "images/context.png")
    (domain / "labels/context.txt").write_text("2 0.5 0.5 0.2 0.2\n", "utf-8")
    (domain / "sources.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "file_name": "context.png",
                        "source_group": "rato_residuos",
                        "source_url": "https://example.org/context",
                        "source_license": "CC BY 4.0",
                        "annotation_note": "reviewed full-frame fixture",
                        "split": split,
                    }
                ]
            }
        ),
        "utf-8",
    )
    return domain


def _rows(root):
    with (root / "data_manifest.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_v3_preserves_heldout_pixels_and_source_labels_while_replacing_tight_rat(
    tmp_path, monkeypatch
):
    baseline, old_rows = _fixture(tmp_path, monkeypatch)
    baseline_bytes = {
        p.relative_to(baseline): p.read_bytes()
        for p in baseline.rglob("*")
        if p.is_file()
    }
    rng = np.random.default_rng(130)
    domain = _domain(
        tmp_path, Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8))
    )
    output = tmp_path / "v3"
    report = prepare(baseline, output, [domain], None, tmp_path / "audit")
    assert report["validation"]["valid"]
    assert report["images_by_split"] == {"train": 4, "val": 2, "test": 2}
    held_old = {r.output_image: r for r in old_rows if r.split != "train"}
    held_new = {r["output_image"]: r for r in _rows(output) if r["split"] != "train"}
    assert held_old.keys() == held_new.keys()
    for name, row in held_old.items():
        assert (baseline / name).read_bytes() == (output / name).read_bytes()
        assert held_new[name]["sha256"] == row.sha256
        assert held_new[name]["source_group"] == row.source_group
        old_boxes = read_yolo(baseline / row.output_label, class_count=3)
        new_boxes = read_yolo(output / row.output_label, class_count=3)
        if row.source_key == "stagnant_water":
            assert len(old_boxes) == 2 and len(new_boxes) == 1
        else:
            assert old_boxes == new_boxes
    assert all(
        (baseline / p).read_bytes() == data for p, data in baseline_bytes.items()
    )
    assert len(report["removed_tight_rat_crops"]) == 1
    rat_rows = [r for r in _rows(output) if r["source_group"] == "rato_residuos"]
    assert len(rat_rows) == 1 and rat_rows[0]["split"] == "train"


@pytest.mark.parametrize("declared_split", ["val", "test"])
def test_v3_rejects_supplement_declared_as_heldout(
    tmp_path, monkeypatch, declared_split
):
    baseline, _ = _fixture(tmp_path, monkeypatch)
    domain = _domain(tmp_path, Image.new("RGB", (80, 64), "blue"), declared_split)
    with pytest.raises(ValueError, match="training-only"):
        prepare(baseline, tmp_path / "v3", [domain], None, tmp_path / "audit")


def test_v3_does_not_copy_a_heldout_image_into_training(tmp_path, monkeypatch):
    baseline, rows = _fixture(tmp_path, monkeypatch)
    heldout = next(row for row in rows if row.split == "test")
    with Image.open(baseline / heldout.output_image) as opened:
        domain = _domain(tmp_path, opened.copy())
    report = prepare(baseline, tmp_path / "v3", [domain], None, tmp_path / "audit")
    merged = _rows(tmp_path / "v3")
    assert report["skipped"]["duplicate_image"] == 1
    assert [r["split"] for r in merged if r["sha256"] == heldout.sha256] == ["test"]


def test_v3_excludes_near_duplicate_heldout_pixels_with_different_file_hash(
    tmp_path, monkeypatch
):
    baseline, rows = _fixture(tmp_path, monkeypatch)
    heldout = next(row for row in rows if row.split == "test")
    with Image.open(baseline / heldout.output_image) as opened:
        pixels = np.asarray(opened).copy()
    pixels[0, 0, 0] ^= 1
    domain = _domain(tmp_path, Image.fromarray(pixels))
    report = prepare(baseline, tmp_path / "v3", [domain], None, tmp_path / "audit")
    assert report["skipped"]["near_heldout_image"] == 1
    assert report["added_training_images"] == {}


def test_v3_excludes_unreliable_source_water_negatives_by_rule_in_all_splits(
    tmp_path, monkeypatch
):
    baseline, rows = _fixture(tmp_path, monkeypatch)
    excluded = []
    for index, split in enumerate(("train", "val", "test")):
        raw = tmp_path / f"wet_negative_{split}.png"
        pixels = np.random.default_rng(230 + index).integers(
            0, 256, (64, 80, 3), dtype=np.uint8
        )
        Image.fromarray(pixels).save(raw)
        candidate = SourceCandidate(
            raw,
            (),
            "stagnant_water",
            raw.name,
            "wet_negative_" + split,
            "Unreliable native wet-only empty label",
        )
        row = _write_candidate(candidate, split, baseline, {})
        rows.append(row)
        excluded.append(row)
    write_manifest(rows, baseline / "data_manifest.csv")
    options = {}

    def no_additional_negatives(*_args, **kwargs):
        options.update(kwargs)
        return [], {}

    monkeypatch.setattr(
        prepare_water_v3, "select_extra_water_training", no_additional_negatives
    )
    report = prepare(baseline, tmp_path / "v3", [], None, tmp_path / "audit")
    assert options["negative_count"] == 0
    assert {r["split"] for r in report["excluded_source_wet_only"]} == {
        "train",
        "val",
        "test",
    }
    assert report["images_by_split"] == {"train": 3, "val": 2, "test": 2}
    selected_hashes = {r["sha256"] for r in _rows(tmp_path / "v3")}
    assert not selected_hashes.intersection(row.sha256 for row in excluded)
    assert all((baseline / row.output_image).exists() for row in excluded)
