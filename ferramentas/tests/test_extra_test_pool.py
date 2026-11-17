from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ferramentas.scripts.prepare_extra_test import (
    apply_blinded_review,
    exclusion_reason,
    prepare,
)


def test_extra_pool_rejects_exact_group_and_near_duplicates() -> None:
    assert exclusion_reason("same", "new", 0, {"same"}, set(), [], 4) == (
        "exact_image_hash",
        None,
    )
    assert exclusion_reason("new", "same-group", 0, set(), {"same-group"}, [], 4) == (
        "source_group",
        "same-group",
    )
    assert exclusion_reason(
        "new", "new", 0b1111, set(), set(), [("training.jpg", 0)], 4
    ) == ("perceptual_similarity", "training.jpg")
    assert (
        exclusion_reason("new", "new", 0b11111, set(), set(), [("training.jpg", 0)], 4)
        is None
    )


def test_prepared_pool_preserves_native_images_and_stays_separate_from_training(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    annotations = raw / "rodent/annotations"
    annotations.mkdir(parents=True)
    reference = tmp_path / "excluded"
    (reference / "images/train").mkdir(parents=True)
    records = []
    generator = np.random.default_rng(42)
    for index, name in enumerate(("used.png", "fresh.png"), 1):
        Image.fromarray(generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            raw / name
        )
        records.append({"id": index, "file_name": name, "width": 32, "height": 32})
    (reference / "images/train/used.png").write_bytes((raw / "used.png").read_bytes())
    (reference / "data_manifest.csv").write_text("source_group\nirodent::used.png\n")
    (annotations / "test.json").write_text(
        json.dumps(
            {
                "images": records,
                "annotations": [
                    {
                        "id": index,
                        "image_id": index,
                        "bbox": [1, 1, 25, 25],
                        "keypoints": [4, 4, 2, 24, 24, 2],
                    }
                    for index in (1, 2)
                ],
            }
        )
    )
    output = tmp_path / "extra_test"
    report = prepare(raw, output, [reference], 4)
    assert report["status"] == "prepared_without_inference"
    assert report["accepted_images"] == 1
    assert report["excluded_by_reason"] == {"exact_image_hash": 1}
    assert (
        list((output / "images/test").iterdir())[0].read_bytes()
        == (raw / "fresh.png").read_bytes()
    )
    with (output / "data_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["split"] == "test"
    assert rows[0]["source_path"] == "annotations/test.json::fresh.png"
    assert not (output / "images/train").exists()
    assert "train:" not in (output / "dataset.yaml").read_text()
    with pytest.raises(FileExistsError):
        prepare(raw, output, [reference], 4)


def test_blinded_review_archives_uncertain_inputs_and_removes_them_from_test_inventory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pool"
    (output / "images/test").mkdir(parents=True)
    (output / "labels/test").mkdir(parents=True)
    rows = []
    for name, count in (("water", 1), ("uncertain", 0)):
        source = output / f"images/test/{name}.png"
        source.write_bytes(name.encode())
        (output / f"labels/test/{name}.txt").write_text("")
        rows.append(
            {
                "source_path": name,
                "source_key": "stagnant_water",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_image": f"images/test/{name}.png",
                "output_label": f"labels/test/{name}.txt",
                "boxes_class_0": 0,
                "boxes_class_1": count,
                "boxes_class_2": 0,
            }
        )
    with (output / "data_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "preparation_report.json").write_text("{}")
    review = tmp_path / "review.json"
    decision = {
        "source_path": "uncertain",
        "sha256": rows[1]["sha256"],
        "reason": "Blurry source label; cannot certify a negative.",
    }
    review.write_text(
        json.dumps({"reviewed_before_inference": False, "exclusions": [decision]})
    )
    with pytest.raises(ValueError, match="no predictions"):
        apply_blinded_review(output, review)
    assert (output / rows[1]["output_image"]).is_file()
    review.write_text(
        json.dumps({"reviewed_before_inference": True, "exclusions": [decision]})
    )
    report = apply_blinded_review(output, review)
    assert report["accepted_images"] == 1
    assert report["blinded_review"]["retained_wet_negative_images"] == 0
    assert not (output / rows[1]["output_image"]).exists()
    assert (
        output / "review_excluded" / rows[1]["output_image"]
    ).read_bytes() == b"uncertain"
    assert (output / rows[0]["output_image"]).read_bytes() == b"water"
    with (output / "data_manifest.csv").open(newline="") as stream:
        assert [row["source_path"] for row in csv.DictReader(stream)] == ["water"]
