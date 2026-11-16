import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from drenavision.data.converter import SourceCandidate, _write_candidate, write_manifest
from drenavision.data.yolo import YoloBox
from ferramentas.scripts import convert_annotations
from ferramentas.scripts.prepare_adaptation import prepare_adaptation


@pytest.mark.parametrize("recover_from_raw", [False, True])
def test_v2_adds_only_native_training_images_and_preserves_baseline(
    tmp_path: Path, recover_from_raw: bool
):
    raw_root = tmp_path / "raw"
    source = raw_root / "rodent"
    (source / "annotations").mkdir(parents=True)
    (source / "images").mkdir()
    rng = np.random.default_rng(42)
    for split, names in (
        ("train", [f"train{index}.jpg" for index in range(5)]),
        ("test", ["test.jpg", "unused_test.jpg"]),
    ):
        images = []
        annotations = []
        for index, name in enumerate(names):
            Image.fromarray(rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)).save(
                source / "images" / name
            )
            images.append({"id": index, "file_name": name, "width": 50, "height": 40})
            annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "bbox": [0, 0, 35, 30],
                    "keypoints": [0, 0, 0, 15, 10, 2, 35, 30, 2],
                }
            )
        (source / "annotations" / (split + ".json")).write_text(
            json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
        )
    baseline = tmp_path / "baseline"
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (baseline / kind / split).mkdir(parents=True)
    rows = []
    for split, name in (
        ("train", "train0.jpg"),
        ("val", "train1.jpg"),
        ("test", "test.jpg"),
    ):
        native_split = "test" if split == "test" else "train"
        candidate = SourceCandidate(
            source / "images" / name,
            (YoloBox(2, 0.35, 0.375, 0.7, 0.75),),
            "irodent",
            f"annotations/{native_split}.json::{name}",
            f"irodent::{name}",
            "original-source-box",
        )
        rows.append(_write_candidate(candidate, split, baseline, {}))
    for class_id, source_key in ((0, "click_trash"), (1, "stagnant_water")):
        image_path = source / "images" / f"{source_key}.jpg"
        Image.fromarray(rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)).save(
            image_path
        )
        source_path = (
            f"images/train/{image_path.name}"
            if source_key == "click_trash"
            else image_path.name
        )
        candidate = SourceCandidate(
            image_path,
            (YoloBox(class_id, 0.5, 0.5, 0.3, 0.3),),
            source_key,
            source_path,
            source_key,
            "fixture",
        )
        rows.append(_write_candidate(candidate, "train", baseline, {}))
        if source_key == "click_trash":
            raw_image = raw_root / "click_trash/sample" / source_path
            raw_label = (
                raw_root
                / "click_trash/sample/labels/train"
                / (image_path.stem + ".txt")
            )
            for split in ("train", "val", "test"):
                (raw_root / "click_trash/sample/images" / split).mkdir(
                    parents=True, exist_ok=True
                )
                (raw_root / "click_trash/sample/labels" / split).mkdir(
                    parents=True, exist_ok=True
                )
        else:
            raw_image = (
                raw_root
                / "stagnant_water/extracted/Stagnant Water and Wet Surface Dataset"
                / source_path
            )
            raw_label = raw_image.with_suffix(".txt")
            raw_image.parent.mkdir(parents=True, exist_ok=True)
        raw_image.write_bytes(image_path.read_bytes())
        raw_label.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    baseline_manifest = tmp_path / "baseline_manifest.csv"
    write_manifest(rows, baseline_manifest)
    original_labels = {path: path.read_bytes() for path in baseline.rglob("*.txt")}
    v2_root = tmp_path / "v2"
    report = prepare_adaptation(
        tmp_path / "absent_baseline" if recover_from_raw else baseline,
        baseline_manifest,
        raw_root,
        v2_root,
        tmp_path / "audit",
        raw_root=raw_root,
    )
    assert report["images_by_split"] == {"train": 6, "val": 1, "test": 1}
    assert report["additional_native_training_images"] == 3
    assert report["baseline_items_recovered_from_raw"] == (2 if recover_from_raw else 0)
    assert report["validation"]["valid"] is True
    manifest_text = (v2_root / "data_manifest.csv").read_text(encoding="utf-8")
    assert "unused_test.jpg" not in manifest_text
    assert "visible-keypoints-v2" in manifest_text
    assert all(
        path.read_bytes() == original for path, original in original_labels.items()
    )


def test_conversion_cli_defaults_do_not_overwrite_historical_v1(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["convert_annotations.py"])

    def fake_convert(**kwargs):
        assert kwargs["output_root"] == Path("dados/processed/drenavision_corrected")
        assert kwargs["manifest_path"] == Path(
            "dados/manifests/dataset_manifest_corrected.csv"
        )
        return {"status": "contract-checked"}

    monkeypatch.setattr(convert_annotations, "convert_dataset", fake_convert)
    assert convert_annotations.main() == 0


def test_conversion_cli_rejects_explicit_historical_output(tmp_path, monkeypatch):
    monkeypatch.setattr(convert_annotations, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_annotations.py",
            "--output-root",
            str(tmp_path / "dados/processed/drenavision"),
        ],
    )
    monkeypatch.setattr(
        convert_annotations,
        "convert_dataset",
        lambda **_kwargs: pytest.fail("must preserve v1 before conversion starts"),
    )
    with pytest.raises(SystemExit) as exc:
        convert_annotations.main()
    assert exc.value.code == 2
