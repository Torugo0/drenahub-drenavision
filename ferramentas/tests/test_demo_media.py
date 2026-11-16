from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ferramentas.scripts.build_demo_video import build_demo_video, select_test_slides


def _fixture_dataset(
    tmp_path: Path, *, omit_class: int | None = None
) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    image_dir = root / "images" / "test"
    label_dir = root / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    fields = [
        "output_image",
        "output_label",
        "split",
        "source_key",
        "boxes_class_0",
        "boxes_class_1",
        "boxes_class_2",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for class_id in range(3):
            if class_id == omit_class:
                continue
            name = f"class_{class_id}"
            image_path = image_dir / f"{name}.jpg"
            label_path = label_dir / f"{name}.txt"
            image = np.full((90 + class_id * 5, 120, 3), 50 + class_id * 60, np.uint8)
            assert cv2.imwrite(str(image_path), image)
            label_path.write_text(
                f"{class_id} 0.500000 0.500000 0.400000 0.400000\n",
                encoding="utf-8",
            )
            row = {
                "output_image": image_path.relative_to(root).as_posix(),
                "output_label": label_path.relative_to(root).as_posix(),
                "split": "test",
                "source_key": f"source_{class_id}",
                "boxes_class_0": "0",
                "boxes_class_1": "0",
                "boxes_class_2": "0",
            }
            row[f"boxes_class_{class_id}"] = "1"
            writer.writerow(row)
    return root, manifest


def test_builds_truthful_video_and_sanitized_sidecar(tmp_path: Path) -> None:
    dataset, manifest = _fixture_dataset(tmp_path)
    output = tmp_path / "demo.mp4"
    payload = build_demo_video(
        dataset,
        manifest,
        output,
        seconds_per_image=0.5,
        fps=4,
        size=(320, 240),
    )

    assert output.is_file() and output.stat().st_size > 0
    assert payload["contains_model_predictions"] is False
    assert payload["is_recorded_video"] is False
    assert [item["class_id"] for item in payload["slides"]] == [0, 1, 2]
    assert payload["video"]["total_frames"] == 6
    assert str(tmp_path.resolve()) not in json.dumps(payload)

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    capture.release()
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["output_sha256"] == payload["output_sha256"]


def test_requires_ground_truth_for_all_classes(tmp_path: Path) -> None:
    dataset, manifest = _fixture_dataset(tmp_path, omit_class=2)
    with pytest.raises(ValueError, match="animal_visivel"):
        select_test_slides(dataset, manifest)
