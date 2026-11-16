import json
from pathlib import Path

import pytest
from PIL import Image

from drenavision.data.converter import (
    SourceCandidate,
    _grouped_select,
    _write_candidate,
    collect_water_candidates,
    select_irodent,
)
from drenavision.data.yolo import YoloBox


def _image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color).save(path)


def _candidate(path: Path, index: int, group: str) -> SourceCandidate:
    _image(path, (index, index + 1, index + 2))
    return SourceCandidate(
        image=path,
        boxes=(YoloBox(0, 0.5, 0.5, 0.2, 0.2),),
        source_key="click_trash",
        source_path=path.name,
        source_group=group,
        annotation_note="fixture",
    )


def test_grouped_selection_is_deterministic_and_has_no_group_leakage(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(tmp_path / f"{index}.jpg", index, f"group-{index // 2}")
        for index in range(12)
    ]
    quotas = {"train": 4, "val": 2, "test": 2}
    first = _grouped_select(candidates, quotas, seed=42)
    second = _grouped_select(candidates, quotas, seed=42)
    assert {
        split: [candidate.source_path for candidate in values]
        for split, values in first.items()
    } == {
        split: [candidate.source_path for candidate in values]
        for split, values in second.items()
    }
    group_splits: dict[str, set[str]] = {}
    for split, values in first.items():
        assert len(values) == quotas[split]
        for candidate in values:
            group_splits.setdefault(candidate.source_group, set()).add(split)
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_water_mapping_excludes_raw_and_wet_surface_boxes(tmp_path: Path) -> None:
    root = tmp_path / "Stagnant Water and Wet Surface Dataset"
    valid = root / "Outdoor" / "Side view" / "image10"
    _image(valid.with_suffix(".jpg"))
    valid.with_suffix(".txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    wet_only = root / "Outdoor" / "Side view" / "image20"
    _image(wet_only.with_suffix(".jpg"))
    wet_only.with_suffix(".txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    raw = root / "Raw data" / "image30"
    _image(raw.with_suffix(".jpg"))
    raw.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    positives, negatives = collect_water_candidates(tmp_path)
    assert len(positives) == 1
    assert positives[0].boxes[0].class_id == 1
    assert len(negatives) == 1
    assert negatives[0].boxes == ()


def test_irodent_keeps_native_test_split(tmp_path: Path) -> None:
    annotation_root = tmp_path / "rodent" / "annotations"
    image_root = tmp_path / "rodent" / "images"
    annotation_root.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(5):
        name = f"train-{index}.jpg"
        _image(image_root / name, (index + 1, 1, 1))
        images.append({"id": index, "file_name": name, "width": 40, "height": 30})
        annotations.append(
            {"id": index, "image_id": index, "category_id": 1, "bbox": [1, 1, 10, 10]}
        )
    (annotation_root / "train.json").write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )
    _image(image_root / "test.jpg", (20, 1, 1))
    (annotation_root / "test.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 99, "file_name": "test.jpg", "width": 40, "height": 30}
                ],
                "annotations": [
                    {"id": 99, "image_id": 99, "category_id": 1, "bbox": [1, 1, 10, 10]}
                ],
            }
        ),
        encoding="utf-8",
    )

    selected = select_irodent(
        tmp_path, seed=42, quotas={"train": 2, "val": 1, "test": 1}
    )
    assert all("train.json" in item.source_path for item in selected["train"])
    assert all("train.json" in item.source_path for item in selected["val"])
    assert all("test.json" in item.source_path for item in selected["test"])


def test_writer_rejects_exact_duplicate_content(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    candidate = _candidate(source, 1, "one")
    output = tmp_path / "output"
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (output / kind / split).mkdir(parents=True)
    seen: dict[str, str] = {}
    _write_candidate(candidate, "train", output, seen)
    with pytest.raises(RuntimeError, match="exact duplicate"):
        _write_candidate(candidate, "val", output, seen)
