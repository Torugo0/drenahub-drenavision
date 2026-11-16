from pathlib import Path

import pytest

from drenavision.data.yolo import YoloBox, coco_bbox_to_yolo, read_yolo, write_yolo


def test_yolo_round_trip_and_empty_negative(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    boxes = [YoloBox(2, 0.5, 0.25, 0.2, 0.1)]
    write_yolo(label, boxes)
    assert read_yolo(label, class_count=3) == boxes

    empty = tmp_path / "negative.txt"
    write_yolo(empty, [])
    assert empty.read_text(encoding="utf-8") == ""
    assert read_yolo(empty, class_count=3) == []


@pytest.mark.parametrize(
    "line",
    [
        "0.5 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0 0.2",
        "0 0.95 0.5 0.2 0.2",
        "3 0.5 0.5 0.2",
        "0 nan 0.5 0.2 0.2",
    ],
)
def test_yolo_parser_rejects_malformed_or_outside_boxes(line: str) -> None:
    with pytest.raises(ValueError):
        box = YoloBox.parse(line)
        box.validate(class_count=3)


def test_coco_box_is_clipped_to_image_boundaries() -> None:
    box = coco_bbox_to_yolo([-10, 20, 50, 60], 100, 100, class_id=2)
    assert box == YoloBox(2, 0.2, 0.5, 0.4, 0.6)


def test_coco_box_fully_outside_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        coco_bbox_to_yolo([150, 20, 10, 10], 100, 100, class_id=2)
