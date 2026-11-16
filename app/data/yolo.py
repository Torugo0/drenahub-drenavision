"""Strict YOLO bounding-box parsing and conversion primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def parse(cls, line: str, *, line_number: int = 1) -> YoloBox:
        fields = line.strip().split()
        if len(fields) != 5:
            raise ValueError(
                f"line {line_number}: expected 5 fields, found {len(fields)}"
            )
        try:
            class_value = float(fields[0])
            coords = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric value") from exc
        if not class_value.is_integer():
            raise ValueError(f"line {line_number}: class id must be an integer")
        if not all(math.isfinite(value) for value in (class_value, *coords)):
            raise ValueError(f"line {line_number}: values must be finite")
        box = cls(int(class_value), *coords)
        box.validate(line_number=line_number)
        return box

    def validate(
        self,
        *,
        line_number: int = 1,
        class_count: int | None = None,
        epsilon: float = 1e-6,
    ) -> None:
        if self.class_id < 0 or (
            class_count is not None and self.class_id >= class_count
        ):
            raise ValueError(f"line {line_number}: invalid class id {self.class_id}")
        values = (self.x_center, self.y_center, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"line {line_number}: values must be finite")
        if not (0 <= self.x_center <= 1 and 0 <= self.y_center <= 1):
            raise ValueError(f"line {line_number}: center must be normalized")
        if not (0 < self.width <= 1 and 0 < self.height <= 1):
            raise ValueError(f"line {line_number}: width/height must be in (0, 1]")
        left = self.x_center - self.width / 2
        right = self.x_center + self.width / 2
        top = self.y_center - self.height / 2
        bottom = self.y_center + self.height / 2
        if min(left, top) < -epsilon or max(right, bottom) > 1 + epsilon:
            raise ValueError(f"line {line_number}: box extends outside the image")

    def remap(self, class_id: int) -> YoloBox:
        return YoloBox(class_id, self.x_center, self.y_center, self.width, self.height)

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def read_yolo(path: Path, *, class_count: int | None = None) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        box = YoloBox.parse(line, line_number=line_number)
        box.validate(line_number=line_number, class_count=class_count)
        boxes.append(box)
    return boxes


def parse_yolo_bytes(data: bytes, *, class_count: int | None = None) -> list[YoloBox]:
    text = data.decode("utf-8-sig")
    boxes: list[YoloBox] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        box = YoloBox.parse(line, line_number=line_number)
        box.validate(line_number=line_number, class_count=class_count)
        boxes.append(box)
    return boxes


def write_yolo(path: Path, boxes: Iterable[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [box.to_line() for box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def coco_bbox_to_yolo(
    bbox: Iterable[float], image_width: int, image_height: int, class_id: int
) -> YoloBox:
    x, y, width, height = (float(value) for value in bbox)
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        raise ValueError("COCO box and image dimensions must be positive")
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("COCO box is outside the image")
    box = YoloBox(
        class_id=class_id,
        x_center=((x1 + x2) / 2) / image_width,
        y_center=((y1 + y2) / 2) / image_height,
        width=(x2 - x1) / image_width,
        height=(y2 - y1) / image_height,
    )
    box.validate(class_count=3)
    return box
