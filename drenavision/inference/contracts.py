"""Framework-neutral inference contracts shared by CLI, API, and UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space ``xyxy`` box, clamped to the decoded image bounds."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    track_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.as_dict(),
        }
        if self.track_id is not None:
            value["track_id"] = self.track_id
        return value


@dataclass(slots=True)
class PredictionResult:
    """One real model result; ``annotated_image`` is an RGB ndarray."""

    detections: list[Detection]
    annotated_image: Any
    inference_ms: float
    model_version: str
    image_width: int
    image_height: int
    image_size: int = 640

    @property
    def annotated_bgr(self) -> Any:
        """Compatibility view for OpenCV-only callers."""

        return self.annotated_image[:, :, ::-1]

    def as_dict(self) -> dict[str, Any]:
        serialized = []
        for item in self.detections:
            if hasattr(item, "model_dump"):
                serialized.append(item.model_dump(mode="json"))
            else:
                serialized.append(item.as_dict())
        return {
            "model_version": self.model_version,
            "inference_ms": self.inference_ms,
            "image": {"width": self.image_width, "height": self.image_height},
            "inference": {"imgsz": self.image_size},
            "detections": serialized,
        }
