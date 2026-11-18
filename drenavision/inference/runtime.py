"""One auditable image/video policy for learned object and water predictions."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from drenavision.core.schemas import Detection

from .contracts import PredictionResult
from .predictor import YoloPredictor
from .utils import box_iou, sha256_file


def merge_object_detections(
    master_detections: list[Detection],
    specialist_detections: list[Detection],
    *,
    mode: str = "animal_replace",
) -> list[Detection]:
    """Fuse real neural proposals; no invented, expanded or union coordinates.

    Specialist coordinates have priority for duplicate objects. This policy
    does not compare the calibration of scores from different architectures.
    Animal replacement avoids retaining an additional partial-body master box.
    Water proposals never participate in object NMS.
    """
    if mode not in {"augment", "replace", "animal_replace"}:
        raise ValueError("invalid object fusion mode")
    other = [item for item in master_detections if item.class_id not in {0, 2}]
    candidates = [item for item in specialist_detections if item.class_id in {0, 2}]
    if mode != "replace":
        candidates.extend(
            item
            for item in master_detections
            if item.class_id == 0 or (item.class_id == 2 and mode == "augment")
        )
    retained: list[Detection] = []
    for item in candidates:
        if not any(
            item.class_id == previous.class_id
            and box_iou(item.bounding_box, previous.bounding_box) >= 0.5
            for previous in retained
        ):
            retained.append(item)
    return other + retained


def filter_detections(
    detections: list[Detection],
    *,
    residue_confidence: float = 0.35,
    animal_confidence: float = 0.60,
) -> list[Detection]:
    """Use declared class operating points; never alter coordinates."""
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in (residue_confidence, animal_confidence)
    ):
        raise ValueError("class confidence thresholds must be finite and within [0, 1]")
    return [
        item
        for item in detections
        if item.confidence
        >= {0: residue_confidence, 2: animal_confidence}.get(item.class_id, 0.0)
    ]


def _area(item: Detection) -> float:
    x1, y1, x2, y2 = item.bounding_box
    return (x2 - x1) * (y2 - y1)


def merge_water_detections(
    master_detections: list[Detection],
    water_detections: list[Detection],
    *,
    mode: str = "augment",
) -> list[Detection]:
    """Retain actual water proposals, suppress contained duplicate envelopes.

    No box is expanded or invented. Independent components can have overlapping
    envelopes; these are visual marks, not a count of physical water bodies.
    """
    if mode not in {"augment", "replace"}:
        raise ValueError("water mode must be augment or replace")
    other = [item for item in master_detections if item.class_id != 1]
    water = [item for item in water_detections if item.class_id == 1]
    if mode == "augment":
        water.extend(item for item in master_detections if item.class_id == 1)
    retained: list[Detection] = []
    for item in sorted(
        water, key=lambda detection: (-_area(detection), -detection.confidence)
    ):
        x1, y1, x2, y2 = item.bounding_box
        duplicate = False
        for previous in retained:
            p1, q1, p2, q2 = previous.bounding_box
            intersection = max(0.0, min(x2, p2) - max(x1, p1)) * max(
                0.0, min(y2, q2) - max(y1, q1)
            )
            if intersection / min(_area(item), _area(previous)) >= 0.8:
                duplicate = True
                break
        if not duplicate:
            retained.append(item)
    return other + retained


def profile_version(
    master_version: str,
    water_provenance: dict[str, Any] | None,
    parameters: dict[str, Any],
) -> str:
    content = json.dumps(
        {
            "master": master_version,
            "water": water_provenance,
            "policy": parameters,
            "fusion": "contained-water-0.8-area-v1",
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ).encode()
    return "pipeline-" + hashlib.sha256(content).hexdigest()[:16]


def overlay_water(image_rgb: Any, mask: Any, *, alpha: float = 0.18) -> Any:
    """Tint only pixels accepted by the real water segmentation model."""
    import numpy as np

    result = image_rgb.copy()
    if mask is not None:
        selected = np.asarray(mask) > 0
        if selected.shape != result.shape[:2]:
            raise ValueError("water mask must match original image dimensions")
        result[selected] = (
            (1 - alpha) * result[selected].astype(np.float32)
            + alpha * np.asarray([2, 132, 199], dtype=np.float32)
        ).astype(np.uint8)
    return result


class InspectionPredictor:
    """Compose real proposals with a shared, versioned operating policy."""

    def __init__(
        self,
        master: YoloPredictor,
        water: Any = None,
        *,
        residue_confidence: float = 0.35,
        animal_confidence: float = 0.60,
        water_mode: str = "augment",
        objects: Any = None,
        object_mode: str = "animal_replace",
    ) -> None:
        self.master = master
        self.water = water
        self.objects = objects
        self.object_mode = object_mode
        if object_mode not in {"augment", "replace", "animal_replace"}:
            raise ValueError("invalid object fusion mode")
        self.parameters = {
            "residue_confidence": residue_confidence,
            "animal_confidence": animal_confidence,
            "water_mode": water_mode,
            "imgsz": master.image_size,
            "confidence": master.confidence_threshold,
            "iou": master.iou_threshold,
        }

    @property
    def is_ready(self) -> bool:
        return (
            self.master.is_ready
            and (self.water is None or self.water.is_ready)
            and (self.objects is None or self.objects.is_ready)
        )

    @property
    def detail(self) -> str:
        if not self.master.is_ready:
            return self.master.detail
        if self.water is not None and not self.water.is_ready:
            return self.water.detail
        if self.objects is not None and not self.objects.is_ready:
            return self.objects.detail
        return "ready"

    @property
    def model_version(self) -> str:
        parameters = dict(self.parameters)
        if self.objects is not None:
            parameters.update(objects=self.objects.provenance, object_mode=self.object_mode)
        return profile_version(
            sha256_file(self.master.model_path),
            self.water.provenance if self.water is not None else None,
            parameters,
        )

    def predict(self, image_rgb: Any) -> PredictionResult:
        from .video import annotate_video_frame

        started = time.perf_counter()
        base = self.master.predict(image_rgb)
        detections = filter_detections(
            base.detections,
            residue_confidence=self.parameters["residue_confidence"],
            animal_confidence=self.parameters["animal_confidence"],
        )
        if self.objects is not None:
            object_result = self.objects.predict(image_rgb)
            detections = merge_object_detections(
                detections, object_result.detections, mode=self.object_mode
            )
        display = image_rgb
        if self.water is not None:
            water = self.water.predict(image_rgb)
            detections = merge_water_detections(
                detections,
                water.detections,
                mode=self.parameters["water_mode"],
            )
            display = overlay_water(image_rgb, water.mask)
        # Always redraw from the original pixels; no old YOLO box can survive
        # filtering via the model's pre-rendered annotated image.
        annotated_rgb = annotate_video_frame(display[:, :, ::-1].copy(), detections)[
            :, :, ::-1
        ].copy()
        return PredictionResult(
            detections=detections,
            annotated_image=annotated_rgb,
            inference_ms=(time.perf_counter() - started) * 1000,
            model_version=self.model_version,
            image_width=base.image_width,
            image_height=base.image_height,
            image_size=base.image_size,
        )


def create_predictor(
    model_path: str | Path,
    *,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    device: str = "auto",
    image_size: int = 640,
    water_model_path: str | Path | None = None,
    water_pixel_confidence: float = 0.5,
    water_min_area_fraction: float = 0.001,
    water_mode: str = "augment",
    residue_confidence: float = 0.35,
    animal_confidence: float = 0.60,
    object_model_path: str | Path | None = None,
    object_verifier_path: str | Path | None = None,
    object_residue_min_margin: float = 0.0,
    object_animal_min_margin: float = 0.0,
    object_residue_confidence: float = 0.35,
    object_animal_confidence: float = 0.35,
    object_text_confidence: float = 0.20,
    object_mode: str = "animal_replace",
    object_shortest_edge: int = 800,
    object_longest_edge: int = 1333,
) -> InspectionPredictor:
    if object_verifier_path is not None and object_model_path is None:
        raise ValueError("an object verifier requires a configured object detector")
    master = YoloPredictor(
        model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
        image_size=image_size,
    )
    water = None
    if water_model_path is not None:
        from .water import SemanticWaterPredictor

        water = SemanticWaterPredictor(
            water_model_path,
            device=device,
            pixel_confidence=water_pixel_confidence,
            min_area_fraction=water_min_area_fraction,
            processing_size=512,
        )
    objects = None
    if object_model_path is not None:
        from .objects import GroundedObjectPredictor

        objects = GroundedObjectPredictor(
            object_model_path,
            device=device,
            residue_confidence=object_residue_confidence,
            animal_confidence=object_animal_confidence,
            text_confidence=object_text_confidence,
            shortest_edge=object_shortest_edge,
            longest_edge=object_longest_edge,
            verifier_model_path=object_verifier_path,
            residue_min_margin=object_residue_min_margin,
            animal_min_margin=object_animal_min_margin,
        )
    return InspectionPredictor(
        master,
        water,
        residue_confidence=residue_confidence,
        animal_confidence=animal_confidence,
        water_mode=water_mode,
        objects=objects,
        object_mode=object_mode,
    )
