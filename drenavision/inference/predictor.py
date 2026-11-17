"""Fail-closed custom-checkpoint image inference."""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from typing import Any

from drenavision.core.config import validate_image_size
from drenavision.core.schemas import Detection
from drenavision.data.constants import CLASS_NAMES

from .contracts import (
    PredictionResult,
)
from .device import describe_device, select_device
from .errors import ClassMapMismatchError, InferenceError, ModelUnavailableError
from .utils import sha256_file, validate_class_map


def _tensor_list(value: Any) -> list[Any]:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _core_detection(
    class_id: int,
    class_name: str,
    confidence: float,
    coordinates: tuple[float, float, float, float],
    track_id: int | None = None,
) -> Any:
    """Build the canonical API Pydantic schema."""

    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=coordinates,
        track_id=track_id,
    )


class YoloPredictor:
    """Lazy Ultralytics predictor that only accepts the custom three-class model."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        image_size: int = 640,
        **legacy: Any,
    ) -> None:
        if "confidence" in legacy:
            confidence_threshold = float(legacy.pop("confidence"))
        if "iou" in legacy:
            iou_threshold = float(legacy.pop("iou"))
        if legacy:
            raise TypeError(f"argumentos desconhecidos: {', '.join(legacy)}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold deve estar entre 0 e 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold deve estar entre 0 e 1")

        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.requested_device = device
        self.image_size = validate_image_size(image_size)
        self._device: str | int | None = None
        self._model: Any | None = None
        self._model_version: str | None = None
        self._detail: str = "modelo ainda nao carregado"
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except InferenceError as exc:
            self._detail = str(exc)
            return False
        except (ImportError, RuntimeError, ValueError) as exc:
            self._detail = str(exc)
            return False

    @property
    def ready(self) -> bool:
        return self.is_ready

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def device(self) -> str | None:
        return describe_device(self._device) if self._device is not None else None

    @property
    def model_version(self) -> str:
        if self._model_version is None:
            if not self.model_path.is_file():
                return "unavailable"
            self._model_version = (
                f"{self.model_path.stem}-{sha256_file(self.model_path)[:12]}"
            )
        return self._model_version

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                raise ModelUnavailableError(
                    "checkpoint customizado ausente em "
                    f"{self.model_path}; treine o modelo e configure o caminho correto"
                )
            try:
                ultralytics = importlib.import_module("ultralytics")
            except ImportError as exc:
                raise ModelUnavailableError(
                    "Ultralytics nao esta instalado; instale as dependencias do projeto"
                ) from exc
            try:
                device = select_device(self.requested_device)
                model = ultralytics.YOLO(str(self.model_path))
                validate_class_map(model.names)
            except ClassMapMismatchError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(
                    f"nao foi possivel carregar o checkpoint customizado: {exc}"
                ) from exc
            self._device = device
            self._model = model
            self._detail = "ready"

    def predict(self, image_rgb: Any) -> PredictionResult:
        """Run real inference on an RGB uint8 ``H x W x 3`` image."""

        if image_rgb is None or not hasattr(image_rgb, "shape"):
            raise ValueError("image_rgb deve ser um ndarray RGB")
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb deve ter formato H x W x 3")
        if int(image_rgb.shape[0]) <= 0 or int(image_rgb.shape[1]) <= 0:
            raise ValueError("imagem vazia")
        if str(getattr(image_rgb, "dtype", "")) != "uint8":
            raise ValueError("image_rgb deve usar dtype uint8")

        self._ensure_loaded()
        image_height, image_width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        image_bgr = image_rgb[:, :, ::-1].copy()
        started = time.perf_counter()
        with self._lock:
            try:
                results = self._model.predict(
                    source=image_bgr,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self._device,
                    imgsz=self.image_size,
                    verbose=False,
                )
            except Exception as exc:
                raise InferenceError(f"falha durante a inferencia: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not results:
            raise InferenceError("Ultralytics nao retornou resultado para a imagem")

        result = results[0]
        detections = detections_from_result(result, image_width, image_height)
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1].copy()
        speed = getattr(result, "speed", None)
        if isinstance(speed, dict) and isinstance(speed.get("inference"), (int, float)):
            inference_ms = float(speed["inference"])
        else:
            inference_ms = elapsed_ms
        return PredictionResult(
            detections=detections,
            annotated_image=annotated_rgb,
            inference_ms=inference_ms,
            model_version=self.model_version,
            image_width=image_width,
            image_height=image_height,
            image_size=self.image_size,
        )


def detections_from_result(
    result: Any, image_width: int, image_height: int
) -> list[Any]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _tensor_list(boxes.xyxy)
    confidences = _tensor_list(boxes.conf)
    classes = _tensor_list(boxes.cls)
    ids_value = getattr(boxes, "id", None)
    track_ids = _tensor_list(ids_value) if ids_value is not None else [None] * len(xyxy)

    detections: list[Any] = []
    for coordinates, confidence, class_value, track_value in zip(
        xyxy, confidences, classes, track_ids, strict=True
    ):
        class_id = int(class_value)
        if class_id not in CLASS_NAMES:
            raise ClassMapMismatchError(
                f"resultado retornou class_id inesperado: {class_id}"
            )
        x_min = min(max(float(coordinates[0]), 0.0), float(image_width))
        y_min = min(max(float(coordinates[1]), 0.0), float(image_height))
        x_max = min(max(float(coordinates[2]), 0.0), float(image_width))
        y_max = min(max(float(coordinates[3]), 0.0), float(image_height))
        if x_max <= x_min or y_max <= y_min:
            continue
        track_id = int(track_value) if track_value is not None else None
        detections.append(
            _core_detection(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=min(max(float(confidence), 0.0), 1.0),
                coordinates=(x_min, y_min, x_max, y_max),
                track_id=track_id,
            )
        )
    return detections


# Preferred integration spelling plus an all-caps compatibility alias.
YOLOPredictor = YoloPredictor

__all__ = [
    "ClassMapMismatchError",
    "InferenceError",
    "ModelUnavailableError",
    "PredictionResult",
    "YOLOPredictor",
    "YoloPredictor",
    "detections_from_result",
]
