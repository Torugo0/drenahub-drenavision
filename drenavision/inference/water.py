"""Local pretrained ADE20K water segmentation with explicit, bounded processing.

This complements object detection: water is a surface rather than a discrete
instance. Components are neural mask regions, not distinct puddles, and their
confidence is mean water-family posterior, not a calibrated risk probability.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drenavision.core.schemas import Detection

from .device import describe_device, select_device
from .errors import InferenceError, ModelUnavailableError

WATER_LABEL_NAMES = frozenset(
    {"water", "river", "sea", "lake", "swimming pool", "waterfall"}
)
POLICY_VERSION = "ade-water-mask-v1"
SUPPORTED_REPOSITORIES = frozenset(
    {
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
    }
)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@dataclass(slots=True)
class WaterPrediction:
    detections: list[Detection]
    mask: np.ndarray
    inference_ms: float
    metadata: dict[str, Any]


def _regions_from_labels(
    labels: np.ndarray,
    water_probability: np.ndarray,
    water_ids: list[int],
    pixel_confidence: float,
    min_area_fraction: float,
) -> tuple[list[Detection], np.ndarray, list[dict[str, Any]]]:
    """Filter original-resolution neural pixels without filling or dilation."""
    import cv2

    if labels.ndim != 2 or labels.shape != water_probability.shape:
        raise ValueError("water labels/probabilities must have the same H x W shape")
    height, width = labels.shape
    pixels = np.isin(labels, water_ids) & (water_probability >= pixel_confidence)
    count, components, statistics, _ = cv2.connectedComponentsWithStats(
        pixels.astype(np.uint8), connectivity=8
    )
    minimum_pixels = max(1, math.ceil(height * width * min_area_fraction))
    retained = np.zeros((height, width), dtype=np.uint8)
    detections, regions = [], []
    for component_id in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in statistics[component_id]
        )
        if area < minimum_pixels:
            continue
        selected = components == component_id
        retained[selected] = 255
        confidence = float(np.clip(water_probability[selected].mean(), 0, 1))
        detections.append(
            Detection(
                class_id=1,
                class_name="agua_acumulada",
                confidence=confidence,
                bounding_box=(x, y, x + component_width, y + component_height),
            )
        )
        regions.append(
            {
                "bounding_box": [x, y, x + component_width, y + component_height],
                "mask_pixels": area,
                "mask_fraction": area / (height * width),
                "mean_water_probability": confidence,
            }
        )
    return detections, retained, regions


def regions_from_maps(
    water_probability: np.ndarray,
    semantic_water_label_mask: np.ndarray,
    pixel_confidence: float = 0.5,
    min_area_fraction: float = 0.001,
) -> tuple[list[Detection], np.ndarray, list[dict[str, Any]]]:
    """Apply a policy to cached posterior/argmax maps without another forward."""
    if (
        water_probability.ndim != 2
        or water_probability.shape != semantic_water_label_mask.shape
        or min(water_probability.shape) < 1
        or not np.isfinite(water_probability).all()
        or not ((water_probability >= 0) & (water_probability <= 1)).all()
    ):
        raise ValueError("water posterior and semantic mask must be finite H x W maps")
    if not math.isfinite(pixel_confidence) or not 0 <= pixel_confidence <= 1:
        raise ValueError("pixel_confidence must be finite and within [0, 1]")
    if not math.isfinite(min_area_fraction) or not 0 <= min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be finite and within [0, 1]")
    return _regions_from_labels(
        semantic_water_label_mask.astype(np.uint8),
        water_probability,
        [1],
        pixel_confidence,
        min_area_fraction,
    )


class SemanticWaterPredictor:
    """Lazy, offline-only SegFormer B0/B1 inference; never downloads weights."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        pixel_confidence: float = 0.5,
        min_area_fraction: float = 0.001,
        processing_size: int = 512,
    ) -> None:
        if not math.isfinite(pixel_confidence) or not 0 <= pixel_confidence <= 1:
            raise ValueError("pixel_confidence must be finite and within [0, 1]")
        if not math.isfinite(min_area_fraction) or not 0 <= min_area_fraction <= 1:
            raise ValueError("min_area_fraction must be finite and within [0, 1]")
        if (
            isinstance(processing_size, bool)
            or not isinstance(processing_size, int)
            or not 128 <= processing_size <= 512
            or processing_size % 32
        ):
            raise ValueError(
                "processing_size must be a multiple of 32 within [128, 512]"
            )
        self.model_dir = Path(model_dir)
        self.requested_device = device
        self._pixel_confidence = float(pixel_confidence)
        self._min_area_fraction = float(min_area_fraction)
        self._processing_size = processing_size
        self._water_ids: list[int] = []
        self._metadata: dict[str, Any] | None = None
        self._version: str | None = None
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: str | int | None = None
        self._detail = "segmentador de água ainda não carregado"
        self._lock = threading.RLock()

    @property
    def pixel_confidence(self) -> float:
        return self._pixel_confidence

    @property
    def min_area_fraction(self) -> float:
        return self._min_area_fraction

    @property
    def processing_size(self) -> int:
        return self._processing_size

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:
            self._detail = str(exc)
            return False

    @property
    def model_version(self) -> str:
        self._prepare_metadata()
        assert self._version is not None
        return self._version

    @property
    def provenance(self) -> dict[str, Any]:
        self._prepare_metadata()
        assert self._metadata is not None
        return copy.deepcopy(self._metadata)

    def _prepare_metadata(self) -> None:
        if self._metadata is not None:
            return
        with self._lock:
            if self._metadata is not None:
                return
            try:
                required = (
                    "model.safetensors",
                    "config.json",
                    "preprocessor_config.json",
                    "download_manifest.json",
                    "LICENSE-NVIDIA-SegFormer.txt",
                )
                for name in required:
                    if not (self.model_dir / name).is_file():
                        raise ValueError(
                            f"arquivo local do segmentador ausente: {name}"
                        )
                config = json.loads(
                    (self.model_dir / "config.json").read_text(encoding="utf-8")
                )
                if config.get("model_type") != "segformer":
                    raise ValueError(
                        "o segmentador de água exige uma configuração SegFormer"
                    )
                labels = {
                    int(key): str(value).strip().lower()
                    for key, value in config["id2label"].items()
                }
                if set(labels) != set(range(150)):
                    raise ValueError(
                        "o segmentador exige o mapa completo de 150 classes ADE20K"
                    )
                self._water_ids = sorted(
                    key for key, value in labels.items() if value in WATER_LABEL_NAMES
                )
                if (
                    len(self._water_ids) != len(WATER_LABEL_NAMES)
                    or {labels[key] for key in self._water_ids} != WATER_LABEL_NAMES
                ):
                    raise ValueError(
                        "classes de água ADE20K ausentes ou duplicadas; não é permitido presumir IDs"
                    )
                source = json.loads(
                    (self.model_dir / "download_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if source.get(
                    "repository"
                ) not in SUPPORTED_REPOSITORIES or not re.fullmatch(
                    r"[0-9a-f]{40}", str(source.get("revision", ""))
                ):
                    raise ValueError(
                        "manifesto exige repositório oficial B0/B1 e revisão fixa"
                    )
                expected = {item["name"]: item["sha256"] for item in source["files"]}
                hashes = {name: _digest(self.model_dir / name) for name in required}
                for name in required[:3]:
                    if hashes[name] != expected.get(name):
                        raise ValueError(
                            f"hash do segmentador não confere com o manifesto: {name}"
                        )
                policy = {
                    "version": POLICY_VERSION,
                    "processing_size": self.processing_size,
                    "pixel_confidence": self.pixel_confidence,
                    "min_area_fraction": self.min_area_fraction,
                    "water_labels": {str(key): labels[key] for key in self._water_ids},
                    "postprocessing": "bilinear logits at bounded processing_size; nearest labels and bilinear water-family posterior to original size; pixel confidence gate; 8-connected components; actual pixel-area gate; no fill/dilation",
                }
                metadata = {
                    "repository": source["repository"],
                    "revision": source["revision"],
                    "license": source.get(
                        "license", "NVIDIA non-commercial research/evaluation license"
                    ),
                    "license_url": source.get("license_url"),
                    "file_sha256": hashes,
                    "policy": policy,
                }
                profile = hashlib.sha256(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                metadata["profile_sha256"] = profile
                metadata["model_version"] = f"water-{profile[:12]}"
                self._version = metadata["model_version"]
                self._metadata = metadata
            except Exception as exc:
                raise ModelUnavailableError(
                    f"segmentador de água local inválido: {exc}"
                ) from exc

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._prepare_metadata()
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
                device = select_device(self.requested_device)
                processor = transformers.SegformerImageProcessor.from_pretrained(
                    self.model_dir, local_files_only=True, trust_remote_code=False
                )
                model = transformers.SegformerForSemanticSegmentation.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                )
                actual = {
                    int(key): str(value).strip().lower()
                    for key, value in model.config.id2label.items()
                }
                if {actual.get(key) for key in self._water_ids} != WATER_LABEL_NAMES:
                    raise ValueError(
                        "o mapa de classes carregado diverge da configuração verificada"
                    )
                model.to(device)
                model.eval()
                self._torch, self._device = torch, device
                self._processor, self._model = processor, model
                self._detail = "ready"
            except Exception as exc:
                raise ModelUnavailableError(
                    f"não foi possível carregar o segmentador de água: {exc}"
                ) from exc

    def infer_maps(
        self,
        image_rgb: np.ndarray,
        *,
        original_resolution: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return water posterior and water-family argmax mask before filtering.

        Set original_resolution=False to cache bounded processing_size maps.
        To reproduce original resolution exactly: bilinear-resize posterior and
        nearest-resize the boolean argmax mask before regions_from_maps().
        No confidence or area gate is applied here; one call is one forward.
        """
        if (
            not isinstance(image_rgb, np.ndarray)
            or image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or min(image_rgb.shape[:2]) < 1
        ):
            raise ValueError("image_rgb deve ser ndarray RGB uint8 H x W x 3 não vazio")
        self._ensure_loaded()
        import cv2

        height, width = image_rgb.shape[:2]
        with self._lock:
            try:
                inputs = self._processor(
                    images=image_rgb,
                    return_tensors="pt",
                    size={
                        "height": self.processing_size,
                        "width": self.processing_size,
                    },
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with self._torch.inference_mode():
                    logits = self._model(**inputs).logits
                    # Never allocate 150-channel logits at original photo size.
                    bounded = self._torch.nn.functional.interpolate(
                        logits,
                        size=(self.processing_size, self.processing_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    labels = bounded.argmax(dim=1)[0].cpu().numpy().astype(np.int16)
                    # logsumexp avoids a second full 150-channel softmax buffer.
                    normalizer = bounded.logsumexp(dim=1)
                    water_log_probability = (
                        bounded[:, self._water_ids].logsumexp(dim=1) - normalizer
                    )
                    probability = water_log_probability.exp().clamp(0, 1)[0].cpu().numpy()
                semantic_mask = np.isin(labels, self._water_ids)
                if original_resolution:
                    semantic_mask = cv2.resize(
                        semantic_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    probability = cv2.resize(
                        probability, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                probability = probability.astype(np.float32, copy=False)
                if not np.isfinite(probability).all():
                    raise ValueError("segmentador retornou posterior não finito")
                return probability, semantic_mask
            except Exception as exc:
                raise InferenceError(f"falha na segmentação de água: {exc}") from exc

    def predict(self, image_rgb: np.ndarray) -> WaterPrediction:
        self._ensure_loaded()
        started = time.perf_counter()
        probability, semantic_mask = self.infer_maps(image_rgb)
        height, width = image_rgb.shape[:2]
        detections, mask, regions = regions_from_maps(
            probability, semantic_mask, self.pixel_confidence, self.min_area_fraction
        )
        return WaterPrediction(
            detections=detections,
            mask=mask,
            inference_ms=(time.perf_counter() - started) * 1000,
            metadata={
                "model_version": self.model_version,
                "profile_sha256": self.provenance["profile_sha256"],
                "device": describe_device(self._device),
                "image_width": width,
                "image_height": height,
                "processing_size": self.processing_size,
                "mask_pixels": int(np.count_nonzero(mask)),
                "mask_fraction": float(np.count_nonzero(mask) / (width * height)),
                "minimum_component_pixels": max(
                    1, math.ceil(width * height * self.min_area_fraction)
                ),
                "confidence_definition": "mean water-family posterior over retained predicted pixels; not calibrated risk probability",
                "counting_note": "connected predicted water regions, not distinct puddles or water volume",
                "regions": regions,
            },
        )


def draw_mask_on_rgb(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    color: tuple[int, int, int] = (14, 165, 233),
) -> np.ndarray:
    """Tint only predicted mask pixels; preserve the input and all mask holes."""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be RGB uint8")
    if mask.ndim != 2 or mask.shape != image_rgb.shape[:2]:
        raise ValueError("water mask shape must match the original image")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and within [0, 1]")
    if len(color) != 3 or any(not 0 <= channel <= 255 for channel in color):
        raise ValueError("mask color must contain three RGB channels within [0, 255]")
    output = image_rgb.copy()
    selected = mask > 0
    output[selected] = np.clip(
        image_rgb[selected] * (1 - alpha) + np.asarray(color) * alpha, 0, 255
    ).astype(np.uint8)
    return output


__all__ = [
    "SemanticWaterPredictor",
    "WaterPrediction",
    "draw_mask_on_rgb",
    "regions_from_maps",
]
