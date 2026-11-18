"""Offline, prompt-fixed Grounding DINO objects with auditable postprocessing."""

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

from drenavision.core.constants import CLASS_NAMES
from drenavision.core.schemas import Detection

from . import object_artifacts
from .device import select_device
from .errors import InferenceError, ModelUnavailableError

OBJECT_PROMPT = (
    "a rat. a mouse. a rodent. litter. plastic waste. discarded bottle. "
    "discarded plastic bag. discarded cup. discarded can."
)
POLICY_VERSION = "grounded-objects-v1"
_ANIMAL = re.compile(r"\b(?:rats?|mice|mouse|rodents?)\b", re.IGNORECASE)
_RESIDUE = re.compile(r"\b(?:litter|waste|bottles?|bags?|cups?|cans?)\b", re.IGNORECASE)


@dataclass(slots=True)
class ObjectPrediction:
    detections: list[Detection]
    raw_candidates: list[dict[str, Any]]
    inference_ms: float


def classify_phrase(phrase: str) -> int | None:
    """Map fixed-prompt terms; mixed animal/residue and unknown phrases fail closed."""
    animal, residue = bool(_ANIMAL.search(phrase)), bool(_RESIDUE.search(phrase))
    if animal == residue:
        return None
    return 2 if animal else 0


def _valid_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} deve ser finito e estar entre 0 e 1")
    return float(value)


def _box_iou(first: tuple, second: tuple) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def postprocess_candidates(
    scores: list[float],
    boxes: list[list[float]],
    phrases: list[str],
    *,
    image_width: int,
    image_height: int,
    residue_confidence: float = 0.35,
    animal_confidence: float = 0.35,
    iou_threshold: float = 0.5,
) -> tuple[list[Detection], list[dict[str, Any]]]:
    """Clip only to image bounds, threshold by family and suppress same-class IoU.

    Raw records retain every rescaled neural box before clipping/threshold/NMS,
    including unmapped or mixed-family phrases. No filename, hash, frame index,
    hand-drawn rectangle or case-specific region participates in this policy.
    """
    residue_confidence = _valid_threshold(residue_confidence, "residue_confidence")
    animal_confidence = _valid_threshold(animal_confidence, "animal_confidence")
    iou_threshold = _valid_threshold(iou_threshold, "iou_threshold")
    if image_width < 1 or image_height < 1:
        raise ValueError("dimensões da imagem devem ser positivas")
    if not len(scores) == len(boxes) == len(phrases):
        raise ValueError("scores, boxes e phrases devem ter o mesmo tamanho")
    raw, eligible = [], []
    for index, (score, box, phrase) in enumerate(
        zip(scores, boxes, phrases, strict=True)
    ):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("o detector retornou confiança inválida")
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("o detector retornou caixa inválida")
        coordinates = tuple(float(v) for v in box)
        class_id = classify_phrase(str(phrase))
        clipped = (
            max(0.0, min(float(image_width), coordinates[0])),
            max(0.0, min(float(image_height), coordinates[1])),
            max(0.0, min(float(image_width), coordinates[2])),
            max(0.0, min(float(image_height), coordinates[3])),
        )
        raw.append(
            {
                "candidate_index": index,
                "phrase": str(phrase),
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id is not None else None,
                "confidence": score,
                "bounding_box": list(coordinates),
                "clipped_bounding_box": list(clipped),
            }
        )
        if class_id is None or clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        threshold = animal_confidence if class_id == 2 else residue_confidence
        if score < threshold:
            continue
        eligible.append((index, class_id, score, clipped))
    kept = []
    for candidate in sorted(eligible, key=lambda item: (-item[2], item[0])):
        if any(
            candidate[1] == retained[1]
            and _box_iou(candidate[3], retained[3]) > iou_threshold
            for retained in kept
        ):
            continue
        kept.append(candidate)
    detections = [
        Detection(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            confidence=score,
            bounding_box=box,
        )
        for _, class_id, score, box in kept
    ]
    return detections, raw


def _tensor_list(value: Any) -> list:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class GroundedObjectPredictor:
    """Lazy and thread-safe, with no model downloads or executable model code."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        residue_confidence: float = 0.35,
        animal_confidence: float = 0.35,
        text_confidence: float = 0.25,
        shortest_edge: int = 800,
        longest_edge: int = 1333,
        iou_threshold: float = 0.5,
        verifier_model_path: str | Path | None = None,
        residue_min_margin: float = 0.0,
        animal_min_margin: float = 0.0,
    ) -> None:
        self._residue_confidence = _valid_threshold(
            residue_confidence, "residue_confidence"
        )
        self._animal_confidence = _valid_threshold(
            animal_confidence, "animal_confidence"
        )
        self._text_confidence = _valid_threshold(text_confidence, "text_confidence")
        self._iou_threshold = _valid_threshold(iou_threshold, "iou_threshold")
        if any(not math.isfinite(v) or not -2 <= v <= 2 for v in (residue_min_margin, animal_min_margin)):
            raise ValueError("verification margins must be finite in [-2, 2]")
        self._verification_margins = {0: residue_min_margin, 2: animal_min_margin}
        self._verifier = None
        if verifier_model_path is not None:
            from .object_verifier import CropObjectVerifier

            self._verifier = CropObjectVerifier(verifier_model_path, device=device)
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (shortest_edge, longest_edge)
            )
            or not 128 <= shortest_edge <= longest_edge <= 2048
        ):
            raise ValueError(
                "arestas de processamento devem ser inteiros entre 128 e 2048, em ordem"
            )
        self._shortest_edge, self._longest_edge = shortest_edge, longest_edge
        self.model_dir = Path(model_dir)
        self.requested_device = device
        self._lock = threading.RLock()
        self._metadata: dict[str, Any] | None = None
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device: str | int | None = None
        self._detail = "detector de objetos ainda não carregado"

    @property
    def residue_confidence(self) -> float:
        return self._residue_confidence

    @property
    def animal_confidence(self) -> float:
        return self._animal_confidence

    @property
    def text_confidence(self) -> float:
        return self._text_confidence

    @property
    def detail(self) -> str:
        if self._verifier is not None and not self._verifier.is_ready:
            return self._verifier.detail
        return self._detail

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_loaded()
            return self._verifier is None or self._verifier.is_ready
        except Exception as exc:
            self._detail = str(exc)
            return False

    @property
    def model_version(self) -> str:
        self._prepare_metadata()
        assert self._metadata is not None
        return self._metadata["model_version"]

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
                hashes = object_artifacts.verify_local_model(self.model_dir)
                config = json.loads(
                    (self.model_dir / "config.json").read_text(encoding="utf-8")
                )
                if config.get("model_type") != "grounding-dino" or config.get(
                    "auto_map"
                ):
                    raise ValueError(
                        "configuração exige Grounding DINO embutido, sem auto_map remoto"
                    )
                metadata = {
                    "repository": object_artifacts.REPOSITORY,
                    "revision": object_artifacts.REVISION,
                    "license": "Apache-2.0",
                    "license_url": object_artifacts.LICENSE_URL,
                    "file_sha256": hashes,
                    "policy": {
                        "version": POLICY_VERSION,
                        "prompt": OBJECT_PROMPT,
                        "residue_confidence": self.residue_confidence,
                        "animal_confidence": self.animal_confidence,
                        "text_confidence": self.text_confidence,
                        "shortest_edge": self._shortest_edge,
                        "longest_edge": self._longest_edge,
                        "class_nms_iou": self._iou_threshold,
                        "mapping": {
                            "animal_visivel": _ANIMAL.pattern,
                            "obstrucao_visivel": _RESIDUE.pattern,
                        },
                        "mixed_or_unknown_phrases": "discard",
                        "postprocessing": "rescale neural xyxy to original size; clip to bounds only; per-class confidence and NMS; no expansion or fabricated boxes",
                        "weights_format": "safetensors",
                        "local_files_only": True,
                        "trust_remote_code": False,
                        "disable_custom_kernels": True,
                    },
                }
                if self._verifier is not None:
                    metadata["verifier"] = {
                        "model": self._verifier.provenance,
                        "residue_min_margin": self._verification_margins[0],
                        "animal_min_margin": self._verification_margins[2],
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
                metadata["model_version"] = f"objects-{profile[:12]}"
                self._metadata = metadata
            except Exception as exc:
                raise ModelUnavailableError(
                    f"detector de objetos local inválido: {exc}"
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
                processor = transformers.AutoProcessor.from_pretrained(
                    self.model_dir, local_files_only=True, trust_remote_code=False
                )
                model = (
                    transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                        self.model_dir,
                        local_files_only=True,
                        trust_remote_code=False,
                        use_safetensors=True,
                        disable_custom_kernels=True,
                    )
                )
                if (
                    model.config.model_type != "grounding-dino"
                    or not model.config.disable_custom_kernels
                ):
                    raise ValueError(
                        "modelo carregado diverge da configuração offline verificada"
                    )
                model.to(device)
                model.eval()
                self._torch, self._device = torch, device
                self._processor, self._model = processor, model
                self._detail = "ready"
            except Exception as exc:
                raise ModelUnavailableError(
                    f"não foi possível carregar o detector de objetos: {exc}"
                ) from exc

    def predict(self, image_rgb: np.ndarray) -> ObjectPrediction:
        if (
            not isinstance(image_rgb, np.ndarray)
            or image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or min(image_rgb.shape[:2]) < 1
        ):
            raise ValueError("image_rgb deve ser ndarray RGB uint8 H x W x 3 não vazio")
        self._ensure_loaded()
        height, width = image_rgb.shape[:2]
        started = time.perf_counter()
        with self._lock:
            try:
                inputs = self._processor(
                    images=image_rgb,
                    text=OBJECT_PROMPT,
                    return_tensors="pt",
                    input_data_format="channels_last",
                    size={
                        "shortest_edge": self._shortest_edge,
                        "longest_edge": self._longest_edge,
                    },
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with self._torch.inference_mode():
                    output = self._model(**inputs)
                processed = self._processor.post_process_grounded_object_detection(
                    output,
                    input_ids=inputs["input_ids"],
                    threshold=min(self.residue_confidence, self.animal_confidence),
                    text_threshold=self.text_confidence,
                    target_sizes=[(height, width)],
                )[0]
                detections, raw = postprocess_candidates(
                    _tensor_list(processed["scores"]),
                    _tensor_list(processed["boxes"]),
                    list(processed["text_labels"]),
                    image_width=width,
                    image_height=height,
                    residue_confidence=self.residue_confidence,
                    animal_confidence=self.animal_confidence,
                    iou_threshold=self._iou_threshold,
                )
                if self._verifier is not None:
                    eligible_indices = [
                        index for index, item in enumerate(raw)
                        if item["class_id"] in {0, 2}
                        and item["confidence"] >= {
                            0: self.residue_confidence, 2: self.animal_confidence
                        }[item["class_id"]]
                        and item["clipped_bounding_box"][2] > item["clipped_bounding_box"][0]
                        and item["clipped_bounding_box"][3] > item["clipped_bounding_box"][1]
                    ]
                    checked = self._verifier.score(image_rgb, [raw[index] for index in eligible_indices])
                    retained = []
                    for index, item in zip(eligible_indices, checked, strict=True):
                        raw[index] = item
                        accepted = item["verification"]["margin"] >= self._verification_margins[item["class_id"]]
                        raw[index]["verification"]["accepted"] = accepted
                        if accepted:
                            retained.append(item)
                    detections, _ = postprocess_candidates(
                        [item["confidence"] for item in retained],
                        [item["bounding_box"] for item in retained],
                        [item["phrase"] for item in retained],
                        image_width=width, image_height=height,
                        residue_confidence=self.residue_confidence,
                        animal_confidence=self.animal_confidence,
                        iou_threshold=self._iou_threshold,
                    )
            except Exception as exc:
                raise InferenceError(f"falha na detecção de objetos: {exc}") from exc
        return ObjectPrediction(detections, raw, (time.perf_counter() - started) * 1000)
