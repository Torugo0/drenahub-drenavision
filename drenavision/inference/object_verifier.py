"""Offline SigLIP 2 crop similarities, without a decision threshold or risk score.

The full checkpoint is loaded on CPU. Fixed text features are encoded once in
FP32; only the vision tower is retained/moved to the requested inference device.
Crop context never modifies the original detector box or any candidate field.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import verifier_artifacts
from .device import select_device
from .errors import InferenceError, ModelUnavailableError

POLICY_VERSION = "siglip2-crop-cosine-v1"
FAMILY_PROMPTS = {
    2: {
        "positive": (
            "photograph of a rat",
            "photograph of a mouse",
            "photograph of a rodent",
            "photograph of a beaver",
            "photograph of a muskrat",
        ),
        "negative": (
            "photograph of plastic litter",
            "photograph of a plastic bottle",
            "photograph of a stone",
            "photograph of a leaf",
            "photograph of a drain pipe",
            "photograph of a shadow",
            "photograph of water",
            "photograph of discarded paper",
        ),
    },
    0: {
        "positive": (
            "photograph of discarded litter or garbage",
            "photograph of a discarded plastic bottle",
            "photograph of a discarded plastic bag",
            "photograph of a discarded can",
            "photograph of discarded food packaging",
            "photograph of discarded paper",
        ),
        "negative": (
            "photograph of a leaf",
            "photograph of a stone",
            "photograph of a branch",
            "photograph of water",
            "photograph of a concrete drain",
            "photograph of a rodent",
            "photograph of a clean pipe",
            "photograph of a plant",
        ),
    },
}
FIXED_TEXTS = tuple(
    dict.fromkeys(
        text
        for family in FAMILY_PROMPTS.values()
        for labels in family.values()
        for text in labels
    )
)


def crop_bounds(
    candidate: dict[str, Any], image_width: int, image_height: int, padding: float
) -> tuple[int, int, int, int]:
    """Add context only for the classifier input, retaining detector coordinates."""
    if image_width < 1 or image_height < 1:
        raise ValueError("dimensões da imagem devem ser positivas")
    if isinstance(padding, bool) or not math.isfinite(padding) or not 0 <= padding <= 1:
        raise ValueError("crop_padding deve estar entre 0 e 1")
    box = candidate.get("clipped_bounding_box")
    if box is None:
        box = candidate.get("bounding_box")
    if box is None or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("candidato exige caixa xyxy finita")
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(image_width, x1)), max(0.0, min(image_width, x2))
    y1, y2 = max(0.0, min(image_height, y1)), max(0.0, min(image_height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("candidato exige caixa de área positiva dentro da imagem")
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(image_width, math.ceil(x2 + dx)),
        min(image_height, math.ceil(y2 + dy)),
    )


def verification_from_similarities(
    similarities: np.ndarray,
    class_id: int,
    *,
    text_labels: tuple[str, ...] = FIXED_TEXTS,
    prompts: dict | None = None,
) -> dict[str, Any]:
    """Contrast maxima of fixed positive/negative cosine similarities only."""
    prompts = FAMILY_PROMPTS if prompts is None else prompts
    if class_id not in prompts:
        raise ValueError("o verificador aceita somente classes 0 e 2")
    vector = np.asarray(similarities, dtype=np.float32)
    if vector.shape != (len(text_labels),) or not np.isfinite(vector).all():
        raise ValueError("similaridades devem ser um vetor finito por texto fixo")
    indices = {text: index for index, text in enumerate(text_labels)}
    family = prompts[class_id]
    positive = max(family["positive"], key=lambda text: float(vector[indices[text]]))
    negative = max(family["negative"], key=lambda text: float(vector[indices[text]]))
    positive_similarity = float(vector[indices[positive]])
    negative_similarity = float(vector[indices[negative]])
    return {
        "margin": positive_similarity - negative_similarity,
        "positive_similarity": positive_similarity,
        "negative_similarity": negative_similarity,
        "best_positive": positive,
        "best_negative": negative,
    }


def _normalized_cpu_features(torch_module: Any, value: Any) -> Any:
    features = value.detach().to(device="cpu", dtype=torch_module.float32)
    if features.ndim != 2 or not bool(torch_module.isfinite(features).all()):
        raise ValueError("o verificador retornou features inválidas")
    norms = features.norm(p=2, dim=-1, keepdim=True)
    if bool((norms <= 0).any()):
        raise ValueError("o verificador retornou embedding sem norma")
    return (features / norms).contiguous()


class CropObjectVerifier:
    """Lazy, thread-safe, sealed local image/text crop verification."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        batch_size: int = 16,
        crop_padding: float = 0.10,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 128
        ):
            raise ValueError("batch_size deve ser inteiro entre 1 e 128")
        if (
            isinstance(crop_padding, bool)
            or not math.isfinite(crop_padding)
            or not 0 <= crop_padding <= 1
        ):
            raise ValueError("crop_padding deve ser finito e estar entre 0 e 1")
        self.model_dir = Path(model_dir)
        self.requested_device = device
        self._batch_size = batch_size
        self._crop_padding = float(crop_padding)
        self._prompts = copy.deepcopy(FAMILY_PROMPTS)
        self._text_labels = FIXED_TEXTS
        self._metadata: dict[str, Any] | None = None
        self._processor: Any = None
        self._vision_model: Any = None
        self._text_features: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._detail = "verificador de objetos ainda não carregado"
        self._lock = threading.RLock()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def crop_padding(self) -> float:
        return self._crop_padding

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
                hashes = verifier_artifacts.verify_local_model(self.model_dir)
                config = json.loads(
                    (self.model_dir / "config.json").read_text(encoding="utf-8")
                )
                if config.get("model_type") != "siglip" or config.get("auto_map"):
                    raise ValueError(
                        "o verificador exige SigLIP FixRes local, sem código remoto"
                    )
                metadata = {
                    "repository": verifier_artifacts.REPOSITORY,
                    "revision": verifier_artifacts.REVISION,
                    "license": "Apache-2.0",
                    "license_url": verifier_artifacts.LICENSE_URL,
                    "file_sha256": hashes,
                    "policy": {
                        "version": POLICY_VERSION,
                        "candidate_classes": [0, 2],
                        "family_prompts": self._prompts,
                        "fixed_texts": self._text_labels,
                        "text_padding": "max_length",
                        "max_text_length": 64,
                        "text_embedding_dtype": "float32",
                        "text_embedding_device": "cpu",
                        "text_encoder_execution": "once, full model CPU only, retain vision tower and normalized text features",
                        "vision_embedding_dtype": "float32",
                        "crop_padding_fraction_each_side": self.crop_padding,
                        "batch_size": self.batch_size,
                        "coordinate_policy": "context only for crop input; original candidate boxes and fields unchanged",
                        "score_definition": "maximum positive cosine minus maximum negative cosine; no sigmoid/softmax; no threshold or risk probability",
                        "trust_remote_code": False,
                        "local_files_only": True,
                        "weights_format": "safetensors",
                    },
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
                metadata["model_version"] = f"object-verifier-{profile[:12]}"
                self._metadata = metadata
            except Exception as exc:
                raise ModelUnavailableError(
                    f"verificador de objetos local inválido: {exc}"
                ) from exc

    def _ensure_loaded(self) -> None:
        if self._vision_model is not None:
            return
        with self._lock:
            if self._vision_model is not None:
                return
            self._prepare_metadata()
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
                processor = transformers.AutoProcessor.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=True,
                )
                model = transformers.AutoModel.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    dtype=torch.float32,
                )
                if model.config.model_type != "siglip":
                    raise ValueError(
                        "a arquitetura do verificador carregado diverge do selo"
                    )
                model.to(device="cpu", dtype=torch.float32)
                model.eval()
                inputs = processor(
                    text=list(self._text_labels),
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                text_inputs = {
                    key: value.to("cpu")
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask", "position_ids"}
                }
                with torch.inference_mode():
                    text_features = _normalized_cpu_features(
                        torch, model.get_text_features(**text_inputs)
                    )
                if text_features.shape[0] != len(self._text_labels):
                    raise ValueError("o número de embeddings diverge dos textos fixos")
                # SiglipModel.get_image_features returns this exact tower's pooler_output.
                # Release the full-model/text-tower reference before moving vision to CUDA.
                vision_model = model.vision_model
                del model, text_inputs, inputs
                device = select_device(self.requested_device)
                vision_model.to(device=device, dtype=torch.float32)
                vision_model.eval()
                self._processor, self._vision_model = processor, vision_model
                self._text_features, self._torch, self._device = (
                    text_features,
                    torch,
                    device,
                )
                self._detail = "ready"
            except Exception as exc:
                raise ModelUnavailableError(
                    f"não foi possível carregar o verificador de objetos: {exc}"
                ) from exc

    def score(self, image_rgb: np.ndarray, candidates: list[dict]) -> list[dict]:
        if (
            not isinstance(image_rgb, np.ndarray)
            or image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or min(image_rgb.shape[:2]) < 1
        ):
            raise ValueError("image_rgb deve ser ndarray RGB uint8 H x W x 3 não vazio")
        if not isinstance(candidates, list):
            raise ValueError("candidates deve ser uma lista de objetos")
        height, width = image_rgb.shape[:2]
        output, crops = [], []
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("class_id") not in (
                0,
                2,
            ):
                raise ValueError("cada candidato exige class_id 0 ou 2")
            bounds = crop_bounds(candidate, width, height, self.crop_padding)
            x1, y1, x2, y2 = bounds
            crops.append(image_rgb[y1:y2, x1:x2].copy())
            output.append(copy.deepcopy(candidate))
        if not output:
            return []
        self._ensure_loaded()
        with self._lock:
            try:
                for start in range(0, len(crops), self.batch_size):
                    inputs = self._processor(
                        images=crops[start : start + self.batch_size],
                        input_data_format="channels_last",
                        return_tensors="pt",
                    )
                    pixels = inputs["pixel_values"].to(
                        device=self._device, dtype=self._torch.float32
                    )
                    with self._torch.inference_mode():
                        image_features = self._vision_model(
                            pixel_values=pixels
                        ).pooler_output
                        image_features = _normalized_cpu_features(
                            self._torch, image_features
                        )
                        similarities = (image_features @ self._text_features.T).numpy()
                    for index, vector in enumerate(similarities):
                        item = output[start + index]
                        item["verification"] = verification_from_similarities(
                            vector,
                            item["class_id"],
                            text_labels=self._text_labels,
                            prompts=self._prompts,
                        )
            except Exception as exc:
                raise InferenceError(f"falha na verificação de objetos: {exc}") from exc
        return output
