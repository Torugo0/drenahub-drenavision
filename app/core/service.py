"""Application service that turns predictions into auditable events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

import numpy as np

from .image_io import encode_jpeg
from .schemas import Detection, PredictionEvent
from .storage import EventRepository, EvidenceStore


class PredictionOutput(Protocol):
    detections: list[Detection]
    annotated_image: np.ndarray
    inference_ms: float
    model_version: str


class Predictor(Protocol):
    @property
    def is_ready(self) -> bool: ...

    def predict(self, image_rgb: np.ndarray) -> PredictionOutput: ...


class PredictionService:
    def __init__(
        self,
        predictor: Predictor,
        repository: EventRepository,
        evidence_store: EvidenceStore,
    ) -> None:
        self.predictor = predictor
        self.repository = repository
        self.evidence_store = evidence_store

    def predict_image(
        self, image_rgb: np.ndarray, *, point_id: str | None = None
    ) -> PredictionEvent:
        result = self.predictor.predict(image_rgb)
        event_id = str(uuid4())
        evidence_uri = self.evidence_store.write(
            event_id, encode_jpeg(result.annotated_image)
        )
        inspection_recommended = bool(result.detections)
        event = PredictionEvent(
            event_id=event_id,
            point_id=point_id,
            timestamp=datetime.now(UTC),
            input_type="image",
            model_version=result.model_version,
            detections=result.detections,
            inspection_recommended=inspection_recommended,
            status=(
                "revisao_recomendada" if inspection_recommended else "sem_deteccao"
            ),
            evidence_uri=evidence_uri,
            inference_ms=result.inference_ms,
        )
        try:
            self.repository.save(event)
        except Exception:
            self.evidence_store.delete(evidence_uri)
            raise
        return event
