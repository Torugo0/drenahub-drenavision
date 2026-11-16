"""Stable API and persistence contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import CLASS_NAMES, DISCLAIMER

Coordinate = Annotated[float, Field(ge=0)]


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0, le=2)
    class_name: Literal["obstrucao_visivel", "agua_acumulada", "animal_visivel"]
    confidence: float = Field(ge=0, le=1)
    bounding_box: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    bounding_box_format: Literal["xyxy_pixels"] = "xyxy_pixels"
    track_id: int | None = Field(default=None, ge=0)
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_detection_contract(self) -> Self:
        if CLASS_NAMES[self.class_id] != self.class_name:
            raise ValueError("class_id e class_name sao incompatíveis")
        x1, y1, x2, y2 = self.bounding_box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bounding_box deve ter área positiva")
        if (self.frame_index is None) != (self.timestamp_seconds is None):
            raise ValueError("frame_index e timestamp_seconds devem aparecer juntos")
        return self


class PredictionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    point_id: str | None = None
    timestamp: datetime
    input_type: Literal["image", "video"]
    model_version: str
    detections: list[Detection]
    inspection_recommended: bool
    status: Literal["sem_deteccao", "revisao_recomendada", "revisado"]
    evidence_uri: str | None = None
    inference_ms: float = Field(ge=0)
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def validate_event_contract(self) -> Self:
        has_detections = bool(self.detections)
        if self.inspection_recommended != has_detections:
            raise ValueError("inspection_recommended deve refletir detections")
        if has_detections and self.status == "sem_deteccao":
            raise ValueError("evento com deteccao nao pode ter status sem_deteccao")
        if not has_detections and self.status == "revisao_recomendada":
            raise ValueError("evento vazio nao pode recomendar revisao")
        return self


class EventSummary(BaseModel):
    event_id: str
    point_id: str | None = None
    timestamp: datetime
    input_type: Literal["image", "video"]
    model_version: str
    detection_count: int = Field(ge=0)
    detected_classes: list[str]
    max_confidence: float | None = Field(default=None, ge=0, le=1)
    inspection_recommended: bool
    status: str
    evidence_uri: str | None = None


class EventListResponse(BaseModel):
    items: list[EventSummary]
    total: int = Field(ge=0)


class ClassDefinition(BaseModel):
    class_id: int
    name: str
    label_pt_br: str


class ClassesResponse(BaseModel):
    classes: list[ClassDefinition]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_path: str
    database_ready: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str
