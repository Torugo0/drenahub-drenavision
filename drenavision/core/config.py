"""Environment-backed runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_LOCAL_PATHS = {
    "artifacts/models/best.pt": "resultados/local/models/best.pt",
    "artifacts/events/drenavision.db": "resultados/local/events/drenavision.db",
    "artifacts/events/evidence": "resultados/local/events/evidence",
    "resultados/models/best.pt": "resultados/local/models/best.pt",
    "resultados/events/drenavision.db": "resultados/local/events/drenavision.db",
    "resultados/events/evidence": "resultados/local/events/evidence",
}


def validate_image_size(value: int | str) -> int:
    """Keep inference resizing explicit instead of Ultralytics rounding silently."""
    message = "image_size/imgsz deve ser inteiro múltiplo de 32, entre 320 e 2048"
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(message)
    try:
        size = int(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not 320 <= size <= 2048 or size % 32:
        raise ValueError(message)
    return size


class Settings(BaseSettings):
    """DrenaVision settings with conservative local defaults."""

    model_config = SettingsConfigDict(
        env_prefix="DRENAVISION_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DrenaVision"
    environment: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    model_path: Path = Path("resultados/local/models/best.pt")
    database_path: Path = Path("resultados/local/events/drenavision.db")
    evidence_dir: Path = Path("resultados/local/events/evidence")
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: str = "auto"
    image_size: int = 640
    water_model_path: Path | None = Path("resultados/local/models/water_segformer_b1_ade")
    water_pixel_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    water_min_area_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    water_mode: str = "augment"
    animal_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    video_water_confirmation_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    max_image_bytes: int = Field(default=12 * 1024 * 1024, ge=1024)
    max_video_bytes: int = Field(default=500 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=40_000_000, ge=1_000_000)
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ]
    )

    @field_validator("model_path", "database_path", "evidence_dir", mode="after")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        expanded = value.expanduser()
        resolved = (
            expanded if expanded.is_absolute() else PROJECT_ROOT / expanded
        ).resolve()
        # Accept old default paths after upgrading the repository. Preserve
        # existing files and custom locations, including external workspaces.
        if not resolved.exists():
            try:
                relative = resolved.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return resolved
            migrated = LEGACY_LOCAL_PATHS.get(relative)
            if migrated:
                return (PROJECT_ROOT / migrated).resolve()
        return resolved

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("device cannot be empty")
        return normalized

    @field_validator("image_size", mode="before")
    @classmethod
    def validate_inference_image_size(cls, value: int | str) -> int:
        return validate_image_size(value)

    @field_validator("water_model_path", mode="after")
    @classmethod
    def resolve_water_path(cls, value: Path | None) -> Path | None:
        return cls.resolve_project_path(value) if value is not None else None

    @field_validator("water_mode")
    @classmethod
    def validate_water_mode(cls, value: str) -> str:
        if value not in {"augment", "replace"}:
            raise ValueError("water_mode deve ser augment ou replace")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
