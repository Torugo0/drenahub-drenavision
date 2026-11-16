"""Environment-backed runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    model_path: Path = Path("artifacts/models/best.pt")
    database_path: Path = Path("artifacts/events/drenavision.db")
    evidence_dir: Path = Path("artifacts/events/evidence")
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: str = "auto"
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
        if expanded.is_absolute():
            return expanded.resolve()
        return (PROJECT_ROOT / expanded).resolve()

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("device cannot be empty")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
