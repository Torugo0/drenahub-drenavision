"""Validated YAML configuration for reproducible train/evaluate runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML nao esta instalado") from exc
    if not path.is_file():
        raise FileNotFoundError(f"configuracao ausente: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuracao YAML invalida: {path}")
    return payload


def _merged(path: Path, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _yaml_mapping(path)
    if overrides:
        payload.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    return payload


@dataclass(frozen=True, slots=True)
class TrainConfig:
    model: str = "yolo26n.pt"
    data: Path = Path("datasets/processed/drenavision/dataset.yaml")
    project: Path = Path("artifacts/training")
    name: str = "yolo26n_drenavision"
    published_checkpoint: Path = Path("artifacts/models/best.pt")
    resume: Path | None = None
    epochs: int = 100
    imgsz: int = 640
    batch: int | float = 8
    patience: int = 20
    workers: int = 4
    seed: int = 42
    device: str = "auto"
    optimizer: str = "auto"
    deterministic: bool = True
    amp: bool = True
    cache: bool | str = False
    pretrained: bool = True
    plots: bool = True
    close_mosaic: int = 10

    @classmethod
    def load(
        cls, path: Path, overrides: Mapping[str, Any] | None = None
    ) -> TrainConfig:
        values = _merged(path, overrides)
        for key in ("data", "project", "published_checkpoint", "resume"):
            if key in values and values[key] is not None:
                values[key] = Path(values[key])
        config = cls(**values)
        if config.model != "yolo26n.pt" and not Path(config.model).is_file():
            raise ValueError("model deve ser yolo26n.pt ou um caminho local existente")
        if config.epochs < 1 or config.imgsz < 32 or config.workers < 0:
            raise ValueError("epochs/imgsz/workers invalidos")
        if config.seed != 42:
            raise ValueError("o protocolo reproduzivel do projeto exige seed 42")
        if config.resume is not None and not config.resume.is_file():
            raise ValueError(f"checkpoint de resume ausente: {config.resume}")
        return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluateConfig:
    checkpoint: Path = Path("artifacts/models/best.pt")
    data: Path = Path("datasets/processed/drenavision/dataset.yaml")
    project: Path = Path("artifacts/evaluation")
    name: str = "test_yolo26n_drenavision"
    imgsz: int = 640
    batch: int | float = 8
    confidence: float = 0.001
    iou: float = 0.60
    match_iou: float = 0.50
    example_confidence: float = 0.35
    workers: int = 4
    device: str = "auto"
    plots: bool = True
    save_json: bool = True
    error_examples_per_type: int = 6
    max_error_images: int = 0

    @classmethod
    def load(
        cls, path: Path, overrides: Mapping[str, Any] | None = None
    ) -> EvaluateConfig:
        values = _merged(path, overrides)
        for key in ("checkpoint", "data", "project"):
            if key in values:
                values[key] = Path(values[key])
        config = cls(**values)
        if (
            not 0 <= config.confidence <= 1
            or not 0 <= config.example_confidence <= 1
            or not 0 <= config.iou <= 1
        ):
            raise ValueError("confidence/iou devem estar entre 0 e 1")
        if not 0 < config.match_iou <= 1:
            raise ValueError("match_iou deve estar em (0, 1]")
        if config.error_examples_per_type < 0 or config.max_error_images < 0:
            raise ValueError("limites de exemplos devem ser nao negativos")
        return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
