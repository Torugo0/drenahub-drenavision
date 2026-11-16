"""Held-out test evaluation and evidence export for the custom detector."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.data.constants import CLASS_NAMES

from .config import EvaluateConfig
from .device import describe_device, select_device
from .error_analysis import collect_error_examples
from .utils import (
    dataset_fingerprint,
    environment_manifest,
    json_safe,
    preflight_dataset,
    safe_artifact_path,
    sha256_file,
    strict_dataset_validation,
    utc_now_iso,
    validate_class_map,
    write_json,
)


def _sequence(value: Any, length: int) -> list[float | None]:
    if value is None:
        return [None] * length
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        values = list(value)
    except TypeError:
        return [None] * length
    normalized: list[float | None] = []
    for item in values[:length]:
        try:
            normalized.append(float(item))
        except (TypeError, ValueError):
            normalized.append(None)
    return normalized + [None] * (length - len(normalized))


def extract_detection_metrics(metrics: Any) -> dict[str, Any]:
    """Extract the stable metric contract from an Ultralytics DetMetrics object."""

    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics nao retornou metricas de deteccao")
    class_count = len(CLASS_NAMES)
    class_indexes_raw = getattr(box, "ap_class_index", None)
    if class_indexes_raw is None:
        class_indexes = list(range(class_count))
    else:
        if hasattr(class_indexes_raw, "tolist"):
            class_indexes_raw = class_indexes_raw.tolist()
        class_indexes = [int(value) for value in class_indexes_raw]

    metric_length = len(class_indexes)
    raw_metrics = {
        "precision": _sequence(getattr(box, "p", None), metric_length),
        "recall": _sequence(getattr(box, "r", None), metric_length),
        "map50": _sequence(getattr(box, "ap50", None), metric_length),
    }
    full_maps = _sequence(getattr(box, "maps", None), class_count)
    values_by_class: dict[int, dict[str, float | None]] = {
        class_id: {
            **{key: values[position] for key, values in raw_metrics.items()},
            "map50_95": full_maps[class_id],
        }
        for position, class_id in enumerate(class_indexes)
        if class_id in CLASS_NAMES
    }
    per_class = {
        CLASS_NAMES[index]: {
            "class_id": index,
            **values_by_class.get(
                index,
                {
                    "precision": None,
                    "recall": None,
                    "map50": None,
                    "map50_95": None,
                },
            ),
        }
        for index in CLASS_NAMES
    }
    return {
        "aggregate": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": json_safe(getattr(metrics, "speed", {})),
    }


def run_evaluation(
    config_path: str | Path = "configs/evaluate.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate only the held-out test split and save metrics plus plot inventory."""

    config = EvaluateConfig.load(Path(config_path), overrides)
    if not config.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint customizado ausente: {config.checkpoint}; execute o treino primeiro"
        )
    preflight = preflight_dataset(config.data, ("train", "val", "test"))
    validation = strict_dataset_validation(config.data)
    fingerprint = dataset_fingerprint(preflight)
    device = select_device(config.device)
    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLO(str(config.checkpoint))
    validate_class_map(model.names)

    started = utc_now_iso()
    val_args = {
        "data": str(config.data.resolve()),
        "split": "test",
        "imgsz": config.imgsz,
        "batch": config.batch,
        "conf": config.confidence,
        "iou": config.iou,
        "workers": config.workers,
        "device": device,
        "plots": config.plots,
        "save_json": config.save_json,
        "project": str(config.project.resolve()),
        "name": config.name,
        "exist_ok": False,
        "verbose": True,
    }
    results = model.val(**val_args)
    metrics_payload = extract_detection_metrics(results)
    result_save_dir = getattr(results, "save_dir", None)
    save_dir = (
        Path(result_save_dir) if result_save_dir else config.project / config.name
    )
    if not save_dir.is_dir():
        validator = getattr(model, "validator", None)
        validator_save_dir = getattr(validator, "save_dir", None)
        save_dir = (
            Path(validator_save_dir)
            if validator_save_dir
            else config.project / config.name
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_files = sorted(
        path.name
        for path in save_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    error_analysis = collect_error_examples(
        model,
        preflight["images"]["test"],
        save_dir,
        device=device,
        confidence=config.example_confidence,
        iou=config.iou,
        match_iou=config.match_iou,
        per_type=config.error_examples_per_type,
        max_images=config.max_error_images,
    )
    manifest = {
        "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": utc_now_iso(),
        "protocol": "held-out test split; Ultralytics detection metrics",
        "checkpoint": safe_artifact_path(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "device": describe_device(device),
        "dataset": fingerprint,
        "dataset_validation": validation,
        "arguments": {
            **json_safe(val_args),
            "data": safe_artifact_path(val_args["data"]),
            "project": safe_artifact_path(val_args["project"]),
        },
        "metrics": metrics_payload,
        "plots": plot_files,
        "error_analysis": error_analysis,
        "environment": environment_manifest(),
    }
    write_json(save_dir / "metrics.json", manifest)
    return manifest
