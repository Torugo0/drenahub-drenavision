"""Ultralytics YOLO26n fine-tuning and safe resume with auditable evidence."""

from __future__ import annotations

import importlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import TrainConfig
from .device import describe_device, select_device
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


def _model_checkpoint_path(model: Any, configured: str) -> Path | None:
    for value in (getattr(model, "ckpt_path", None), configured):
        if value and Path(str(value)).is_file():
            return Path(str(value)).resolve()
    return None


def _checkpoint_epoch(model: Any) -> int:
    checkpoint = getattr(model, "ckpt", None)
    if not isinstance(checkpoint, dict):
        return -1
    try:
        return int(checkpoint.get("epoch", -1))
    except (TypeError, ValueError):
        return -1


def snapshot_interrupted_run(
    resume_checkpoint: Path, project: Path, *, completed_epoch: int
) -> Path:
    """Preserve a run before Ultralytics mutates it during resume."""

    resume_checkpoint = resume_checkpoint.resolve()
    if (
        resume_checkpoint.name.lower() != "last.pt"
        or resume_checkpoint.parent.name.lower() != "weights"
    ):
        raise ValueError("--resume deve apontar para weights/last.pt")
    run_dir = resume_checkpoint.parent.parent
    if not (run_dir / "results.csv").is_file():
        raise ValueError(f"run de resume nao possui results.csv: {run_dir.name}")
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    destination = (
        project.resolve()
        / "interrupted"
        / f"{run_dir.name}_after_epoch_{completed_epoch + 1}_{stamp}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, destination)
    write_json(
        destination / "interrupted_manifest.json",
        {
            "status": "interrupted",
            "recorded_at_utc": utc_now_iso(),
            "source_run": safe_artifact_path(run_dir),
            "last_completed_epoch": completed_epoch + 1,
            "resume_checkpoint_sha256": sha256_file(resume_checkpoint),
            "note": "Snapshot criado antes da retomada; nao e entrega concluida.",
        },
    )
    return destination


def _fresh_arguments(config: TrainConfig, device: str | int) -> dict[str, Any]:
    return {
        "data": str(config.data.resolve()),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "patience": config.patience,
        "workers": config.workers,
        "seed": config.seed,
        "device": device,
        "optimizer": config.optimizer,
        "deterministic": config.deterministic,
        "amp": config.amp,
        "cache": config.cache,
        "pretrained": config.pretrained,
        "plots": config.plots,
        "close_mosaic": config.close_mosaic,
        "project": str(config.project.resolve()),
        "name": config.name,
        "exist_ok": False,
        "verbose": True,
    }


def _safe_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    safe = dict(arguments)
    for key in (
        "data",
        "project",
        "model",
        "resume",
        "checkpoint",
        "published_checkpoint",
    ):
        value = safe.get(key)
        if isinstance(value, (str, Path)) and value not in {"", "True", "False"}:
            safe[key] = safe_artifact_path(value)
    return json_safe(safe)


def run_training(
    config_path: str | Path = "ferramentas/configs/train.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or resume fine-tuning; publish only after all checks pass."""

    config = TrainConfig.load(Path(config_path), overrides)
    preflight = preflight_dataset(config.data, ("train", "val", "test"))
    validation = strict_dataset_validation(config.data)
    fingerprint = dataset_fingerprint(preflight)
    device = select_device(config.device)
    ultralytics = importlib.import_module("ultralytics")
    started = utc_now_iso()

    base_checkpoint = (
        Path(config.model).resolve() if Path(config.model).is_file() else None
    )
    base_sha256 = sha256_file(base_checkpoint) if base_checkpoint else None
    resume_snapshot: Path | None = None
    resume_epoch = -1
    mode = "fresh"

    if config.resume is not None:
        mode = "resume"
        initial_checkpoint = config.resume.resolve()
        model = ultralytics.YOLO(str(initial_checkpoint))
        validate_class_map(model.names)
        resume_epoch = _checkpoint_epoch(model)
        if resume_epoch < 0:
            raise RuntimeError("last.pt nao informa uma epoca retomavel")
        checkpoint_args = getattr(model, "ckpt", {}).get("train_args", {})
        if int(checkpoint_args.get("epochs", -1)) != config.epochs:
            raise RuntimeError(
                "meta de epochs do last.pt diverge de ferramentas/configs/train.yaml"
            )
        resume_snapshot = snapshot_interrupted_run(
            initial_checkpoint, config.project, completed_epoch=resume_epoch
        )
        train_args: dict[str, Any] = {"resume": True, "device": device}
    else:
        model = ultralytics.YOLO(config.model)
        initial_checkpoint = _model_checkpoint_path(model, config.model)
        if initial_checkpoint is None:
            raise RuntimeError("pesos iniciais sem caminho verificavel para hash")
        train_args = _fresh_arguments(config, device)

    initial_sha256 = sha256_file(initial_checkpoint)
    try:
        results = model.train(**train_args)
    except Exception as exc:
        if resume_snapshot is not None:
            write_json(
                resume_snapshot / "resume_failure.json",
                {
                    "status": "resume_failed",
                    "failed_at_utc": utc_now_iso(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics terminou sem expor o trainer")
    save_dir = Path(trainer.save_dir)
    best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    last = Path(getattr(trainer, "last", save_dir / "weights" / "last.pt"))
    if not best.is_file() or not last.is_file():
        raise RuntimeError("treino terminou sem best.pt/last.pt verificaveis")
    validate_class_map(ultralytics.YOLO(str(best)).names)
    validate_class_map(ultralytics.YOLO(str(last)).names)

    best_sha256 = sha256_file(best)
    last_sha256 = sha256_file(last)
    if initial_sha256 == last_sha256:
        raise RuntimeError("last.pt nao mudou; retomada/treino nao comprovado")
    changed_from_base = base_sha256 != best_sha256 if base_sha256 else None
    if changed_from_base is False:
        raise RuntimeError("best.pt e identico ao yolo26n.pt inicial")
    completed_epochs = int(getattr(trainer, "epoch", -1)) + 1
    if mode == "resume" and completed_epochs <= resume_epoch + 1:
        raise RuntimeError("a retomada nao avancou alem da epoca salva")

    config.published_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, config.published_checkpoint)
    published_sha256 = sha256_file(config.published_checkpoint)
    if published_sha256 != best_sha256:
        raise RuntimeError("hash publicado diverge de best.pt")

    metrics = getattr(results, "results_dict", None) or getattr(
        trainer, "metrics", None
    )
    manifest = {
        "status": "completed",
        "training_mode": mode,
        "started_at_utc": started,
        "completed_at_utc": utc_now_iso(),
        "target_epochs": config.epochs,
        "completed_epochs": completed_epochs,
        "stopped_early": completed_epochs < config.epochs,
        "base_model": config.model,
        "base_checkpoint": safe_artifact_path(base_checkpoint)
        if base_checkpoint
        else None,
        "base_sha256": base_sha256,
        "initial_checkpoint": safe_artifact_path(initial_checkpoint),
        "initial_sha256": initial_sha256,
        "resume_from_epoch": resume_epoch + 1 if mode == "resume" else None,
        "interrupted_snapshot": (
            safe_artifact_path(resume_snapshot) if resume_snapshot else None
        ),
        "last_checkpoint": safe_artifact_path(last),
        "last_sha256": last_sha256,
        "best_checkpoint": safe_artifact_path(best),
        "best_sha256": best_sha256,
        "published_checkpoint": safe_artifact_path(config.published_checkpoint),
        "published_sha256": published_sha256,
        "weights_changed_during_invocation": initial_sha256 != last_sha256,
        "weights_changed_from_base": changed_from_base,
        "device": describe_device(device),
        "seed": config.seed,
        "dataset": fingerprint,
        "dataset_validation": validation,
        "train_arguments": _safe_arguments(train_args),
        "final_metrics": json_safe(metrics),
        "environment": environment_manifest(),
    }
    write_json(save_dir / "training_manifest.json", manifest)
    try:
        import yaml

        (save_dir / "effective_config.yaml").write_text(
            yaml.safe_dump(
                _safe_arguments(config.as_dict()),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    except ImportError:
        pass
    return manifest
