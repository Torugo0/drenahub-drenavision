"""Frame-by-frame video inference with optional ByteTrack identity counting."""

from __future__ import annotations

import importlib
import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.schemas import Detection, PredictionEvent
from app.core.storage import EventRepository, EvidenceStore
from app.data.constants import CLASS_NAMES

from .device import describe_device, select_device
from .predictor import detections_from_result
from .utils import sha256_file, utc_now_iso, validate_class_map, write_json


def _serialize_detection(detection: Any) -> dict[str, Any]:
    if hasattr(detection, "model_dump"):
        return detection.model_dump(mode="json")
    return detection.as_dict()


def persist_video_event(
    manifest: dict[str, Any],
    repository: EventRepository,
    evidence_store: EvidenceStore,
    *,
    point_id: str | None = None,
) -> PredictionEvent:
    """Persist one auditable summary event for a completed video run."""

    if manifest.get("status") != "completed":
        raise ValueError("somente videos processados podem gerar evento")
    detections = [
        Detection.model_validate(item)
        for item in manifest.get("representative_detections", [])
    ]
    frame_times = [
        float(frame["inference_ms"])
        for frame in manifest.get("frames", [])
        if isinstance(frame.get("inference_ms"), (int, float))
    ]
    inspection_recommended = bool(detections)
    normalized_point = point_id.strip() if point_id and point_id.strip() else None
    event_id = str(uuid4())
    evidence_payload = {
        key: manifest.get(key)
        for key in (
            "status",
            "created_at_utc",
            "model_version",
            "model_sha256",
            "input_sha256",
            "video",
            "tracking",
            "counts",
            "representative_detections",
            "device",
        )
    }
    evidence_payload["event_id"] = event_id
    evidence_uri = evidence_store.write(
        event_id,
        json.dumps(evidence_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        suffix=".json",
    )
    event = PredictionEvent(
        event_id=event_id,
        point_id=normalized_point,
        timestamp=datetime.now(UTC),
        input_type="video",
        model_version=str(manifest["model_version"]),
        detections=detections,
        inspection_recommended=inspection_recommended,
        status=("revisao_recomendada" if inspection_recommended else "sem_deteccao"),
        evidence_uri=evidence_uri,
        inference_ms=sum(frame_times) / len(frame_times) if frame_times else 0.0,
    )
    try:
        repository.initialize()
        repository.save(event)
    except Exception:
        evidence_store.delete(evidence_uri)
        raise
    return event


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    *,
    confidence: float = 0.35,
    iou: float = 0.45,
    device: str = "auto",
    track: bool = False,
    max_frames: int = 18_000,
) -> dict[str, Any]:
    """Annotate one video and write a structured JSON sidecar.

    With tracking enabled, counts use unique tracker IDs. Without IDs, the
    conservative maximum simultaneous count is reported instead of summing the
    same object once per frame.
    """

    source = Path(input_path)
    destination = Path(output_path)
    checkpoint = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"video de entrada ausente: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint customizado ausente: {checkpoint}")
    if source.resolve() == destination.resolve():
        raise ValueError("input e output devem ser arquivos diferentes")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("output deve usar extensao .mp4")
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf/iou devem estar entre 0 e 1")
    if max_frames < 1:
        raise ValueError("max_frames deve ser positivo")

    cv2 = importlib.import_module("cv2")
    ultralytics = importlib.import_module("ultralytics")
    selected_device = select_device(device)
    model = ultralytics.YOLO(str(checkpoint))
    validate_class_map(model.names)

    scratch_handle = tempfile.TemporaryDirectory(prefix="drenavision_video_")
    scratch = Path(scratch_handle.name)
    local_source = scratch / f"input{source.suffix.lower()}"
    partial = scratch / "annotated.mp4"
    try:
        shutil.copyfile(source, local_source)
    except Exception:
        scratch_handle.cleanup()
        raise

    capture = cv2.VideoCapture(str(local_source))
    if not capture.isOpened():
        scratch_handle.cleanup()
        raise RuntimeError(f"OpenCV nao abriu o video: {source.name}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        scratch_handle.cleanup()
        raise RuntimeError("video nao informou dimensoes validas")
    if not 0 < fps < 1000:
        fps = 25.0
    if declared_frames > max_frames:
        capture.release()
        scratch_handle.cleanup()
        raise RuntimeError(
            f"video declara {declared_frames} frames e excede o limite de {max_frames}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        scratch_handle.cleanup()
        raise RuntimeError(
            "OpenCV nao criou MP4; instale um build com codec mp4v ou use FFmpeg"
        )

    unique_ids: dict[int, set[int]] = {class_id: set() for class_id in CLASS_NAMES}
    fallback_peak: dict[int, int] = {class_id: 0 for class_id in CLASS_NAMES}
    tracked_representatives: dict[int, dict[int, Any]] = {
        class_id: {} for class_id in CLASS_NAMES
    }
    peak_representatives: dict[int, list[Any]] = {
        class_id: [] for class_id in CLASS_NAMES
    }
    untracked_frames = 0
    frames: list[dict[str, Any]] = []
    frame_index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= max_frames:
                raise RuntimeError(
                    f"video excede o limite operacional de {max_frames} frames"
                )
            frame_started = time.perf_counter()
            if track:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence,
                    iou=iou,
                    device=selected_device,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=confidence,
                    iou=iou,
                    device=selected_device,
                    verbose=False,
                )
            if not results:
                raise RuntimeError(f"modelo nao retornou o frame {frame_index}")
            result = results[0]
            detections = [
                item.model_copy(
                    update={
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps,
                    }
                )
                for item in detections_from_result(result, width, height)
            ]
            for class_id in CLASS_NAMES:
                class_detections = [
                    item for item in detections if int(item.class_id) == class_id
                ]
                if len(class_detections) > fallback_peak[class_id]:
                    fallback_peak[class_id] = len(class_detections)
                    peak_representatives[class_id] = class_detections
            has_untracked = False
            for item in detections:
                if item.track_id is None:
                    has_untracked = True
                else:
                    unique_ids[int(item.class_id)].add(int(item.track_id))
                    previous = tracked_representatives[int(item.class_id)].get(
                        int(item.track_id)
                    )
                    if previous is None or item.confidence > previous.confidence:
                        tracked_representatives[int(item.class_id)][
                            int(item.track_id)
                        ] = item
            if has_untracked:
                untracked_frames += 1
            annotated = result.plot()
            writer.write(annotated)
            speed = getattr(result, "speed", {})
            inference_ms = (
                float(speed["inference"])
                if isinstance(speed, dict)
                and isinstance(speed.get("inference"), (int, float))
                else (time.perf_counter() - frame_started) * 1000.0
            )
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "inference_ms": inference_ms,
                    "detections": [_serialize_detection(item) for item in detections],
                }
            )
            frame_index += 1
    except Exception:
        writer.release()
        capture.release()
        scratch_handle.cleanup()
        raise
    writer.release()
    capture.release()
    if frame_index == 0:
        scratch_handle.cleanup()
        raise RuntimeError("video nao contem frames decodificaveis")
    try:
        shutil.copy2(partial, destination)
    finally:
        scratch_handle.cleanup()

    counts: dict[str, dict[str, Any]] = {}
    representative_detections: list[dict[str, Any]] = []
    for class_id, class_name in CLASS_NAMES.items():
        tracked_count = len(unique_ids[class_id])
        peak_count = fallback_peak[class_id]
        if track and tracked_count:
            count = max(tracked_count, peak_count)
            method = (
                "unique_track_ids"
                if tracked_count >= peak_count
                else "hybrid_max_tracks_and_frame_peak"
            )
            representatives = list(tracked_representatives[class_id].values())
            if len(representatives) < count:
                tracked_ids = {
                    int(item.track_id)
                    for item in representatives
                    if item.track_id is not None
                }
                supplements = [
                    item
                    for item in peak_representatives[class_id]
                    if item.track_id is None or int(item.track_id) not in tracked_ids
                ]
                representatives.extend(supplements[: count - len(representatives)])
        else:
            count = peak_count
            method = "max_simultaneous_per_frame"
            representatives = peak_representatives[class_id]
        counts[class_name] = {"count": count, "method": method}
        representative_detections.extend(
            _serialize_detection(item) for item in representatives[:count]
        )
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "model_version": f"{checkpoint.stem}-{sha256_file(checkpoint)[:12]}",
        "model_sha256": sha256_file(checkpoint),
        "input_sha256": sha256_file(source),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frames": declared_frames,
            "processed_frames": frame_index,
            "max_frames": max_frames,
            "elapsed_seconds": elapsed,
            "processing_fps": frame_index / elapsed if elapsed > 0 else None,
            "audio_preserved": False,
        },
        "tracking": {
            "enabled": track,
            "tracker": "bytetrack.yaml" if track else None,
            "frames_with_untracked_detections": untracked_frames,
            "counting_note": (
                "O total por classe e o maior entre IDs unicos e o pico por quadro; "
                "deteccoes nao sao somadas indiscriminadamente entre frames."
            ),
        },
        "counts": counts,
        "representative_detections": representative_detections,
        "device": describe_device(selected_device),
        "frames": frames,
    }
    write_json(destination.with_suffix(".json"), manifest)
    return manifest
