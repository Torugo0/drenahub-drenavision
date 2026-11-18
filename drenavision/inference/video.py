"""Frame-by-frame video inference with optional ByteTrack identity counting."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from drenavision.core.annotations import place_label
from drenavision.core.config import validate_image_size
from drenavision.core.constants import OVERLAY_CLASS_LABELS
from drenavision.core.schemas import Detection, PredictionEvent
from drenavision.core.storage import EventRepository, EvidenceStore
from drenavision.data.constants import CLASS_NAMES

from .device import describe_device, select_device
from .errors import InferenceError
from .media_io import write_image
from .objects import GroundedObjectPredictor
from .predictor import detections_from_result
from .utils import sha256_file, utc_now_iso, validate_class_map, write_json
from .video_codec import prepare_browser_video
from .water import SemanticWaterPredictor


def _serialize_detection(detection: Any) -> dict[str, Any]:
    if hasattr(detection, "model_dump"):
        return detection.model_dump(mode="json")
    return detection.as_dict()


def annotate_video_frame(frame_bgr: Any, detections: list[Detection]) -> Any:
    """Draw readable Portuguese labels and high-contrast boxes for the public UI."""

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    colors = {0: "#F97316", 1: "#0284C7", 2: "#A855F7"}
    image = Image.fromarray(frame_bgr[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    size = max(16, min(36, image.width // 38))
    font = ImageFont.load_default(size=size)
    for font_name in ("DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            font = ImageFont.truetype(font_name, size)
            break
        except OSError:
            continue
    thickness = max(3, image.width // 220)
    # Draw every outline first: a later overlapping box must not strike through
    # a label that was already rendered. Coordinates and detections stay intact.
    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        draw.rectangle(
            (x1, y1, x2, y2), outline=colors[detection.class_id], width=thickness
        )
    occupied_labels = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bounding_box
        color = colors[detection.class_id]
        label = OVERLAY_CLASS_LABELS[detection.class_name]
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        label_x, label_y = place_label(
            image.size,
            (x1, y1, x2, y2),
            (text_width + 16, text_height + 16),
            occupied_labels,
        )
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 16, label_y + text_height + 16),
            fill=color,
        )
        draw.text(
            (label_x + 8 - left, label_y + 8 - top), label, fill="white", font=font
        )
    return np.asarray(image)[:, :, ::-1].copy()


def water_temporal_policy(
    frames: list[dict[str, Any]], fps: float, confirmation_seconds: float
) -> dict[str, Any]:
    """Confirm persistent water presence without changing any image prediction."""
    current_run = longest_run = raw_water_frames = raw_water_boxes = 0
    for frame in frames:
        water = [item for item in frame["detections"] if item["class_id"] == 1]
        raw_water_boxes += len(water)
        if water:
            raw_water_frames += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    required = math.ceil(fps * confirmation_seconds) if confirmation_seconds > 0 else 0
    confirmed = confirmation_seconds == 0 or longest_run >= required
    suppressed = raw_water_frames if not confirmed else 0
    return {
        "enabled": confirmation_seconds > 0,
        "confirmation_seconds": confirmation_seconds,
        "fps": fps,
        "required_consecutive_frames": required,
        "raw_water_frames": raw_water_frames,
        "raw_water_detection_count": raw_water_boxes,
        "longest_consecutive_water_run": longest_run,
        "confirmed": confirmed,
        "suppressed_frames": suppressed,
        "rerendered_from_original": suppressed > 0,
        "policy": "video-water-longest-run-v1; require a consecutive run of ceil(fps*seconds); otherwise suppress all water for this clip, preserve raw proposals, and redraw from original decoded source frames",
        "scope": "video-only engineering confirmation; image predictions and the frozen image profile remain unchanged",
    }


def _summarize_video_counts(
    track: bool,
    unique_ids: dict[int, set[int]],
    fallback_peak: dict[int, int],
    tracked_representatives: dict[int, dict[int, Any]],
    peak_representatives: dict[int, list[Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    counts, representative_detections = {}, []
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
    return counts, representative_detections


def persist_video_event(
    manifest: dict[str, Any],
    repository: EventRepository,
    evidence_store: EvidenceStore,
    *,
    point_id: str | None = None,
    event_id: str | None = None,
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
    event_id = str(UUID(str(event_id))) if event_id is not None else str(uuid4())
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
            "inference",
            "pipeline",
            "base_pipeline_version",
            "raw_counts",
            "water_temporal_policy",
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
    image_size: int = 640,
    track: bool = False,
    max_frames: int = 18_000,
    water_model_path: str | Path | None = None,
    water_pixel_confidence: float = 0.5,
    water_min_area_fraction: float = 0.001,
    water_mode: str = "augment",
    residue_confidence: float = 0.35,
    animal_confidence: float = 0.35,
    water_confirmation_seconds: float = 0.0,
    object_model_path: str | Path | None = None,
    object_residue_confidence: float = 0.35,
    object_animal_confidence: float = 0.35,
    object_text_confidence: float = 0.20,
    object_mode: str = "animal_replace",
    object_shortest_edge: int = 800,
    object_longest_edge: int = 1333,
    object_verifier_path: str | Path | None = None,
    object_residue_min_margin: float = 0.0,
    object_animal_min_margin: float = 0.0,
    shared_predictor: Any = None,
) -> dict[str, Any]:
    """Annotate one video and write a structured JSON sidecar.

    With tracking enabled, counts use unique tracker IDs. Without IDs, the
    conservative maximum simultaneous count is reported instead of summing the
    same object once per frame.
    """

    image_size = validate_image_size(image_size)
    from .runtime import (
        filter_detections,
        merge_object_detections,
        merge_water_detections,
        overlay_water,
        profile_version,
    )

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
    if water_mode not in {"augment", "replace"}:
        raise ValueError("water_mode deve ser augment ou replace")
    if object_mode not in {"augment", "replace", "animal_replace"}:
        raise ValueError("object_mode deve ser augment, replace ou animal_replace")
    for name, value in (
        ("water_pixel_confidence", water_pixel_confidence),
        ("water_min_area_fraction", water_min_area_fraction),
        ("residue_confidence", residue_confidence),
        ("animal_confidence", animal_confidence),
        ("object_residue_confidence", object_residue_confidence),
        ("object_animal_confidence", object_animal_confidence),
        ("object_text_confidence", object_text_confidence),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} deve ser finito e estar entre 0 e 1")
    if (
        not math.isfinite(water_confirmation_seconds)
        or not 0 <= water_confirmation_seconds <= 10
    ):
        raise ValueError(
            "water_confirmation_seconds deve ser finito e estar entre 0 e 10"
        )
    water_confirmation_seconds = float(water_confirmation_seconds)

    expected_parameters = {
        "imgsz": image_size,
        "confidence": float(confidence),
        "iou": float(iou),
        "residue_confidence": float(residue_confidence),
        "animal_confidence": float(animal_confidence),
        "water_mode": water_mode,
    }
    objects = None
    object_provenance = None
    water = None
    water_provenance = None
    if shared_predictor is not None:
        if not shared_predictor.is_ready:
            raise InferenceError("perfil compartilhado indisponível para vídeo")
        if getattr(shared_predictor, "parameters", None) != expected_parameters:
            raise InferenceError(
                "parâmetros do vídeo divergem do perfil de imagem compartilhado"
            )
        shared_master = getattr(shared_predictor, "master", None)
        if shared_master is None or sha256_file(
            shared_master.model_path
        ) != sha256_file(checkpoint):
            raise InferenceError("checkpoint do vídeo diverge do perfil compartilhado")
        water = getattr(shared_predictor, "water", None)
        objects = getattr(shared_predictor, "objects", None)
        if (water is None) != (water_model_path is None) or (objects is None) != (
            object_model_path is None
        ):
            raise InferenceError(
                "componentes do vídeo divergem do perfil compartilhado"
            )
        if water is not None:
            water_provenance = water.provenance
            policy = water_provenance["policy"]
            if any(
                policy.get(key) != value
                for key, value in {
                    "pixel_confidence": water_pixel_confidence,
                    "min_area_fraction": water_min_area_fraction,
                    "processing_size": 512,
                }.items()
            ):
                raise InferenceError(
                    "política de água do vídeo diverge do perfil compartilhado"
                )
            if (
                hasattr(water, "model_dir")
                and Path(water.model_dir).resolve() != Path(water_model_path).resolve()
            ):
                raise InferenceError(
                    "modelo de água do vídeo diverge do perfil compartilhado"
                )
        if objects is not None:
            object_provenance = objects.provenance
            policy = object_provenance["policy"]
            if (
                any(
                    policy.get(key) != value
                    for key, value in {
                        "residue_confidence": object_residue_confidence,
                        "animal_confidence": object_animal_confidence,
                        "text_confidence": object_text_confidence,
                        "shortest_edge": object_shortest_edge,
                        "longest_edge": object_longest_edge,
                    }.items()
                )
                or shared_predictor.object_mode != object_mode
            ):
                raise InferenceError(
                    "política de objetos do vídeo diverge do perfil compartilhado"
                )
            verification = object_provenance.get("verifier")
            if (verification is None) != (object_verifier_path is None):
                raise InferenceError(
                    "verificador do vídeo diverge do perfil compartilhado"
                )
            if verification is not None and (
                verification.get("residue_min_margin") != object_residue_min_margin
                or verification.get("animal_min_margin") != object_animal_min_margin
            ):
                raise InferenceError(
                    "margens do vídeo divergem do perfil compartilhado"
                )
            if (
                hasattr(objects, "model_dir")
                and Path(objects.model_dir).resolve()
                != Path(object_model_path).resolve()
            ):
                raise InferenceError(
                    "modelo de objetos do vídeo diverge do perfil compartilhado"
                )
    elif object_model_path is not None:
        objects = GroundedObjectPredictor(
            object_model_path,
            device=device,
            residue_confidence=object_residue_confidence,
            animal_confidence=object_animal_confidence,
            text_confidence=object_text_confidence,
            shortest_edge=object_shortest_edge,
            longest_edge=object_longest_edge,
            verifier_model_path=object_verifier_path,
            residue_min_margin=object_residue_min_margin,
            animal_min_margin=object_animal_min_margin,
        )
        if not objects.is_ready:
            raise RuntimeError(
                f"especialista de objetos configurado indisponível: {objects.detail}"
            )
        object_provenance = objects.provenance

    if shared_predictor is None and water_model_path is not None:
        water = SemanticWaterPredictor(
            water_model_path,
            device=device,
            pixel_confidence=water_pixel_confidence,
            min_area_fraction=water_min_area_fraction,
            processing_size=512,
        )
        if not water.is_ready:
            raise RuntimeError(
                f"segmentador de água configurado indisponível: {water.detail}"
            )
        water_provenance = water.provenance

    model_sha256 = sha256_file(checkpoint)
    master_version = f"{checkpoint.stem}-{model_sha256[:12]}"
    operating_parameters = dict(expected_parameters)
    # With no configured specialist, preserve the exact established v3 profile.
    if objects is not None:
        operating_parameters["objects"] = object_provenance
        operating_parameters["object_mode"] = object_mode
    base_pipeline_version = profile_version(
        model_sha256, water_provenance, operating_parameters
    )
    if (
        shared_predictor is not None
        and shared_predictor.model_version != base_pipeline_version
    ):
        raise InferenceError(
            "versão de imagem diverge do perfil compartilhado do vídeo"
        )
    video_parameters = {
        **operating_parameters,
        "video_water_confirmation_seconds": water_confirmation_seconds,
    }
    pipeline_version = (
        profile_version(model_sha256, water_provenance, video_parameters)
        if water_confirmation_seconds > 0
        else base_pipeline_version
    )
    pipeline = {
        "model_version": pipeline_version,
        "base_pipeline_version": base_pipeline_version,
        "master_model_version": master_version,
        "master_model_sha256": model_sha256,
        "water": water_provenance,
        "operating_parameters": video_parameters
        if water_confirmation_seconds > 0
        else operating_parameters,
        "water_counting_policy": "maximum simultaneous merged water regions; no persistent IDs or unique-puddle claims"
        if water is not None
        else "legacy YOLO tracking/peak counting",
        "water_merge_policy": "same-class contained duplicates only; keep an actual predicted box; never invent a union"
        if water is not None
        else None,
    }
    if objects is not None:
        pipeline.update(
            {
                "objects": object_provenance,
                "object_processing_policy": "one real specialist forward on every current original RGB frame; no stride, frame-result cache or propagated boxes",
                "object_counting_policy": "specialist detections have no persistent IDs; use simultaneous-frame peak fallback, retaining genuine master tracker IDs where present",
            }
        )
    if shared_predictor is not None:
        pipeline["model_ownership"] = (
            "borrowed API water/object/verifier instances; independent lightweight master tracker in the same process"
        )

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
    best_frame: Any = None
    best_frame_score = -1.0
    class_highlights: dict[int, tuple[float, Any, dict[str, Any]]] = {}
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
            inference_arguments = {
                "source": frame,
                "conf": confidence,
                "iou": iou,
                "device": selected_device,
                "imgsz": image_size,
                "verbose": False,
            }
            if water is not None:
                inference_arguments["classes"] = (
                    [0, 2] if water_mode == "replace" else [0, 1, 2]
                )
            if track:
                results = model.track(
                    persist=True,
                    tracker="bytetrack.yaml",
                    **inference_arguments,
                )
            else:
                results = model.predict(**inference_arguments)
            if not results:
                raise RuntimeError(f"modelo nao retornou o frame {frame_index}")
            result = results[0]
            master_detections = filter_detections(
                detections_from_result(result, width, height),
                residue_confidence=residue_confidence,
                animal_confidence=animal_confidence,
            )
            object_record = None
            rgb = (
                frame[:, :, ::-1].copy()
                if objects is not None or water is not None
                else None
            )
            if objects is not None:
                object_started = time.perf_counter()
                object_result = objects.predict(rgb.copy())
                object_forward_ms = (time.perf_counter() - object_started) * 1000.0
                # A specialist forward is object detection, not a tracker.
                specialist_detections = [
                    item.model_copy(update={"track_id": None})
                    for item in object_result.detections
                ]
                master_detections = merge_object_detections(
                    master_detections, specialist_detections, mode=object_mode
                )
                object_record = {
                    "model_version": objects.model_version,
                    "inference_ms": float(object_result.inference_ms),
                    "forward_wall_ms": object_forward_ms,
                    "detections": [
                        _serialize_detection(item) for item in specialist_detections
                    ],
                    "raw_candidates": getattr(object_result, "raw_candidates", []),
                    "processing": "forward_on_current_original_frame",
                }
            water_result = None
            display_frame = frame
            if water is not None:
                water_result = water.predict(rgb)
                master_detections = merge_water_detections(
                    master_detections, water_result.detections, mode=water_mode
                )
                # Water surfaces are not persistent instances, including a
                # master box retained while augmenting semantic regions.
                master_detections = [
                    item.model_copy(update={"track_id": None})
                    if item.class_id == 1
                    else item
                    for item in master_detections
                ]
                display_frame = overlay_water(rgb, water_result.mask)[:, :, ::-1].copy()
            detections = [
                item.model_copy(
                    update={
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps,
                    }
                )
                for item in master_detections
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
            annotated = annotate_video_frame(display_frame, detections)
            writer.write(annotated)
            frame_score = sum(item.confidence for item in detections)
            if frame_score > best_frame_score:
                best_frame = annotated.copy()
                best_frame_score = frame_score
            for class_id in CLASS_NAMES:
                class_items = [item for item in detections if item.class_id == class_id]
                if not class_items:
                    continue
                score = max(item.confidence for item in class_items)
                previous_highlight = class_highlights.get(class_id)
                if previous_highlight is None or score > previous_highlight[0]:
                    class_highlights[class_id] = (
                        score,
                        annotated.copy(),
                        {
                            "class_name": CLASS_NAMES[class_id],
                            "frame_index": frame_index,
                            "timestamp_seconds": frame_index / fps,
                            "detections": [
                                _serialize_detection(item) for item in detections
                            ],
                        },
                    )
            inference_ms = (time.perf_counter() - frame_started) * 1000.0
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "inference_ms": inference_ms,
                    "detections": [_serialize_detection(item) for item in detections],
                    "water": water_result.metadata
                    if water_result is not None
                    else None,
                    **({"objects": object_record} if object_record is not None else {}),
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
    raw_counts, raw_representatives = _summarize_video_counts(
        track, unique_ids, fallback_peak, tracked_representatives, peak_representatives
    )
    temporal = water_temporal_policy(frames, fps, water_confirmation_seconds)
    pipeline["video_temporal_policy"] = temporal
    try:
        if temporal["enabled"]:
            for frame in frames:
                frame["raw_water_detections"] = [
                    item for item in frame["detections"] if item["class_id"] == 1
                ]
                frame["water_temporally_suppressed"] = bool(
                    frame["raw_water_detections"] and temporal["suppressed_frames"]
                )
        if temporal["suppressed_frames"]:
            # Decode the original recording again. No annotated/JPEG frame is
            # reused, so removed boxes and blue masks cannot leave ghost pixels.
            rerender_source = cv2.VideoCapture(str(local_source))
            rerender_path = scratch / "temporally_confirmed.mp4"
            rerender_writer = cv2.VideoWriter(
                str(rerender_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            best_frame, best_frame_score, class_highlights = None, -1.0, {}
            try:
                if not rerender_source.isOpened() or not rerender_writer.isOpened():
                    raise RuntimeError(
                        "não foi possível reconstruir o vídeo a partir do original"
                    )
                for frame_record in frames:
                    ok, original = rerender_source.read()
                    if not ok:
                        raise RuntimeError(
                            "reconstrução temporal perdeu quadros do original"
                        )
                    render_started = time.perf_counter()
                    frame_record["detections"] = [
                        item
                        for item in frame_record["detections"]
                        if item["class_id"] != 1
                    ]
                    detections = [
                        Detection.model_validate(item)
                        for item in frame_record["detections"]
                    ]
                    annotated = annotate_video_frame(original, detections)
                    rerender_writer.write(annotated)
                    frame_score = sum(item.confidence for item in detections)
                    if frame_score > best_frame_score:
                        best_frame, best_frame_score = annotated.copy(), frame_score
                    for class_id in CLASS_NAMES:
                        class_items = [
                            item for item in detections if item.class_id == class_id
                        ]
                        if not class_items:
                            continue
                        score = max(item.confidence for item in class_items)
                        if (
                            class_id not in class_highlights
                            or score > class_highlights[class_id][0]
                        ):
                            class_highlights[class_id] = (
                                score,
                                annotated.copy(),
                                {
                                    "class_name": CLASS_NAMES[class_id],
                                    "frame_index": frame_record["frame_index"],
                                    "timestamp_seconds": frame_record[
                                        "timestamp_seconds"
                                    ],
                                    "detections": frame_record["detections"],
                                },
                            )
                    frame_record["raw_inference_ms"] = frame_record["inference_ms"]
                    frame_record["temporal_rerender_ms"] = (
                        time.perf_counter() - render_started
                    ) * 1000
                    frame_record["inference_ms"] += frame_record["temporal_rerender_ms"]
                if rerender_source.read()[0]:
                    raise RuntimeError(
                        "reconstrução temporal encontrou quadros não registrados"
                    )
            finally:
                rerender_writer.release()
                rerender_source.release()
            partial = rerender_path
            unique_ids[1].clear()
            tracked_representatives[1].clear()
            fallback_peak[1] = 0
            peak_representatives[1] = []
            untracked_frames = sum(
                any(item["track_id"] is None for item in frame["detections"])
                for frame in frames
            )
        encoding = prepare_browser_video(partial, destination)
    finally:
        scratch_handle.cleanup()

    # The still frame lets the user see the marked object immediately, even
    # before pressing play. Highlights are genuine model output from this run.
    poster_path = destination.with_suffix(".jpg")
    write_image(poster_path, best_frame)
    review_frames = []
    for class_id, (_score, highlighted_frame, details) in class_highlights.items():
        highlight_path = destination.with_name(
            f"{destination.stem}_{CLASS_NAMES[class_id]}.jpg"
        )
        write_image(highlight_path, highlighted_frame)
        review_frames.append({**details, "image_filename": highlight_path.name})

    counts, representative_detections = _summarize_video_counts(
        track, unique_ids, fallback_peak, tracked_representatives, peak_representatives
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "model_version": pipeline_version,
        "base_pipeline_version": base_pipeline_version,
        "model_sha256": model_sha256,
        "pipeline": pipeline,
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "poster_filename": poster_path.name,
        "review_frames": review_frames,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frames": declared_frames,
            "processed_frames": frame_index,
            "max_frames": max_frames,
            "elapsed_seconds": elapsed,
            "processing_fps": frame_index / elapsed if elapsed > 0 else None,
            "duration_seconds": frame_index / fps,
            **encoding,
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
        "raw_counts": raw_counts,
        "raw_water_representative_detections": [
            item for item in raw_representatives if item["class_id"] == 1
        ],
        "water_temporal_policy": temporal,
        "representative_detections": representative_detections,
        "device": describe_device(selected_device),
        "inference": {
            **operating_parameters,
            "video_water_confirmation_seconds": water_confirmation_seconds,
            "water_pixel_confidence": water_pixel_confidence
            if water is not None
            else None,
            "water_min_area_fraction": water_min_area_fraction
            if water is not None
            else None,
            "water_processing_size": 512 if water is not None else None,
            "timing_note": "wall time per frame includes master inference, configured object/water forwards, filtering, merging, mask/label annotation and writer call",
        },
        "frames": frames,
    }
    write_json(destination.with_suffix(".json"), manifest)
    return manifest
