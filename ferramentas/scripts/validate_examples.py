"""Run the real model on every public example and verify encoded video outputs."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio_ffmpeg  # noqa: E402

from drenavision.core.config import Settings, validate_image_size  # noqa: E402
from drenavision.data.constants import CLASS_NAMES  # noqa: E402
from drenavision.inference.media_io import read_image_bgr, write_image  # noqa: E402
from drenavision.inference.runtime import create_predictor, profile_version  # noqa: E402
from drenavision.inference.utils import sha256_file, write_json  # noqa: E402
from drenavision.inference.video import process_video  # noqa: E402


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def probe_browser_video(path: Path) -> dict[str, Any]:
    """Check the real bitstream and decode it independently with FFmpeg."""

    tags = []
    with path.open("rb") as stream:
        while header := stream.read(8):
            size, tag = struct.unpack(">I4s", header)
            if size < 8:
                raise ValueError("estrutura MP4 inválida")
            tags.append(tag)
            stream.seek(size - 8, 1)
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    return {
        "codec": "h264" if b"avcC" in path.read_bytes() else "unknown",
        "faststart": tags.index(b"moov") < tags.index(b"mdat"),
        "decoded_frames": count,
        "decoded_seconds": seconds,
    }


def validate_examples(
    catalog_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    confidence: float | None = None,
    iou: float | None = None,
    device: str | None = None,
    image_size: int | None = None,
    water_model_path: Path | None = None,
    no_water: bool = False,
    water_pixel_confidence: float | None = None,
    water_min_area_fraction: float | None = None,
    water_mode: str | None = None,
    residue_confidence: float | None = None,
    animal_confidence: float | None = None,
    water_confirmation_seconds: float | None = None,
    expected_model_version: str | None = None,
    expected_video_model_version: str | None = None,
) -> dict[str, Any]:
    runtime = Settings()
    confidence = runtime.confidence_threshold if confidence is None else confidence
    iou = runtime.iou_threshold if iou is None else iou
    device = runtime.device if device is None else device
    image_size = (
        runtime.image_size if image_size is None else validate_image_size(image_size)
    )
    water_model_path = (
        None if no_water else water_model_path or runtime.water_model_path
    )
    water_pixel_confidence = (
        runtime.water_pixel_confidence
        if water_pixel_confidence is None
        else water_pixel_confidence
    )
    water_min_area_fraction = (
        runtime.water_min_area_fraction
        if water_min_area_fraction is None
        else water_min_area_fraction
    )
    water_mode = runtime.water_mode if water_mode is None else water_mode
    residue_confidence = (
        confidence if residue_confidence is None else residue_confidence
    )
    animal_confidence = (
        runtime.animal_confidence_threshold
        if animal_confidence is None
        else animal_confidence
    )
    water_confirmation_seconds = float(
        runtime.video_water_confirmation_seconds
        if water_confirmation_seconds is None
        else water_confirmation_seconds
    )
    if (
        not math.isfinite(water_confirmation_seconds)
        or not 0 <= water_confirmation_seconds <= 10
    ):
        raise ValueError("water_confirmation_seconds deve estar entre 0 e 10")
    predictor = create_predictor(
        checkpoint,
        confidence_threshold=confidence,
        iou_threshold=iou,
        device=device,
        image_size=image_size,
        water_model_path=water_model_path,
        water_pixel_confidence=water_pixel_confidence,
        water_min_area_fraction=water_min_area_fraction,
        water_mode=water_mode,
        residue_confidence=residue_confidence,
        animal_confidence=animal_confidence,
    )
    model_version = predictor.model_version
    if expected_model_version is not None and model_version != expected_model_version:
        raise ValueError(
            f"Perfil divergente antes da avaliação: {model_version} != {expected_model_version}"
        )
    water_provenance = (
        predictor.water.provenance if predictor.water is not None else None
    )
    master_sha256 = sha256_file(checkpoint)
    video_model_version = (
        profile_version(
            master_sha256,
            water_provenance,
            {
                **predictor.parameters,
                "video_water_confirmation_seconds": water_confirmation_seconds,
            },
        )
        if water_confirmation_seconds > 0
        else model_version
    )
    if (
        expected_video_model_version is not None
        and video_model_version != expected_video_model_version
    ):
        raise ValueError(
            f"Perfil de vídeo divergente antes da avaliação: {video_model_version} != {expected_video_model_version}"
        )
    if not predictor.is_ready:
        raise RuntimeError(f"Pipeline configurado indisponível: {predictor.detail}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "model_version": model_version,
        "master_checkpoint_sha256": master_sha256,
        "water": water_provenance,
        "operating_parameters": predictor.parameters,
        "video_model_version": video_model_version,
        "video_water_confirmation_seconds": water_confirmation_seconds,
    }
    report: dict[str, Any] = {
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": sha256_file(checkpoint),
        "model_version": model_version,
        "image_model_version": model_version,
        "video_model_version": video_model_version,
        "pipeline": pipeline,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "image_size": image_size,
        "tracking": True,
        "protocol": "All eight catalog originals, complete videos without frame sampling; frozen image detector plus separately versioned video-only temporal water confirmation. Original media and image thresholds remain unchanged.",
        "note": (
            "Known regression/adaptation files, not independent accuracy. Catalog tags describe subjects, not ground truth. "
            "Missing subject classes and unexpected predictions need visual review; no precision/recall/mAP is inferred. "
            "Animal/residue tracker IDs need not represent distinct objects; water counts are simultaneous visual regions, not distinct puddles."
        ),
        "items": [],
    }
    items = []
    for entry in catalog["items"]:
        source = catalog_path.parent / entry["path"]
        output = (
            output_dir
            / f"{entry['id']}_annotated{'.mp4' if entry['kind'] == 'video' else '.jpg'}"
        )
        row: dict[str, Any] = {
            "id": entry["id"],
            "kind": entry["kind"],
            "input": project_path(source),
            "output": project_path(output),
            "content_tags": entry["content_tags"],
            "evaluation_role": entry.get("evaluation_role"),
        }
        try:
            row["input_sha256"] = sha256_file(source)
            if entry.get("sha256") and row["input_sha256"] != entry["sha256"]:
                raise ValueError(
                    "O arquivo original diverge do hash selado no catálogo"
                )
            if entry["kind"] == "video":
                manifest = process_video(
                    source,
                    output,
                    checkpoint,
                    confidence=confidence,
                    iou=iou,
                    device=device,
                    image_size=image_size,
                    track=True,
                    water_model_path=water_model_path,
                    water_pixel_confidence=water_pixel_confidence,
                    water_min_area_fraction=water_min_area_fraction,
                    water_mode=water_mode,
                    residue_confidence=residue_confidence,
                    animal_confidence=animal_confidence,
                    water_confirmation_seconds=water_confirmation_seconds,
                )
                row["model_version"] = manifest["model_version"]
                row["base_pipeline_version"] = manifest.get(
                    "base_pipeline_version",
                    manifest.get("pipeline", {}).get(
                        "base_pipeline_version", manifest["model_version"]
                    ),
                )
                if row["base_pipeline_version"] != model_version:
                    raise ValueError(
                        "O perfil de vídeo diverge do pipeline de imagem configurado"
                    )
                if manifest["model_version"] != video_model_version:
                    raise ValueError(
                        "O perfil temporal do vídeo diverge dos parâmetros configurados"
                    )
                row["counts"] = {
                    name: value["count"] for name, value in manifest["counts"].items()
                }
                row["raw_counts"] = manifest.get("raw_counts", manifest["counts"])
                row["water_temporal_policy"] = manifest.get(
                    "water_temporal_policy",
                    manifest.get("pipeline", {}).get("video_temporal_policy"),
                )
                row["processed_frames"] = manifest["video"]["processed_frames"]
                if entry.get("frames") and row["processed_frames"] != entry["frames"]:
                    raise RuntimeError(
                        "O processamento não cobriu todos os quadros do original"
                    )
                row["counting_methods"] = {
                    name: value["method"] for name, value in manifest["counts"].items()
                }
                row["frames_with_detections"] = sum(
                    bool(frame["detections"]) for frame in manifest["frames"]
                )
                row["video_validation"] = probe_browser_video(output)
                if row["video_validation"]["decoded_frames"] != row["processed_frames"]:
                    raise RuntimeError("o vídeo exportado perdeu quadros")
                if row["video_validation"]["codec"] != "h264":
                    raise RuntimeError("o vídeo exportado não usa H264")
                if not row["video_validation"]["faststart"]:
                    raise RuntimeError("o vídeo exportado não usa faststart")
                detections = [
                    detection
                    for frame in manifest["frames"]
                    for detection in frame["detections"]
                ]
                row["poster"] = project_path(output.with_suffix(".jpg"))
                row["review_frames"] = manifest["review_frames"]
                row["manifest"] = project_path(output.with_suffix(".json"))
                row["frames_with_class"] = {
                    name: sum(
                        any(
                            detection["class_name"] == name
                            for detection in frame["detections"]
                        )
                        for frame in manifest["frames"]
                    )
                    for name in CLASS_NAMES.values()
                }
                row["frames_with_retained_water_mask"] = sum(
                    bool((frame.get("water") or {}).get("mask_pixels", 0))
                    and not frame.get("water_temporally_suppressed", False)
                    for frame in manifest["frames"]
                )
                row["frames_with_raw_semantic_water_mask"] = sum(
                    bool((frame.get("water") or {}).get("mask_pixels", 0))
                    for frame in manifest["frames"]
                )
                row["raw_water_detection_frames"] = sum(
                    bool(
                        frame.get(
                            "raw_water_detections",
                            [
                                item
                                for item in frame["detections"]
                                if item["class_id"] == 1
                            ],
                        )
                    )
                    for frame in manifest["frames"]
                )
            else:
                frame_bgr = read_image_bgr(source)
                result = predictor.predict(frame_bgr[:, :, ::-1].copy())
                row["model_version"] = result.model_version
                row["base_pipeline_version"] = result.model_version
                if result.model_version != model_version:
                    raise ValueError(
                        "O perfil da imagem diverge do pipeline configurado"
                    )
                # Use the real pipeline image: it includes the neural mask and
                # fresh boxes. Redrawing only the YOLO results would lose water.
                write_image(output, result.annotated_bgr)
                detections = [
                    item.model_dump(mode="json") for item in result.detections
                ]
                row["counts"] = {
                    name: sum(item["class_name"] == name for item in detections)
                    for name in CLASS_NAMES.values()
                }
                row["detections"] = detections
                write_json(
                    output.with_suffix(".json"),
                    {
                        **result.as_dict(),
                        "input_sha256": row["input_sha256"],
                        "pipeline": pipeline,
                    },
                )
            row["max_confidence_by_class"] = {
                name: max(
                    (
                        item["confidence"]
                        for item in detections
                        if item["class_name"] == name
                    ),
                    default=None,
                )
                for name in CLASS_NAMES.values()
            }
            row["missing_content_classes"] = [
                name for name in entry["content_tags"] if not row["counts"].get(name)
            ]
            row["unexpected_class_predictions"] = [
                name
                for name, count in row["counts"].items()
                if count and name not in entry["content_tags"]
            ]
            row["inspection_recommended"] = any(row["counts"].values())
            row["output_sha256"] = sha256_file(output)
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc).replace(str(PROJECT_ROOT), "<project>")
        items.append(row)
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in ("id", "status", "counts", "missing_content_classes")
                }
            ),
            flush=True,
        )
        report["items"] = items
        write_json(output_dir / "validation_report.json", report)
    report["status"] = (
        "completed"
        if all(item["status"] == "completed" for item in items)
        else "completed_with_failures"
    )
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["summary"] = {
        "examples": len(items),
        "successful_examples": sum(item["status"] == "completed" for item in items),
        "processed_video_frames": sum(
            item.get("processed_frames", 0) for item in items
        ),
        "all_outputs_match_pipeline": all(
            item.get("base_pipeline_version") == model_version
            and item.get("model_version")
            == (video_model_version if item["kind"] == "video" else model_version)
            for item in items
        ),
        "base_pipeline_version": model_version,
        "video_model_version": video_model_version,
    }
    write_json(output_dir / "validation_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    runtime = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("dados/examples/sources.json")
    )
    parser.add_argument("--model", type=Path, default=runtime.model_path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados/local/validation_examples_v3_confirmed"),
    )
    parser.add_argument("--conf", type=float, default=runtime.confidence_threshold)
    parser.add_argument("--iou", type=float, default=runtime.iou_threshold)
    parser.add_argument("--device", default=runtime.device)
    parser.add_argument("--imgsz", type=validate_image_size, default=runtime.image_size)
    parser.add_argument("--water-model", type=Path, default=runtime.water_model_path)
    parser.add_argument(
        "--no-water",
        action="store_true",
        help="Desativa o segmentador para reproduzir comparações históricas; demais limiares são explícitos",
    )
    parser.add_argument(
        "--water-pixel-confidence", type=float, default=runtime.water_pixel_confidence
    )
    parser.add_argument(
        "--water-min-area-fraction", type=float, default=runtime.water_min_area_fraction
    )
    parser.add_argument(
        "--water-mode", choices=("augment", "replace"), default=runtime.water_mode
    )
    parser.add_argument(
        "--residue-conf", type=float, default=runtime.confidence_threshold
    )
    parser.add_argument(
        "--animal-conf", type=float, default=runtime.animal_confidence_threshold
    )
    parser.add_argument(
        "--expected-model-version",
        help="Interrompe antes da inferência se o perfil congelado divergir",
    )
    parser.add_argument(
        "--expected-video-model-version",
        help="Exige o identificador congelado da política temporal de vídeo",
    )
    parser.add_argument(
        "--water-confirmation-seconds",
        type=float,
        default=runtime.video_water_confirmation_seconds,
        help="Confirmação temporal somente para vídeo; zero reproduz o resultado bruto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_examples(
        args.catalog,
        args.model,
        args.output_dir,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        image_size=args.imgsz,
        water_model_path=args.water_model,
        no_water=args.no_water,
        water_pixel_confidence=args.water_pixel_confidence,
        water_min_area_fraction=args.water_min_area_fraction,
        water_mode=args.water_mode,
        residue_confidence=args.residue_conf,
        animal_confidence=args.animal_conf,
        expected_model_version=args.expected_model_version,
        expected_video_model_version=args.expected_video_model_version,
        water_confirmation_seconds=args.water_confirmation_seconds,
    )
    return int(any(item["status"] != "completed" for item in report["items"]))


if __name__ == "__main__":
    raise SystemExit(main())
