"""Run the real model on every public example and verify encoded video outputs."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio_ffmpeg  # noqa: E402

from drenavision.data.constants import CLASS_NAMES  # noqa: E402
from drenavision.inference.media_io import read_image_bgr, write_image  # noqa: E402
from drenavision.inference.predictor import YoloPredictor  # noqa: E402
from drenavision.inference.utils import sha256_file, write_json  # noqa: E402
from drenavision.inference.video import (  # noqa: E402
    annotate_video_frame,
    process_video,
)


def project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


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
    confidence: float = 0.35,
) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = YoloPredictor(checkpoint, confidence_threshold=confidence)
    items = []
    for entry in catalog["items"]:
        source = catalog_path.parent / entry["path"]
        output = output_dir / f"{entry['id']}_annotated{source.suffix}"
        row: dict[str, Any] = {
            "id": entry["id"],
            "kind": entry["kind"],
            "input": project_path(source),
            "input_sha256": sha256_file(source),
            "output": project_path(output),
            "content_tags": entry["content_tags"],
        }
        try:
            if entry["kind"] == "video":
                manifest = process_video(
                    source, output, checkpoint, confidence=confidence, track=True
                )
                row["counts"] = {
                    name: value["count"] for name, value in manifest["counts"].items()
                }
                row["processed_frames"] = manifest["video"]["processed_frames"]
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
            else:
                frame_bgr = read_image_bgr(source)
                result = predictor.predict(frame_bgr[:, :, ::-1].copy())
                write_image(output, annotate_video_frame(frame_bgr, result.detections))
                detections = [
                    item.model_dump(mode="json") for item in result.detections
                ]
                row["counts"] = {
                    name: sum(item["class_name"] == name for item in detections)
                    for name in CLASS_NAMES.values()
                }
                row["detections"] = detections
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
        write_json(output_dir / "validation_report.json", {"items": items})
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": sha256_file(checkpoint),
        "confidence_threshold": confidence,
        "iou_threshold": 0.45,
        "tracking": True,
        "note": (
            "Teste funcional externo sem caixas de referência revisadas. As tags "
            "descrevem o assunto, não ground truth. Classes de assunto ausentes "
            "são candidatas a falsos negativos visuais e exigem inspeção; "
            "previsões inesperadas exigem revisão de falsos positivos. "
            "Este relatório não estima precisão, recall ou mAP."
        ),
        "items": items,
    }
    write_json(output_dir / "validation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("dados/examples/sources.json")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("resultados/local/models/best.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("resultados/local/validation_examples")
    )
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    report = validate_examples(
        args.catalog, args.model, args.output_dir, confidence=args.conf
    )
    return int(any(item["status"] != "completed" for item in report["items"]))


if __name__ == "__main__":
    raise SystemExit(main())
