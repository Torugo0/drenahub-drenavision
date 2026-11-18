"""Validate the warmed shared API with original catalogue inputs, sequentially.

HTTPX, OpenCV and the standard library only: this client never imports project
model code or Torch, instantiates networks, or runs GPU inference itself.
Known catalogue sources are functional regression examples, not independent GT.
Use --execute only after the separate UI upload has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IMAGE_VERSION = "pipeline-4c5c993223219929"
CLASS_NAMES = {0: "obstrucao_visivel", 1: "agua_acumulada", 2: "animal_visivel"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def decode_video(path: Path, selected: set[int] | None = None) -> tuple[dict, dict]:
    """Decode to EOF, with an ASCII temporary path for Windows OpenCV codecs."""
    selected = selected or set()
    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="drenavision_client_decode_") as scratch:
        directory = Path(scratch).resolve()
        if directory.parent != temp_root or not directory.name.startswith(
            "drenavision_client_decode_"
        ):
            raise ValueError("Unexpected temporary decoding directory")
        local = directory / "input.mp4"
        shutil.copyfile(path, local)
        capture = cv2.VideoCapture(str(local))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open {path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        count, saved = 0, {}
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape[:2] != (height, width):
                    raise ValueError("Decoded frame dimensions changed mid-video")
                if count in selected:
                    saved[count] = image.copy()
                count += 1
        finally:
            capture.release()
        if not math.isfinite(fps) or fps <= 0 or count < 1 or count != declared:
            raise ValueError(
                f"Video did not decode completely: {path.name} ({count}/{declared})"
            )
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "declared_frames": declared,
        "decoded_frames": count,
        "duration_seconds": count / fps,
    }, saved


def image_dimensions(path: Path) -> tuple[int, int]:
    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="drenavision_client_image_") as scratch:
        directory = Path(scratch).resolve()
        if directory.parent != temp_root or not directory.name.startswith(
            "drenavision_client_image_"
        ):
            raise ValueError("Unexpected temporary image directory")
        local = directory / "input.jpg"
        shutil.copyfile(path, local)
        image = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"JPEG could not be decoded: {path.name}")
        height, width = image.shape[:2]
    return width, height


def mp4_structure(path: Path) -> dict:
    atoms = []
    total = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < total:
            position = stream.tell()
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated MP4 root atom")
            size, tag = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError("Truncated extended MP4 root atom")
                size, header_size = struct.unpack(">Q", extended)[0], 16
            if size == 0:
                size = total - position
            if size < header_size or position + size > total:
                raise ValueError("Invalid MP4 root atom size")
            atoms.append(tag)
            stream.seek(position + size)
    if b"moov" not in atoms or b"mdat" not in atoms:
        raise ValueError("MP4 lacks moov/mdat")
    data = path.read_bytes()
    if b"avcC" not in data or atoms.index(b"moov") > atoms.index(b"mdat"):
        raise ValueError("Video output is not H.264 faststart")
    return {"h264_avc_configuration_present": True, "faststart_moov_before_mdat": True}


def frame_selection(records: list[dict]) -> list[int]:
    """Prefer up to three real detected frames, then fill gaps with timeline frames."""
    detected = [row["frame_index"] for row in records if row["detections"]]
    pool = detected or [row["frame_index"] for row in records]
    chosen = list(dict.fromkeys(pool[round(i * (len(pool) - 1) / 2)] for i in range(3)))
    fallback = [
        records[round(i * (len(records) - 1) / 2)]["frame_index"] for i in range(3)
    ]
    for index in fallback:
        if len(chosen) >= 3:
            break
        if index not in chosen:
            chosen.append(index)
    return sorted(chosen)


def video_presence(records: list[dict]) -> dict:
    counts = Counter({name: 0 for name in CLASS_NAMES.values()})
    empty = 0
    for frame in records:
        classes = {item["class_id"] for item in frame["detections"]}
        empty += not bool(classes)
        for class_id in classes:
            counts[CLASS_NAMES[class_id]] += 1
    return {
        "frames_with_detection_by_class": dict(counts),
        "empty_frames": empty,
        "total_frames": len(records),
    }


def validate_video_manifest(manifest: dict, entry: dict, frozen: dict) -> str:
    if (
        manifest.get("status") != "completed"
        or manifest.get("base_pipeline_version") != EXPECTED_IMAGE_VERSION
    ):
        raise ValueError("Video did not use the completed frozen image pipeline")
    if manifest.get("input_sha256") != entry["sha256"]:
        raise ValueError("API input hash differs from the original catalogue file")
    pipeline = manifest["pipeline"]
    ownership = pipeline.get("model_ownership", "")
    if (
        not ownership.startswith("borrowed API ")
        or "water/object/verifier" not in ownership
    ):
        raise ValueError("Video does not declare borrowed shared API model instances")
    if (
        pipeline.get("objects") != frozen["objects"]
        or pipeline.get("water") != frozen["water"]
    ):
        raise ValueError(
            "Video specialist/water provenance differs from the frozen models"
        )
    if manifest["water_temporal_policy"]["confirmation_seconds"] != 0.5:
        raise ValueError("Expected frozen 0.5-second video water confirmation")
    if manifest["inference"]["video_water_confirmation_seconds"] != 0.5:
        raise ValueError("Video inference parameters have a different water policy")
    records = manifest["frames"]
    if (
        len(records) != entry["frames"]
        or manifest["video"]["processed_frames"] != entry["frames"]
    ):
        raise ValueError("The API truncated or changed the original frame count")
    for index, frame in enumerate(records):
        if frame["frame_index"] != index:
            raise ValueError("Video frame records must be complete and ordered")
        objects = frame.get("objects", {})
        if (
            objects.get("model_version") != frozen["objects"]["model_version"]
            or objects.get("processing") != "forward_on_current_original_frame"
        ):
            raise ValueError("A frame lacks a current-frame specialist forward record")
    created = datetime.fromisoformat(manifest["created_at_utc"].replace("Z", "+00:00"))
    freeze_time = datetime.fromisoformat(frozen["frozen_at_utc"].replace("Z", "+00:00"))
    if created < freeze_time:
        raise ValueError("The reused API/UI execution predates the frozen candidate")
    event_id = str(UUID(manifest.get("event_id") or manifest["event"]["event_id"]))
    if manifest.get("video_url") != f"/vision/videos/{event_id}":
        raise ValueError("Video URL differs from the canonical event UUID")
    return event_id


def download_video(
    client: httpx.Client, event_id: str, manifest: dict, destination: Path
) -> dict:
    endpoint = f"/vision/videos/{event_id}"
    with client.stream("GET", endpoint) as response:
        response.raise_for_status()
        if response.status_code != 200 or not response.headers.get(
            "content-type", ""
        ).startswith("video/mp4"):
            raise ValueError("Video GET must return MP4 and HTTP 200")
        with destination.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
    if digest(destination) != manifest["output_sha256"]:
        raise ValueError("Downloaded MP4 differs from the processed output hash")
    size = destination.stat().st_size
    upper = min(1023, size - 1)
    response = client.get(endpoint, headers={"Range": f"bytes=0-{upper}"})
    response.raise_for_status()
    if (
        response.status_code != 206
        or response.headers.get("content-range") != f"bytes 0-{upper}/{size}"
    ):
        raise ValueError(
            "Video byte-range request did not return the expected HTTP 206"
        )
    with destination.open("rb") as stream:
        expected = stream.read(upper + 1)
    if response.content != expected:
        raise ValueError("Range bytes differ from the actual downloaded MP4")
    return {
        "status_code": 206,
        "content_range": response.headers["content-range"],
        "verified_bytes": len(expected),
    }


def assert_decoded_metadata(actual: dict, expected: dict) -> None:
    for key in ("width", "height"):
        if actual[key] != expected[key]:
            raise ValueError(f"Decoded {key} differs from source/manifest")
    frames = expected.get(
        "processed_frames", expected.get("frames", expected.get("decoded_frames"))
    )
    if (
        actual["decoded_frames"] != frames
        or abs(actual["fps"] - expected["fps"]) > 1e-3
    ):
        raise ValueError("Decoded frame count or FPS differs from source/manifest")
    if (
        abs(actual["duration_seconds"] - expected["duration_seconds"])
        > 1 / actual["fps"] + 1e-3
    ):
        raise ValueError("Decoded duration differs from source/manifest")


def save_sheet(path: Path, frames: dict, manifest: dict) -> None:
    tiles = []
    for index, image in sorted(frames.items()):
        ratio = min(480 / image.shape[1], 340 / image.shape[0])
        tile = cv2.resize(
            image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio))
        )
        top, left = (340 - tile.shape[0]) // 2, (480 - tile.shape[1]) // 2
        tile = cv2.copyMakeBorder(
            tile,
            top + 40,
            340 - tile.shape[0] - top,
            left,
            480 - tile.shape[1] - left,
            cv2.BORDER_CONSTANT,
            value=(41, 23, 11),
        )
        count = len(manifest["frames"][index]["detections"])
        cv2.putText(
            tile,
            f"frame {index} / {index / manifest['video']['fps']:.2f}s / {count} marks",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    sheet = cv2.hconcat(tiles)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode the video review sheet")
    path.write_bytes(encoded.tobytes())


def validate_product(
    api_url: str,
    catalog_path: Path,
    output: Path,
    frozen_path: Path,
    reused_ui: Path | None,
) -> dict:
    catalog, frozen = read_json(catalog_path), read_json(frozen_path)
    if frozen["model_version"] != EXPECTED_IMAGE_VERSION:
        raise ValueError("Unexpected frozen candidate")
    seals = {
        "catalog_sha256": digest(catalog_path),
        "frozen_manifest_sha256": digest(frozen_path),
    }
    entries = catalog["items"]
    if reused_ui is not None:
        # Check/download the explicit fresh UI result before starting new jobs.
        entries = sorted(entries, key=lambda entry: entry["id"] != "rato_residuos")
    for entry in entries:
        source = catalog_path.parent / entry["path"]
        if (
            digest(source) != entry["sha256"]
            or entry.get("contains_model_predictions") is not False
        ):
            raise ValueError(
                f"Catalogue original differs or is pre-annotated: {entry['path']}"
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "run_seal.json",
        {
            **seals,
            "saved_before_HTTP_inference": True,
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    timeout = httpx.Timeout(1800.0, connect=15.0, pool=15.0)
    summary = {
        "status": "running",
        "scope": "Functional known-catalogue regression; some sources supplied training adaptation frames. Topic tags are not ground truth and no animal accuracy is inferred from them.",
        "model_version": EXPECTED_IMAGE_VERSION,
        "client_imports_no_model_runtime": True,
        "seals": seals,
        "entries": [],
    }
    with httpx.Client(
        base_url=api_url.rstrip("/"), timeout=timeout, trust_env=False
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("status") != "ok" or not health.json().get("model_loaded"):
            raise ValueError("The shared API is not warmed and ready")
        summary["health"] = health.json()
        for entry in entries:
            source = catalog_path.parent / entry["path"]
            started = time.perf_counter()
            print(
                json.dumps(
                    {"event": "starting", "input": entry["id"], "kind": entry["kind"]}
                ),
                flush=True,
            )
            response_path = output / f"{entry['id']}.response.json"
            reused = entry["id"] == "rato_residuos" and reused_ui is not None
            if reused:
                response_path.write_bytes(reused_ui.read_bytes())
            else:
                endpoint = (
                    "/vision/predict/video"
                    if entry["kind"] == "video"
                    else "/vision/predict"
                )
                with source.open("rb") as stream:
                    response = client.post(
                        endpoint,
                        files={
                            "file": (
                                source.name,
                                stream,
                                "video/mp4"
                                if entry["kind"] == "video"
                                else "image/jpeg",
                            )
                        },
                        data={"point_id": "Validação catálogo v0.5"},
                    )
                response.raise_for_status()
                response_path.write_bytes(response.content)
            result = read_json(response_path)
            row = {
                "id": entry["id"],
                "input": str(source.relative_to(ROOT)).replace("\\", "/"),
                "input_sha256": entry["sha256"],
                "kind": entry["kind"],
                "evaluation_role": entry.get("evaluation_role"),
                "topic_tags_not_ground_truth": entry.get("content_tags", []),
                "response_manifest": str(response_path.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "response_sha256": digest(response_path),
                "execution": "reused_fresh_UI_API_execution"
                if reused
                else "fresh_sequential_API_POST",
            }
            if reused:
                row["reused_UI_manifest"] = str(reused_ui.resolve())
                row["reused_UI_manifest_sha256"] = digest(reused_ui)
            if entry["kind"] == "video":
                event_id = validate_video_manifest(result, entry, frozen)
                target = output / f"{entry['id']}.mp4"
                range_result = download_video(client, event_id, result, target)
                source_metadata, _ = decode_video(source)
                selected = frame_selection(result["frames"])
                decoded, images = decode_video(target, set(selected))
                assert_decoded_metadata(source_metadata, entry)
                assert_decoded_metadata(decoded, entry)
                assert_decoded_metadata(decoded, result["video"])
                if len(images) != len(selected):
                    raise ValueError("A requested real review frame was not decoded")
                sheet = output / f"{entry['id']}_review.jpg"
                save_sheet(sheet, images, result)
                row.update(
                    {
                        "event_id": event_id,
                        "video_model_version": result["model_version"],
                        "base_pipeline_version": result["base_pipeline_version"],
                        "model_ownership": result["pipeline"]["model_ownership"],
                        "water_confirmation_seconds": 0.5,
                        "water_temporal_policy": result["water_temporal_policy"],
                        "processing_seconds": result["video"]["elapsed_seconds"],
                        "processing_fps": result["video"]["processing_fps"],
                        "decoded_output": decoded,
                        "decoded_source": source_metadata,
                        "bitstream": mp4_structure(target),
                        "range": range_result,
                        **video_presence(result["frames"]),
                        "review_frame_indices": selected,
                        "review_sheet": str(sheet.relative_to(ROOT)).replace("\\", "/"),
                        "output_mp4": str(target.relative_to(ROOT)).replace("\\", "/"),
                        "output_sha256": digest(target),
                    }
                )
            else:
                if (
                    result.get("model_version") != EXPECTED_IMAGE_VERSION
                    or result.get("input_type") != "image"
                ):
                    raise ValueError("Photo did not use the frozen image pipeline")
                event_id = str(UUID(result["event_id"]))
                evidence = client.get(f"/vision/evidence/{event_id}")
                evidence.raise_for_status()
                if not evidence.headers.get("content-type", "").startswith(
                    "image/jpeg"
                ):
                    raise ValueError("Photo evidence did not return JPEG")
                target = output / f"{entry['id']}.jpg"
                target.write_bytes(evidence.content)
                width, height = image_dimensions(target)
                if (width, height) != (entry["width"], entry["height"]):
                    raise ValueError("Photo evidence dimensions differ from source")
                counts = Counter(item["class_id"] for item in result["detections"])
                row.update(
                    {
                        "event_id": event_id,
                        "detections": result["detections"],
                        "detection_counts": {
                            name: counts[key] for key, name in CLASS_NAMES.items()
                        },
                        "inspection_recommended": result["inspection_recommended"],
                        "inference_ms": result["inference_ms"],
                        "evidence_jpeg": str(target.relative_to(ROOT)).replace(
                            "\\", "/"
                        ),
                        "evidence_sha256": digest(target),
                        "width": width,
                        "height": height,
                    }
                )
            event = client.get(f"/vision/events/{event_id}")
            event.raise_for_status()
            if event.json()["input_type"] != entry["kind"]:
                raise ValueError(
                    "Persisted event type differs from the uploaded source"
                )
            row["persisted_event_verified"] = True
            health = client.get("/health")
            health.raise_for_status()
            if health.json().get("status") != "ok" or not health.json().get("model_loaded"):
                raise ValueError("The shared API lost readiness between catalogue cases")
            row["health_after_case"] = health.json()
            row["client_elapsed_seconds"] = time.perf_counter() - started
            summary["entries"].append(row)
            write_json(output / "progress_summary.json", summary)
            print(
                json.dumps(
                    {
                        "event": "completed",
                        "input": entry["id"],
                        "client_elapsed_seconds": round(
                            row["client_elapsed_seconds"], 2
                        ),
                        "execution": row["execution"],
                    }
                ),
                flush=True,
            )
    if (
        len(summary["entries"]) != 8
        or sum(row.get("total_frames", 0) for row in summary["entries"]) != 1388
    ):
        raise ValueError(
            "Expected all eight catalogue inputs and all 1388 original video frames"
        )
    if any(
        digest(catalog_path.parent / entry["path"]) != entry["sha256"]
        for entry in entries
    ):
        raise ValueError("An original catalogue input changed during validation")
    if (
        digest(catalog_path) != seals["catalog_sha256"]
        or digest(frozen_path) != seals["frozen_manifest_sha256"]
    ):
        raise ValueError("Catalogue/frozen manifest changed during validation")
    summary.update(
        status="completed",
        completed_at_utc=datetime.now(UTC).isoformat(),
        decoded_video_frames=1388,
    )
    write_json(output / "product_validation.json", summary)
    write_json(ROOT / "resultados/metrics/v4/product_validation.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "dados/examples/sources.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "resultados/local/v4/product_validation",
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=ROOT / "resultados/metrics/v4/frozen_candidate.json",
    )
    parser.add_argument("--reuse-ui-manifest", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly start sequential API uploads; otherwise print the plan without requests",
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "plan_only_no_requests_no_models",
                    "api": args.api_url,
                    "inputs": [
                        {"id": entry["id"], "kind": entry["kind"]}
                        for entry in read_json(args.catalog)["items"]
                    ],
                    "reuse_UI_manifest": str(args.reuse_ui_manifest)
                    if args.reuse_ui_manifest
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = validate_product(
        args.api_url,
        args.catalog.resolve(),
        args.output_dir.resolve(),
        args.frozen_manifest.resolve(),
        args.reuse_ui_manifest,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "entries": len(result["entries"]),
                "decoded_video_frames": result["decoded_video_frames"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
