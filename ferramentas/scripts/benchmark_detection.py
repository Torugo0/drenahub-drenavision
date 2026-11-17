"""Compare actual detections on saved split images and complete video frames.

This is an operating-point audit, not mAP evaluation. Product topic tags are never
converted into ground truth. Use validation, rather than test, to select changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.data.constants import CLASS_NAMES  # noqa: E402
from drenavision.data.yolo import read_yolo  # noqa: E402
from drenavision.inference.device import describe_device, select_device  # noqa: E402
from drenavision.inference.media_io import read_image_bgr, write_image  # noqa: E402
from drenavision.inference.predictor import detections_from_result  # noqa: E402
from drenavision.inference.utils import (  # noqa: E402
    box_iou,
    greedy_class_matches,
    label_path_for_image,
    preflight_dataset,
    safe_artifact_path,
    sha256_file,
    utc_now_iso,
    validate_class_map,
    write_json,
    xywhn_to_xyxy,
)


def stratified_paths(paths: list[Path], per_class: int, seed: int) -> list[Path]:
    """Select before inference by class presence; include labelled negatives."""

    if per_class <= 0:
        return sorted(paths)
    strata: dict[int, list[Path]] = {key: [] for key in (*CLASS_NAMES, -1)}
    for path in paths:
        labels = read_yolo(label_path_for_image(path), class_count=len(CLASS_NAMES))
        for class_id in {item.class_id for item in labels} or {-1}:
            strata[class_id].append(path)

    def order(path: Path) -> str:
        return hashlib.sha256(f"{seed}/{path.name}".encode()).hexdigest()

    return sorted(
        {
            path
            for group in strata.values()
            for path in sorted(group, key=order)[:per_class]
        }
    )


def split_samples(data: Path, split: str, per_class: int, seed: int) -> list[dict]:
    preflight = preflight_dataset(data, (split,))
    paths = stratified_paths(preflight["images"][split], per_class, seed)
    manifest = data.parent / "data_manifest.csv"
    provenance = {}
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            provenance = {
                Path(row["output_image"]).name: row for row in csv.DictReader(stream)
            }
    samples = []
    for path in paths:
        image = read_image_bgr(path)
        height, width = image.shape[:2]
        label_path = label_path_for_image(path)
        # A missing label raises, instead of being silently treated as a negative.
        truth = [
            {
                "class_id": box.class_id,
                "bounding_box": xywhn_to_xyxy(
                    box.x_center, box.y_center, box.width, box.height, width, height
                ),
            }
            for box in read_yolo(label_path, class_count=len(CLASS_NAMES))
        ]
        row = provenance.get(path.name, {})
        samples.append(
            {
                "id": f"{split}/{path.stem}",
                "group": "labelled_split",
                "input": safe_artifact_path(path),
                "read_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "split": split,
                "source_key": row.get("source_key"),
                "source_group": row.get("source_group"),
                "annotation_note": row.get("annotation_note"),
                "label_sha256": sha256_file(label_path),
                "truth": truth,
            }
        )
    return samples


def frame_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0 or count <= 0:
        return []
    if count == 1:
        return [0]
    return sorted(
        {round(index * (frame_count - 1) / (count - 1)) for index in range(count)}
    )


def catalog_samples(
    catalog_path: Path, output_dir: Path, count: int
) -> tuple[list[dict], list[dict]]:
    import cv2

    samples, videos = [], []
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for entry in catalog["items"]:
        path = catalog_path.parent / entry["path"]
        source_hash = sha256_file(path)
        base = {
            "group": "product_without_reference_boxes",
            "input": safe_artifact_path(path),
            "sha256": source_hash,
            "catalog_id": entry["id"],
            "evaluation_role": entry.get("evaluation_role"),
            "topic_tags_not_ground_truth": entry.get("content_tags", []),
            "truth": None,
        }
        if entry["kind"] == "image":
            image = read_image_bgr(path)
            height, width = image.shape[:2]
            samples.append(
                {
                    **base,
                    "id": f"product/{entry['id']}",
                    "read_path": str(path.resolve()),
                    "width": width,
                    "height": height,
                }
            )
            continue
        capture = cv2.VideoCapture(str(path.resolve()))
        if not capture.isOpened():
            raise ValueError(f"video cannot be opened: {path.name}")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if expected <= 0 or fps <= 0:
            capture.release()
            raise ValueError(f"video has invalid frame metadata: {path.name}")
        targets = set(frame_indices(expected, count))
        decoded, selected = 0, []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if decoded in targets:
                destination = output_dir / "inputs" / f"{entry['id']}_{decoded:06d}.png"
                write_image(destination, image)
                height, width = image.shape[:2]
                samples.append(
                    {
                        **base,
                        "id": f"product/{entry['id']}/{decoded:06d}",
                        "read_path": str(destination.resolve()),
                        "frame_index": decoded,
                        "timestamp_seconds": decoded / fps,
                        "width": width,
                        "height": height,
                        "decoded_pixels_sha256": hashlib.sha256(
                            image.tobytes()
                        ).hexdigest(),
                    }
                )
                selected.append(decoded)
            decoded += 1
        capture.release()
        if decoded != expected or set(selected) != targets:
            raise ValueError(
                f"incomplete video decoding {path.name}: {decoded}/{expected}"
            )
        videos.append(
            {
                "id": entry["id"],
                "input": safe_artifact_path(path),
                "sha256": source_hash,
                "fps": fps,
                "metadata_frames": expected,
                "decoded_frames": decoded,
                "sampled_frame_indices": selected,
                "sampling": "uniform full frames; no crops",
            }
        )
    return samples, videos


def tile_starts(length: int, size: int, overlap: float) -> list[int]:
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile size must be positive and overlap in [0, 1)")
    if length <= size:
        return [0]
    stride = max(1, round(size * (1 - overlap)))
    return sorted({*range(0, length - size + 1, stride), length - size})


def merge_predictions(predictions: list[dict], iou: float) -> list[dict]:
    """Class-aware NMS in original image coordinates, including tile offsets."""

    kept = []
    for item in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
        if not any(
            item["class_id"] == other["class_id"]
            and box_iou(item["bounding_box"], other["bounding_box"]) > iou
            for other in kept
        ):
            kept.append(item)
    return kept


def predict(
    model: Any,
    image: Any,
    device: str | int,
    imgsz: int,
    confidence: float,
    iou: float,
    tile_size: int,
    overlap: float,
) -> list[dict]:
    height, width = image.shape[:2]
    crops = [(0, 0, width, height)]
    if tile_size and (width > tile_size or height > tile_size):
        crops += [
            (x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in tile_starts(height, tile_size, overlap)
            for x in tile_starts(width, tile_size, overlap)
        ]
    predictions = []
    for x1, y1, x2, y2 in crops:
        results = model.predict(
            image[y1:y2, x1:x2],
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            device=device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("model returned no result object")
        for detection in detections_from_result(results[0], x2 - x1, y2 - y1):
            a, b, c, d = detection.bounding_box
            predictions.append(
                {
                    "class_id": detection.class_id,
                    "confidence": detection.confidence,
                    "bounding_box": [a + x1, b + y1, c + x1, d + y1],
                }
            )
    return merge_predictions(predictions, iou) if tile_size else predictions


def predict_pipeline(pipeline: Any, image_bgr: Any) -> tuple[list[dict], Any]:
    """Use the exact application pipeline without resizing or altering its boxes."""
    result = pipeline.predict(image_bgr[:, :, ::-1].copy())
    predictions = [
        {
            "class_id": item.class_id,
            "confidence": item.confidence,
            "bounding_box": list(item.bounding_box),
        }
        for item in result.detections
    ]
    return predictions, result.annotated_image[:, :, ::-1].copy()


def score(truth: list[dict], predictions: list[dict], match_iou: float) -> dict:
    gt = [(row["class_id"], row["bounding_box"]) for row in truth]
    pred = [
        (row["class_id"], row["confidence"], row["bounding_box"]) for row in predictions
    ]
    matched_gt, matched_pred = greedy_class_matches(gt, pred, match_iou)
    counts = {}
    for class_id, name in CLASS_NAMES.items():
        tp = sum(pred[index][0] == class_id for index in matched_pred)
        fp = sum(
            item[0] == class_id
            for index, item in enumerate(pred)
            if index not in matched_pred
        )
        fn = sum(
            item[0] == class_id
            for index, item in enumerate(gt)
            if index not in matched_gt
        )
        counts[name] = {"tp": tp, "fp": fp, "fn": fn}
    return {
        "per_class": counts,
        "matched_truth_indexes": sorted(matched_gt),
        "matched_prediction_indexes": sorted(matched_pred),
    }


def totals(rows: list[dict]) -> dict:
    aggregate = {name: Counter(tp=0, fp=0, fn=0) for name in CLASS_NAMES.values()}
    for row in rows:
        if row["score"] is not None:
            for name, counts in row["score"]["per_class"].items():
                aggregate[name].update(counts)
    return {
        name: {
            **counts,
            "precision": counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"]
            else None,
            "recall": counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"]
            else None,
            "f1": 2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])
            if 2 * counts["tp"] + counts["fp"] + counts["fn"]
            else None,
        }
        for name, counts in aggregate.items()
    }


def contact_sheets(
    samples: list[dict],
    rows: list[dict],
    output_dir: Path,
    annotated_paths: dict[str, Path] | None = None,
) -> list[str]:
    import cv2
    import numpy as np

    pages = []
    cell_width, cell_height = 400, 300
    for offset in range(0, len(rows), 12):
        page_rows = (min(12, len(rows) - offset) + 2) // 3
        page = np.full(
            (cell_height * page_rows, cell_width * 3, 3), 246, dtype=np.uint8
        )
        for position, (sample, row) in enumerate(
            zip(samples[offset : offset + 12], rows[offset : offset + 12], strict=True)
        ):
            annotated_path = (annotated_paths or {}).get(sample["id"])
            image = (
                read_image_bgr(annotated_path)
                if annotated_path
                else read_image_bgr(sample["read_path"])
            )
            scale = min(
                cell_width / image.shape[1], (cell_height - 52) / image.shape[0]
            )
            resized = cv2.resize(image, None, fx=scale, fy=scale)
            matched_gt = (
                set(row["score"]["matched_truth_indexes"]) if row["score"] else set()
            )
            matched_pred = (
                set(row["score"]["matched_prediction_indexes"])
                if row["score"]
                else set()
            )
            for is_truth, boxes in (
                (True, sample["truth"] or []),
                (False, [] if annotated_path else row["predictions"]),
            ):
                for index, item in enumerate(boxes):
                    if is_truth:
                        color = (40, 170, 40) if index in matched_gt else (20, 20, 230)
                        prefix = "GT" if index in matched_gt else "FN"
                    else:
                        color = (
                            (0, 160, 255)
                            if row["score"] is None or index in matched_pred
                            else (200, 0, 200)
                        )
                        prefix = (
                            "P"
                            if row["score"] is None
                            else ("TP" if index in matched_pred else "FP")
                        )
                    x1, y1, x2, y2 = [
                        round(value * scale) for value in item["bounding_box"]
                    ]
                    cv2.rectangle(resized, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(
                        resized,
                        f"{prefix}:{item['class_id']}",
                        (x1, max(11, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            top, left = position // 3 * cell_height, position % 3 * cell_width
            page[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
            title = sample["id"].split("/")[-1]
            cv2.putText(
                page,
                title[:48],
                (left + 4, top + cell_height - 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            detail = (
                f"GT={len(sample['truth'])}"
                if sample["truth"] is not None
                else "NO REFERENCE BOXES"
            )
            cv2.putText(
                page,
                f"{detail}  predictions={len(row['predictions'])}",
                (left + 4, top + cell_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
        path = output_dir / f"contact_{offset // 12 + 1:02d}.jpg"
        write_image(path, page)
        pages.append(safe_artifact_path(path))
    return pages


def demo_reference_coverage(samples: list[dict]) -> list[dict]:
    by_hash = {sample["sha256"]: sample["id"] for sample in samples}
    references = []
    for path in sorted((PROJECT_ROOT / "resultados/demo").glob("*_prediction.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        references.append(
            {
                "saved_prediction": safe_artifact_path(path),
                "input_name": record.get("input_name"),
                "input_sha256": record.get("input_sha256"),
                "covered_sample_id": by_hash.get(record.get("input_sha256")),
                "note": "Original input is tested once. Annotated output is never fed back as input.",
            }
        )
    return references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    parser.add_argument(
        "--data", type=Path, default=Path("dados/processed/drenavision_v2/dataset.yaml")
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="0: all images; >0: deterministic class strata + negatives",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--catalog", type=Path, default=Path("dados/examples/sources.json")
    )
    catalog_mode = parser.add_mutually_exclusive_group()
    catalog_mode.add_argument("--no-catalog", action="store_true")
    catalog_mode.add_argument("--only-catalog", action="store_true")
    parser.add_argument("--video-frames", type=int, default=6)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Also compare full-frame + overlapping tiles",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--water-model", type=Path)
    parser.add_argument("--water-pixel-confidence", type=float, default=0.5)
    parser.add_argument("--water-min-area-fraction", type=float, default=0.001)
    parser.add_argument(
        "--water-mode", choices=("augment", "replace"), default="augment"
    )
    parser.add_argument("--animal-conf", type=float)
    parser.add_argument("--residue-conf", type=float)
    parser.add_argument(
        "--expected-profile",
        help="Fail before predictions if the shared runtime version differs",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save original-resolution actual prediction renderings",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime_enabled = (
        args.water_model is not None
        or args.animal_conf is not None
        or args.residue_conf is not None
    )
    if runtime_enabled and args.tile_size:
        parser.error(
            "tiled composed-runtime benchmarks are unsupported; use complete images"
        )
    if args.expected_profile and not runtime_enabled:
        parser.error("expected-profile requires the shared runtime controls")
    for key in (
        "water_pixel_confidence",
        "water_min_area_fraction",
        "animal_conf",
        "residue_conf",
    ):
        value = getattr(args, key)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{key} must be a finite probability")
    if not 0 <= args.conf <= 1 or not 0 < args.match_iou <= 1 or not 0 < args.iou <= 1:
        parser.error("conf, iou and match-iou must be valid probabilities")
    if (
        args.tile_size < 0
        or not 0 <= args.tile_overlap < 1
        or any(value < 32 for value in args.imgsz)
    ):
        parser.error("invalid image or tile size/overlap")
    if args.video_frames < 1 or args.per_class < 0:
        parser.error("video-frames must be positive and per-class nonnegative")
    checkpoints = []
    for value in args.model:
        label, separator, path_text = value.partition("=")
        if (
            not separator
            or not label
            or not path_text
            or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for char in label
            )
        ):
            parser.error("model must be LABEL=PATH with a filesystem-safe label")
        path = Path(path_text)
        if not path.is_file():
            parser.error(f"checkpoint missing: {path}")
        if label in {item[0] for item in checkpoints}:
            parser.error(f"duplicate model label: {label}")
        checkpoints.append((label, path))
    if (args.output_dir / "benchmark.json").exists():
        parser.error("output already contains a benchmark; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = (
        []
        if args.only_catalog
        else split_samples(args.data, args.split, args.per_class, args.seed)
    )
    videos = []
    if not args.no_catalog:
        product, videos = catalog_samples(
            args.catalog, args.output_dir, args.video_frames
        )
        samples.extend(product)
    if not samples:
        parser.error("no input samples selected")
    report = {
        "created_at_utc": utc_now_iso(),
        "status": "running",
        "device": None,
        "coordinate_format": "xyxy pixels in the original complete image/frame",
        "protocol": {
            "split": None if args.only_catalog else args.split,
            "selection": "all split images"
            if args.per_class == 0
            else "pre-inference deterministic strata",
            "per_class_limit": args.per_class,
            "seed": args.seed,
            "confidence": args.conf,
            "nms_iou": args.iou,
            "matching_iou": args.match_iou,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "prediction_engine": "application_runtime"
            if runtime_enabled
            else "pure_yolo",
            "water_model": safe_artifact_path(args.water_model)
            if args.water_model
            else None,
            "water_pixel_confidence": args.water_pixel_confidence
            if args.water_model
            else None,
            "water_min_area_fraction": args.water_min_area_fraction
            if args.water_model
            else None,
            "water_mode": args.water_mode if runtime_enabled else None,
            "animal_confidence": (0.6 if args.animal_conf is None else args.animal_conf)
            if runtime_enabled
            else None,
            "residue_confidence": (
                0.35 if args.residue_conf is None else args.residue_conf
            )
            if runtime_enabled
            else None,
            "expected_profile": args.expected_profile,
        },
        "caveats": [
            "TP/FP/FN compare predictions with the stored source annotations, not independently certified deployment truth.",
            "Single-source datasets can omit other categories present; unmatched detections require visual review before being called real-world false positives.",
            "Water annotations can describe containers/breeding sites rather than complete visible water surfaces; inspect source notes.",
            "Rodent source boxes derived from visible pose points can cover only part of the body.",
            "Product topic tags are not reference boxes and do not produce TP/FP/FN or accuracy claims.",
            "Detection counts are boxes per sampled frame, not unique animals or objects.",
            "Use validation for model selection; inspecting the test set makes future claims exploratory.",
        ],
        "sample_count": len(samples),
        "labelled_image_count": sum(item["truth"] is not None for item in samples),
        "product_frame_or_image_count": sum(item["truth"] is None for item in samples),
        "samples": [
            {key: value for key, value in item.items() if key != "read_path"}
            for item in samples
        ],
        "video_decoding": videos,
        "saved_demo_reference_coverage": demo_reference_coverage(samples),
        "variants": [],
    }
    write_json(args.output_dir / "benchmark.json", report)
    from ultralytics import YOLO

    device = select_device(args.device)
    report["device"] = describe_device(device)
    for label, checkpoint in checkpoints:
        model = None if runtime_enabled else YOLO(str(checkpoint))
        if model is not None:
            validate_class_map(model.names)
        checkpoint_hash = sha256_file(checkpoint)
        for imgsz in dict.fromkeys(args.imgsz):
            pipeline = None
            if runtime_enabled:
                from drenavision.inference.runtime import create_predictor

                pipeline = create_predictor(
                    checkpoint,
                    confidence_threshold=args.conf,
                    iou_threshold=args.iou,
                    image_size=imgsz,
                    device=args.device,
                    water_model_path=args.water_model,
                    water_pixel_confidence=args.water_pixel_confidence,
                    water_min_area_fraction=args.water_min_area_fraction,
                    water_mode=args.water_mode,
                    residue_confidence=0.35
                    if args.residue_conf is None
                    else args.residue_conf,
                    animal_confidence=0.6
                    if args.animal_conf is None
                    else args.animal_conf,
                )
                if not pipeline.is_ready:
                    raise RuntimeError(
                        f"configured application pipeline unavailable: {pipeline.detail}"
                    )
                if (
                    args.expected_profile
                    and pipeline.model_version != args.expected_profile
                ):
                    raise ValueError(
                        f"pipeline version {pipeline.model_version} differs from frozen {args.expected_profile}"
                    )
            for tile_size in [0, args.tile_size] if args.tile_size else [0]:
                name = f"{label}_s{imgsz}_{'full' if not tile_size else f'tile{tile_size}'}"
                rows = []
                annotated_paths = {}
                started = time.perf_counter()
                for sample in samples:
                    image = read_image_bgr(sample["read_path"])
                    annotated = None
                    if pipeline is not None:
                        predictions, annotated = predict_pipeline(pipeline, image)
                    else:
                        predictions = predict(
                            model,
                            image,
                            device,
                            imgsz,
                            args.conf,
                            args.iou,
                            tile_size,
                            args.tile_overlap,
                        )
                    row = {
                        "sample_id": sample["id"],
                        "predictions": predictions,
                        "score": score(sample["truth"], predictions, args.match_iou)
                        if sample["truth"] is not None
                        else None,
                    }
                    if args.save_annotated:
                        if annotated is None:
                            from drenavision.core.schemas import Detection
                            from drenavision.inference.video import annotate_video_frame

                            annotated = annotate_video_frame(
                                image,
                                [
                                    Detection(
                                        class_id=item["class_id"],
                                        class_name=CLASS_NAMES[item["class_id"]],
                                        confidence=item["confidence"],
                                        bounding_box=item["bounding_box"],
                                    )
                                    for item in predictions
                                ],
                            )
                        destination = (
                            args.output_dir
                            / name
                            / "predictions"
                            / (sample["id"].replace("/", "_") + ".jpg")
                        )
                        write_image(destination, annotated, jpeg_quality=92)
                        row["annotated_image"] = safe_artifact_path(destination)
                        annotated_paths[sample["id"]] = destination
                    rows.append(row)
                elapsed = time.perf_counter() - started
                per_class = totals(rows)
                class_f1 = [
                    item["f1"] for item in per_class.values() if item["f1"] is not None
                ]
                variant = {
                    "name": name,
                    "checkpoint": safe_artifact_path(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "model_version": pipeline.model_version
                    if pipeline is not None
                    else f"{checkpoint.stem}-{checkpoint_hash[:12]}",
                    "operating_parameters": dict(pipeline.parameters)
                    if pipeline is not None
                    else {"confidence": args.conf, "iou": args.iou, "imgsz": imgsz},
                    "water_provenance": pipeline.water.provenance
                    if pipeline is not None and pipeline.water is not None
                    else None,
                    "imgsz": imgsz,
                    "tile_size": tile_size,
                    "elapsed_seconds_inference_and_io": elapsed,
                    "per_class": per_class,
                    "macro_f1": sum(class_f1) / len(class_f1) if class_f1 else None,
                    "rows": rows,
                    "contact_sheets": contact_sheets(
                        samples, rows, args.output_dir / name, annotated_paths
                    ),
                }
                report["variants"].append(variant)
                write_json(args.output_dir / "benchmark.json", report)
                print(
                    json.dumps(
                        {
                            key: variant[key]
                            for key in (
                                "name",
                                "per_class",
                                "macro_f1",
                                "elapsed_seconds_inference_and_io",
                            )
                        }
                    ),
                    flush=True,
                )
    report["status"] = "completed"
    report["completed_at_utc"] = utc_now_iso()
    write_json(args.output_dir / "benchmark.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
