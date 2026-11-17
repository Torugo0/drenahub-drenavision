"""Calibrate one pinned SegFormer B1 on sealed, image-level VAL references.

No model search or test inputs: one bounded neural forward per clear VAL image,
then an explicit pixel/area grid. The chosen water fusion policy is selected
only from this validation reference, before any held-out test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.inference.utils import utc_now_iso, write_json  # noqa: E402
from drenavision.inference.water import (  # noqa: E402
    SemanticWaterPredictor,
    regions_from_maps,
)

SEALED_REFERENCE_SHA256 = (
    "9735e2e01718608fe07389d0dc4c487b01217c30a97d14bd9d2740682f97be7c"
)
PIXEL_GRID = (0.4, 0.5, 0.6, 0.7)
AREA_GRID = (0.001, 0.003, 0.01)
EXTENDED_PIXEL_GRID = PIXEL_GRID + (0.8, 0.9, 0.95)
EXTENDED_AREA_GRID = AREA_GRID + (0.02, 0.03, 0.05)
SPECIFICITY_TARGET = 0.90
REPOSITORY = "nvidia/segformer-b1-finetuned-ade-512-512"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def presence_metrics(truth: list[bool], predicted: list[bool]) -> dict[str, Any]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError(
            "Presence truth and predictions must be nonempty equal lengths"
        )
    tp = sum(a and b for a, b in zip(truth, predicted, strict=True))
    fp = sum(not a and b for a, b in zip(truth, predicted, strict=True))
    fn = sum(a and not b for a, b in zip(truth, predicted, strict=True))
    tn = sum(not a and not b for a, b in zip(truth, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "images": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _review(path: Path, dataset_root: Path) -> tuple[dict, list[dict]]:
    if digest(path) != SEALED_REFERENCE_SHA256:
        raise ValueError(
            "Sealed VAL presence review changed; do not tune labels to predictions"
        )
    review = json.loads(path.read_text("utf-8"))
    if review["split"] != "val" or len(review["records"]) != 60:
        raise ValueError("Only the sealed60-image VAL review is supported")
    if digest(dataset_root / "data_manifest.csv") != review["dataset_manifest_sha256"]:
        raise ValueError("VAL dataset manifest differs from the blinded reference")
    records = []
    for record in review["records"]:
        image_path = Path(record["image_path"])
        if image_path.parts[:2] != ("images", "val") or image_path.is_absolute():
            raise ValueError("Only VAL images may be calibrated")
        if digest(dataset_root / image_path) != record["image_sha256"]:
            raise ValueError(f"Image changed: {image_path}")
        if record["water_presence"] not in {"yes", "no", "uncertain"}:
            raise ValueError("Invalid sealed presence label")
        if record["water_presence"] != "uncertain":
            records.append(record)
    if Counter(r["water_presence"] for r in records) != {"yes": 24, "no": 35}:
        raise ValueError("Expected24 positive and35 negative clear VAL images")
    return review, records


def _yolo_presence(path: Path, records: list[dict]) -> tuple[dict, list[bool]]:
    benchmark = json.loads(path.read_text("utf-8"))
    if benchmark["protocol"]["split"] != "val":
        raise ValueError("YOLO comparison must use VAL")
    if benchmark["protocol"]["confidence"] != 0.35:
        raise ValueError("YOLO operating confidence must remain0.35")
    variant = next(v for v in benchmark["variants"] if v["name"] == "v3_s640_full")
    if variant["imgsz"] != 640 or variant["tile_size"] != 0:
        raise ValueError("YOLO comparison must be full-frame640")
    samples = {sample["id"]: sample for sample in benchmark["samples"]}
    rows = {row["sample_id"]: row for row in variant["rows"]}
    present = []
    for record in records:
        key = "val/" + Path(record["filename"]).stem
        if samples[key]["sha256"] != record["image_sha256"]:
            raise ValueError("YOLO comparison image identity mismatch")
        present.append(
            any(
                prediction["class_id"] == 1 and prediction["confidence"] >= 0.35
                for prediction in rows[key]["predictions"]
            )
        )
    return {
        "benchmark_sha256": digest(path),
        "variant": variant["name"],
        "checkpoint": variant["checkpoint"],
        "checkpoint_sha256": variant["checkpoint_sha256"],
        "confidence": 0.35,
        "nms_iou": benchmark["protocol"]["nms_iou"],
        "imgsz": 640,
    }, present


def choose_candidate(candidates: list[dict]) -> dict | None:
    eligible = [
        c for c in candidates if c["metrics"]["specificity"] >= SPECIFICITY_TARGET
    ]
    if not eligible:
        return None
    # Declared before inference: F1 first with a specificity floor. Exact score
    # ties prefer specificity/precision, replacement simplicity and stricter gates.
    return max(
        eligible,
        key=lambda c: (
            c["metrics"]["f1"],
            c["metrics"]["specificity"],
            c["metrics"]["precision"],
            c["fusion_policy"] == "replace_yolo_water",
            c["pixel_confidence"],
            c["min_area_fraction"],
        ),
    )


def calibrate(args: argparse.Namespace) -> dict:
    review, records = _review(args.presence_review, args.dataset_root)
    yolo_provenance, yolo_present = _yolo_presence(args.yolo_benchmark, records)
    if (args.output / "calibration_report.json").exists():
        raise ValueError("Calibration already complete; select a new output directory")
    predictor = SemanticWaterPredictor(
        args.model_dir,
        device=args.device,
        processing_size=512,
        pixel_confidence=0.5,
        min_area_fraction=0.001,
    )
    provenance = predictor.provenance
    if provenance["repository"] != REPOSITORY:
        raise ValueError("Only the pinned B1 experiment is permitted; no model search")
    truth = [r["water_presence"] == "yes" for r in records]
    pixel_grid = EXTENDED_PIXEL_GRID if args.extended_grid else PIXEL_GRID
    area_grid = EXTENDED_AREA_GRID if args.extended_grid else AREA_GRID
    cache_root = args.cache_source or args.output
    cache_dir = cache_root / "bounded_maps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "sealed_reference_sha256": digest(args.presence_review),
        "model_provenance": provenance,
        "processing_size": 512,
        "maps": "unthresholded bounded water-family probability plus semantic-water argmax mask",
        "device_requested": args.device,
    }
    identity_path = args.output / "cache_identity.json"
    source_cache_hashes = {}
    if args.cache_source:
        previous_identity = json.loads(
            (args.cache_source / "cache_identity.json").read_text("utf-8")
        )
        if previous_identity != identity:
            raise ValueError(
                "Source cached maps have different reference/model/profile"
            )
        previous_report = json.loads(
            (args.cache_source / "calibration_report.json").read_text("utf-8")
        )
        source_cache_hashes = {
            r["image_sha256"]: r["cache_sha256"] for r in previous_report["cache_rows"]
        }
    if identity_path.exists():
        if json.loads(identity_path.read_text("utf-8")) != identity:
            raise ValueError(
                "Cached maps belong to a different model/reference/profile"
            )
    else:
        write_json(identity_path, identity)
    cache_rows = []
    largest_components = []
    started = time.perf_counter()
    forward_count = 0
    for index, record in enumerate(records):
        image_path = args.dataset_root / record["image_path"]
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        height, width = image.shape[:2]
        cached = cache_dir / f"{record['image_sha256']}.npz"
        if args.cache_source and (
            not cached.is_file()
            or digest(cached) != source_cache_hashes.get(record["image_sha256"])
        ):
            raise ValueError(
                "Source maps missing/changed; cached-only iteration forbids new inference"
            )
        if not cached.exists():
            probability, labels = predictor.infer_maps(image, original_resolution=False)
            if probability.shape != (512, 512) or labels.shape != (512, 512):
                raise ValueError("Calibration cache must preserve bounded512 maps")
            np.savez_compressed(
                cached,
                water_probability=probability.astype(np.float32),
                water_labels=labels.astype(np.bool_),
            )
            forward_count += 1
        with np.load(cached, allow_pickle=False) as maps:
            bounded_probability = maps["water_probability"]
            bounded_labels = maps["water_labels"]
        if (
            bounded_probability.shape != (512, 512)
            or not np.isfinite(bounded_probability).all()
        ):
            raise ValueError("Invalid cached bounded probability map")
        probability = cv2.resize(
            bounded_probability, (width, height), interpolation=cv2.INTER_LINEAR
        )
        labels = cv2.resize(
            bounded_labels.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        row = {"pixels": width * height, "by_pixel_confidence": {}}
        for threshold in pixel_grid:
            pixels = labels & (probability >= threshold)
            count, _, statistics, _ = cv2.connectedComponentsWithStats(
                pixels.astype(np.uint8), connectivity=8
            )
            maximum = int(statistics[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
            row["by_pixel_confidence"][str(threshold)] = maximum
        largest_components.append(row)
        cache_rows.append(
            {
                **record,
                "original_width": width,
                "original_height": height,
                "yolo_water_present": yolo_present[index],
                "cache_file": os.path.relpath(cached, args.output).replace("\\", "/"),
                "cache_sha256": digest(cached),
                "max_component_pixels_by_threshold": row["by_pixel_confidence"],
            }
        )
        print(f"VAL cached {index + 1}/{len(records)}", flush=True)
    candidates = []
    for threshold in pixel_grid:
        for minimum in area_grid:
            semantic = [
                row["by_pixel_confidence"][str(threshold)]
                >= math.ceil(row["pixels"] * minimum)
                for row in largest_components
            ]
            for policy in ("replace_yolo_water", "augment_yolo_water_union"):
                present = (
                    semantic
                    if policy == "replace_yolo_water"
                    else [a or b for a, b in zip(semantic, yolo_present, strict=True)]
                )
                candidates.append(
                    {
                        "pixel_confidence": threshold,
                        "min_area_fraction": minimum,
                        "fusion_policy": policy,
                        "metrics": presence_metrics(truth, present),
                    }
                )
    chosen = choose_candidate(candidates)
    selected_rows = []
    if chosen is not None:
        masks_dir = args.output / "chosen_val_masks"
        masks_dir.mkdir(exist_ok=True)
        for record in cache_rows:
            with np.load(
                args.output / record["cache_file"], allow_pickle=False
            ) as maps:
                probability = cv2.resize(
                    maps["water_probability"],
                    (record["original_width"], record["original_height"]),
                    interpolation=cv2.INTER_LINEAR,
                )
                labels = cv2.resize(
                    maps["water_labels"].astype(np.uint8),
                    (record["original_width"], record["original_height"]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            detections, mask, regions = regions_from_maps(
                probability,
                labels,
                chosen["pixel_confidence"],
                chosen["min_area_fraction"],
            )
            mask_path = masks_dir / (Path(record["filename"]).stem + ".png")
            Image.fromarray(mask).save(mask_path)
            semantic = bool(detections)
            final_presence = (
                semantic
                if chosen["fusion_policy"] == "replace_yolo_water"
                else semantic or record["yolo_water_present"]
            )
            selected_rows.append(
                {
                    "filename": record["filename"],
                    "image_sha256": record["image_sha256"],
                    "truth": record["water_presence"],
                    "semantic_water_present": semantic,
                    "yolo_water_present": record["yolo_water_present"],
                    "pipeline_water_present": final_presence,
                    "mask_file": mask_path.relative_to(args.output).as_posix(),
                    "mask_sha256": digest(mask_path),
                    "mask_pixels": int(np.count_nonzero(mask)),
                    "regions": regions,
                }
            )
        selected_profile = SemanticWaterPredictor(
            args.model_dir,
            device=args.device,
            processing_size=512,
            pixel_confidence=chosen["pixel_confidence"],
            min_area_fraction=chosen["min_area_fraction"],
        ).provenance
    else:
        selected_profile = None
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "status": "completed",
        "protocol": "B1 only; sealed59 clear VAL image-level presence references; no test or public-regression tuning",
        "presence_reference": {
            "path": args.presence_review.as_posix(),
            "sha256": digest(args.presence_review),
            "counts": review["counts"],
        },
        "excluded_uncertain_images": [
            r for r in review["records"] if r["water_presence"] == "uncertain"
        ],
        "model_provenance": provenance,
        "selected_model_provenance": selected_profile,
        "yolo_comparison": yolo_provenance,
        "yolo_only_metrics": presence_metrics(truth, yolo_present),
        "grid": {
            "pixel_confidence": pixel_grid,
            "min_area_fraction": area_grid,
            "processing_size": 512,
            "morphology": "none",
        },
        "selection_rule": "Specificity>=0.90; maximize presenceF1, then specificity, precision, replacement simplicity, higher pixel threshold and larger minimum component area. If none meet specificity floor, no acceptable profile is selected.",
        "candidates": candidates,
        "chosen": chosen,
        "eligible_candidate_count": sum(
            c["metrics"]["specificity"] >= SPECIFICITY_TARGET for c in candidates
        ),
        "cache_rows": cache_rows,
        "selected_rows": selected_rows,
        "new_neural_forwards": forward_count,
        "cached_source_iteration": (
            {
                "source": args.cache_source.as_posix(),
                "prior_report_sha256": digest(
                    args.cache_source / "calibration_report.json"
                ),
                "prior_chosen": previous_report["chosen"],
                "purpose": "Additional validation-only parameter calibration; first grid preserved unchanged; sealed test and reference labels unchanged",
            }
            if args.cache_source
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Presence is image-level, assistant-reviewed validation truth; this does not establish pixel IoU, bounding-box IoU or calibrated risk.",
            "Parameter/model-fusion selection makes these validation scores optimistic; the119-image sealed test remains separate and unused here.",
            "No morphology, gap filling, per-image threshold or known-regression model selection is applied.",
            "Water components are predicted surface regions, not distinct puddles, water volume or sanitary risk.",
        ],
    }
    write_json(args.output / "calibration_report.json", report)
    print(
        json.dumps(
            {"chosen": chosen, "yolo_only": report["yolo_only_metrics"]}, indent=2
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presence-review",
        type=Path,
        default=ROOT / "resultados/metrics/v3/val_water_presence_review.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dados/processed/drenavision_v3_clean",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "resultados/local/models/water_segformer_b1_ade",
    )
    parser.add_argument(
        "--yolo-benchmark",
        type=Path,
        default=ROOT
        / "resultados/local/benchmark_v3/validation_candidate_v3_clean/benchmark.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "resultados/local/water_calibration_v3"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--extended-grid",
        action="store_true",
        help="Documented second VAL iteration: pixel gates through0.95, region areas through0.05",
    )
    parser.add_argument(
        "--cache-source",
        type=Path,
        help="Reuse an immutable completed calibration cache; missing/changed maps fail instead of running new forwards",
    )
    return parser


if __name__ == "__main__":
    calibrate(build_parser().parse_args())
