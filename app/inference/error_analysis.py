"""Mine genuine correct, false-positive, and false-negative test examples."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.data.constants import CLASS_NAMES
from app.data.yolo import read_yolo

from .media_io import read_image_bgr, write_image
from .utils import (
    greedy_class_matches,
    label_path_for_image,
    write_json,
    xywhn_to_xyxy,
)


def _tolist(value: Any) -> list[Any]:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _predictions(result: Any) -> list[tuple[int, float, tuple[float, ...]]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    return [
        (int(class_id), float(confidence), tuple(float(v) for v in coordinates))
        for class_id, confidence, coordinates in zip(
            _tolist(boxes.cls),
            _tolist(boxes.conf),
            _tolist(boxes.xyxy),
            strict=True,
        )
    ]


def _truth(
    image_path: Path, width: int, height: int
) -> list[tuple[int, tuple[float, ...]]]:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        return []
    boxes = read_yolo(label_path, class_count=len(CLASS_NAMES))
    return [
        (
            box.class_id,
            xywhn_to_xyxy(
                box.x_center, box.y_center, box.width, box.height, width, height
            ),
        )
        for box in boxes
    ]


def collect_error_examples(
    model: Any,
    image_paths: Sequence[Path],
    save_dir: Path,
    *,
    device: str | int,
    confidence: float,
    iou: float,
    match_iou: float,
    per_type: int,
    max_images: int = 0,
) -> dict[str, Any]:
    """Run genuine predictions and export evidence categories without fabrication."""

    cv2 = importlib.import_module("cv2")
    root = save_dir / "error_examples"
    categories = {
        "true_positive": root / "true_positive",
        "correct": root / "correct",
        "false_positive": root / "false_positive",
        "false_negative": root / "false_negative",
    }
    for directory in categories.values():
        directory.mkdir(parents=True, exist_ok=True)

    selected: dict[str, dict[str, list[str]]] = {
        category: {class_name: [] for class_name in CLASS_NAMES.values()}
        for category in categories
    }
    totals = {"images": 0, "tp": 0, "fp": 0, "fn": 0}
    candidates = list(image_paths[:max_images] if max_images > 0 else image_paths)
    for index, image_path in enumerate(candidates):
        image = read_image_bgr(image_path)
        height, width = image.shape[:2]
        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            device=device,
            verbose=False,
        )
        if not results:
            raise RuntimeError(f"predicao sem resultado para {image_path}")
        result = results[0]
        truth = _truth(image_path, width, height)
        predictions = _predictions(result)
        matched_truth, matched_predictions = greedy_class_matches(
            truth, predictions, match_iou
        )
        tp = len(matched_predictions)
        fp = len(predictions) - tp
        fn = len(truth) - len(matched_truth)
        totals["images"] += 1
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        applicable: dict[str, set[int]] = {name: set() for name in categories}
        applicable["true_positive"].update(
            predictions[prediction_index][0] for prediction_index in matched_predictions
        )
        if truth and fp == 0 and fn == 0:
            applicable["correct"].update(class_id for class_id, _box in truth)
        applicable["false_positive"].update(
            predictions[prediction_index][0]
            for prediction_index in range(len(predictions))
            if prediction_index not in matched_predictions
        )
        applicable["false_negative"].update(
            truth[truth_index][0]
            for truth_index in range(len(truth))
            if truth_index not in matched_truth
        )
        needed = {
            category: {
                class_id
                for class_id in class_ids
                if len(selected[category][CLASS_NAMES[class_id]]) < per_type
            }
            for category, class_ids in applicable.items()
        }
        if not any(needed.values()):
            continue

        canvas = result.plot()
        for truth_index, (class_id, coordinates) in enumerate(truth):
            x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
            missed = truth_index not in matched_truth
            color = (0, 0, 255) if missed else (0, 180, 0)
            label = ("FN " if missed else "GT ") + CLASS_NAMES[class_id]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        for pred_index, (_class_id, _confidence, coordinates) in enumerate(predictions):
            if pred_index in matched_predictions:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                canvas,
                "FP",
                (x1, min(height - 4, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        filename = f"{index:05d}_{image_path.stem}.jpg"
        for category, class_ids in needed.items():
            for class_id in sorted(class_ids):
                class_name = CLASS_NAMES[class_id]
                destination = categories[category] / class_name / filename
                write_image(destination, canvas)
                selected[category][class_name].append(
                    destination.relative_to(save_dir).as_posix()
                )

    summary = {
        "matching_iou": match_iou,
        "prediction_confidence": confidence,
        "scanned": totals,
        "examples": selected,
        "note": (
            "TP/FP/FN totals cover every scanned candidate. Examples are selected "
            "separately by category and class; official aggregate metrics remain "
            "those from the complete Ultralytics test evaluation."
        ),
    }
    write_json(root / "summary.json", summary)
    return summary
