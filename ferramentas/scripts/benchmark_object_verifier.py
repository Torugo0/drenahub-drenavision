"""Score fixed validation DINO crops and compare confidence/margin operating points.

Only validation selects parameters. Unknown presence is excluded; box metrics
use the20 CLICK validation images, never partial iRodent/body references. Raw
GroundingDINO and master files are read-only. No application setting is changed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.core.schemas import Detection  # noqa: E402
from drenavision.inference.media_io import read_image_bgr  # noqa: E402
from drenavision.inference.runtime import merge_object_detections  # noqa: E402
from drenavision.inference.utils import (  # noqa: E402
    greedy_class_matches,
    sha256_file,
    utc_now_iso,
)

CONFIDENCE_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
MARGIN_GRID = (-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16)
CANDIDATE_FLOOR = 0.20


def eligible_candidates(candidates: list[dict]) -> list[dict]:
    """Retain mapped class0/2 proposals in original order, without editing them."""
    selected = []
    for item in candidates:
        class_id = item.get("class_id")
        if type(class_id) is not int or class_id not in {0, 2}:
            continue
        confidence = float(item["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid raw candidate confidence")
        if confidence >= CANDIDATE_FLOOR:
            selected.append(copy.deepcopy(item))
    return selected


def validate_verifier_result(original: list[dict], scored: list[dict]) -> None:
    if len(original) != len(scored):
        raise ValueError("Verifier must preserve every candidate and its order")
    for before, after in zip(original, scored, strict=True):
        if {
            key: value for key, value in after.items() if key != "verification"
        } != before:
            raise ValueError("Verifier changed a candidate, its box or its order")
        verification = after.get("verification", {})
        for key in ("margin", "positive_similarity", "negative_similarity"):
            if not math.isfinite(float(verification[key])):
                raise ValueError("Verifier returned a non-finite score")
        if not all(
            isinstance(verification[key], str)
            for key in ("best_positive", "best_negative")
        ):
            raise ValueError("Verifier returned invalid prompt identities")


def master_index(payload: dict) -> dict[str, dict]:
    indexed = {}
    for row in payload["records"]:
        digest = row["sha256"]
        if digest in indexed:
            raise ValueError("Duplicate source image hash in master predictions")
        indexed[digest] = row
    return indexed


def accepted_candidates(
    row: dict, class_id: int, threshold: float, margin: float
) -> list[dict]:
    return [
        item
        for item in row["verified_candidates"]
        if item["class_id"] == class_id
        and float(item["confidence"]) >= threshold
        and float(item["verification"]["margin"]) >= margin
    ]


def _master_candidates(row: dict, class_id: int) -> list[dict]:
    threshold = {0: 0.35, 2: 0.60}[class_id]
    return [
        item
        for item in row["predictions"]
        if item["class_id"] == class_id and float(item["confidence"]) >= threshold
    ]


def _metrics(tp: int, fp: int, fn: int, tn: int | None = None) -> dict:
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "f2": 5 * tp / (5 * tp + fp + 4 * fn) if tp + fp + fn else 0.0,
    }
    if tn is not None:
        result.update(
            tn=tn, negative_false_alert_rate=fp / (fp + tn) if fp + tn else None
        )
    return result


def presence_metrics(
    records: list[dict],
    masters: dict[str, dict],
    class_id: int,
    *,
    threshold: float = 0.35,
    margin: float = 0.0,
    mode: str = "replace",
) -> dict:
    if mode not in {"replace", "augment", "master_only"} or class_id not in {0, 2}:
        raise ValueError("Invalid class or comparison mode")
    if class_id == 2 and mode == "augment":
        raise ValueError("Animal candidates always replace the master animal branch")
    tp = fp = fn = tn = unknown = 0
    for row in records:
        truth = row["presence"].get(str(class_id))
        if truth is None:
            unknown += 1
            continue
        if not isinstance(truth, bool):
            raise ValueError("Presence must be boolean or null")
        master_found = bool(_master_candidates(masters[row["sha256"]], class_id))
        found = (
            master_found
            if mode == "master_only"
            else bool(accepted_candidates(row, class_id, threshold, margin))
        )
        if mode == "augment":
            found = (
                found or master_found
            )  # Legacy litter is deliberately not crop-verified.
        tp += int(truth and found)
        fp += int(not truth and found)
        fn += int(truth and not found)
        tn += int(not truth and not found)
    return {
        **_metrics(tp, fp, fn, tn),
        "unknown_excluded": unknown,
        "known_images": tp + fp + fn + tn,
    }


def _detection(item: dict) -> Detection:
    return Detection(
        class_id=item["class_id"],
        class_name={0: "obstrucao_visivel", 2: "animal_visivel"}[item["class_id"]],
        confidence=float(item["confidence"]),
        bounding_box=item.get("clipped_bounding_box", item["bounding_box"]),
    )


def litter_bbox_metrics(
    records: list[dict],
    masters: dict[str, dict],
    *,
    threshold: float = 0.35,
    margin: float = 0.0,
    mode: str = "replace",
) -> dict:
    """Micro box F1 on CLICK only, after the exact shared class-aware fusion."""
    if mode not in {"replace", "augment", "master_only"}:
        raise ValueError("Invalid litter comparison mode")
    tp = fp = fn = images = reference_boxes = 0
    for row in records:
        master = masters[row["sha256"]]
        if master["source_key"] != "click_trash":
            continue
        if master.get("split") != "val":
            raise ValueError("Bounding-box selection requires CLICK validation only")
        images += 1
        truth = [
            (0, item["bounding_box"])
            for item in master["truth"]
            if item["class_id"] == 0
        ]
        reference_boxes += len(truth)
        legacy = [_detection(item) for item in _master_candidates(master, 0)]
        proposed = (
            []
            if mode == "master_only"
            else [
                _detection(item)
                for item in sorted(
                    accepted_candidates(row, 0, threshold, margin),
                    key=lambda item: -float(item["confidence"]),
                )
            ]
        )
        # Shared fusion deduplicates DINO proposals before matching and prioritizes
        # their actual coordinates; no union/enlarged box is fabricated.
        fused = (
            legacy
            if mode == "master_only"
            else merge_object_detections(legacy, proposed, mode=mode)
        )
        predictions = [
            (item.class_id, item.confidence, item.bounding_box) for item in fused
        ]
        matched_truth, matched_predictions = greedy_class_matches(
            truth, predictions, 0.5
        )
        tp += len(matched_truth)
        fp += len(predictions) - len(matched_predictions)
        fn += len(truth) - len(matched_truth)
    return {
        **_metrics(tp, fp, fn),
        "images": images,
        "reference_boxes": reference_boxes,
        "match_iou": 0.5,
        "scope": "CLICK primary-class validation boxes only; no partial iRodent or cross-source box scoring",
    }


def select_grid_row(
    rows: list[dict],
    class_id: int,
    *,
    residue_selection: str,
    max_presence_fp: int,
    min_residue_box_precision: float = 0.70,
) -> dict | None:
    eligible = [
        row for row in rows if class_id != 0 or row["presence"]["fp"] <= max_presence_fp
    ]
    if class_id == 0 and residue_selection == "bbox-f2":
        eligible = [
            row
            for row in eligible
            if row["bbox"]["precision"] >= min_residue_box_precision
        ]
    if not eligible:
        return None

    def key(row: dict) -> tuple:
        primary = (
            row["bbox"]
            if class_id == 0 and residue_selection in {"bbox", "bbox-f2"}
            else row["presence"]
        )
        return (
            primary["f2" if class_id == 0 and residue_selection == "bbox-f2" else "f1"],
            -primary["fp"],
            -row["presence"]["fp"],
            row["margin"],
            row["threshold"],
            row["mode"] == "replace",
        )

    return max(eligible, key=key)


def evaluate_grid(
    records: list[dict],
    masters: dict[str, dict],
    *,
    residue_selection: str = "bbox",
    max_extra_residue_false_alerts: int = 2,
    min_residue_box_precision: float = 0.70,
) -> dict:
    if (
        residue_selection not in {"bbox", "presence", "bbox-f2"}
        or max_extra_residue_false_alerts < 0
        or not math.isfinite(min_residue_box_precision)
        or not 0 <= min_residue_box_precision <= 1
    ):
        raise ValueError("Invalid selection policy")
    baseline = {
        str(cls): {
            "presence": presence_metrics(records, masters, cls, mode="master_only")
        }
        for cls in (0, 2)
    }
    baseline["0"]["bbox"] = litter_bbox_metrics(records, masters, mode="master_only")
    max_fp = baseline["0"]["presence"]["fp"] + max_extra_residue_false_alerts
    grid, selected = {}, {}
    for class_id in (0, 2):
        rows = []
        for mode in ("replace", "augment") if class_id == 0 else ("replace",):
            for threshold in CONFIDENCE_GRID:
                for margin in MARGIN_GRID:
                    row = {
                        "threshold": threshold,
                        "margin": margin,
                        "mode": mode,
                        "presence": presence_metrics(
                            records,
                            masters,
                            class_id,
                            threshold=threshold,
                            margin=margin,
                            mode=mode,
                        ),
                    }
                    if class_id == 0:
                        row["bbox"] = litter_bbox_metrics(
                            records,
                            masters,
                            threshold=threshold,
                            margin=margin,
                            mode=mode,
                        )
                        row["passes_presence_guard"] = row["presence"]["fp"] <= max_fp
                        row["passes_box_precision_guard"] = (
                            row["bbox"]["precision"] >= min_residue_box_precision
                        )
                    rows.append(row)
        grid[str(class_id)] = rows
        selected[str(class_id)] = select_grid_row(
            rows,
            class_id,
            residue_selection=residue_selection,
            max_presence_fp=max_fp,
            min_residue_box_precision=min_residue_box_precision,
        )
    return {
        "baseline": baseline,
        "grid": grid,
        "selected": selected,
        "selection_policy": {
            "residue_primary_metric": "CLICK box F2"
            if residue_selection == "bbox-f2"
            else f"{residue_selection} F1",
            "residue_min_box_precision_when_f2": min_residue_box_precision,
            "selection_scope": "engineering objective chosen on validation, not an independent test result; freeze parameters before hold-out evaluation",
            "residue_presence_false_alert_guard": {
                "baseline_fp": baseline["0"]["presence"]["fp"],
                "max_extra_false_alerts": max_extra_residue_false_alerts,
                "max_fp": max_fp,
                "known_negative_images": baseline["0"]["presence"]["fp"]
                + baseline["0"]["presence"]["tn"],
            },
            "animal_primary_metric": "presence F1; specialist replaces master animals",
            "tie_breaking": "fewer primary-metric FP, fewer presence FP, larger margin, larger DINO threshold, replace mode",
            "master_operating_points": {
                "residue_confidence": 0.35,
                "animal_confidence": 0.60,
            },
            "class0_augment": "unverified legacy litter retained at confidence>=0.35; only DINO crops are verified",
            "deployment": "report only; does not change active weights, thresholds, settings or deployment",
        },
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_cached_prefix(
    cached: dict, source: dict, source_hash: str, master_hash: str
) -> None:
    if (
        cached.get("source_predictions_sha256") != source_hash
        or cached.get("master_predictions_sha256") != master_hash
    ):
        raise ValueError(
            "Cached source/master hashes do not match; preserve output and choose another path"
        )
    if len(cached["records"]) > len(source["records"]):
        raise ValueError("Cache contains more records than the fixed source cohort")
    for previous, original in zip(cached["records"], source["records"], strict=False):
        if any(previous.get(key) != value for key, value in original.items()):
            raise ValueError(
                "Cached records are not an unchanged ordered source prefix"
            )
        validate_verifier_result(
            eligible_candidates(original["raw_candidates"]),
            previous["verified_candidates"],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--master-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument(
        "--residue-selection", choices=("bbox", "presence", "bbox-f2"), default="bbox"
    )
    parser.add_argument("--min-residue-box-precision", type=float, default=0.70)
    parser.add_argument("--max-extra-residue-false-alerts", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the matching ordered checkpoint; preserve all completed scores",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Recalculate metrics only; no model loading or GPU work",
    )
    args = parser.parse_args(argv)
    if args.output.resolve() in {
        args.predictions.resolve(),
        args.master_predictions.resolve(),
    }:
        parser.error("Output must differ from both read-only prediction inputs")
    source, master = _read(args.predictions), _read(args.master_predictions)
    if "validation" not in source.get("scope", "").lower():
        parser.error(
            "Selection requires fixed validation predictions, not test/regression data"
        )
    if (
        not source.get("cohort_sha256")
        or not source.get("model")
        or not master.get("checkpoint_sha256")
    ):
        parser.error("Input provenance lacks fixed cohort/model/master hashes")
    masters = master_index(master)
    if master.get("evaluation_role") != "validation_for_parameter_selection":
        parser.error(
            "Master predictions must be explicitly marked validation for parameter selection"
        )
    for row in source["records"]:
        if row["sha256"] not in masters:
            parser.error(f"Source image missing from master cohort: {row['image']}")
        if masters[row["sha256"]].get("split") != "val":
            parser.error(
                "Test/regression source images cannot drive verifier parameter selection"
            )
    source_hash, master_hash = (
        sha256_file(args.predictions),
        sha256_file(args.master_predictions),
    )
    if args.cached or args.resume:
        payload = _read(args.output)
        validate_cached_prefix(payload, source, source_hash, master_hash)
    else:
        if args.output.exists():
            parser.error(
                "Output already exists; use --resume/--cached or choose a new output"
            )
        payload = {
            "created_at_utc": utc_now_iso(),
            "status": "in_progress",
            "scope": "validation-only verifier/threshold selection; presence and CLICK box F1 are development metrics, not independent field accuracy or AP",
            "source_predictions": str(args.predictions),
            "source_predictions_sha256": source_hash,
            "master_predictions": str(args.master_predictions),
            "master_predictions_sha256": master_hash,
            "cohort": source["cohort"],
            "cohort_sha256": source["cohort_sha256"],
            "grounded_model": source["model"],
            "master_checkpoint_sha256": master["checkpoint_sha256"],
            "master_data_manifest_sha256": master.get("data_manifest_sha256"),
            "candidate_policy": {
                "minimum_dino_confidence": CANDIDATE_FLOOR,
                "class_ids": [0, 2],
                "source_record_order_preserved": True,
                "raw_candidates_preserved": True,
            },
            "records": [],
        }
    if not args.cached:
        from drenavision.inference.object_verifier import CropObjectVerifier

        verifier = CropObjectVerifier(
            args.verifier_model,
            device=args.device,
            batch_size=args.batch_size,
            crop_padding=args.crop_padding,
        )
        if not verifier.is_ready:
            raise RuntimeError(
                f"Configured crop verifier unavailable: {verifier.detail}"
            )
        provenance = verifier.provenance
        if args.resume and payload.get("verifier") != provenance:
            raise ValueError("Verifier differs from the preserved checkpoint")
        payload["verifier"] = provenance
        try:
            for index in range(len(payload["records"]), len(source["records"])):
                row = source["records"][index]
                image_path = ROOT / row["image"]
                if sha256_file(image_path) != row["sha256"]:
                    raise ValueError(f"Source image hash changed: {row['image']}")
                candidates = eligible_candidates(row["raw_candidates"])
                started = time.perf_counter()
                scored = (
                    verifier.score(
                        read_image_bgr(image_path)[:, :, ::-1].copy(), candidates
                    )
                    if candidates
                    else []
                )
                validate_verifier_result(candidates, scored)
                payload["records"].append(
                    {
                        **copy.deepcopy(row),
                        "verified_candidates": scored,
                        "verification_wall_ms": (time.perf_counter() - started)
                        * 1000.0,
                    }
                )
                if (index + 1) % 10 == 0:
                    _write(args.output, payload)
                    print(f"Verified {index + 1}/{len(source['records'])}", flush=True)
        except Exception as exc:
            payload["status"], payload["error"] = "failed", str(exc)
            _write(args.output, payload)
            raise
    if len(payload["records"]) != len(source["records"]):
        raise ValueError(
            "Incomplete validation checkpoint; use --resume before calculating selection"
        )
    payload.update(
        evaluate_grid(
            payload["records"],
            masters,
            residue_selection=args.residue_selection,
            max_extra_residue_false_alerts=args.max_extra_residue_false_alerts,
            min_residue_box_precision=args.min_residue_box_precision,
        )
    )
    payload.update(status="completed", completed_at_utc=utc_now_iso())
    payload.pop("error", None)
    if (
        sha256_file(args.predictions) != source_hash
        or sha256_file(args.master_predictions) != master_hash
    ):
        raise ValueError("Read-only raw inputs changed during the benchmark")
    _write(args.output, payload)
    print(json.dumps(payload["selected"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
