"""Compare learned rodent/litter proposals on a reviewed validation cohort.

This is presence evaluation, not AP or risk prediction. A cohort must exist
before inference; unknown class presence is excluded instead of scored negative.
New user examples must not be used as calibration data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.inference.media_io import read_image_bgr  # noqa: E402
from drenavision.inference.utils import sha256_file, utc_now_iso, write_json  # noqa: E402


def presence_score(records: list[dict], class_id: int, threshold: float) -> dict:
    tp = fp = fn = tn = unknown = 0
    for row in records:
        truth = row["presence"].get(str(class_id))
        if truth is None:
            unknown += 1
            continue
        predicted = any(
            item["class_id"] == class_id and item["confidence"] >= threshold
            for item in row["raw_candidates"]
        )
        tp += bool(truth and predicted)
        fn += bool(truth and not predicted)
        fp += bool(not truth and predicted)
        tn += bool(not truth and not predicted)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown_excluded": unknown,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
        "negative_false_alert_rate": fp / (fp + tn) if fp + tn else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path("resultados/local/models/grounding_dino_tiny")
    )
    parser.add_argument("--shortest-edge", type=int, default=800)
    parser.add_argument("--longest-edge", type=int, default=1333)
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    if args.cached:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        from drenavision.inference.objects import GroundedObjectPredictor

        cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
        predictor = GroundedObjectPredictor(
            args.model,
            residue_confidence=0.1,
            animal_confidence=0.1,
            text_confidence=0.20,
            shortest_edge=args.shortest_edge,
            longest_edge=args.longest_edge,
        )
        payload = {
            "created_at_utc": utc_now_iso(),
            "scope": "validation-only threshold selection; assistant-reviewed image presence, not independent human ground truth or AP",
            "cohort": str(args.cohort),
            "cohort_sha256": sha256_file(args.cohort),
            "model": predictor.provenance,
            "records": [],
        }
        for index, row in enumerate(cohort["records"]):
            path = ROOT / row["image"]
            result = predictor.predict(read_image_bgr(path)[:, :, ::-1].copy())
            payload["records"].append(
                {
                    "image": row["image"],
                    "sha256": sha256_file(path),
                    "presence": row["presence"],
                    "raw_candidates": result.raw_candidates,
                    "inference_ms": result.inference_ms,
                }
            )
            if (index + 1) % 10 == 0:
                write_json(args.output, payload)
                print(f"Evaluated {index + 1}/{len(cohort['records'])}", flush=True)
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    payload["selection_rule"] = "maximize class presence F1 on validation; break ties by fewer false positives then higher threshold"
    payload["grid"] = {
        str(class_id): [
            presence_score(payload["records"], class_id, threshold)
            for threshold in thresholds
        ]
        for class_id in (0, 2)
    }
    payload["selected"] = {
        key: max(rows, key=lambda r: (r["f1"], -r["fp"], r["threshold"]))
        for key, rows in payload["grid"].items()
    }
    write_json(args.output, payload)
    print(json.dumps(payload["selected"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
