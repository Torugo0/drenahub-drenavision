"""Evaluate a frozen object revision without selecting or changing parameters.

The new CLICK holdout and the previously inspected 179-image regression are
separate phases. Raw predictions stay local; compact metrics retain cohort,
model/configuration seals, reference limitations and water-invariance checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.yolo import read_yolo  # noqa: E402
from drenavision.inference.media_io import read_image_bgr  # noqa: E402
from drenavision.inference.runtime import InspectionPredictor, create_predictor  # noqa: E402
from drenavision.inference.utils import (  # noqa: E402
    sha256_file,
    utc_now_iso,
    xywhn_to_xyxy,
)
from ferramentas.scripts.benchmark_detection import score, split_samples, totals  # noqa: E402

BASELINE_VERSION = "pipeline-1314fcfa94f5f8cf"
CANDIDATE_VERSION = "pipeline-4c5c993223219929"
REFERENCE_NOTE = (
    "Native source boxes can be incomplete or visible-keypoint body envelopes. "
    "Box FP may include genuine unannotated objects. Source-box absence is a "
    "label proxy, not reviewed absence, except the sealed holdout presence review."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def presence_metrics(rows: list[dict], model: str, class_id: int) -> dict:
    counts = Counter(tp=0, fp=0, tn=0, fn=0, unknown=0)
    for row in rows:
        truth = row["presence"].get(str(class_id))
        if truth is None:
            counts["unknown"] += 1
            continue
        present = any(item["class_id"] == class_id for item in row[model]["detections"])
        counts[
            "tp" if truth and present else "fn" if truth else "fp" if present else "tn"
        ] += 1
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {
        **counts,
        "positive_images": tp + fn,
        "negative_images": fp + tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_alert_rate_on_negative_images": fp / (fp + tn) if fp + tn else None,
    }


def assert_water_equal(
    baseline: list[dict], candidate: list[dict], tolerance: float = 1e-5
) -> None:
    def water(predictions):
        return sorted(
            [float(item["confidence"]), *map(float, item["bounding_box"])]
            for item in predictions
            if item["class_id"] == 1
        )

    before, after = water(baseline), water(candidate)
    if (
        len(before) != len(after)
        or before
        and not np.allclose(before, after, atol=tolerance, rtol=0)
    ):
        raise AssertionError(
            "Water detections changed outside the frozen object-revision scope"
        )


def holdout_samples(frozen: dict) -> tuple[list[dict], dict]:
    directory = ROOT / "dados/raw/v4/click_holdout"
    review_path, manifest_path = (
        directory / "holdout_presence_review.json",
        directory / "holdout_manifest.json",
    )
    review, manifest = read_json(review_path), read_json(manifest_path)
    if sha256_file(review_path) != frozen["new_holdout_review_sha256"]:
        raise ValueError(
            "New holdout review differs from the pre-prediction frozen seal"
        )
    if sha256_file(manifest_path) != review["holdout_manifest_sha256"]:
        raise ValueError("New holdout manifest differs from the reviewed source seal")
    reviewed = {row["sha256"]: row for row in review["records"]}
    samples = []
    for index, entry in enumerate(manifest["records"]):
        path, label = ROOT / entry["image"], ROOT / entry["label"]
        if (
            sha256_file(path) != entry["sha256"]
            or sha256_file(label) != entry["label_sha256"]
        ):
            raise ValueError(f"Holdout image/label changed: {entry['image']}")
        with Image.open(path) as source:
            width, height = source.size
        truth = [
            {
                "class_id": box.class_id,
                "bounding_box": list(
                    xywhn_to_xyxy(
                        box.x_center, box.y_center, box.width, box.height, width, height
                    )
                ),
            }
            for box in read_yolo(label, class_count=3)
        ]
        samples.append(
            {
                "id": f"holdout40/{index:03d}",
                "cohort": "holdout40",
                "read_path": str(path.resolve()),
                "input": entry["image"],
                "sha256": entry["sha256"],
                "label_sha256": entry["label_sha256"],
                "truth": truth,
                "presence": reviewed[entry["sha256"]]["presence"],
                "source_key": entry["source_key"],
                "source_group": entry["source_group"],
                "source_url": entry["source_url"],
                "source_license": entry["source_license"],
            }
        )
    if len(samples) != 40 or sum(len(row["truth"]) for row in samples) != 60:
        raise ValueError(
            "Expected exactly 40 new holdout images and 60 native litter boxes"
        )
    return samples, {
        "holdout_review_sha256": sha256_file(review_path),
        "holdout_manifest_sha256": sha256_file(manifest_path),
        "review_method": review["review_method"],
        "scope": review["scope"],
        "presence_reference": "source-only reviewed presence; unknown excluded; no animal-positive images",
    }


def regression_samples() -> tuple[list[dict], dict]:
    dataset = ROOT / "dados/processed/drenavision_v3_clean/dataset.yaml"
    test60 = split_samples(dataset, "test", 0, 42)
    extra_path = (
        ROOT / "resultados/local/v4/diagnosis/master_extra119_raw_conf005_s640.json"
    )
    extra = read_json(extra_path)["records"]
    samples = []
    for cohort, records in (("test60", test60), ("extra119", extra)):
        for entry in records:
            sample = {
                key: entry[key]
                for key in (
                    "id",
                    "read_path",
                    "input",
                    "sha256",
                    "truth",
                    "source_key",
                    "source_group",
                )
            }
            sample["id"], sample["cohort"] = f"{cohort}/{entry['id']}", cohort
            sample["read_path"] = str((ROOT / entry["input"]).resolve())
            sample["presence"] = {
                str(class_id): any(
                    box["class_id"] == class_id for box in entry["truth"]
                )
                for class_id in range(3)
            }
            if sha256_file(Path(sample["read_path"])) != sample["sha256"]:
                raise ValueError(f"Regression source image changed: {sample['input']}")
            samples.append(sample)
    if len(test60) != 60 or len(extra) != 119:
        raise ValueError("Expected 60 prior test and 119 extra regression images")
    return samples, {
        "test_dataset_yaml_sha256": sha256_file(dataset),
        "extra119_source_report_sha256": sha256_file(extra_path),
        "scope": "Previously evaluated 179-image engineering regression; not a newly unseen test or parameter selection",
        "presence_reference": "Positive source labels; absent labels treated as a proxy negative with incomplete cross-class annotation caveat",
    }


def model_result(predictor, rgb: np.ndarray) -> tuple[dict, np.ndarray]:
    started = time.perf_counter()
    result = predictor.predict(rgb)
    return {
        "model_version": result.model_version,
        "inference_ms": float(result.inference_ms),
        "wall_ms": (time.perf_counter() - started) * 1000,
        "detections": [item.model_dump(mode="json") for item in result.detections],
    }, result.annotated_image


def save_overlay(
    output: Path,
    name: str,
    rgb: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    truth: list[dict],
) -> str:
    reference = Image.fromarray(rgb)
    draw = ImageDraw.Draw(reference)
    for item in truth:
        draw.rectangle(
            item["bounding_box"],
            outline="#16a34a",
            width=max(2, reference.width // 500),
        )
    panel = Image.new("RGB", (1500, 420), "#0b1729")
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
    for column, (title, source) in enumerate(
        (
            ("Source reference boxes (incomplete)", reference),
            ("Baseline v3", Image.fromarray(baseline)),
            ("Frozen candidate v4", Image.fromarray(candidate)),
        )
    ):
        tile = ImageOps.contain(source, (490, 370))
        panel.paste(
            tile,
            (column * 500 + (500 - tile.width) // 2, 40 + (370 - tile.height) // 2),
        )
        ImageDraw.Draw(panel).text(
            (column * 500 + 8, 8), title, font=font, fill="white"
        )
    destination = output / "overlays" / f"{name}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination, quality=94)
    return destination.relative_to(ROOT).as_posix()


def summarize(rows: list[dict], scope: dict) -> dict:
    cohorts = {}
    for cohort in dict.fromkeys(row["cohort"] for row in rows):
        selected = [row for row in rows if row["cohort"] == cohort]
        models = {}
        for model in ("baseline", "candidate"):
            box_rows = [{"score": row[model]["box_score"]} for row in selected]
            timings = np.asarray([row[model]["inference_ms"] for row in selected])
            models[model] = {
                "presence": {
                    str(c): presence_metrics(selected, model, c) for c in range(3)
                },
                "source_bbox_iou_0_5": totals(box_rows),
                "latency_ms": {
                    "mean": float(timings.mean()),
                    "median": float(np.median(timings)),
                    "p95": float(np.percentile(timings, 95)),
                },
            }
        cohorts[cohort] = {"images": len(selected), **models}
    return {
        "scope": scope,
        "reference_caveat": REFERENCE_NOTE,
        "cohorts": cohorts,
        "water_unchanged_images": len(rows),
        "water_tolerance": 1e-5,
    }


def run_phase(phase: str, output_dir: Path, frozen_path: Path) -> dict:
    import torch

    torch.set_num_threads(4)
    frozen_hash = sha256_file(frozen_path)
    frozen = read_json(frozen_path)
    configuration = ROOT / frozen["configuration"]
    if sha256_file(configuration) != frozen["configuration_sha256"]:
        raise ValueError("Runtime configuration differs from frozen candidate")
    if frozen["model_version"] != CANDIDATE_VERSION:
        raise ValueError(
            "This evaluation script requires the explicitly sealed v4 candidate"
        )
    parameters = read_json(configuration)
    if parameters != frozen["runtime_parameters"]:
        raise ValueError("Frozen/runtime parameter dictionaries differ")
    samples, scope = (
        holdout_samples(frozen) if phase == "holdout40" else regression_samples()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_seal = hashlib.sha256(
        json.dumps(samples, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    write_json(
        output_dir / f"{phase}_run_seal.json",
        {
            "saved_before_model_initialization_or_inference": True,
            "started_at_utc": utc_now_iso(),
            "phase": phase,
            "frozen_manifest_sha256": frozen_hash,
            "configuration_sha256": frozen["configuration_sha256"],
            "source_samples_sha256": sample_seal,
            "expected_baseline": BASELINE_VERSION,
            "expected_candidate": CANDIDATE_VERSION,
        },
    )
    candidate = create_predictor(**parameters)
    baseline = InspectionPredictor(
        candidate.master,
        candidate.water,
        residue_confidence=0.35,
        animal_confidence=0.60,
        water_mode="augment",
    )
    if (
        baseline.model_version != BASELINE_VERSION
        or candidate.model_version != CANDIDATE_VERSION
    ):
        raise ValueError(
            "Instantiated model profiles differ from the frozen baseline/candidate"
        )
    print(
        json.dumps(
            {
                "phase": phase,
                "images": len(samples),
                "baseline": baseline.model_version,
                "candidate": candidate.model_version,
                "frozen_manifest_sha256": frozen_hash,
            }
        ),
        flush=True,
    )
    rows, qa_counts, overlays = [], Counter(), []
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        bgr = read_image_bgr(sample["read_path"])
        rgb = bgr[:, :, ::-1].copy()
        base, base_overlay = model_result(baseline, rgb)
        revised, candidate_overlay = model_result(candidate, rgb)
        assert_water_equal(base["detections"], revised["detections"])
        base["box_score"], revised["box_score"] = (
            score(sample["truth"], base["detections"], 0.5),
            score(sample["truth"], revised["detections"], 0.5),
        )
        row = {
            **sample,
            "baseline": base,
            "candidate": revised,
            "water_unchanged": True,
        }
        rows.append(row)
        if phase == "holdout40":
            before = base["box_score"]["per_class"]["obstrucao_visivel"]
            after = revised["box_score"]["per_class"]["obstrucao_visivel"]
            categories = [
                name
                for name, condition in (
                    ("improved_tp", after["tp"] > before["tp"]),
                    ("fn", after["fn"] > 0),
                    ("source_reference_fp", after["fp"] > 0),
                )
                if condition and qa_counts[name] < 3
            ]
            if index < 8 or categories:
                overlays.append(
                    save_overlay(
                        output_dir,
                        f"holdout_{index:03d}",
                        rgb,
                        base_overlay,
                        candidate_overlay,
                        sample["truth"],
                    )
                )
                qa_counts.update(categories)
        with (output_dir / f"{phase}_records.jsonl").open(
            "a" if index else "w", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (index + 1) % 10 == 0 or index + 1 == len(samples):
            if (
                sha256_file(frozen_path) != frozen_hash
                or sha256_file(configuration) != frozen["configuration_sha256"]
            ):
                raise ValueError(
                    "The candidate seal changed during inference; stop without tuning"
                )
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed": index + 1,
                        "images": len(samples),
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    summary = {
        "evaluated_at_utc": utc_now_iso(),
        "phase": phase,
        "frozen_manifest_sha256_before_inference": frozen_hash,
        "configuration_sha256": frozen["configuration_sha256"],
        "baseline_model_version": baseline.model_version,
        "candidate_model_version": candidate.model_version,
        "selection_used_results": False,
        "source_samples_sha256": sample_seal,
        "elapsed_seconds": time.perf_counter() - started,
        **summarize(rows, scope),
        "overlays": overlays,
    }
    detailed = {**summary, "runtime_parameters": parameters, "records": rows}
    write_json(output_dir / f"{phase}_evaluation.json", detailed)
    report_path = ROOT / "resultados/metrics/v4" / f"evaluation_{phase}.json"
    summary["detailed_report"] = str(
        (output_dir / f"{phase}_evaluation.json").relative_to(ROOT)
    ).replace("\\", "/")
    summary["detailed_report_sha256"] = sha256_file(
        output_dir / f"{phase}_evaluation.json"
    )
    write_json(report_path, summary)
    print(
        json.dumps(
            {"phase": phase, "report": str(report_path), "summary": summary},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("holdout40", "regression179"), required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "resultados/local/v4/evaluation"
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=ROOT / "resultados/metrics/v4/frozen_candidate.json",
    )
    args = parser.parse_args()
    run_phase(args.phase, args.output_dir.resolve(), args.frozen_manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
