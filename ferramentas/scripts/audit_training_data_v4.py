"""Read-only audit of the active training data; never trains or tunes a model.

Writes statistics and labeled sample sheets to a local audit directory. The
two user regression photos are fingerprinted and checked against every split;
they are never copied into a training/calibration dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": ordered[round((len(ordered) - 1) * 0.1)],
        "median": statistics.median(ordered),
        "p90": ordered[round((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
    }


def sample_sheet(
    rows: list[dict], dataset: Path, class_id: int, output: Path, split: str = "train"
) -> None:
    candidates = [
        row
        for row in rows
        if row["split"] == split and int(row[f"boxes_class_{class_id}"]) > 0
    ]
    chosen = random.Random(20261117 + class_id).sample(
        candidates, min(20, len(candidates))
    )
    cell_w, cell_h = 380, 310
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 5), "#eef4fa")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    for index, row in enumerate(chosen):
        with Image.open(dataset / row["output_image"]) as opened:
            image = opened.convert("RGB")
        pen = ImageDraw.Draw(image)
        for line in (dataset / row["output_label"]).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            cls, xc, yc, w, h = map(float, line.split())
            if int(cls) != class_id:
                continue
            box = (
                (xc - w / 2) * image.width,
                (yc - h / 2) * image.height,
                (xc + w / 2) * image.width,
                (yc + h / 2) * image.height,
            )
            pen.rectangle(box, width=max(2, min(image.size) // 160), outline="#f97316")
        thumbnail = ImageOps.contain(image, (cell_w, cell_h - 42))
        x, y = index % 4 * cell_w, index // 4 * cell_h
        sheet.paste(thumbnail, (x + (cell_w - thumbnail.width) // 2, y))
        draw.text(
            (x + 5, y + cell_h - 39), row["source_key"], font=font, fill="#12213a"
        )
        draw.text(
            (x + 5, y + cell_h - 22),
            Path(row["source_path"].split("::")[-1]).name[:47],
            font=font,
            fill="#12213a",
        )
    sheet.save(output, quality=92)


def audit(dataset: Path, output: Path) -> dict:
    rows = list(
        csv.DictReader((dataset / "data_manifest.csv").open(encoding="utf-8-sig"))
    )
    output.mkdir(parents=True, exist_ok=True)
    per_split_source = Counter(f"{r['split']}::{r['source_key']}" for r in rows)
    boxes = defaultdict(list)
    label_counts = Counter()
    negative_rows = []
    hashes = {row["sha256"] for row in rows}
    for row in rows:
        count = 0
        width, height = int(row["width"]), int(row["height"])
        scale_640 = 640 / max(width, height)
        for line in (dataset / row["output_label"]).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            cls, _xc, _yc, w, h = map(float, line.split())
            cls = int(cls)
            boxes[(row["split"], cls)].append(
                {
                    "fraction": w * h,
                    "min_side_640": min(w * width, h * height) * scale_640,
                }
            )
            count += 1
            label_counts[f"{row['split']}::class_{cls}"] += 1
        if not count:
            negative_rows.append(row["output_image"])
    regression = []
    for name in ("teste.jpg", "teste1.jpg"):
        path = ROOT / "dados/examples" / name
        if path.is_file():
            sha = digest(path)
            with Image.open(path) as image:
                size = list(image.size)
            regression.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha,
                    "size": size,
                    "present_in_training_or_calibration_or_test": sha in hashes,
                    "role": "user_reported_regression_only; do not train/calibrate on these images",
                }
            )
    rodent_root = ROOT / "dados/raw/irodent/extracted/rodent"
    annotation_by_file = {}
    for split in ("train", "test"):
        payload = json.loads(
            (rodent_root / f"annotations/{split}.json").read_text("utf-8")
        )
        names = {image["id"]: image["file_name"] for image in payload["images"]}
        for item in payload["annotations"]:
            annotation_by_file[names[item["image_id"]]] = item
    rodent_counts = Counter()
    for row in rows:
        if row["source_key"] != "irodent" or row["split"] != "train":
            continue
        name = row["source_path"].split("::")[-1]
        annotation = annotation_by_file[name]
        points = [annotation["keypoints"][i : i + 3] for i in range(0, 60, 3)]
        visible_head = sum(p[2] > 0 for p in points[:6])
        visible_body = sum(p[2] > 0 for p in points[6:])
        rodent_counts["images"] += 1
        rodent_counts["no_body_keypoints_visible"] += visible_body == 0
        rodent_counts["at_most_one_body_keypoint_visible"] += visible_body <= 1
        rodent_counts["at_most_four_total_keypoints_visible"] += (
            visible_head + visible_body <= 4
        )
        rodent_counts["has_segmentation"] += bool(annotation.get("segmentation"))
    inventories = []
    for selection in (ROOT / "dados/raw").rglob("selection.json"):
        payload = json.loads(selection.read_text("utf-8"))
        if payload.get("source") != "click_trash":
            continue
        inventories.append(
            {
                "manifest": selection.relative_to(ROOT).as_posix(),
                "selection_by_split": dict(
                    Counter(x["split"] for x in payload["items"])
                ),
                "local_pair_count": sum(
                    (selection.parent / x["image_local"]).is_file()
                    and (selection.parent / x["label_local"]).is_file()
                    for x in payload["items"]
                ),
            }
        )
    result = {
        "dataset": dataset.relative_to(ROOT).as_posix(),
        "dataset_manifest_sha256": digest(dataset / "data_manifest.csv"),
        "images_by_split_source": dict(per_split_source),
        "boxes_by_split_class": dict(label_counts),
        "group_counts_by_source": {
            source: len({r["source_group"] for r in rows if r["source_key"] == source})
            for source in sorted({r["source_key"] for r in rows})
        },
        "negative_images": negative_rows,
        "negative_review": {
            "nominal_empty_labels": len(negative_rows),
            "previously_reviewed_background_crops": 5,
            "uncertain_empty_source_image": "images/train/click_trash_3f055cd0459a2095.jpg",
            "uncertain_note": "Source trash(1045) has tiny blue/red anthropogenic-looking fragments among leaves; its empty label must not be treated as a confidently clean all-class negative without further review.",
        },
        "negative_train_fraction": sum("/train/" in x for x in negative_rows)
        / sum(r["split"] == "train" for r in rows),
        "box_geometry": {
            f"{split}::class_{cls}": {
                "relative_area": distribution([b["fraction"] for b in records]),
                "min_side_at_640": distribution([b["min_side_640"] for b in records]),
                "min_side_under_12px_at_640": sum(
                    b["min_side_640"] < 12 for b in records
                ),
            }
            for (split, cls), records in sorted(boxes.items())
        },
        "irodent_training_keypoint_coverage": dict(rodent_counts),
        "click_local_inventories": inventories,
        "user_regressions": regression,
        "method": "Read-only manifest/label/keypoint geometry audit; no model inference or tuning. Sample sheets show source-derived annotations, not predictions.",
    }
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
    )
    for class_id, name in (
        (0, "trash_training_labels.jpg"),
        (2, "rodent_training_labels.jpg"),
    ):
        sample_sheet(rows, dataset, class_id, output / name)
    for class_id in range(3):
        sample_sheet(
            rows, dataset, class_id, output / f"validation_class_{class_id}.jpg", "val"
        )
    return result


def freeze_calibration(dataset: Path, output: Path) -> dict:
    """Freeze the reviewed, existing validation images before fusion selection.

    The explicit review below is annotation metadata, not a prediction rule.
    No user photo, training image or test-pool image enters this cohort.
    """
    expected_manifest = (
        "7bf12ddc7a164b1d6e36f20170afdb2757d2d7dcfc7d6190e2bf311922c32878"
    )
    if digest(dataset / "data_manifest.csv") != expected_manifest:
        raise ValueError(
            "Dataset identities changed; repeat visual review before freezing"
        )
    rows = [
        row
        for row in csv.DictReader(
            (dataset / "data_manifest.csv").open(encoding="utf-8-sig")
        )
        if row["split"] == "val"
    ]
    if Counter(row["source_key"] for row in rows) != {
        "click_trash": 20,
        "irodent": 20,
        "stagnant_water": 20,
    }:
        raise ValueError(
            "The reviewed validation cohort must contain exactly 60 images"
        )
    target = output / "calibration_presence_v4_v2.json"
    if target.exists():
        raise FileExistsError(
            "The frozen calibration file already exists; never overwrite it"
        )
    records = []
    water_in_rodent = {
        "obs_id_00654.jpg",
        "new_obs_id_00222.jpg",
        "obs_id_00443.jpg",
        "obs_id_00815.jpg",
    }
    ambiguous_litter = {
        "image1686.jpeg",
        "image1687.jpeg",
        "image552.jpeg",
        "image554.jpeg",
        "image558.jpeg",
        "image892.jpeg",
        "image893.jpeg",
        "image896.jpeg",
        "image897.jpeg",
    }
    for row in rows:
        path = dataset / row["output_image"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Source image changed since manifest: {path}")
        source = row["source_key"]
        native_name = Path(row["source_path"].split("::")[-1]).name
        presence: dict[str, bool | None] = {
            "0": source == "click_trash",
            "1": source == "stagnant_water",
            "2": source == "irodent",
        }
        note = "All three class presences reviewed on the validation contact sheets; source annotations confirm the primary positive."
        if source == "irodent" and native_name in water_in_rodent:
            presence["1"] = True
            note += " Visible water surface independently reviewed; native rodent-only YOLO label omits water."
        if source == "irodent" and native_name == "new_obs_id_00213.jpg":
            presence["0"] = True
            note += " Original full-resolution image reviewed: discarded/torn plastic visible at the right; native rodent-only label omits litter."
        if source == "stagnant_water" and native_name in ambiguous_litter:
            presence["0"] = None
            note += " Distant/ambiguous debris is not confidently litter; unknown, never a negative."
        records.append(
            {
                "image": path.relative_to(ROOT).as_posix(),
                "label": (dataset / row["output_label"]).relative_to(ROOT).as_posix(),
                "sha256": row["sha256"],
                "label_sha256": digest(dataset / row["output_label"]),
                "source_key": source,
                "source_group": row["source_group"],
                "source_path": row["source_path"],
                "source_split": "val",
                "presence": presence,
                "review_note": note,
                "bbox_reference_complete": False,
                "bbox_scope": "Native source-derived boxes retained separately for diagnostics, not complete manually reviewed full-body/all-litter ground truth.",
            }
        )
    counts = {
        str(cls): {
            state: sum(row["presence"][str(cls)] is value for row in records)
            for state, value in (
                ("positive", True),
                ("negative", False),
                ("unknown", None),
            )
        }
        for cls in range(3)
    }
    data = {
        "schema_version": 1,
        "review_revision": 2,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Fusion threshold/prompt development calibration; not independent test and not a new generalization score.",
        "source_dataset_manifest_sha256": expected_manifest,
        "images": len(records),
        "class_ids": {
            "0": "obstrucao_visivel",
            "1": "agua_acumulada",
            "2": "animal_visivel",
        },
        "presence_counts": counts,
        "review_method": "Assistant visual review of all 60 validation images on full-scene contact sheets plus full-resolution inspection of ambiguous cross-class examples; no detector predictions used for the review; not independent human ground truth.",
        "absence_definition": "No target visibly confirmed in the available image; no claim that hidden or subpixel objects are absent.",
        "label_scope": "Presence-only. True=confirmed visible target, False=visually reviewed absence, null=ambiguous; never convert null to False.",
        "exclusions": [
            "all training images",
            "60 source test images",
            "119 additional held-out test images",
            "user teste.jpg and teste1.jpg",
            "all demonstration clips",
        ],
        "selection_policy": "Existing val identities fixed; no model-score-based selection; keep this cohort out of training and do not alter labels after looking at candidate predictions.",
        "revision_note": "Source-only reconciliation with a second assistant review before observing candidate fusion predictions. v1 remains archived; nine ambiguous water-source litter presences are unknown, never negative.",
        "previous_review_sha256": "19c5b54e19e589a893d39dcbf5b0cdc3fef4862f5cf29cc6d7f35243bb803d2a",
        "records": records,
    }
    serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    target.write_text(serialized, "utf-8")
    target.with_suffix(".sha256").write_text(digest(target) + "\n", "ascii")
    return {
        "manifest": target.relative_to(ROOT).as_posix(),
        "sha256": digest(target),
        "presence_counts": counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "dados/processed/drenavision_v3_clean"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "resultados/local/v4/data_audit"
    )
    parser.add_argument("--freeze-calibration", action="store_true")
    args = parser.parse_args()
    result = audit(args.dataset.resolve(), args.output.resolve())
    if args.freeze_calibration:
        result["frozen_calibration"] = freeze_calibration(
            args.dataset.resolve(), args.output.resolve()
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
