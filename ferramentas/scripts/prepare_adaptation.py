"""Build a separate v2 dataset with repaired iRodent labels and domain frames.

The original processed dataset and published metrics are never overwritten.
Validation/test keep the original source-image IDs, with annotation-derived
rodent boxes corrected using visible pose landmarks. This changes ground truth,
so v1 and v2 mAP values are not interchangeable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.constants import IMAGE_SUFFIXES, SPLITS  # noqa: E402
from drenavision.data.converter import (  # noqa: E402
    ManifestRow,
    SourceCandidate,
    _write_candidate,
    collect_click,
    collect_irodent,
    collect_water_candidates,
    irodent_annotation_box,
    sha256_file,
    write_dataset_yaml,
    write_manifest,
)
from drenavision.data.validator import validate_dataset  # noqa: E402
from drenavision.data.yolo import read_yolo, write_yolo  # noqa: E402


def audit_irodent(raw_root: Path, output_dir: Path) -> dict:
    """Measure the source defect and draw original/derived boxes together."""
    source_root = raw_root / "rodent"
    records = []
    counts: Counter = Counter()
    for split in ("train", "test"):
        payload = json.loads(
            (source_root / "annotations" / f"{split}.json").read_text(encoding="utf-8")
        )
        images = {image["id"]: image for image in payload["images"]}
        for annotation in payload["annotations"]:
            image = images[annotation["image_id"]]
            points = [
                annotation["keypoints"][offset : offset + 3]
                for offset in range(0, len(annotation["keypoints"]), 3)
            ]
            visible = [point for point in points if point[2] > 0]
            source_box = annotation["bbox"]
            all_bound = [
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points) - min(point[0] for point in points),
                max(point[1] for point in points) - min(point[1] for point in points),
            ]
            invisible = sum(point[2] == 0 for point in points)
            counts["annotations"] += 1
            counts["source_bbox_at_origin"] += source_box[0] == 0 and source_box[1] == 0
            counts["source_bbox_equals_bound_including_invisible_points"] += (
                source_box == all_bound and invisible > 0
            )
            record = {
                "native_split": split,
                "file_name": image["file_name"],
                "source_bbox_xywh": source_box,
                "visible_keypoints": len(visible),
                "invisible_keypoints": invisible,
                "width": image["width"],
                "height": image["height"],
            }
            try:
                _box, note = irodent_annotation_box(
                    annotation, image["width"], image["height"]
                )
                details = json.loads(note)
                record["derived_bbox_xywh"] = details["corrected_bbox_xywh"]
                record["status"] = "usable"
                counts["usable_v2"] += 1
            except ValueError as exc:
                record.update(status="excluded", reason=str(exc))
                counts["excluded_insufficient_geometry"] += 1
            records.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        "obs_id_00138.jpg",
        "obs_id_00803.jpg",
        "obs_id_00845.jpg",
        "obs_id_00184.jpg",
    }
    usable = [record for record in records if record["status"] == "usable"]
    chosen = [record for record in usable if record["file_name"] in wanted]
    pool = [record for record in usable if record["file_name"] not in wanted]
    chosen.extend(random.Random(42).sample(pool, 12 - len(chosen)))
    cell_width, cell_height = 420, 330
    sheet = Image.new("RGB", (cell_width * 3, cell_height * 4 + 42), "#F1F5F9")
    pen = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    except OSError:
        font = ImageFont.load_default(size=14)
    pen.text(
        (12, 8),
        "Vermelho: bbox publicada | Azul: v2, pontos visiveis + margem de 10%",
        fill="#0F172A",
        font=font,
    )
    for index, record in enumerate(chosen):
        path = source_root / "images" / Path(record["file_name"]).name
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        thickness = max(3, min(image.size) // 160)
        for key, color in (
            ("source_bbox_xywh", "#EF4444"),
            ("derived_bbox_xywh", "#0284C7"),
        ):
            x, y, width, height = record[key]
            draw.rectangle(
                (x, y, x + width, y + height), outline=color, width=thickness
            )
        image = ImageOps.contain(image, (cell_width, cell_height - 25))
        x, y = (index % 3) * cell_width, (index // 3) * cell_height + 42
        sheet.paste(image, (x + (cell_width - image.width) // 2, y))
        pen.text(
            (x + 8, y + cell_height - 23),
            f"{record['native_split']}/{record['file_name']} | {record['visible_keypoints']} pts",
            fill="#0F172A",
            font=font,
        )
    preview = output_dir / "irodent_source_vs_v2.jpg"
    sheet.save(preview, quality=92)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "finding": "Published iRodent boxes frequently equal the bounds of all pose landmarks, including absent landmarks encoded as (0,0,0).",
        "counts": dict(counts),
        "v2_method": "Bounds of visibility > 0 annotated landmarks; 10% per-side padding (minimum 2px); clipping to image; exclude fewer than two visible points or zero-area geometry.",
        "limitation": "Boxes are derived from annotated pose points and are not hand-reviewed full-animal masks. Sparse or occluded landmarks can still under-cover an animal.",
        "preview": preview.as_posix(),
        "preview_source_images": [record["file_name"] for record in chosen],
        "records": records,
    }
    (output_dir / "annotation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _domain_rows(
    domain_root: Path, output_root: Path, seen: dict[str, str]
) -> list[ManifestRow]:
    metadata = json.loads((domain_root / "sources.json").read_text(encoding="utf-8"))
    rows = []
    for item in metadata["items"]:
        if item.get("split", "train") != "train":
            raise ValueError(
                "Domain frames are training-only; validation/test images remain held out"
            )
        image = domain_root / "images" / item["file_name"]
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported domain image: {image.name}")
        boxes = read_yolo(domain_root / "labels" / (image.stem + ".txt"), class_count=3)
        digest = sha256_file(image)
        if digest in seen:
            raise ValueError(f"Duplicate domain image: {image.name}")
        seen[digest] = image.name
        stem = "domain_" + digest[:16]
        suffix = image.suffix.lower().replace(".jpeg", ".jpg")
        destination = output_root / "images/train" / (stem + suffix)
        shutil.copy2(image, destination)
        label = output_root / "labels/train" / (stem + ".txt")
        write_yolo(label, boxes)
        with Image.open(image) as opened:
            width, height = opened.size
        counts = Counter(box.class_id for box in boxes)
        rows.append(
            ManifestRow(
                destination.relative_to(output_root).as_posix(),
                label.relative_to(output_root).as_posix(),
                "train",
                "domain_reviewed",
                "1",
                item["source_url"],
                item["source_license"],
                item["file_name"],
                item["source_group"],
                digest,
                width,
                height,
                counts[0],
                counts[1],
                counts[2],
                item["annotation_note"],
            )
        )
    return rows


def _recover_baseline_candidate(
    row: dict[str, str], raw_root: Path, pools: dict[str, dict[str, SourceCandidate]]
) -> SourceCandidate:
    """Recover the original manifest item from raw sources on a fresh clone."""
    source_key = row["source_key"]
    if source_key not in pools:
        if source_key == "click_trash":
            split_pools = collect_click(raw_root / "click_trash/sample")
            candidates = [
                candidate for pool in split_pools.values() for candidate in pool
            ]
        elif source_key == "stagnant_water":
            positives, negatives = collect_water_candidates(
                raw_root / "stagnant_water/extracted"
            )
            candidates = positives + negatives
        else:
            raise ValueError(f"No raw-source resolver for {source_key}")
        pools[source_key] = {
            candidate.source_path: candidate for candidate in candidates
        }
    candidate = pools[source_key].get(row["source_path"])
    if candidate is None:
        raise FileNotFoundError(
            f"Manifest image not found in raw {source_key}: {row['source_path']}; run download_datasets.py first"
        )
    if sha256_file(candidate.image) != row["sha256"]:
        raise ValueError(
            f"Raw source does not match historical manifest hash: {row['source_path']}"
        )
    counts = Counter(box.class_id for box in candidate.boxes)
    if any(
        counts[class_id] != int(row[f"boxes_class_{class_id}"]) for class_id in range(3)
    ):
        raise ValueError(
            f"Raw annotation counts differ from historical manifest: {row['source_path']}"
        )
    return SourceCandidate(
        candidate.image,
        candidate.boxes,
        source_key,
        row["source_path"],
        row["source_group"],
        row["annotation_note"]
        + "; recovered from raw source with verified historical image hash",
    )


def prepare_adaptation(
    baseline_root: Path,
    baseline_manifest: Path,
    raw_irodent: Path,
    output_root: Path,
    audit_dir: Path,
    domain_root: Path | None = None,
    raw_root: Path | None = None,
) -> dict:
    if output_root.resolve() == baseline_root.resolve():
        raise ValueError(
            "V2 must use a new output root, preserving the baseline dataset"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            "V2 output already contains files; choose a new output root rather than overwrite evidence"
        )
    baseline = list(
        csv.DictReader(baseline_manifest.open(encoding="utf-8", newline=""))
    )
    annotation_root = raw_irodent / "rodent/annotations"
    native = {
        name: collect_irodent(raw_irodent, annotation_root / (name + ".json"))
        for name in ("train", "test")
    }
    corrected = {
        candidate.source_path: candidate
        for pool in native.values()
        for candidate in pool
    }
    held_out_hashes = {row["sha256"] for row in baseline if row["split"] != "train"}
    held_out_groups = {
        row["source_group"] for row in baseline if row["split"] != "train"
    }
    selections: dict[str, list[SourceCandidate]] = {split: [] for split in SPLITS}
    skipped = []
    raw_root = raw_root or raw_irodent.parents[1]
    recovery_pools: dict[str, dict[str, SourceCandidate]] = {}
    recovered_baseline = 0
    baseline_groups = {row["source_group"] for row in baseline}
    for row in baseline:
        if row["source_key"] == "irodent":
            candidate = corrected.get(row["source_path"])
            if candidate is None:
                skipped.append(
                    {
                        "image": row["output_image"],
                        "split": row["split"],
                        "reason": "insufficient visible pose landmarks",
                    }
                )
                continue
        else:
            image = baseline_root / row["output_image"]
            label = baseline_root / row["output_label"]
            if image.is_file() and label.is_file():
                candidate = SourceCandidate(
                    image,
                    tuple(read_yolo(label, class_count=3)),
                    row["source_key"],
                    row["source_path"],
                    row["source_group"],
                    row["annotation_note"],
                )
            else:
                candidate = _recover_baseline_candidate(row, raw_root, recovery_pools)
                recovered_baseline += 1
        selections[row["split"]].append(candidate)
    extras = []
    for candidate in native["train"]:
        if (
            candidate.source_group in baseline_groups
            or candidate.source_group in held_out_groups
        ):
            continue
        if sha256_file(candidate.image) in held_out_hashes:
            continue
        extras.append(candidate)
    selections["train"].extend(extras)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (output_root / kind / split).mkdir(parents=True, exist_ok=True)
    rows = []
    seen: dict[str, str] = {}
    for split in SPLITS:
        for candidate in selections[split]:
            rows.append(_write_candidate(candidate, split, output_root, seen))
    if domain_root:
        rows.extend(_domain_rows(domain_root, output_root, seen))
    manifest = output_root / "data_manifest.csv"
    write_manifest(rows, manifest)
    write_manifest(rows, audit_dir / "data_manifest.csv")
    yaml_path = write_dataset_yaml(output_root)
    validation = validate_dataset(
        output_root,
        yaml_path=yaml_path,
        manifest_path=manifest,
        use_phash=True,
        phash_threshold=4,
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "Original source-image split assignments retained for baseline items; iRodent labels rederived from visible annotated landmarks. Additional images come only from native iRodent train, excluding baseline held-out IDs/groups/hashes. Reviewed domain frames are train-only.",
        "metric_comparability": "The rodent ground-truth boxes change. Evaluate both baseline and candidate against v2 labels; do not compare v1 mAP to v2 mAP directly.",
        "dataset_yaml": yaml_path.as_posix(),
        "manifest": manifest.as_posix(),
        "images_by_split": dict(Counter(row.split for row in rows)),
        "images_by_source": dict(Counter(row.source_key for row in rows)),
        "boxes_by_class": {
            str(class_id): sum(getattr(row, f"boxes_class_{class_id}") for row in rows)
            for class_id in range(3)
        },
        "additional_native_training_images": len(extras),
        "excluded_baseline_images": skipped,
        "baseline_items_recovered_from_raw": recovered_baseline,
        "domain_root": domain_root.as_posix() if domain_root else None,
        "validation": validation.to_dict(),
    }
    (audit_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not validation.valid:
        raise ValueError(
            f"V2 validation failed; inspect {audit_dir / 'preparation_report.json'}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path, default=Path("dados/processed/drenavision")
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("dados/manifests/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--raw-irodent", type=Path, default=Path("dados/raw/irodent/extracted")
    )
    parser.add_argument("--raw-root", type=Path, default=Path("dados/raw"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("dados/processed/drenavision_v2")
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("resultados/local/adaptation")
    )
    parser.add_argument("--domain-root", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = audit_irodent(args.raw_irodent, args.audit_dir)
    print(json.dumps(report["counts"], indent=2), flush=True)
    if not args.audit_only:
        prepared = prepare_adaptation(
            args.baseline_root,
            args.baseline_manifest,
            args.raw_irodent,
            args.output_root,
            args.audit_dir,
            args.domain_root,
            args.raw_root,
        )
        print(
            json.dumps(
                {key: value for key, value in prepared.items() if key != "validation"},
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
