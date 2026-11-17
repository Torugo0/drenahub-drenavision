"""Prepare versioned water annotations and additional training-only sources.

The source often boxes several overlapping portions of one puddle. V3 derives
one bounding envelope per connected component of touching/overlapping source
water boxes. It introduces no padding or gap threshold, and never uses model
predictions to label training/evaluation data. Original files stay immutable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.converter import (  # noqa: E402
    SourceCandidate,
    collect_water_candidates,
    sha256_file,
)
from drenavision.data.validator import _phash  # noqa: E402
from drenavision.data.yolo import YoloBox, coco_bbox_to_yolo, read_yolo, write_yolo  # noqa: E402

WATER_POLICY = "water-v3: connected original water boxes; intersection or exact edge/corner contact; zero gap; min/max component envelope; no padding"


def _xyxy(box: YoloBox) -> tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2,
        box.y_center - box.height / 2,
        box.x_center + box.width / 2,
        box.y_center + box.height / 2,
    )


def merge_source_water_boxes(
    boxes: tuple[YoloBox, ...] | list[YoloBox],
) -> tuple[YoloBox, ...]:
    """Merge class 1 source components; leave disconnected/other boxes intact.

    Connectivity is calculated on the ORIGINAL boxes, never on growing bounding
    envelopes. Otherwise a rectangle could incorrectly absorb a separate pool
    that only intersects empty space inside a previous merged envelope.
    """
    water = [box for box in boxes if box.class_id == 1]
    other = [box for box in boxes if box.class_id != 1]
    corners = [_xyxy(box) for box in water]
    remaining = set(range(len(water)))
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, pending = [seed], [seed]
        while pending:
            current = pending.pop()
            left, top, right, bottom = corners[current]
            joined = [
                index
                for index in sorted(remaining)
                if min(right, corners[index][2]) >= max(left, corners[index][0])
                and min(bottom, corners[index][3]) >= max(top, corners[index][1])
            ]
            for index in joined:
                remaining.remove(index)
                component.append(index)
                pending.append(index)
        components.append(component)
    merged = []
    for component in components:
        left = max(0.0, min(corners[index][0] for index in component))
        top = max(0.0, min(corners[index][1] for index in component))
        right = min(1.0, max(corners[index][2] for index in component))
        bottom = min(1.0, max(corners[index][3] for index in component))
        box = YoloBox(
            1, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
        )
        box.validate(class_count=3)
        merged.append(box)
    return tuple(other + merged)


def select_extra_water_training(
    raw_extracted: Path,
    baseline_manifest: Path,
    baseline_root: Path,
    *,
    positive_count: int = 400,
    negative_count: int = 0,
    seed: int = 43,
) -> tuple[list[SourceCandidate], dict]:
    """Use remaining raw water images, protecting all held-out source groups.

    The water source has no native split: these are newly assigned TRAIN items,
    not a pre-existing official training set. Baseline val/test image identities
    remain fixed. Near duplicates within four pHash bits of held-out images are
    excluded as well as exact hashes and numbered source blocks.
    """
    if min(positive_count, negative_count) < 0:
        raise ValueError("Requested counts must not be negative")
    if negative_count:
        raise ValueError(
            "Source wet-surface-only labels are not reliable visible-water negatives; "
            "use negative_count=0 and separately reviewed background crops"
        )
    baseline = list(
        csv.DictReader(baseline_manifest.open(encoding="utf-8", newline=""))
    )
    held = [row for row in baseline if row["split"] != "train"]
    held_groups = {row["source_group"] for row in held}
    used_hashes = {row["sha256"] for row in baseline}
    held_phashes = [_phash(baseline_root / row["output_image"]) for row in held]
    positive, negative = collect_water_candidates(raw_extracted)
    counts: Counter = Counter()
    selected: list[SourceCandidate] = []
    selected_records = []
    seen = set(used_hashes)
    rng = random.Random(seed)
    for kind, pool, requested in (
        ("positive", positive, positive_count),
        ("wet_surface_negative", negative, negative_count),
    ):
        # Shuffle source groups, then round-robin across folders and groups.
        # This prevents a large repeated scene from filling the whole quota.
        grouped: dict[str, list[SourceCandidate]] = defaultdict(list)
        for candidate in pool:
            grouped[candidate.source_group].append(candidate)
        folders: dict[str, list[list[SourceCandidate]]] = defaultdict(list)
        for group in sorted(grouped):
            candidates = sorted(
                grouped[group], key=lambda candidate: candidate.source_path
            )
            rng.shuffle(candidates)
            folders[str(Path(candidates[0].source_path).parent)].append(candidates)
        folder_queues = []
        for folder in sorted(folders):
            rng.shuffle(folders[folder])
            folder_queues.append(deque(deque(group) for group in folders[folder]))
        chosen = 0
        while folder_queues and chosen < requested:
            next_folders = []
            for groups in folder_queues:
                if chosen >= requested:
                    break
                group = groups.popleft()
                candidate = group.popleft()
                if group:
                    groups.append(group)
                if groups:
                    next_folders.append(groups)
                if candidate.source_group in held_groups:
                    counts[f"{kind}_held_group_excluded"] += 1
                    continue
                digest = sha256_file(candidate.image)
                if digest in seen:
                    counts[f"{kind}_exact_hash_excluded"] += 1
                    continue
                perceptual = _phash(candidate.image)
                if any(
                    (perceptual ^ held_hash).bit_count() <= 4
                    for held_hash in held_phashes
                ):
                    counts[f"{kind}_held_phash_excluded"] += 1
                    continue
                selected.append(
                    replace(
                        candidate,
                        boxes=merge_source_water_boxes(candidate.boxes),
                        annotation_note=candidate.annotation_note
                        + "; "
                        + WATER_POLICY
                        + "; extra v3 training-only; fixed held-out groups/hashes/pHash protected",
                    )
                )
                seen.add(digest)
                selected_records.append(
                    {
                        "kind": kind,
                        "source_path": candidate.source_path,
                        "source_group": candidate.source_group,
                        "sha256": digest,
                        "original_boxes": len(candidate.boxes),
                        "merged_boxes": len(selected[-1].boxes),
                    }
                )
                chosen += 1
            folder_queues = next_folders
        if chosen != requested:
            raise ValueError(
                f"Only {chosen} safe {kind} images available, requested {requested}"
            )
        counts[f"{kind}_selected"] = chosen
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": WATER_POLICY,
        "seed": seed,
        "baseline_manifest_sha256": sha256_file(baseline_manifest),
        "held_out_images": len(held),
        "held_out_source_groups": len(held_groups),
        "phash_max_distance_excluded": 4,
        "counts": dict(counts),
        "source_folder_counts": dict(
            Counter(str(Path(candidate.source_path).parent) for candidate in selected)
        ),
        "records": selected_records,
        "limitation": "Additional source labels are retained/merged annotations, not individually human-certified masks. The source is 256x256 and does not substitute for new full-frame drainage evaluation.",
    }
    return selected, report


# Full decoded frames inspected at all eight instants. Each rectangle encloses
# the visible continuous water surface. Canal boxes include some diagonal bank,
# an unavoidable limitation of an axis-aligned envelope, not a water mask.
CONTEXT_REVIEW = (
    ("drenagem_chuva", 0, (0, 128, 960, 540)),
    ("drenagem_chuva", 3, (0, 154, 960, 540)),
    ("drenagem_chuva", 6, (0, 0, 960, 540)),
    ("drenagem_chuva", 9, (0, 0, 960, 540)),
    ("canal_residuos", 0, (0, 145, 406, 607)),
    ("canal_residuos", 3, (0, 132, 406, 575)),
    ("canal_residuos", 6, (0, 123, 406, 571)),
    ("canal_residuos", 9, (0, 121, 406, 549)),
)


def prepare_water_context(catalog_path: Path, output_root: Path) -> dict:
    """Reproduce eight manually inspected, fully contextual water frames."""
    catalog = {
        item["id"]: item
        for item in json.loads(catalog_path.read_text(encoding="utf-8"))["items"]
    }
    images, labels = output_root / "images", output_root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1600, 900), "#F1F5F9")
    pen = ImageDraw.Draw(sheet)
    items = []
    for index, (source_id, seconds, coordinates) in enumerate(CONTEXT_REVIEW):
        source = catalog[source_id]
        source_path = catalog_path.parent / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Reviewed water video changed: {source_id}")
        capture = cv2.VideoCapture(str(source_path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError(f"Cannot decode {source_id} at {seconds}s")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = coordinates
        box = coco_bbox_to_yolo((x1, y1, x2 - x1, y2 - y1), width, height, 1)
        stem = f"water_context_{source_id}_{seconds:02}"
        file_name = stem + ".jpg"
        encoded_ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not encoded_ok:
            raise RuntimeError(f"Cannot encode {stem}")
        encoded.tofile(str(images / file_name))
        write_yolo(labels / (stem + ".txt"), [box])
        review = "Visible continuous flowing/flood water; no definite loose trash or animal seen. Fixed roadside bin is infrastructure, not loose litter. One axis-aligned water envelope; not a segmentation mask nor proof that the water is stagnant."
        items.append(
            {
                "file_name": file_name,
                "split": "train",
                "source_id": source_id,
                "timestamp_seconds": seconds,
                "source_group": source_id,
                "source_url": source["source_url"],
                "source_license": source["license"],
                "source_sha256": source["sha256"],
                "author": source["author"],
                "license_url": source["license_url"],
                "image_sha256": sha256_file(images / file_name),
                "water_bbox_xyxy_pixels": list(coordinates),
                "annotation_note": "Manual visual review of full decoded frame, water-visible-v3; "
                + review,
                "evaluation_role": "training_adaptation; entire source clip is excluded from independent test",
            }
        )
        preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(preview).rectangle(coordinates, outline="#0284C7", width=4)
        preview = ImageOps.contain(preview, (400, 420))
        x, y = index % 4 * 400, index // 4 * 450
        sheet.paste(preview, (x + (400 - preview.width) // 2, y + 25))
        pen.text(
            (x + 6, y + 5), f"{source_id} {seconds}s | manual water box", fill="#0F172A"
        )
    sheet.save(output_root / "context_annotations_review.jpg", quality=92)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Eight reviewed full-context water training frames; no model labels",
        "class_semantics": "Canonical class 1 agua_acumulada retained for API compatibility; visual target is visible standing OR flowing water, not a claim of stagnancy/flood hazard",
        "excluded_regression_sources": [
            "sarjeta_residuos",
            "bueiro_agua",
            "ralo_lixo",
            "rato_rua",
        ],
        "items": items,
    }
    (output_root / "sources.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_source_annotation_review(dataset_root: Path, output_path: Path) -> dict:
    """Render fixed source-water examples before/after without editing labels."""
    rows = [
        row
        for row in csv.DictReader(
            (dataset_root / "data_manifest.csv").open(encoding="utf-8", newline="")
        )
        if row["source_key"] == "stagnant_water" and int(row["boxes_class_1"]) > 1
    ]
    chosen = []
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        chosen.extend(
            sorted(
                (row for row in rows if row["split"] == split),
                key=lambda row: row["sha256"],
            )[:count]
        )
    sheet = Image.new("RGB", (1600, 1100), "#F1F5F9")
    pen = ImageDraw.Draw(sheet)
    records = []
    for index, row in enumerate(chosen):
        old = read_yolo(dataset_root / row["output_label"], class_count=3)
        new = merge_source_water_boxes(old)
        with Image.open(dataset_root / row["output_image"]) as opened:
            original = opened.convert("RGB")
        for column, (boxes, color) in enumerate(((old, "#EF4444"), (new, "#0284C7"))):
            image = original.copy()
            draw = ImageDraw.Draw(image)
            for box in boxes:
                left, top, right, bottom = _xyxy(box)
                draw.rectangle(
                    (
                        left * image.width,
                        top * image.height,
                        right * image.width,
                        bottom * image.height,
                    ),
                    outline=color,
                    width=2,
                )
            x, y = index % 4 * 400 + column * 200, index // 4 * 550
            image = ImageOps.contain(image, (198, 500))
            sheet.paste(image, (x, y + 35))
            pen.text(
                (x + 3, y + 5),
                f"{row['split']} {'source' if column == 0 else 'components'} {len(boxes)}",
                fill="#0F172A",
            )
        records.append(
            {
                "image": row["output_image"],
                "source_path": row["source_path"],
                "before": len(old),
                "after": len(new),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "policy": WATER_POLICY,
        "preview": output_path.as_posix(),
        "records": records,
        "limitation": "Annotation-derived envelopes may include dry holes between connected fragments. Visual review is a sample, not a certification of all source annotations.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path("dados/examples/sources.json")
    )
    parser.add_argument(
        "--context-output", type=Path, default=Path("dados/interim/water_v3")
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=Path("dados/processed/drenavision_v2")
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("resultados/local/water_v3")
    )
    parser.add_argument("--extra-positive", type=int, default=400)
    parser.add_argument("--extra-negative", type=int, default=0)
    args = parser.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    context = prepare_water_context(args.catalog, args.context_output)
    review = build_source_annotation_review(
        args.baseline_root, args.audit_dir / "source_water_before_after.jpg"
    )
    _, extra = select_extra_water_training(
        Path("dados/raw/stagnant_water/extracted"),
        args.baseline_root / "data_manifest.csv",
        args.baseline_root,
        positive_count=args.extra_positive,
        negative_count=args.extra_negative,
    )
    report = {
        "context_frames": len(context["items"]),
        "annotation_review": review,
        "extra_source_selection": extra,
    }
    (args.audit_dir / "water_preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "context_frames": len(context["items"]),
                "extra_counts": extra["counts"],
                "review_preview": review["preview"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
