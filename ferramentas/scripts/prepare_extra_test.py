"""Prepare unused native iRodent test images without running a detector.

Images already present in any excluded dataset are rejected by exact hash,
source group and perceptual similarity. The pool must stay out of model tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.data.constants import CLASS_NAMES, IMAGE_SUFFIXES, SOURCES  # noqa: E402
from drenavision.data.converter import (  # noqa: E402
    ManifestRow,
    SourceCandidate,
    collect_irodent,
    collect_water_candidates,
)
from drenavision.data.validator import _phash  # noqa: E402
from drenavision.data.yolo import write_yolo  # noqa: E402
from drenavision.inference.utils import (  # noqa: E402
    safe_artifact_path,
    sha256_file,
    utc_now_iso,
    write_json,
)


def exclusion_reason(
    image_hash: str,
    source_group: str,
    perceptual_hash: int,
    hashes: set[str],
    groups: set[str],
    perceptual_hashes: list[tuple[str, int]],
    threshold: int,
) -> tuple[str, str | None] | None:
    if image_hash in hashes:
        return "exact_image_hash", None
    if source_group in groups:
        return "source_group", source_group
    for reference, value in perceptual_hashes:
        if (value ^ perceptual_hash).bit_count() <= threshold:
            return "perceptual_similarity", reference
    return None


def exclusion_inventory(
    roots: list[Path],
) -> tuple[set[str], set[str], list[tuple[str, int]], list[dict]]:
    hashes, groups, perceptual, inventories = set(), set(), [], []
    seen_paths = set()
    for value in roots:
        root = value.parent if value.is_file() else value
        root = root.resolve()
        if root in seen_paths:
            continue
        seen_paths.add(root)
        manifest = root / "data_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"excluded dataset has no manifest: {root}")
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            groups.add(row["source_group"])
        paths = sorted(
            path
            for path in (root / "images").rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"excluded dataset has no images: {root}")
        fingerprint = hashlib.sha256()
        for path in paths:
            digest = sha256_file(path)
            hashes.add(digest)
            relative = safe_artifact_path(path)
            perceptual.append((relative, _phash(path)))
            fingerprint.update(
                f"{path.relative_to(root).as_posix()}\0{digest}\n".encode()
            )
        inventories.append(
            {
                "root": safe_artifact_path(root),
                "image_count": len(paths),
                "manifest_sha256": sha256_file(manifest),
                "image_inventory_sha256": fingerprint.hexdigest(),
            }
        )
    return hashes, groups, perceptual, inventories


def reproduce_sealed(
    manifest: Path,
    review_path: Path,
    protocol_path: Path,
    raw_irodent: Path,
    raw_water: Path,
    output: Path,
) -> dict[str, Any]:
    """Rebuild the exact pre-inference pool, independent of later train selection."""
    from ferramentas.scripts.prepare_water_v3 import merge_source_water_boxes

    if (output / "dataset.yaml").exists() or (output / "data_manifest.csv").exists():
        raise FileExistsError(f"pool already prepared: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256_file(manifest) != protocol["manifest_sha256"]:
        raise ValueError("sealed manifest hash differs from its protocol")
    if sha256_file(review_path) != protocol["blinded_review_sha256"]:
        raise ValueError("sealed blinded review hash differs from its protocol")
    annotation = raw_irodent / "rodent/annotations/test.json"
    if sha256_file(annotation) != protocol["source_irodent_annotation_sha256"]:
        raise ValueError("native iRodent source annotations changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != protocol["accepted_images"]:
        raise ValueError("sealed image count differs from its protocol")
    rodent = collect_irodent(raw_irodent, annotation)
    water_positive, water_negative = collect_water_candidates(raw_water)
    pools = {
        "irodent": {candidate.source_path: candidate for candidate in rodent},
        "stagnant_water": {
            candidate.source_path: candidate
            for candidate in water_positive + water_negative
        },
    }
    prepared = []
    for row in rows:
        candidate = pools[row["source_key"]][row["source_path"]]
        if candidate.source_group != row["source_group"]:
            raise ValueError(f"source group changed: {row['source_path']}")
        if sha256_file(candidate.image) != row["sha256"]:
            raise ValueError(f"source image changed: {row['source_path']}")
        boxes = (
            merge_source_water_boxes(candidate.boxes)
            if candidate.source_key == "stagnant_water"
            else candidate.boxes
        )
        text = "\n".join(box.to_line() for box in boxes) + ("\n" if boxes else "")
        if protocol.get("label_line_ending") == "CRLF":
            text = text.replace("\n", "\r\n")
        label_bytes = text.encode("utf-8")
        if hashlib.sha256(label_bytes).hexdigest() != row["label_sha256"]:
            raise ValueError(f"derived label changed: {row['source_path']}")
        with Image.open(candidate.image) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"source dimensions changed: {row['source_path']}")
        prepared.append((row, candidate.image, label_bytes))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    excluded = []
    for item in review["exclusions"]:
        candidate = pools["stagnant_water"][item["source_path"]]
        if sha256_file(candidate.image) != item["sha256"]:
            raise ValueError("reviewed exclusion source changed")
        if candidate.boxes:
            raise ValueError(
                "reviewed source-negative exclusion unexpectedly has water labels"
            )
        excluded.append((item, candidate.image))
    # All identity/label checks above finish before creating any dataset files.
    for row, source_image, label_bytes in prepared:
        image = output / row["output_image"]
        label = output / row["output_label"]
        if not image.resolve().is_relative_to(
            output.resolve()
        ) or not label.resolve().is_relative_to(output.resolve()):
            raise ValueError("sealed destination is outside the intended pool")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image)
        label.write_bytes(label_bytes)
    for item, source_image in excluded:
        name = "water_extra_" + item["sha256"][:16]
        destination = (
            output
            / "review_excluded/images/test"
            / (name + source_image.suffix.lower())
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination)
        label = output / "review_excluded/labels/test" / (name + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_bytes(b"")
    fields = list(ManifestRow.__dataclass_fields__)
    with (output / "data_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    (output / "dataset.yaml").write_text(
        "# Additional test only: native iRodent remainder + new water source groups. Never train here.\ntest: images/test\nnames:\n"
        + names
        + "\n",
        encoding="utf-8",
    )
    report = {
        "created_at_utc": utc_now_iso(),
        "status": "prepared_without_inference",
        "reproduction": "exact sealed identities and source-derived label hashes; no randomized reselection",
        "sealed_manifest": safe_artifact_path(manifest),
        "sealed_manifest_sha256": sha256_file(manifest),
        "sealed_protocol": protocol,
        "accepted_images": len(rows),
        "accepted_boxes": {
            name: sum(int(row[f"boxes_class_{class_id}"]) for row in rows)
            for class_id, name in CLASS_NAMES.items()
        },
        "additional_water_or_trash_images": sum(
            row["source_key"] == "stagnant_water" for row in rows
        ),
        "blinded_review": review,
        "review_archived_images": len(excluded),
        "inference_performed": False,
        "isolation_note": "Run --verify-only against every final training dataset before first inference.",
    }
    write_json(output / "blinded_review.json", review)
    write_json(output / "preparation_report.json", report)
    return report


def prepare(
    raw_root: Path, output: Path, exclude_roots: list[Path], threshold: int = 4
) -> dict[str, Any]:
    annotation = raw_root / "rodent/annotations/test.json"
    if not 0 <= threshold <= 64:
        raise ValueError("perceptual threshold must be in [0, 64]")
    if (output / "dataset.yaml").exists() or (output / "data_manifest.csv").exists():
        raise FileExistsError(
            f"pool already prepared: {output}; use a new output directory"
        )
    hashes, groups, perceptual, inventories = exclusion_inventory(exclude_roots)
    candidates = sorted(
        collect_irodent(raw_root, annotation), key=lambda item: item.source_path
    )
    rows, decisions = [], []
    original = json.loads(annotation.read_text(encoding="utf-8"))
    source = SOURCES["irodent"]
    for candidate in candidates:
        digest = sha256_file(candidate.image)
        perceptual_hash = _phash(candidate.image)
        reason = exclusion_reason(
            digest,
            candidate.source_group,
            perceptual_hash,
            hashes,
            groups,
            perceptual,
            threshold,
        )
        if reason:
            decisions.append(
                {
                    "source_path": candidate.source_path,
                    "status": "excluded",
                    "reason": reason[0],
                    "reference": reason[1],
                    "sha256": digest,
                }
            )
            continue
        name = "irodent_extra_" + digest[:16]
        image = output / "images/test" / (name + candidate.image.suffix.lower())
        label = output / "labels/test" / (name + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.image, image)
        write_yolo(label, candidate.boxes)
        with Image.open(candidate.image) as opened:
            width, height = opened.size
        counts = Counter(box.class_id for box in candidate.boxes)
        rows.append(
            ManifestRow(
                image.relative_to(output).as_posix(),
                label.relative_to(output).as_posix(),
                "test",
                "irodent",
                source.version,
                source.landing_url,
                source.license_id,
                candidate.source_path,
                candidate.source_group,
                digest,
                width,
                height,
                counts[0],
                counts[1],
                counts[2],
                candidate.annotation_note,
            )
        )
        # Remove exact/near duplicates within this pool as well as overlap with exclusions.
        hashes.add(digest)
        groups.add(candidate.source_group)
        perceptual.append((safe_artifact_path(image), perceptual_hash))
        decisions.append(
            {
                "source_path": candidate.source_path,
                "status": "accepted",
                "output_image": image.relative_to(output).as_posix(),
                "sha256": digest,
                "phash64": f"{perceptual_hash:016x}",
            }
        )
    if not rows:
        raise ValueError("no unused native-test images survived exclusion checks")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "data_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items())
    (output / "dataset.yaml").write_text(
        "# Additional native test images only. Never train on this pool.\ntest: images/test\nnames:\n"
        + names
        + "\n",
        encoding="utf-8",
    )
    report = {
        "created_at_utc": utc_now_iso(),
        "status": "prepared_without_inference",
        "source_native_split": "test",
        "source_annotation": safe_artifact_path(annotation),
        "source_annotation_sha256": sha256_file(annotation),
        "source_indexed_images": len(original["images"]),
        "usable_candidates": len(candidates),
        "accepted_images": len(rows),
        "accepted_boxes": dict(
            Counter(
                {
                    name: sum(getattr(row, f"boxes_class_{class_id}") for row in rows)
                    for class_id, name in CLASS_NAMES.items()
                }
            )
        ),
        "excluded_by_reason": dict(
            Counter(row["reason"] for row in decisions if row["status"] == "excluded")
        ),
        "excluded_dataset_inventories": inventories,
        "phash_threshold": threshold,
        "within_pool_exact_and_phash_deduplication": True,
        "selection": "All usable unused native test images surviving overlap checks, sorted by source path; no model predictions inspected.",
        "label_caveat": "Boxes are derived from source pose keypoints with visibility >0, with 10% padding. They can cover only a head/partial body and are not independent human-reviewed full-body ground truth.",
        "deployment_caveat": "Same public iRodent domain, not a new urban field study. Once inspected for tuning this pool loses untouched-test status.",
        "additional_water_or_trash_images": 0,
        "decisions": decisions,
    }
    write_json(output / "preparation_report.json", report)
    return report


def diverse_order(
    candidates: list[SourceCandidate], seed: int
) -> list[SourceCandidate]:
    """Round robin folders and source groups before taking another same-group item."""
    rng = random.Random(seed)
    groups: dict[str, list[SourceCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.source_group].append(candidate)
    folders: dict[str, list[deque]] = defaultdict(list)
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda candidate: candidate.source_path)
        rng.shuffle(items)
        folders[str(Path(items[0].source_path).parent)].append(deque(items))
    queues = []
    for folder in sorted(folders):
        rng.shuffle(folders[folder])
        queues.append(deque(folders[folder]))
    ordered = []
    while queues:
        next_queues = []
        for queue in queues:
            group = queue.popleft()
            ordered.append(group.popleft())
            if group:
                queue.append(group)
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return ordered


def water_truth_sheets(output: Path, records: list[dict]) -> list[str]:
    """Render only source references, before inference, for a blinded label review."""
    paths = []
    for offset in range(0, len(records), 12):
        count = min(12, len(records) - offset)
        sheet = Image.new("RGB", (1200, 300 * ((count + 3) // 4)), "#f1f5f9")
        draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(records[offset : offset + 12]):
            from drenavision.data.yolo import read_yolo

            with Image.open(output / row["output_image"]) as opened:
                image = opened.convert("RGB")
            pen = ImageDraw.Draw(image)
            for box in read_yolo(output / row["output_label"], class_count=3):
                pen.rectangle(
                    (
                        (box.x_center - box.width / 2) * image.width,
                        (box.y_center - box.height / 2) * image.height,
                        (box.x_center + box.width / 2) * image.width,
                        (box.y_center + box.height / 2) * image.height,
                    ),
                    outline="#0284c7",
                    width=2,
                )
            image = ImageOps.contain(image, (296, 250))
            x, y = local % 4 * 300, local // 4 * 300
            sheet.paste(image, (x + 2, y))
            draw.text(
                (x + 3, y + 253),
                f"{offset + local:02d} {Path(row['source_path']).name}",
                fill="#0f172a",
            )
            draw.text(
                (x + 3, y + 270),
                "water reference"
                if int(row["boxes_class_1"])
                else "source wet-only negative",
                fill="#0f172a",
            )
        path = output / f"blinded_water_truth_{offset // 12 + 1:02d}.jpg"
        sheet.save(path, quality=94)
        paths.append(safe_artifact_path(path))
    return paths


def append_water(
    output: Path,
    raw_water: Path,
    training_report: Path,
    exclude_roots: list[Path],
    context_root: Path,
    positive_count: int,
    negative_count: int,
    threshold: int = 4,
    seed: int = 44,
) -> dict[str, Any]:
    """Add new source-assigned water references, excluding every planned train group."""
    from ferramentas.scripts.prepare_water_v3 import (
        WATER_POLICY,
        merge_source_water_boxes,
    )

    report_path = output / "preparation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("additional_water_or_trash_images"):
        raise FileExistsError("additional water pool already prepared")
    if min(positive_count, negative_count) < 0:
        raise ValueError("water counts must be nonnegative")
    if negative_count:
        raise ValueError(
            "Unreviewed source wet-only negatives are unreliable; reproduce the sealed reviewed pool instead"
        )
    hashes, groups, perceptual, inventories = exclusion_inventory(
        [*exclude_roots, output]
    )
    original_report = json.loads(training_report.read_text(encoding="utf-8"))
    planned = original_report.get("extra_source_selection", original_report)["records"]
    positives, negatives = collect_water_candidates(raw_water)
    by_source = {
        candidate.source_path: candidate for candidate in positives + negatives
    }
    for item in planned:
        candidate = by_source[item["source_path"]]
        digest = sha256_file(candidate.image)
        if digest != item["sha256"]:
            raise ValueError("planned water-training input hash mismatch")
        hashes.add(digest)
        groups.add(item["source_group"])
        perceptual.append(
            (safe_artifact_path(candidate.image), _phash(candidate.image))
        )
    context_paths = sorted((context_root / "images").glob("*.jpg"))
    for path in context_paths:
        hashes.add(sha256_file(path))
        perceptual.append((safe_artifact_path(path), _phash(path)))
    with (output / "data_manifest.csv").open(encoding="utf-8", newline="") as stream:
        previous = list(csv.DictReader(stream))
    rows, decisions = [], []
    source = SOURCES["stagnant_water"]
    rejected = Counter()
    eligible_counts = {}
    for kind, pool, requested in (
        ("positive", positives, positive_count),
        ("wet_negative", negatives, negative_count),
    ):
        eligible = [
            candidate for candidate in pool if candidate.source_group not in groups
        ]
        eligible_counts[kind] = len(eligible)
        selected = 0
        for candidate in diverse_order(eligible, seed):
            if selected == requested:
                break
            digest = sha256_file(candidate.image)
            phash = _phash(candidate.image)
            reason = exclusion_reason(
                digest,
                candidate.source_group,
                phash,
                hashes,
                groups,
                perceptual,
                threshold,
            )
            if reason:
                rejected[f"{kind}_{reason[0]}"] += 1
                continue
            merged = replace(
                candidate,
                boxes=merge_source_water_boxes(candidate.boxes),
                annotation_note=candidate.annotation_note
                + "; "
                + WATER_POLICY
                + "; extra source-assigned test v3; not used for model selection",
            )
            name = "water_extra_" + digest[:16]
            image = output / "images/test" / (name + candidate.image.suffix.lower())
            label = output / "labels/test" / (name + ".txt")
            shutil.copy2(candidate.image, image)
            write_yolo(label, merged.boxes)
            with Image.open(candidate.image) as opened:
                width, height = opened.size
            rows.append(
                ManifestRow(
                    image.relative_to(output).as_posix(),
                    label.relative_to(output).as_posix(),
                    "test",
                    "stagnant_water",
                    source.version,
                    source.landing_url,
                    source.license_id,
                    candidate.source_path,
                    candidate.source_group,
                    digest,
                    width,
                    height,
                    0,
                    len(merged.boxes),
                    0,
                    merged.annotation_note,
                )
            )
            hashes.add(digest)
            perceptual.append((safe_artifact_path(image), phash))
            # Same source group can recur only within this test pool, never in training.
            decisions.append(
                {
                    "kind": kind,
                    "source_path": candidate.source_path,
                    "source_group": candidate.source_group,
                    "sha256": digest,
                    "output_image": image.relative_to(output).as_posix(),
                    "original_boxes": len(candidate.boxes),
                    "merged_boxes": len(merged.boxes),
                }
            )
            selected += 1
        if selected != requested:
            raise ValueError(
                f"only {selected} safe {kind} inputs available; requested {requested}"
            )
    combined = previous + [asdict(row) for row in rows]
    temporary = output / "data_manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    temporary.replace(output / "data_manifest.csv")
    report.update(
        {
            "updated_at_utc": utc_now_iso(),
            "source_native_split": {
                "irodent": "official native test remainder",
                "stagnant_water": "no official split; new source-assigned test groups",
            },
            "accepted_images": len(combined),
            "usable_native_rodent_candidates": report["usable_candidates"],
            "usable_candidates": report["usable_candidates"]
            + sum(eligible_counts.values()),
            "accepted_boxes": {
                name: sum(int(row[f"boxes_class_{class_id}"]) for row in combined)
                for class_id, name in CLASS_NAMES.items()
            },
            "additional_water_or_trash_images": len(rows),
            "water_selection": {
                "positive_images": positive_count,
                "wet_negative_images": negative_count,
                "seed": seed,
                "eligible_water_inputs_before_similarity_checks": eligible_counts,
                "source_group_count": len({row.source_group for row in rows}),
                "policy": WATER_POLICY,
                "planned_training_report": safe_artifact_path(training_report),
                "planned_training_report_sha256": sha256_file(training_report),
                "planned_training_inputs_excluded": len(planned),
                "context_inputs_excluded": len(context_paths),
                "exclusion_inventories": inventories,
                "rejected": dict(rejected),
                "records": decisions,
                "blinded_truth_sheets": water_truth_sheets(
                    output, [asdict(row) for row in rows]
                ),
                "caveat": "No official water test split exists. Source labels merged by connected-box protocol are not certified water masks; wet-only negatives require blinded visual review for omitted water.",
            },
        }
    )
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        dataset_yaml.read_text(encoding="utf-8").replace(
            "# Additional native test images only. Never train on this pool.",
            "# Additional test only: native iRodent remainder + new water source groups. Never train here.",
        ),
        encoding="utf-8",
    )
    write_json(report_path, report)
    return report


def apply_blinded_review(output: Path, review_path: Path) -> dict[str, Any]:
    """Archive explicitly reviewed uncertain inputs before any predictions are seen."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("reviewed_before_inference") is not True:
        raise ValueError(
            "review must explicitly declare that no predictions were inspected"
        )
    with (output / "data_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_hash = {row["sha256"]: row for row in rows}
    exclusions = review["exclusions"]
    for item in exclusions:
        row = by_hash.get(item["sha256"])
        if row is None or row["source_path"] != item["source_path"]:
            raise ValueError("blinded review does not match a current source input")
        if not item.get("reason"):
            raise ValueError("each blinded exclusion requires a concrete visual reason")
        for key in ("output_image", "output_label"):
            if not (output / row[key]).resolve().is_relative_to(output.resolve()):
                raise ValueError("review source is outside the intended pool")
        image = output / row["output_image"]
        if sha256_file(image) != row["sha256"]:
            raise ValueError("reviewed input changed")
    rejected = {item["sha256"] for item in exclusions}
    retained = [row for row in rows if row["sha256"] not in rejected]
    if not retained:
        raise ValueError("review would remove the entire pool")
    archive = output / "review_excluded"
    for digest in rejected:
        row = by_hash[digest]
        for key in ("output_image", "output_label"):
            source = (output / row[key]).resolve()
            if not source.is_relative_to(output.resolve()):
                raise ValueError("review archive source is outside the intended pool")
            destination = archive / row[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
    with (output / "data_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(retained[0]))
        writer.writeheader()
        writer.writerows(retained)
    report_path = output / "preparation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    water = [row for row in retained if row["source_key"] == "stagnant_water"]
    report.update(
        {
            "updated_at_utc": utc_now_iso(),
            "accepted_images": len(retained),
            "accepted_boxes": {
                name: sum(int(row[f"boxes_class_{class_id}"]) for row in retained)
                for class_id, name in CLASS_NAMES.items()
            },
            "additional_water_or_trash_images": len(water),
            "blinded_review": {
                "reviewer": review.get("reviewer"),
                "review_file": safe_artifact_path(review_path),
                "review_sha256": sha256_file(review_path),
                "reviewed_before_inference": True,
                "excluded_images": len(rejected),
                "exclusions": exclusions,
                "retained_positive_water_images": sum(
                    int(row["boxes_class_1"]) > 0 for row in water
                ),
                "retained_wet_negative_images": sum(
                    int(row["boxes_class_1"]) == 0 for row in water
                ),
                "archive": safe_artifact_path(archive),
            },
        }
    )
    write_json(output / "blinded_review.json", review)
    write_json(report_path, report)
    return report


def verify_pool(
    output: Path, exclude_roots: list[Path], threshold: int = 4
) -> dict[str, Any]:
    hashes, groups, perceptual, inventories = exclusion_inventory(exclude_roots)
    with (output / "data_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    violations = []
    for row in rows:
        path = output / row["output_image"]
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(f"extra test image changed: {path.name}")
        reason = exclusion_reason(
            digest,
            row["source_group"],
            _phash(path),
            hashes,
            groups,
            perceptual,
            threshold,
        )
        if reason:
            violations.append(
                {
                    "image": row["output_image"],
                    "reason": reason[0],
                    "reference": reason[1],
                }
            )
    report = {
        "created_at_utc": utc_now_iso(),
        "status": "passed" if not violations else "failed",
        "pool": safe_artifact_path(output),
        "pool_images": len(rows),
        "manifest_sha256": sha256_file(output / "data_manifest.csv"),
        "excluded_dataset_inventories": inventories,
        "phash_threshold": threshold,
        "violations": violations,
        "inference_performed": False,
    }
    write_json(output / "final_isolation_check.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("dados/raw/irodent/extracted")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dados/processed/extra_test_v3")
    )
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        action="append",
        default=[Path("dados/processed/drenavision_v2")],
        help="Dataset root or dataset.yaml; repeat for every new training dataset. V2 is always included.",
    )
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--apply-blinded-review", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--exploratory-new-selection",
        action="store_true",
        help="Create a different exploratory pool; default reproduces the sealed 119 identities.",
    )
    parser.add_argument(
        "--sealed-manifest",
        type=Path,
        default=Path("resultados/metrics/v3/extra_test_manifest.csv"),
    )
    parser.add_argument(
        "--sealed-review",
        type=Path,
        default=Path("resultados/metrics/v3/extra_test_blinded_review.json"),
    )
    parser.add_argument(
        "--sealed-protocol",
        type=Path,
        default=Path("resultados/metrics/v3/extra_test_protocol.json"),
    )
    parser.add_argument("--water-positive", type=int, default=0)
    parser.add_argument("--water-negative", type=int, default=0)
    parser.add_argument(
        "--water-root", type=Path, default=Path("dados/raw/stagnant_water/extracted")
    )
    parser.add_argument(
        "--water-training-report",
        type=Path,
        default=Path("resultados/local/water_v3/water_preparation_report.json"),
    )
    parser.add_argument(
        "--water-context-root", type=Path, default=Path("dados/interim/water_v3")
    )
    args = parser.parse_args()
    if args.apply_blinded_review:
        report = apply_blinded_review(args.output, args.apply_blinded_review)
        print(
            json.dumps(
                {
                    "accepted_images": report["accepted_images"],
                    "blinded_review": report["blinded_review"],
                }
            )
        )
        return 0
    if args.verify_only:
        report = verify_pool(args.output, args.exclude_dataset, args.phash_threshold)
        print(
            json.dumps(
                {key: report[key] for key in ("status", "pool_images", "violations")}
            )
        )
        return int(report["status"] != "passed")
    if not args.exploratory_new_selection:
        report = reproduce_sealed(
            args.sealed_manifest,
            args.sealed_review,
            args.sealed_protocol,
            args.raw_root,
            args.water_root,
            args.output,
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("status", "accepted_images", "accepted_boxes")
                }
            )
        )
        return 0
    add_water = args.water_positive > 0 or args.water_negative > 0
    if not (args.output / "preparation_report.json").exists() or not add_water:
        report = prepare(
            args.raw_root, args.output, args.exclude_dataset, args.phash_threshold
        )
    if add_water:
        report = append_water(
            args.output,
            args.water_root,
            args.water_training_report,
            args.exclude_dataset,
            args.water_context_root,
            args.water_positive,
            args.water_negative,
            args.phash_threshold,
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "usable_candidates",
                    "accepted_images",
                    "accepted_boxes",
                    "excluded_by_reason",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
