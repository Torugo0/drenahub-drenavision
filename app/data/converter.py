"""Merge approved sources into one leakage-aware YOLO dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image

from .constants import CLASS_NAMES, IMAGE_SUFFIXES, SOURCES, SPLITS
from .yolo import YoloBox, coco_bbox_to_yolo, read_yolo, write_yolo

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    image: Path
    boxes: tuple[YoloBox, ...]
    source_key: str
    source_path: str
    source_group: str
    annotation_note: str


@dataclass(frozen=True, slots=True)
class ManifestRow:
    output_image: str
    output_label: str
    split: str
    source_key: str
    source_version: str
    source_url: str
    source_license: str
    source_path: str
    source_group: str
    sha256: str
    width: int
    height: int
    boxes_class_0: int
    boxes_class_1: int
    boxes_class_2: int
    annotation_note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number_block(path: Path, block_size: int = 10) -> str:
    numbers = re.findall(r"\d+", path.stem)
    block = int(numbers[-1]) // block_size if numbers else path.stem
    return f"{path.parent.as_posix()}::block-{block}"


def _read_source_yolo(path: Path, *, class_count: int) -> tuple[list[YoloBox], int]:
    """Read third-party labels while accounting for degenerate source boxes.

    Final labels are always strict.  A source line with a zero-area or otherwise
    invalid box is excluded explicitly and counted in the per-image audit note.
    """

    boxes: list[YoloBox] = []
    rejected = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            box = YoloBox.parse(line, line_number=line_number)
            box.validate(line_number=line_number, class_count=class_count)
        except ValueError:
            rejected += 1
            continue
        boxes.append(box)
    return boxes, rejected


def _grouped_select(
    candidates: Sequence[SourceCandidate],
    quotas: dict[str, int],
    *,
    seed: int,
) -> dict[str, list[SourceCandidate]]:
    by_group: dict[str, list[SourceCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.source_group].append(candidate)
    group_names = sorted(by_group)
    random.Random(seed).shuffle(group_names)
    selected: dict[str, list[SourceCandidate]] = {split: [] for split in SPLITS}
    cursor = 0
    for split in SPLITS:
        target = quotas.get(split, 0)
        while len(selected[split]) < target and cursor < len(group_names):
            group = sorted(
                by_group[group_names[cursor]], key=lambda item: item.source_path
            )
            cursor += 1
            remaining = target - len(selected[split])
            selected[split].extend(group[:remaining])
        if len(selected[split]) != target:
            raise RuntimeError(
                f"could not select {target} grouped samples for {split}; "
                f"got {len(selected[split])}"
            )
    return selected


def _deduplicate_candidates(
    candidates: Sequence[SourceCandidate],
    seen_hashes: set[str] | None = None,
) -> list[SourceCandidate]:
    """Keep the first deterministic occurrence of each exact image."""

    seen = seen_hashes if seen_hashes is not None else set()
    unique: list[SourceCandidate] = []
    for candidate in candidates:
        digest = sha256_file(candidate.image)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(candidate)
    return unique


def collect_water_candidates(
    extracted_root: Path,
) -> tuple[list[SourceCandidate], list[SourceCandidate]]:
    dataset_root = extracted_root / "Stagnant Water and Wet Surface Dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"water dataset not extracted: {dataset_root}")
    positives: list[SourceCandidate] = []
    negatives: list[SourceCandidate] = []
    for label_path in sorted(dataset_root.rglob("*.txt")):
        if label_path.name == "classes.txt" or "Raw data" in label_path.parts:
            continue
        image_path = next(
            (
                label_path.with_suffix(suffix)
                for suffix in (".jpeg", ".jpg", ".JPG", ".png")
                if label_path.with_suffix(suffix).exists()
            ),
            None,
        )
        if image_path is None:
            LOGGER.warning("skipping orphan water label without image: %s", label_path)
            continue
        source_boxes, rejected_boxes = _read_source_yolo(label_path, class_count=2)
        water_boxes = tuple(box.remap(1) for box in source_boxes if box.class_id == 0)
        relative = image_path.relative_to(dataset_root).as_posix()
        candidate = SourceCandidate(
            image=image_path,
            boxes=water_boxes,
            source_key="stagnant_water",
            source_path=relative,
            source_group=_number_block(Path(relative)),
            annotation_note=(
                "source YOLO class water retained; wet-surface boxes dropped"
                + (
                    f"; {rejected_boxes} invalid source box(es) dropped"
                    if rejected_boxes
                    else ""
                )
            ),
        )
        if water_boxes:
            positives.append(candidate)
        elif source_boxes and all(box.class_id == 1 for box in source_boxes):
            negatives.append(
                replace(candidate, annotation_note="wet-surface-only negative")
            )
    return positives, negatives


def select_water(
    extracted_root: Path,
    *,
    seed: int = 42,
    quotas: dict[str, int] | None = None,
    negative_quotas: dict[str, int] | None = None,
    audit: dict[str, int] | None = None,
) -> dict[str, list[SourceCandidate]]:
    quotas = quotas or {"train": 100, "val": 20, "test": 20}
    negative_quotas = negative_quotas or {"train": 8, "val": 2, "test": 2}
    positives, negatives = collect_water_candidates(extracted_root)
    candidate_count = len(positives) + len(negatives)
    source_hashes: set[str] = set()
    positives = _deduplicate_candidates(positives, source_hashes)
    negatives = _deduplicate_candidates(negatives, source_hashes)
    if audit is not None:
        audit.update(
            usable_candidates=candidate_count,
            exact_duplicates_excluded=candidate_count - len(positives) - len(negatives),
        )
    selected = _grouped_select(positives, quotas, seed=seed)
    negative_selected = _grouped_select(negatives, negative_quotas, seed=seed + 101)
    for split in SPLITS:
        selected[split].extend(negative_selected[split])
    return selected


def collect_click(click_root: Path) -> dict[str, list[SourceCandidate]]:
    selected: dict[str, list[SourceCandidate]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        image_root = click_root / "images" / split
        label_root = click_root / "labels" / split
        if not image_root.exists() or not label_root.exists():
            raise FileNotFoundError(f"CLICK sample missing split: {split}")
        for image_path in sorted(
            path
            for path in image_root.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            label_path = label_root / f"{image_path.stem}.txt"
            boxes = tuple(box.remap(0) for box in read_yolo(label_path, class_count=1))
            selected[split].append(
                SourceCandidate(
                    image=image_path,
                    boxes=boxes,
                    source_key="click_trash",
                    source_path=f"images/{split}/{image_path.name}",
                    source_group=f"click::{image_path.stem}",
                    annotation_note=(
                        "reviewed single-class YOLO trash annotation"
                        if boxes
                        else "published CLICK negative sample with empty label"
                    ),
                )
            )
    return selected


def _find_coco_jsons(extracted_root: Path) -> dict[str, tuple[Path, dict[str, object]]]:
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    for path in extracted_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("images"), list)
            and isinstance(payload.get("annotations"), list)
        ):
            candidates.append((len(payload["images"]), path, payload))
    if not candidates:
        raise FileNotFoundError(f"no COCO annotation JSON found under {extracted_root}")
    return {
        path.stem.lower(): (path, payload)
        for _, path, payload in sorted(candidates, key=lambda item: item[1].as_posix())
    }


def collect_irodent(
    extracted_root: Path, annotation_path: Path | None = None
) -> list[SourceCandidate]:
    if annotation_path is None:
        discovered = _find_coco_jsons(extracted_root)
        json_path, coco = max(
            discovered.values(), key=lambda item: len(item[1].get("images", []))
        )
    else:
        json_path = annotation_path
        coco = json.loads(json_path.read_text(encoding="utf-8"))
    image_paths: dict[str, list[Path]] = defaultdict(list)
    for path in extracted_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths[path.name].append(path)

    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in coco["annotations"]:  # type: ignore[index]
        if (
            isinstance(annotation, dict)
            and "image_id" in annotation
            and "bbox" in annotation
        ):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    candidates: list[SourceCandidate] = []
    for image in coco["images"]:  # type: ignore[index]
        if not isinstance(image, dict):
            continue
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        direct_candidates = [
            json_path.parent / file_name,
            extracted_root / file_name,
        ]
        image_path = next((path for path in direct_candidates if path.exists()), None)
        if image_path is None:
            basename_matches = image_paths.get(Path(file_name).name, [])
            if len(basename_matches) == 1:
                image_path = basename_matches[0]
        if image_path is None:
            raise FileNotFoundError(f"iRodent image not found: {file_name}")
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            with Image.open(image_path) as opened:
                width, height = opened.size
        boxes: list[YoloBox] = []
        for annotation in annotations_by_image.get(image_id, []):
            try:
                boxes.append(coco_bbox_to_yolo(annotation["bbox"], width, height, 2))
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning(
                    "skipping invalid iRodent annotation %s for %s: %s",
                    annotation.get("id", "unknown"),
                    file_name,
                    exc,
                )
                continue
        if not boxes:
            continue
        candidates.append(
            SourceCandidate(
                image=image_path,
                boxes=tuple(boxes),
                source_key="irodent",
                source_path=f"annotations/{json_path.name}::{file_name}",
                source_group=f"irodent::{file_name}",
                annotation_note="published COCO bounding box mapped to animal_visivel",
            )
        )
    if not candidates:
        raise RuntimeError("iRodent COCO file contained no usable bounding boxes")
    return candidates


def select_irodent(
    extracted_root: Path,
    *,
    seed: int = 42,
    quotas: dict[str, int] | None = None,
    audit: dict[str, int] | None = None,
) -> dict[str, list[SourceCandidate]]:
    quotas = quotas or {"train": 100, "val": 20, "test": 20}
    discovered = _find_coco_jsons(extracted_root)
    selected: dict[str, list[SourceCandidate]] = {split: [] for split in SPLITS}
    if {"train", "test"}.issubset(discovered):
        training_pool = collect_irodent(extracted_root, discovered["train"][0])
        test_pool = collect_irodent(extracted_root, discovered["test"][0])
        candidate_count = len(training_pool) + len(test_pool)
        source_hashes: set[str] = set()
        # Preserve the source's held-out test images if an exact duplicate also
        # exists in train; the duplicate training occurrence is the one removed.
        test_pool = _deduplicate_candidates(test_pool, source_hashes)
        training_pool = _deduplicate_candidates(training_pool, source_hashes)
        if audit is not None:
            audit.update(
                usable_candidates=candidate_count,
                exact_duplicates_excluded=(
                    candidate_count - len(training_pool) - len(test_pool)
                ),
            )
        random.Random(seed).shuffle(training_pool)
        random.Random(seed + 1).shuffle(test_pool)
        required_training = quotas.get("train", 0) + quotas.get("val", 0)
        if len(training_pool) < required_training or len(test_pool) < quotas.get(
            "test", 0
        ):
            raise RuntimeError(
                "iRodent native splits do not contain enough usable annotated images"
            )
        train_count = quotas.get("train", 0)
        selected["train"] = training_pool[:train_count]
        selected["val"] = training_pool[train_count:required_training]
        selected["test"] = test_pool[: quotas.get("test", 0)]
    else:
        raw_candidates = collect_irodent(extracted_root)
        candidates = _deduplicate_candidates(raw_candidates)
        if audit is not None:
            audit.update(
                usable_candidates=len(raw_candidates),
                exact_duplicates_excluded=len(raw_candidates) - len(candidates),
            )
        random.Random(seed).shuffle(candidates)
        required = sum(quotas.values())
        if len(candidates) < required:
            raise RuntimeError(
                f"iRodent has {len(candidates)} usable images, fewer than requested {required}"
            )
        cursor = 0
        for split in SPLITS:
            count = quotas.get(split, 0)
            selected[split] = candidates[cursor : cursor + count]
            cursor += count
    return selected


def _prepare_output(output_root: Path) -> None:
    resolved = output_root.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as dataset output")
    for kind in ("images", "labels"):
        target = output_root / kind
        if target.exists():
            for attempt in range(8):
                try:
                    shutil.rmtree(target)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.25 * (attempt + 1))
        for split in SPLITS:
            (target / split).mkdir(parents=True, exist_ok=True)


def _write_candidate(
    candidate: SourceCandidate,
    split: str,
    output_root: Path,
    seen_hashes: dict[str, str],
) -> ManifestRow:
    digest = sha256_file(candidate.image)
    if digest in seen_hashes:
        raise RuntimeError(
            f"exact duplicate across selected data: {candidate.source_path} and "
            f"{seen_hashes[digest]}"
        )
    seen_hashes[digest] = candidate.source_path
    suffix = candidate.image.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    output_stem = f"{candidate.source_key}_{digest[:16]}"
    output_image = output_root / "images" / split / f"{output_stem}{suffix}"
    output_label = output_root / "labels" / split / f"{output_stem}.txt"
    shutil.copy2(candidate.image, output_image)
    write_yolo(output_label, candidate.boxes)
    with Image.open(output_image) as opened:
        width, height = opened.size
    counts = Counter(box.class_id for box in candidate.boxes)
    source = SOURCES[candidate.source_key]
    return ManifestRow(
        output_image=output_image.relative_to(output_root).as_posix(),
        output_label=output_label.relative_to(output_root).as_posix(),
        split=split,
        source_key=candidate.source_key,
        source_version=source.version,
        source_url=source.landing_url,
        source_license=source.license_id,
        source_path=candidate.source_path,
        source_group=candidate.source_group,
        sha256=digest,
        width=width,
        height=height,
        boxes_class_0=counts[0],
        boxes_class_1=counts[1],
        boxes_class_2=counts[2],
        annotation_note=candidate.annotation_note,
    )


def write_manifest(rows: Iterable[ManifestRow], manifest_path: Path) -> None:
    rows = list(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(ManifestRow.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_dataset_yaml(output_root: Path) -> Path:
    """Write the portable YAML consumed directly by Ultralytics.

    Omitting ``path`` makes Ultralytics resolve splits from this YAML's parent,
    so the processed directory can be moved without embedding an absolute path.
    """

    yaml_path = output_root / "dataset.yaml"
    names = "\n".join(
        f"  {class_id}: {class_name}" for class_id, class_name in CLASS_NAMES.items()
    )
    yaml_path.write_text(
        "# Generated by scripts/convert_annotations.py; do not edit by hand.\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def convert_dataset(
    raw_root: Path = Path("datasets/raw"),
    output_root: Path = Path("datasets/processed/drenavision"),
    manifest_path: Path = Path("datasets/manifests/dataset_manifest.csv"),
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Convert all downloaded sources and return measured statistics."""

    water_audit: dict[str, int] = {}
    irodent_audit: dict[str, int] = {}
    selections = {
        "stagnant_water": select_water(
            raw_root / "stagnant_water" / "extracted", seed=seed, audit=water_audit
        ),
        "click_trash": collect_click(raw_root / "click_trash" / "sample"),
        "irodent": select_irodent(
            raw_root / "irodent" / "extracted", seed=seed, audit=irodent_audit
        ),
    }
    _prepare_output(output_root)
    rows: list[ManifestRow] = []
    seen_hashes: dict[str, str] = {}
    for source_key in ("click_trash", "stagnant_water", "irodent"):
        for split in SPLITS:
            for candidate in selections[source_key][split]:
                rows.append(
                    _write_candidate(candidate, split, output_root, seen_hashes)
                )
    dataset_yaml = write_dataset_yaml(output_root)
    write_manifest(rows, manifest_path)
    summary: dict[str, object] = {
        "seed": seed,
        "images": len(rows),
        "images_by_split": dict(Counter(row.split for row in rows)),
        "images_by_source": dict(Counter(row.source_key for row in rows)),
        "boxes_by_class": {
            CLASS_NAMES[class_id]: sum(
                getattr(row, f"boxes_class_{class_id}") for row in rows
            )
            for class_id in CLASS_NAMES
        },
        "negative_images": sum(
            row.boxes_class_0 + row.boxes_class_1 + row.boxes_class_2 == 0
            for row in rows
        ),
        "source_pool_audit": {
            "stagnant_water": water_audit,
            "click_trash": {
                "downloaded_candidates": sum(
                    len(selections["click_trash"][split]) for split in SPLITS
                )
            },
            "irodent": irodent_audit,
        },
        "dataset_yaml": dataset_yaml.as_posix(),
        "manifest": manifest_path.as_posix(),
    }
    summary_path = manifest_path.with_name("dataset_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
