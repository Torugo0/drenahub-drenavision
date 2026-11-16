"""Strict integrity and leakage checks for a YOLO detection dataset."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from .constants import CLASS_NAMES, IMAGE_SUFFIXES, SPLITS
from .yolo import read_yolo


def _portable_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(slots=True)
class ValidationReport:
    dataset_root: str
    issues: list[ValidationIssue] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(
        self, severity: str, code: str, message: str, path: Path | str | None = None
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                path=_portable_path(path) if path is not None else None,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "dataset_root": self.dataset_root,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "statistics": self.statistics,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phash(path: Path) -> int:
    """Return the conventional 64-bit DCT perceptual hash.

    NumPy is imported lazily because exact-hash validation remains dependency
    light and is the default path.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional CLI mode
        raise RuntimeError("--phash requires numpy") from exc

    size = 32
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    transform = np.cos(np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * size))
    coefficients = transform @ pixels @ transform.T
    low = coefficients[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    result = 0
    for bit in (low > median).reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _validate_yaml(
    yaml_path: Path, dataset_root: Path, report: ValidationReport
) -> None:
    if not yaml_path.exists():
        report.add("error", "missing_yaml", "dataset YAML does not exist", yaml_path)
        return
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        report.add("error", "invalid_yaml", str(exc), yaml_path)
        return
    if not isinstance(payload, dict):
        report.add(
            "error", "invalid_yaml", "top-level YAML value must be a mapping", yaml_path
        )
        return
    names = payload.get("names")
    normalized_names: dict[int, str] = {}
    if isinstance(names, dict):
        try:
            normalized_names = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError):
            normalized_names = {}
    elif isinstance(names, list):
        normalized_names = dict(enumerate(str(value) for value in names))
    if normalized_names != CLASS_NAMES:
        report.add(
            "error",
            "class_schema_mismatch",
            f"expected class mapping {CLASS_NAMES}, found {normalized_names}",
            yaml_path,
        )

    configured_root = Path(str(payload.get("path", ".")))
    if not configured_root.is_absolute():
        configured_root = (yaml_path.parent / configured_root).resolve()
    if configured_root != dataset_root.resolve():
        report.add(
            "error",
            "dataset_path_mismatch",
            f"YAML resolves to {configured_root}, validator received {dataset_root.resolve()}",
            yaml_path,
        )
    for split in SPLITS:
        expected = Path("images") / split
        configured = payload.get(split)
        if (
            configured is None
            or Path(str(configured)).as_posix() != expected.as_posix()
        ):
            report.add(
                "error",
                "split_path_mismatch",
                f"{split} must be configured as {expected.as_posix()}",
                yaml_path,
            )


def _validate_manifest(
    manifest_path: Path,
    dataset_root: Path,
    image_hashes: dict[str, str],
    image_splits: dict[str, str],
    report: ValidationReport,
) -> None:
    if not manifest_path.exists():
        report.add(
            "error",
            "missing_manifest",
            "dataset manifest does not exist",
            manifest_path,
        )
        return
    required = {
        "output_image",
        "output_label",
        "split",
        "source_key",
        "source_version",
        "source_url",
        "source_license",
        "source_path",
        "source_group",
        "sha256",
    }
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing_columns = required - set(reader.fieldnames or ())
            if missing_columns:
                report.add(
                    "error",
                    "manifest_columns",
                    f"missing columns: {sorted(missing_columns)}",
                    manifest_path,
                )
                return
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        report.add("error", "invalid_manifest", str(exc), manifest_path)
        return

    manifest_images: set[str] = set()
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, 2):
        image_rel = Path(row["output_image"]).as_posix()
        label_rel = Path(row["output_label"]).as_posix()
        split = row["split"]
        if image_rel in manifest_images:
            report.add(
                "error",
                "duplicate_manifest_row",
                f"duplicate output_image at CSV row {row_number}",
                manifest_path,
            )
        manifest_images.add(image_rel)
        if split not in SPLITS or image_splits.get(image_rel) != split:
            report.add(
                "error",
                "manifest_split_mismatch",
                f"CSV row {row_number} has inconsistent split {split!r}",
                manifest_path,
            )
        if not (dataset_root / label_rel).is_file():
            report.add(
                "error",
                "manifest_missing_label",
                f"CSV row {row_number} points to a missing label",
                dataset_root / label_rel,
            )
        actual_hash = image_hashes.get(image_rel)
        if actual_hash is not None and row["sha256"].lower() != actual_hash:
            report.add(
                "error",
                "manifest_hash_mismatch",
                f"CSV row {row_number} SHA-256 does not match the image",
                dataset_root / image_rel,
            )
        group_splits[(row["source_key"], row["source_group"])].add(split)

    filesystem_images = set(image_hashes)
    for missing in sorted(filesystem_images - manifest_images):
        report.add(
            "error",
            "untracked_image",
            "image is absent from manifest",
            dataset_root / missing,
        )
    for extra in sorted(manifest_images - filesystem_images):
        report.add(
            "error",
            "manifest_missing_image",
            "manifest image does not exist",
            dataset_root / extra,
        )
    for (source, group), splits in sorted(group_splits.items()):
        if len(splits) > 1:
            report.add(
                "error",
                "source_group_leakage",
                f"source group {source}/{group} occurs in splits {sorted(splits)}",
                manifest_path,
            )


def validate_dataset(
    dataset_root: Path,
    *,
    yaml_path: Path | None = None,
    manifest_path: Path | None = None,
    use_phash: bool = False,
    phash_threshold: int = 4,
) -> ValidationReport:
    """Validate layout, images, labels, provenance and cross-split leakage."""

    report = ValidationReport(dataset_root=_portable_path(dataset_root))
    dataset_root = dataset_root.resolve()
    if phash_threshold < 0 or phash_threshold > 64:
        raise ValueError("phash_threshold must be between 0 and 64")

    image_hashes: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    phashes: list[tuple[str, str, int]] = []
    image_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    box_counts: Counter[int] = Counter()
    boxes_by_split: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    negative_images = 0

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            report.add(
                "error",
                "missing_image_dir",
                "image split directory is missing",
                image_dir,
            )
            continue
        if not label_dir.is_dir():
            report.add(
                "error",
                "missing_label_dir",
                "label split directory is missing",
                label_dir,
            )
            continue
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        image_by_stem: dict[str, Path] = {}
        for image_path in images:
            if image_path.stem in image_by_stem:
                report.add(
                    "error",
                    "ambiguous_image_stem",
                    f"multiple image extensions share stem {image_path.stem!r}",
                    image_path,
                )
            image_by_stem[image_path.stem] = image_path
        label_by_stem = {path.stem: path for path in labels}
        for stem in sorted(set(image_by_stem) - set(label_by_stem)):
            report.add(
                "error", "missing_label", "image has no YOLO label", image_by_stem[stem]
            )
        for stem in sorted(set(label_by_stem) - set(image_by_stem)):
            report.add(
                "error",
                "orphan_label",
                "label has no corresponding image",
                label_by_stem[stem],
            )

        image_counts[split] = len(images)
        label_counts[split] = len(labels)
        for image_path in images:
            relative = image_path.relative_to(dataset_root).as_posix()
            try:
                with Image.open(image_path) as opened:
                    opened.verify()
                with Image.open(image_path) as opened:
                    width, height = opened.size
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive dimensions")
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                report.add("error", "corrupt_image", str(exc), image_path)
                continue
            digest = _sha256(image_path)
            image_hashes[relative] = digest
            image_splits[relative] = split
            hashes[digest].append((split, relative))
            if use_phash:
                try:
                    phashes.append((split, relative, _phash(image_path)))
                except (OSError, RuntimeError, UnidentifiedImageError) as exc:
                    report.add("error", "phash_failed", str(exc), image_path)

            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            try:
                boxes = read_yolo(label_path, class_count=len(CLASS_NAMES))
            except (OSError, UnicodeError, ValueError) as exc:
                report.add("error", "invalid_yolo", str(exc), label_path)
                continue
            if not boxes:
                negative_images += 1
            for box in boxes:
                box_counts[box.class_id] += 1
                boxes_by_split[split][box.class_id] += 1

    for _digest, occurrences in hashes.items():
        if len(occurrences) > 1:
            report.add(
                "error",
                "duplicate_sha256",
                f"identical image occurs {len(occurrences)} times: {occurrences}",
            )

    if use_phash:
        for index, (left_split, left_path, left_hash) in enumerate(phashes):
            for right_split, right_path, right_hash in phashes[index + 1 :]:
                if left_split == right_split:
                    continue
                distance = (left_hash ^ right_hash).bit_count()
                if distance <= phash_threshold:
                    report.add(
                        "error",
                        "perceptual_split_leakage",
                        f"pHash distance {distance} across {left_split}/{right_split}: "
                        f"{left_path} and {right_path}",
                    )

    for class_id, class_name in CLASS_NAMES.items():
        if box_counts[class_id] == 0:
            report.add(
                "error", "class_without_boxes", f"no boxes found for {class_name}"
            )
        for split in SPLITS:
            if boxes_by_split[split][class_id] == 0:
                report.add(
                    "warning",
                    "split_without_class",
                    f"{split} contains no {class_name} boxes",
                )

    if yaml_path is not None:
        _validate_yaml(yaml_path.resolve(), dataset_root, report)
    if manifest_path is not None:
        _validate_manifest(
            manifest_path.resolve(), dataset_root, image_hashes, image_splits, report
        )

    report.statistics = {
        "images_by_split": {split: image_counts[split] for split in SPLITS},
        "labels_by_split": {split: label_counts[split] for split in SPLITS},
        "boxes_by_class": {
            CLASS_NAMES[class_id]: box_counts[class_id] for class_id in CLASS_NAMES
        },
        "boxes_by_split_and_class": {
            split: {
                CLASS_NAMES[class_id]: boxes_by_split[split][class_id]
                for class_id in CLASS_NAMES
            }
            for split in SPLITS
        },
        "negative_images": negative_images,
        "exact_duplicate_groups": sum(len(items) > 1 for items in hashes.values()),
        "phash_enabled": use_phash,
        "phash_threshold": phash_threshold if use_phash else None,
    }
    return report
