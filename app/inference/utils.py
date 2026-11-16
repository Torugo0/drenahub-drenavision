"""Reproducibility, dataset, box matching, and JSON helpers."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from app.data.constants import CLASS_NAMES, IMAGE_SUFFIXES

from .errors import ClassMapMismatchError, DatasetUnavailableError


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_class_map(names: Mapping[Any, Any] | Sequence[Any]) -> dict[int, str]:
    if isinstance(names, Mapping):
        try:
            return {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ClassMapMismatchError("mapa de classes possui IDs invalidos") from exc
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    raise ClassMapMismatchError("checkpoint nao expoe um mapa de classes valido")


def validate_class_map(names: Mapping[Any, Any] | Sequence[Any]) -> dict[int, str]:
    normalized = normalize_class_map(names)
    if normalized != CLASS_NAMES:
        raise ClassMapMismatchError(
            f"classes do checkpoint {normalized!r}; esperado {CLASS_NAMES!r}"
        )
    return normalized


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(
            (
                "torch",
                "torchvision",
                "ultralytics",
                "opencv-python",
                "PyYAML",
                "faster-coco-eval",
            )
        ),
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML nao esta instalado") from exc
    if not path.is_file():
        raise DatasetUnavailableError(f"arquivo YAML ausente: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetUnavailableError(f"YAML invalido: {path}")
    return payload


def resolve_dataset_root(dataset_yaml: Path, payload: Mapping[str, Any]) -> Path:
    configured = payload.get("path", ".")
    root = Path(str(configured))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    return root


def _resolve_split_entries(
    dataset_yaml: Path, payload: Mapping[str, Any], split: str
) -> list[Path]:
    value = payload.get(split)
    if value is None:
        raise DatasetUnavailableError(f"dataset.yaml nao define o split {split}")
    values = value if isinstance(value, list) else [value]
    root = resolve_dataset_root(dataset_yaml, payload)
    images: list[Path] = []
    for item in values:
        target = Path(str(item))
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        if target.is_dir():
            images.extend(
                path
                for path in sorted(target.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif target.is_file() and target.suffix.lower() == ".txt":
            for line in target.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip())
                if not image.is_absolute():
                    image = (target.parent / image).resolve()
                images.append(image)
        elif target.is_file() and target.suffix.lower() in IMAGE_SUFFIXES:
            images.append(target)
        else:
            raise DatasetUnavailableError(
                f"split {split} aponta para alvo ausente: {target}"
            )
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise DatasetUnavailableError(
            f"split {split} contem imagem ausente: {missing[0]}"
        )
    if not images:
        raise DatasetUnavailableError(f"split {split} nao contem imagens")
    return sorted(set(images))


def preflight_dataset(
    dataset_yaml: Path, required_splits: Sequence[str]
) -> dict[str, Any]:
    dataset_yaml = dataset_yaml.resolve()
    payload = _read_yaml(dataset_yaml)
    names = payload.get("names")
    if names is None:
        raise DatasetUnavailableError("dataset.yaml nao define names")
    try:
        validate_class_map(names)
    except ClassMapMismatchError as exc:
        raise DatasetUnavailableError(str(exc)) from exc
    images_by_split = {
        split: _resolve_split_entries(dataset_yaml, payload, split)
        for split in required_splits
    }
    return {
        "yaml": dataset_yaml,
        "payload": payload,
        "root": resolve_dataset_root(dataset_yaml, payload),
        "images": images_by_split,
    }


def strict_dataset_validation(dataset_yaml: Path) -> dict[str, Any]:
    """Run provenance, label, exact-duplicate, and group-leakage checks."""

    from app.data.validator import validate_dataset

    dataset_yaml = dataset_yaml.resolve()
    dataset_root = dataset_yaml.parent
    manifest: Path | None = None
    if len(dataset_yaml.parents) > 2:
        candidate = dataset_yaml.parents[2] / "manifests" / "dataset_manifest.csv"
        if candidate.is_file():
            manifest = candidate
    report = validate_dataset(
        dataset_root,
        yaml_path=dataset_yaml,
        manifest_path=manifest,
        use_phash=False,
    )
    if not report.valid:
        details = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.errors[:8]
        )
        raise DatasetUnavailableError(f"validacao estrita do dataset falhou: {details}")
    return report.to_dict()


def dataset_fingerprint(preflight: Mapping[str, Any]) -> dict[str, Any]:
    yaml_path = Path(preflight["yaml"])
    root = Path(preflight["root"])
    all_images: list[tuple[str, Path, str]] = []
    for split, paths in preflight["images"].items():
        for image in paths:
            all_images.append(
                (str(split), Path(image), _relative_or_name(Path(image), root))
            )

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    split_counts: dict[str, int] = {}
    for split, image, _relative in sorted(
        all_images, key=lambda item: (item[0], item[2])
    ):
        candidates = [image, label_path_for_image(image)]
        split_counts[split] = split_counts.get(split, 0) + 1
        for path in candidates:
            if not path.is_file():
                continue
            file_hash = sha256_file(path)
            size = path.stat().st_size
            rel = _relative_or_name(path, root)
            digest.update(f"{rel}\0{size}\0{file_hash}\n".encode())
            file_count += 1
            total_bytes += size
    provenance_candidates = {
        yaml_path.parent / "manifest.jsonl",
        yaml_path.parent / "sources.csv",
        root / "manifest.jsonl",
        root / "sources.csv",
        yaml_path.parent.parent / "dataset_sources.csv",
    }
    if len(yaml_path.parents) > 2:
        datasets_dir = yaml_path.parents[2]
        provenance_candidates.add(datasets_dir / "dataset_sources.csv")
        manifests_dir = datasets_dir / "manifests"
        if manifests_dir.is_dir():
            provenance_candidates.update(
                path
                for path in manifests_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".csv", ".json"}
            )
    provenance = {
        _relative_or_name(path, Path.cwd()): sha256_file(path)
        for path in sorted(provenance_candidates)
        if path.is_file()
    }
    return {
        "dataset_yaml": _relative_or_name(yaml_path, Path.cwd()),
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "dataset_sha256": digest.hexdigest(),
        "files_hashed": file_count,
        "bytes_hashed": total_bytes,
        "images_by_split": split_counts,
        "provenance_sha256": provenance,
    }


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def safe_artifact_path(path: str | Path) -> str:
    """Return a repository-relative path, or only the basename if external."""

    return _relative_or_name(Path(path), Path.cwd())


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, value in enumerate(parts) if value.lower() == "images"]
    if indexes:
        parts[indexes[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def xywhn_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    return (
        max(0.0, (x_center - width / 2) * image_width),
        max(0.0, (y_center - height / 2) * image_height),
        min(float(image_width), (x_center + width / 2) * image_width),
        min(float(image_height), (y_center + height / 2) * image_height),
    )


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def greedy_class_matches(
    truth: Sequence[tuple[int, Sequence[float]]],
    predictions: Sequence[tuple[int, float, Sequence[float]]],
    iou_threshold: float = 0.5,
) -> tuple[set[int], set[int]]:
    """Return matched truth and prediction indexes using confidence-first greediness."""

    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    ordered_predictions = sorted(
        enumerate(predictions), key=lambda item: float(item[1][1]), reverse=True
    )
    for pred_index, (pred_class, _confidence, pred_box) in ordered_predictions:
        candidates = [
            (box_iou(pred_box, truth_box), truth_index)
            for truth_index, (truth_class, truth_box) in enumerate(truth)
            if truth_index not in matched_truth and truth_class == pred_class
        ]
        if not candidates:
            continue
        best_iou, truth_index = max(candidates)
        if best_iou >= iou_threshold:
            matched_truth.add(truth_index)
            matched_predictions.add(pred_index)
    return matched_truth, matched_predictions
