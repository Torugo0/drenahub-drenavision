"""Build a truthful MP4 sequence from held-out DrenaVision test images.

The generated input contains no model predictions or bounding boxes. It is a
short, deterministic sequence of three real test images that can subsequently
be passed to ``ferramentas/scripts/process_video.py`` for genuine detector output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.inference.video_codec import prepare_browser_video  # noqa: E402

CLASS_NAMES = {
    0: "obstrucao_visivel",
    1: "agua_acumulada",
    2: "animal_visivel",
}
CAPTION = "Sequencia de imagens ineditas do split test - nao e video capturado"


@dataclass(frozen=True, slots=True)
class DemoSlide:
    class_id: int
    class_name: str
    image_path: Path
    label_path: Path
    ground_truth_boxes: int
    source_key: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        width_text, height_text = normalized.split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size deve usar LARGURAxALTURA") from exc
    if width < 160 or height < 120 or width > 3840 or height > 2160:
        raise argparse.ArgumentTypeError("size deve estar entre 160x120 e 3840x2160")
    if width % 2 or height % 2:
        raise argparse.ArgumentTypeError("largura e altura devem ser pares")
    return width, height


def _label_class_counts(path: Path) -> dict[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"label de teste ausente: {path}")
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"label invalido {path.name}:{line_number}")
        try:
            value = float(fields[0])
        except ValueError as exc:
            raise ValueError(f"classe invalida {path.name}:{line_number}") from exc
        if not value.is_integer() or int(value) not in CLASS_NAMES:
            raise ValueError(f"classe invalida {path.name}:{line_number}")
        counts[int(value)] += 1
    return counts


def select_test_slides(dataset_root: Path, manifest_path: Path) -> list[DemoSlide]:
    """Select one unique manifest-backed test image for every target class."""

    dataset_root = dataset_root.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifesto ausente: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "output_image",
        "output_label",
        "split",
        "source_key",
        "boxes_class_0",
        "boxes_class_1",
        "boxes_class_2",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - (set(rows[0]) if rows else set()))
        raise ValueError(f"manifesto sem colunas obrigatorias: {missing}")

    selected: list[DemoSlide] = []
    used_images: set[Path] = set()
    test_rows = sorted(
        (row for row in rows if row.get("split") == "test"),
        key=lambda row: row["output_image"],
    )
    for class_id, class_name in CLASS_NAMES.items():
        slide: DemoSlide | None = None
        for row in test_rows:
            try:
                manifest_boxes = int(row[f"boxes_class_{class_id}"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "manifesto contem contagem de caixas invalida"
                ) from exc
            if manifest_boxes < 1:
                continue
            image_path = (dataset_root / row["output_image"]).resolve()
            label_path = (dataset_root / row["output_label"]).resolve()
            try:
                image_path.relative_to(dataset_root)
                label_path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError("manifesto aponta para fora do dataset") from exc
            if image_path in used_images:
                continue
            if not image_path.is_file():
                raise FileNotFoundError(f"imagem de teste ausente: {image_path}")
            actual_boxes = _label_class_counts(label_path)[class_id]
            if actual_boxes != manifest_boxes:
                raise ValueError(
                    f"contagem diverge para {image_path.name}: "
                    f"manifesto={manifest_boxes}, label={actual_boxes}"
                )
            slide = DemoSlide(
                class_id=class_id,
                class_name=class_name,
                image_path=image_path,
                label_path=label_path,
                ground_truth_boxes=actual_boxes,
                source_key=row["source_key"],
            )
            break
        if slide is None:
            raise ValueError(f"split test nao possui amostra valida de {class_name}")
        selected.append(slide)
        used_images.add(slide.image_path)
    return selected


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((height, width, 3), (24, 22, 20), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _read_image(path: Path) -> np.ndarray:
    """Decode through bytes because OpenCV paths can fail on Windows Unicode."""

    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"nao foi possivel ler {path.name}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"imagem de teste corrompida: {path.name}")
    return image


def _caption_frame(frame: np.ndarray, slide: DemoSlide) -> np.ndarray:
    result = frame.copy()
    height = result.shape[0]
    scale = max(0.42, min(0.72, result.shape[1] / 1500))
    lines = (
        CAPTION,
        f"Rotulo de referencia (nao e predicao): {slide.class_name}",
    )
    for index, text in enumerate(lines):
        origin = (14, height - 42 + index * 22)
        cv2.putText(
            result,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return result


def build_demo_video(
    dataset_root: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    seconds_per_image: float = 2.0,
    fps: float = 12.0,
    size: tuple[int, int] = (960, 540),
) -> dict[str, object]:
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output deve terminar em .mp4")
    if not math.isfinite(seconds_per_image) or seconds_per_image <= 0:
        raise ValueError("seconds_per_image deve ser positivo")
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("fps deve estar entre 1 e 120")
    width, height = size
    if width % 2 or height % 2 or width < 160 or height < 120:
        raise ValueError("size deve usar dimensoes pares e validas")

    slides = select_test_slides(dataset_root, manifest_path)
    frames_per_slide = max(1, int(round(seconds_per_image * fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_handle = tempfile.TemporaryDirectory(prefix="drenavision_demo_")
    partial = Path(scratch_handle.name) / "frames.mp4"
    writer = cv2.VideoWriter(
        str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        scratch_handle.cleanup()
        raise RuntimeError("OpenCV nao conseguiu inicializar o codec mp4v")
    try:
        for slide in slides:
            image = _read_image(slide.image_path)
            frame = _caption_frame(_letterbox(image, size), slide)
            for _ in range(frames_per_slide):
                writer.write(frame)
    except Exception:
        writer.release()
        scratch_handle.cleanup()
        raise
    writer.release()
    if not partial.is_file() or partial.stat().st_size == 0:
        scratch_handle.cleanup()
        raise RuntimeError("OpenCV nao produziu um MP4 valido")
    try:
        encoding = prepare_browser_video(partial, output_path)
    finally:
        scratch_handle.cleanup()

    payload: dict[str, object] = {
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "truthfulness_note": (
            "Sequencia de imagens reais do split test; nao e video capturado e "
            "nao contem caixas ou predicoes."
        ),
        "is_recorded_video": False,
        "contains_model_predictions": False,
        "dataset_manifest": safe_path(manifest_path),
        "output_video": safe_path(output_path),
        "output_sha256": sha256_file(output_path),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "seconds_per_image": seconds_per_image,
            "frames_per_image": frames_per_slide,
            "total_frames": frames_per_slide * len(slides),
            "duration_seconds": frames_per_slide * len(slides) / fps,
            **encoding,
        },
        "slides": [
            {
                "class_id": slide.class_id,
                "class_name": slide.class_name,
                "ground_truth_boxes": slide.ground_truth_boxes,
                "source_key": slide.source_key,
                "image": safe_path(slide.image_path),
                "label": safe_path(slide.label_path),
                "image_sha256": sha256_file(slide.image_path),
            }
            for slide in slides
        ],
    }
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dados/processed/drenavision_v3_clean"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dados/processed/drenavision_v3_clean/data_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados/local/videos/demo_input_v3_sequence.mp4"),
    )
    parser.add_argument("--seconds-per-image", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--size", type=parse_size, default=(960, 540))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_demo_video(
        args.dataset_root,
        args.manifest,
        args.output,
        seconds_per_image=args.seconds_per_image,
        fps=args.fps,
        size=args.size,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
