"""Prepare five manually reviewed background crops for adaptation training.

Full canal and flooding frames contain water and must not receive empty labels.
Only the fixed, visually reviewed rectangles below are negative examples. Both
source clips become adaptation sources; ``rato_rua`` stays outside this script
and remains a separate regression example, never an adaptation sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
REVIEWED = (
    (
        "negative_cars_00",
        "canal_residuos",
        0,
        (15, 0, 406, 105),
        "Parked cars, buildings, sky and vegetation only; crop excludes canal water and has no visible litter or animal.",
    ),
    (
        "negative_cars_05",
        "canal_residuos",
        5,
        (100, 5, 406, 105),
        "Parked vans and cars with sky/vegetation; crop excludes canal water and has no visible target.",
    ),
    (
        "negative_cars_09",
        "canal_residuos",
        9,
        (130, 20, 406, 117),
        "Cars and vans on the dry upper bank; no water surface, loose litter or animal inside this rectangle.",
    ),
    (
        "negative_grass_00",
        "drenagem_chuva",
        0,
        (5, 18, 160, 115),
        "Grass and fence posts only; crop excludes the bin, pavement and all flood water.",
    ),
    (
        "negative_grass_05",
        "drenagem_chuva",
        5,
        (175, 0, 370, 100),
        "Grass, fence posts and distant building; crop excludes the bin and water and has no visible animal/litter.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_negative_crops(catalog: Path, output_dir: Path) -> dict:
    """Reproduce the reviewed crops with source credits and empty YOLO labels."""
    sources = {
        item["id"]: item
        for item in json.loads(catalog.read_text(encoding="utf-8"))["items"]
    }
    images, labels = output_dir / "images", output_dir / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    records = []
    for name, source_id, seconds, coordinates, review in REVIEWED:
        source = sources[source_id]
        source_path = catalog.parent / source["path"]
        source_hash = sha256(source_path)
        if source_hash != source["sha256"]:
            raise ValueError(f"Source changed since visual review: {source_id}")
        capture = cv2.VideoCapture(str(source_path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"Cannot decode {source_id} at {seconds}s")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
            raise ValueError(f"Reviewed crop outside frame dimensions: {source_id}")
        crop = frame[y1:y2, x1:x2]
        image_path, label_path = images / f"{name}.jpg", labels / f"{name}.txt"
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"Cannot encode reviewed crop: {name}")
        encoded.tofile(str(image_path))
        label_path.write_text("", encoding="utf-8")
        try:
            portable_source = source_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            portable_source = source_path.name
        records.append(
            {
                "image": image_path.relative_to(output_dir).as_posix(),
                "label": label_path.relative_to(output_dir).as_posix(),
                "split": "train",
                "source_group": f"commons_video::{source_id}",
                "source_video": portable_source,
                "source_sha256": source_hash,
                "source_url": source["source_url"],
                "author": source["author"],
                "license": source["license"],
                "license_url": source["license_url"],
                "frame_index": round(seconds * fps),
                "timestamp_seconds": seconds,
                "crop_xyxy_pixels": coordinates,
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
                "sha256": sha256(image_path),
                "labels": [],
                "annotation_method": "manual_visual_review_of_decoded_frame_and_crop",
                "review_note": review,
                "changes": "Decoded frame; rectangular crop; JPEG quality 95. No inserted/removed objects and no model-predicted label.",
            }
        )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Reviewed hard negatives for adaptation training, never validation/test.",
        "reproduce": "python ferramentas/scripts/prepare_negative_crops.py",
        "adaptation_source_groups": [
            "commons_video::canal_residuos",
            "commons_video::drenagem_chuva",
        ],
        "excluded_regression_source": "rato_rua",
        "evaluation_note": "Both source clips belong to adaptation after these crops are used; neither full source video remains an independent holdout. rato_rua is never read or annotated by this script.",
        "reviewed_classes_absent": [
            "obstrucao_visivel",
            "agua_acumulada",
            "animal_visivel",
        ],
        "excluded_ambiguous_candidate": "Roof tile was excluded because loose construction material could be obstruction; full water clips were excluded as negatives because real water is visible.",
        "records": records,
    }
    (output_dir / "negatives_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "dados/examples/sources.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "dados/interim/adaptation"
    )
    args = parser.parse_args()
    manifest = prepare_negative_crops(args.catalog, args.output_dir)
    print(
        f"Prepared {len(manifest['records'])} reviewed negative crops, all train-only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
