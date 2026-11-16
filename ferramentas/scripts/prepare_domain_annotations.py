"""Reproduce visually reviewed training frames; never infer labels from predictions.

The two source clips belong entirely to training. rato_rua remains outside
adaptation. These examples are regression material, not independent accuracy
estimates. Pixel boxes are reviewed against the decoded frames before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TIMES = (0, 2, 4, 6, 8, 10)
REVIEWED_BOXES: dict[str, dict[int, list[tuple[int, int, int, int]]]] = {
    "rato_close": {
        0: [(138, 68, 636, 352)],
        2: [(160, 65, 638, 330)],
        4: [(189, 100, 637, 369)],
        6: [(113, 75, 636, 378)],
        8: [(95, 90, 636, 357)],
        10: [(126, 82, 638, 356)],
    },
    "rato_residuos": {
        0: [(291, 164, 389, 232)],
        2: [(290, 173, 367, 239)],
        4: [(285, 158, 381, 229)],
        6: [(222, 185, 314, 249)],
        10: [(357, 177, 455, 241)],
    },
}


def prepare(output: Path, *, review_only: bool = False) -> None:
    if review_only:
        output = output / "unannotated_review"
    elif not (output / "negatives_manifest.json").is_file():
        from ferramentas.scripts.prepare_negative_crops import prepare_negative_crops

        prepare_negative_crops(ROOT / "dados/examples/sources.json", output)
    catalog = json.loads(
        (ROOT / "dados/examples/sources.json").read_text(encoding="utf-8")
    )
    sources = {item["id"]: item for item in catalog["items"]}
    records = []
    for source_id in ("rato_close", "rato_residuos"):
        source = sources[source_id]
        video = ROOT / "dados/examples" / source["path"]
        capture = cv2.VideoCapture(str(video))
        tiles = []
        for second in TIMES:
            if not review_only and second not in REVIEWED_BOXES[source_id]:
                continue  # Blurred frame at 8s is excluded, never guessed.
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"missing frame {source_id} at {second}s")
            image = Image.fromarray(frame[:, :, ::-1])
            boxes = REVIEWED_BOXES.get(source_id, {}).get(second)
            crop = None
            if not review_only and source_id == "rato_residuos":
                # Tight animal-only view: exclude the incompletely annotated
                # refuse pile. This is explicit training augmentation, not a
                # modified product input or a fabricated prediction.
                x1, y1, x2, y2 = boxes[0]
                crop = (max(0, x1 - 8), max(0, y1 - 8), min(image.width, x2 + 8), y2)
                image = image.crop(crop)
                boxes = [(x1 - crop[0], y1 - crop[1], x2 - crop[0], y2 - crop[1])]
            name = f"domain_{source_id}_{second:02d}"
            image_path = output / "images" / f"{name}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, quality=95)
            if not review_only and boxes is None:
                raise ValueError(f"frame without reviewed boxes: {name}")
            label = output / "labels" / f"{name}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            if boxes is not None:
                rows = []
                for x1, y1, x2, y2 in boxes:
                    rows.append(
                        f"2 {(x1 + x2) / 2 / image.width:.8f} {(y1 + y2) / 2 / image.height:.8f} {(x2 - x1) / image.width:.8f} {(y2 - y1) / image.height:.8f}"
                    )
                label.write_text(
                    "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
                )
            tile = image.copy()
            draw = ImageDraw.Draw(tile)
            for x in range(0, image.width, 80):
                draw.line((x, 0, x, image.height), fill="#00bfff", width=1)
                draw.text(
                    (x + 2, 22),
                    str(x),
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
            for y in range(0, image.height, 80):
                draw.line((0, y, image.width, y), fill="#00bfff", width=1)
                draw.text(
                    (2, y + 2),
                    str(y),
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
            draw.text(
                (10, 4),
                f"{name} {image.width}x{image.height}",
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            for box in boxes or []:
                draw.rectangle(box, outline="#ff5600", width=3)
            tile.thumbnail((640, 480))
            tiles.append(tile)
            records.append(
                {
                    "id": name,
                    "image": f"images/{name}.jpg",
                    "label": f"labels/{name}.txt",
                    "split": "train",
                    "source_group": source_id,
                    "source_video": source["path"],
                    "source_url": source["source_url"],
                    "author": source["author"],
                    "license": source["license"],
                    "license_url": source["license_url"],
                    "timestamp_seconds": second,
                    "crop_xyxy_pixels": crop,
                    "boxes_xyxy": boxes,
                    "annotation_method": "assistant visual review of decoded frame; not independent human ground truth",
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                }
            )
        capture.release()
        sheet = Image.new("RGB", (1280, 480 * 3), "#172033")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 2) * 640, (index // 2) * 480))
        sheet.save(output / f"review_{source_id}.jpg", quality=92)
    (output / "domain_manifest.json").write_text(
        json.dumps(
            {"training_only": True, "held_out_clip": "rato_rua", "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not review_only:
        all_records = list(records)
        negatives = output / "negatives_manifest.json"
        if negatives.is_file():
            all_records.extend(
                json.loads(negatives.read_text(encoding="utf-8"))["records"]
            )
        items = [
            {
                "file_name": Path(record["image"]).name,
                "source_url": record["source_url"],
                "source_license": record["license"],
                "source_group": record["source_group"],
                "annotation_note": record.get(
                    "annotation_method", record.get("review_note", "visually reviewed")
                ),
                "split": "train",
            }
            for record in all_records
        ]
        (output / "sources.json").write_text(
            json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dados/interim/adaptation"
    )
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    prepare(args.output, review_only=args.review_only)
