"""Seal a new native CLICK-test sample before any candidate inference.

No training, pseudo-labeling or score-based selection. Existing train/val/test
and user/example images are excluded by source identity, hash and pHash <= 4.
The source is known to the project: this is a fresh image holdout, not a novel
camera/domain claim. Cross-class absences require visual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from PIL import Image
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.constants import SOURCES  # noqa: E402
from drenavision.data.validator import _phash  # noqa: E402
from drenavision.data.yolo import read_yolo  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path, count: int = 40, seed: int = 20261117) -> dict:
    if (output / "holdout_manifest.json").exists():
        raise FileExistsError(
            "Holdout already sealed; do not overwrite or select again"
        )
    excluded_hashes, excluded_names, references = set(), set(), []
    reference_paths = set()
    for name in ("drenavision_v3_clean", "extra_test_v3"):
        root = ROOT / "dados/processed" / name
        rows = list(
            csv.DictReader((root / "data_manifest.csv").open(encoding="utf-8-sig"))
        )
        for row in rows:
            if row["source_key"] == "click_trash":
                excluded_names.add(PurePosixPath(row["source_path"]).name)
            excluded_hashes.add(row["sha256"])
            path = root / row["output_image"]
            if path.resolve() not in reference_paths:
                reference_paths.add(path.resolve())
                references.append((path.relative_to(ROOT).as_posix(), _phash(path)))
    for path in (ROOT / "dados/examples").glob("*.jpg"):
        excluded_hashes.add(sha256(path))
        references.append((path.relative_to(ROOT).as_posix(), _phash(path)))
    image_root, label_root = output / "images", output / "labels"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    source = SOURCES["click_trash"]
    records, exclusions = [], []
    with RemoteZip(
        source.download_url, headers={"User-Agent": "DrenaVision-ReadOnly-Holdout/1.0"}
    ) as archive:
        names = set(archive.namelist())
        native_pool = sorted(
            name
            for name in names
            if name.startswith("images/test/")
            and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        candidates = [
            name
            for name in native_pool
            if PurePosixPath(name).name not in excluded_names
        ]
        random.Random(seed).shuffle(candidates)
        print(
            f"Native test={len(native_pool)}; unused identities={len(candidates)}; requested={count}",
            flush=True,
        )
        for remote in candidates:
            filename = PurePosixPath(remote).name
            label_remote = f"single class/labels/test/{PurePosixPath(remote).stem}.txt"
            image, label = (
                image_root / filename,
                label_root / f"{Path(filename).stem}.txt",
            )
            if not image.is_file():
                image.write_bytes(archive.read(remote))
            if not label.is_file():
                label.write_bytes(archive.read(label_remote))
            boxes = read_yolo(label, class_count=1)
            digest = sha256(image)
            if digest in excluded_hashes:
                exclusions.append({"image": filename, "reason": "exact_hash"})
                continue
            p_hash = _phash(image)
            near = next(
                (
                    name
                    for name, value in references
                    if (p_hash ^ value).bit_count() <= 4
                ),
                None,
            )
            if near:
                exclusions.append(
                    {"image": filename, "reason": "phash_lte_4", "reference": near}
                )
                continue
            with Image.open(image) as opened:
                size = list(opened.size)
            record = {
                "image": image.relative_to(ROOT).as_posix(),
                "label": label.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "label_sha256": sha256(label),
                "source_key": "click_trash",
                "native_split": "test",
                "source_group": f"click::{Path(filename).stem}",
                "source_path": remote,
                "source_url": source.landing_url,
                "source_license": "CC BY 4.0",
                "size": size,
                "boxes_class_0": len(boxes),
                "source_litter_presence": True if boxes else None,
                "cross_class_presence": {"1": None, "2": None},
                "review_state": "source annotations retained; cross-class review pending before evaluation",
            }
            records.append(record)
            excluded_hashes.add(digest)
            references.append((record["image"], p_hash))
            print(
                f"Accepted {len(records)}/{count}: {filename}; boxes={len(boxes)}",
                flush=True,
            )
            if len(records) == count:
                break
    if len(records) != count:
        raise RuntimeError(
            f"Only {len(records)} images survived; not enough to seal desired cohort"
        )
    manifest = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "New image holdout from known CLICK source for post-selection evaluation, never training or calibration.",
        "scope": "Known-source/camera holdout, not an unseen-domain generalization claim.",
        "selection": "seeded shuffle of unused native test identities, no model inference; exact/hash/pHash<=4 exclusion against current full dataset, extra test, all JPG examples and accepted holdout items",
        "seed": seed,
        "images": len(records),
        "boxes_class_0": sum(row["boxes_class_0"] for row in records),
        "candidate_exclusions": exclusions,
        "images_by_native_split": dict(Counter(row["native_split"] for row in records)),
        "source_annotations_scope": "Native single-class litter boxes; no absence of other target classes may be inferred from missing labels. Review before scoring.",
        "records": records,
    }
    path = output / "holdout_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    path.with_suffix(".sha256").write_text(sha256(path) + "\n", "ascii")
    return {
        "manifest": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "images": len(records),
    }


def review_sealed(output: Path) -> dict:
    """Reproduce the source-only assistant review of the fixed 40 scenes.

    The identity pin prevents these review decisions being silently applied to
    any new image selection. No candidate prediction was used for the review.
    """
    source = output / "holdout_manifest.json"
    expected = "03b7474f03cb4d6545d9d66db7129f693cdb215ef26bcdce004805896340269e"
    if sha256(source) != expected:
        raise ValueError("New holdout identities require a new visual review")
    destination = output / "holdout_presence_review.json"
    if destination.exists():
        raise FileExistsError("The reviewed holdout is already frozen")
    metadata = json.loads(source.read_text("utf-8"))
    rows = []
    ambiguous_animal = {"trash (55).jpg", "trash (665).jpg"}
    for item in metadata["records"]:
        path = ROOT / item["image"]
        if sha256(path) != item["sha256"]:
            raise ValueError("A reviewed source image changed")
        rows.append(
            {
                "image": item["image"],
                "sha256": item["sha256"],
                "source_group": item["source_group"],
                "source_path": item["source_path"],
                "presence": {
                    "0": True,
                    "1": None,
                    "2": None if path.name in ambiguous_animal else False,
                },
                "review_note": "Visible source-annotated litter confirmed on all 40 source scenes. No visible rodent in clear scenes; two strongly blurred scenes excluded from strict rodent-negative scoring. Water unreviewed/out of v4 scope.",
                "bbox_reference_complete": False,
            }
        )
    review = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "holdout_manifest_sha256": expected,
        "purpose": "Post-selection evaluation only; never training, prompting or threshold selection.",
        "review_method": "Assistant source-only review of two full-scene 20-image contact sheets with the native litter annotation overlays, before any candidate predictions. Not independent human ground truth.",
        "scope": "Fresh image holdout within a known source/camera; not proof of new-domain generalization. Source native test identities; zero hash/pHash<=4 overlaps with current data, user examples or the sealed extra test.",
        "presence_definition": "True=recognizable visible target. False=no visible confirmed target in an adequately readable image. null=unreviewed/ambiguous, excluded from scoring; not a negative.",
        "counts": {
            "0": {"positive": 40, "negative": 0, "unknown": 0},
            "1": {"positive": 0, "negative": 0, "unknown": 40},
            "2": {"positive": 0, "negative": 38, "unknown": 2},
        },
        "records": rows,
    }
    destination.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    destination.with_suffix(".sha256").write_text(sha256(destination) + "\n", "ascii")
    return {
        "review": destination.relative_to(ROOT).as_posix(),
        "sha256": sha256(destination),
        "counts": review["counts"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dados/raw/v4/click_holdout"
    )
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--review-sealed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            review_sealed(args.output.resolve())
            if args.review_sealed
            else prepare(args.output.resolve(), args.count, args.seed),
            ensure_ascii=False,
        )
    )
