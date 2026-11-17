"""Build immutable v3 data with water-region targets and full scene context.

V2 positive image split identities remain fixed. Source wet-only empty labels
are excluded after a blinded review found water and ambiguous content. Water
labels change to connected regions, so compare both checkpoints on v3 labels.
Additional source data and reviewed clip frames are restricted to training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.converter import (  # noqa: E402
    ManifestRow,
    _write_candidate,
    collect_click,
    sha256_file,
    write_dataset_yaml,
    write_manifest,
)
from drenavision.data.validator import _phash, validate_dataset  # noqa: E402
from drenavision.data.yolo import read_yolo, write_yolo  # noqa: E402
from drenavision.inference.utils import utc_now_iso, write_json  # noqa: E402


def _row(values: dict[str, str]) -> ManifestRow:
    fields = dict(values)
    for key in ("width", "height", "boxes_class_0", "boxes_class_1", "boxes_class_2"):
        fields[key] = int(fields[key])
    return ManifestRow(**fields)


def prepare(
    baseline: Path,
    output: Path,
    domain_roots: list[Path],
    click_root: Path | None,
    audit_dir: Path,
    raw_water_root: Path = ROOT / "dados/raw/stagnant_water/extracted",
) -> dict:
    from ferramentas.scripts.prepare_water_v3 import (
        merge_source_water_boxes,
        select_extra_water_training,
    )

    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty output; previous datasets are immutable")
    if output.resolve() == baseline.resolve():
        raise ValueError("Output must differ from baseline")
    source_rows = [
        _row(r)
        for r in csv.DictReader(
            (baseline / "data_manifest.csv").open(encoding="utf-8", newline="")
        )
    ]
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
    seen = {}
    rows = []
    removed_tight_crops = []
    excluded_wet_only = []
    before_water = after_water = 0
    for row in source_rows:
        if row.split == "train" and row.source_group == "rato_residuos":
            removed_tight_crops.append(row.output_image)
            continue
        if row.source_key == "stagnant_water" and row.boxes_class_1 == 0:
            excluded_wet_only.append(
                {
                    "image": row.output_image,
                    "split": row.split,
                    "reason": "Unreliable wet-only source negative; excluded before candidate evaluation",
                }
            )
            continue
        shutil.copy2(baseline / row.output_image, output / row.output_image)
        boxes = read_yolo(baseline / row.output_label, class_count=3)
        if row.source_key == "stagnant_water":
            before_water += sum(b.class_id == 1 for b in boxes)
            boxes = merge_source_water_boxes(boxes)
            after_water += sum(b.class_id == 1 for b in boxes)
            row = replace(
                row,
                annotation_note=row.annotation_note
                + "; v3: union bounds of intersecting/touching source water-box components; no positive gap bridging",
            )
        write_yolo(output / row.output_label, boxes)
        counts = Counter(b.class_id for b in boxes)
        row = replace(
            row,
            boxes_class_0=counts[0],
            boxes_class_1=counts[1],
            boxes_class_2=counts[2],
        )
        rows.append(row)
        seen[row.sha256] = row.output_image

    heldout = [r for r in rows if r.split != "train"]
    heldout_hashes = {r.sha256 for r in heldout}
    heldout_groups = {(r.source_key, r.source_group) for r in heldout}
    heldout_phashes = [_phash(output / r.output_image) for r in heldout]
    skipped = Counter()
    added = Counter()
    water_candidates, water_selection = select_extra_water_training(
        raw_water_root,
        baseline / "data_manifest.csv",
        baseline,
        negative_count=0,
    )
    for candidate in water_candidates:
        rows.append(_write_candidate(candidate, "train", output, seen))
        added["stagnant_water"] += 1
    for domain_root in domain_roots:
        items = json.loads((domain_root / "sources.json").read_text("utf-8"))["items"]
        for item in items:
            if item.get("split", "train") != "train":
                raise ValueError("Supplementary data must be training-only")
            image = domain_root / "images" / item["file_name"]
            digest = sha256_file(image)
            source_key = item.get("source_key", "domain_reviewed")
            group = item["source_group"]
            if digest in seen:
                skipped["duplicate_image"] += 1
                continue
            if digest in heldout_hashes or (source_key, group) in heldout_groups:
                raise ValueError(
                    f"Held-out source leaked into adaptation: {image.name}"
                )
            p_hash = _phash(image)
            if any((p_hash ^ other).bit_count() <= 4 for other in heldout_phashes):
                skipped["near_heldout_image"] += 1
                continue
            boxes = read_yolo(
                domain_root / "labels" / (image.stem + ".txt"), class_count=3
            )
            stem = source_key + "_" + digest[:16]
            suffix = image.suffix.lower().replace(".jpeg", ".jpg")
            image_rel = f"images/train/{stem}{suffix}"
            label_rel = f"labels/train/{stem}.txt"
            shutil.copy2(image, output / image_rel)
            write_yolo(output / label_rel, boxes)
            with Image.open(image) as opened:
                width, height = opened.size
            counts = Counter(b.class_id for b in boxes)
            rows.append(
                ManifestRow(
                    image_rel,
                    label_rel,
                    "train",
                    source_key,
                    item.get("source_version", "1"),
                    item["source_url"],
                    item["source_license"],
                    item.get("source_path", item["file_name"]),
                    group,
                    digest,
                    width,
                    height,
                    counts[0],
                    counts[1],
                    counts[2],
                    item["annotation_note"],
                )
            )
            seen[digest] = image_rel
            added[source_key] += 1

    if click_root:
        # The downloader's zero-sized val/test partitions are never training
        # sources. collect_click expects their empty directories to exist.
        for split in ("val", "test"):
            for kind in ("images", "labels"):
                (click_root / kind / split).mkdir(parents=True, exist_ok=True)
        for candidate in collect_click(click_root)["train"]:
            if sha256_file(candidate.image) in seen:
                skipped["duplicate_click_image"] += 1
                continue
            if ("click_trash", candidate.source_group) in heldout_groups:
                raise ValueError("Native CLICK train overlaps held-out source group")
            p_hash = _phash(candidate.image)
            if any((p_hash ^ other).bit_count() <= 4 for other in heldout_phashes):
                skipped["near_heldout_click_image"] += 1
                continue
            rows.append(_write_candidate(candidate, "train", output, seen))
            added["click_trash"] += 1
    manifest = output / "data_manifest.csv"
    write_manifest(rows, manifest)
    yaml_path = write_dataset_yaml(output)
    validation = validate_dataset(
        output,
        yaml_path=yaml_path,
        manifest_path=manifest,
        use_phash=True,
        phash_threshold=4,
    )
    report = {
        "created_at_utc": utc_now_iso(),
        "annotation_protocol": "v3-water-regions",
        "baseline": baseline.as_posix(),
        "dataset_yaml": yaml_path.as_posix(),
        "split_policy": "V2 positive validation/test identities fixed; source wet-only empty labels excluded before candidate evaluation; extras train-only; no hash/group/pHash<=4 leakage",
        "water_policy": "Original source water boxes connected by intersection or exact contact form one instance. There is no positive gap threshold. The final bounding envelope may contain dry holes between fragments; it is not a segmentation mask. API id agua_acumulada retained; means visible water, not proof of stagnation.",
        "comparability": "V3 water targets differ from v2. Re-evaluate BOTH models on v3 ground truth; never compare old v2 metrics directly.",
        "removed_tight_rat_crops": removed_tight_crops,
        "excluded_source_wet_only": excluded_wet_only,
        "baseline_water_boxes_before": before_water,
        "baseline_water_regions_after": after_water,
        "added_training_images": dict(added),
        "skipped": dict(skipped),
        "extra_water_selection": water_selection,
        "images_by_split": dict(Counter(r.split for r in rows)),
        "images_by_source": dict(Counter(r.source_key for r in rows)),
        "validation": validation.to_dict(),
    }
    write_json(audit_dir / "preparation_report.json", report)
    write_manifest(rows, audit_dir / "data_manifest.csv")
    if not validation.valid:
        raise ValueError(f"Dataset validation failed: inspect {audit_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=Path("dados/processed/drenavision_v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dados/processed/drenavision_v3_clean")
    )
    parser.add_argument("--domain-root", type=Path, action="append", default=[])
    parser.add_argument("--click-root", type=Path)
    parser.add_argument(
        "--fetch-click",
        action="store_true",
        help="Baixa/verifica 250 pares nativos de treino CLICK por HTTP Range em dados/raw/supplement_v3",
    )
    parser.add_argument(
        "--raw-water-root",
        type=Path,
        default=ROOT / "dados/raw/stagnant_water/extracted",
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("resultados/local/preparation_v3")
    )
    args = parser.parse_args()
    if args.fetch_click:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Escolha uma saída vazia antes de baixar os dados de treino")
        from drenavision.data.downloader import download_click_sample

        args.click_root = download_click_sample(
            ROOT / "dados/raw/supplement_v3",
            quotas={"train": 250, "val": 0, "test": 0},
            seed=42,
        )
    report = prepare(
        args.baseline,
        args.output,
        args.domain_root,
        args.click_root,
        args.audit_dir,
        args.raw_water_root,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "validation"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
