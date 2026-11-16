import csv
import hashlib
from pathlib import Path

from PIL import Image

from app.data.validator import validate_dataset


def _build_dataset(root: Path, *, duplicate: bool = False) -> tuple[Path, Path]:
    manifest = root.parent / "manifest.csv"
    rows = []
    colors = {"train": (10, 20, 30), "val": (40, 50, 60), "test": (70, 80, 90)}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image = image_dir / f"fixture_{split}.jpg"
        color = colors["train"] if duplicate and split == "val" else colors[split]
        Image.new("RGB", (32, 24), color).save(image, quality=100, subsampling=0)
        label = label_dir / f"fixture_{split}.txt"
        label.write_text(
            "0 0.2 0.2 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.8 0.8 0.2 0.2\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        rows.append(
            {
                "output_image": image.relative_to(root).as_posix(),
                "output_label": label.relative_to(root).as_posix(),
                "split": split,
                "source_key": "fixture",
                "source_version": "1",
                "source_url": "https://example.test/data",
                "source_license": "CC-BY-4.0",
                "source_path": image.name,
                "source_group": f"group-{split}",
                "sha256": digest,
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    yaml_path = root.parent / "dataset.yaml"
    yaml_path.write_text(
        "path: data\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: obstrucao_visivel\n  1: agua_acumulada\n  2: animal_visivel\n",
        encoding="utf-8",
    )
    return yaml_path, manifest


def test_validator_accepts_complete_strict_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    yaml_path, manifest = _build_dataset(dataset)
    report = validate_dataset(dataset, yaml_path=yaml_path, manifest_path=manifest)
    assert report.valid, report.to_dict()
    assert report.statistics["images_by_split"] == {"train": 1, "val": 1, "test": 1}
    assert report.statistics["boxes_by_class"] == {
        "obstrucao_visivel": 3,
        "agua_acumulada": 3,
        "animal_visivel": 3,
    }


def test_validator_rejects_exact_duplicate_across_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    yaml_path, manifest = _build_dataset(dataset, duplicate=True)
    report = validate_dataset(dataset, yaml_path=yaml_path, manifest_path=manifest)
    assert not report.valid
    assert any(issue.code == "duplicate_sha256" for issue in report.errors)


def test_validator_rejects_invalid_box_and_orphan_label(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    yaml_path, manifest = _build_dataset(dataset)
    (dataset / "labels" / "train" / "fixture_train.txt").write_text(
        "7 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset / "labels" / "val" / "orphan.txt").write_text("", encoding="utf-8")
    report = validate_dataset(dataset, yaml_path=yaml_path, manifest_path=manifest)
    codes = {issue.code for issue in report.errors}
    assert "invalid_yolo" in codes
    assert "orphan_label" in codes


def test_manifest_source_group_cannot_cross_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    yaml_path, manifest = _build_dataset(dataset)
    text = manifest.read_text(encoding="utf-8")
    text = text.replace("group-val", "group-train")
    manifest.write_text(text, encoding="utf-8")
    report = validate_dataset(dataset, yaml_path=yaml_path, manifest_path=manifest)
    assert any(issue.code == "source_group_leakage" for issue in report.errors)
