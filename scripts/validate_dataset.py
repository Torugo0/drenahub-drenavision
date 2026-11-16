"""Validate DrenaVision YOLO data and fail on integrity or leakage errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.validator import validate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/processed/drenavision"),
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("datasets/processed/drenavision/dataset.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifests/dataset_manifest.csv"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--phash", action="store_true", help="check near duplicates across splits"
    )
    parser.add_argument("--phash-threshold", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_dataset(
        args.dataset_root,
        yaml_path=args.yaml,
        manifest_path=args.manifest,
        use_phash=args.phash,
        phash_threshold=args.phash_threshold,
    )
    rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
