"""Convert downloaded source annotations into the DrenaVision YOLO schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.converter import convert_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("dados/raw"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("dados/processed/drenavision")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dados/manifests/dataset_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = convert_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        manifest_path=args.manifest,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
