"""Render a deterministic annotation sample for visual quality review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.visualizer import create_contact_sheet, visualize_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/processed/drenavision"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/dataset_preview")
    )
    parser.add_argument("--samples-per-split", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rendered = visualize_dataset(
        args.dataset_root,
        args.output_root,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
    )
    for path in rendered:
        print(path)
    contact_sheet = create_contact_sheet(
        rendered, args.output_root / "annotation_contact_sheet.jpg"
    )
    print(contact_sheet)
    print(
        f"rendered {len(rendered)} previews and one contact sheet in {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
