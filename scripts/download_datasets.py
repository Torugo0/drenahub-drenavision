"""Download approved source datasets without committing their large files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.downloader import (  # noqa: E402
    download_click_sample,
    download_irodent,
    download_stagnant_water,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", "water", "click", "irodent"),
        default="all",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--click-train", type=int, default=100)
    parser.add_argument("--click-val", type=int, default=20)
    parser.add_argument("--click-test", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.source in {"all", "water"}:
        download_stagnant_water(args.raw_root, force=args.force)
    if args.source in {"all", "click"}:
        download_click_sample(
            args.raw_root,
            quotas={
                "train": args.click_train,
                "val": args.click_val,
                "test": args.click_test,
            },
            seed=args.seed,
            force=args.force,
        )
    if args.source in {"all", "irodent"}:
        download_irodent(args.raw_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
