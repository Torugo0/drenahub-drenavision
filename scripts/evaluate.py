"""CLI for held-out DrenaVision test evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.evaluation import run_evaluation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia o checkpoint customizado somente no split test."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--model", "--checkpoint", dest="checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:N ou N")
    parser.add_argument("--name", help="Nome unico do run de avaliacao")
    parser.add_argument(
        "--max-error-images",
        type=int,
        help="Limite de depuracao; 0 usa todo o test split",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "device": args.device,
        "name": args.name,
        "max_error_images": args.max_error_images,
    }
    manifest = run_evaluation(args.config, overrides)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
