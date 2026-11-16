"""CLI for real DrenaVision YOLO26n fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.inference.training import run_training  # noqa: E402


def _batch(value: str) -> int | float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch deve ser numerico") from exc
    return int(number) if number.is_integer() else number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina YOLO26n nas tres classes customizadas do DrenaVision."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("ferramentas/configs/train_v2.yaml")
    )
    parser.add_argument("--data", type=Path, help="Sobrescreve dataset.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=_batch)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:N ou N")
    parser.add_argument("--name", help="Nome unico do run Ultralytics")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Retoma com segurança um last.pt e preserva snapshot do run interrompido",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "name": args.name,
        "resume": args.resume,
    }
    manifest = run_training(args.config, overrides)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
