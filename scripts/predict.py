"""Run one real custom-checkpoint image prediction and export evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.media_io import read_image_bgr, write_image  # noqa: E402
from app.inference.predictor import YoloPredictor  # noqa: E402
from app.inference.utils import (  # noqa: E402
    safe_artifact_path,
    sha256_file,
    utc_now_iso,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa inferencia real em uma imagem JPG/PNG."
    )
    parser.add_argument("--input", type=Path, required=True, help="Imagem JPG/PNG")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=Path,
        default=Path("artifacts/models/best.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/predictions")
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"imagem ausente: {args.input}")
    image_bgr = read_image_bgr(args.input)
    image_rgb = image_bgr[:, :, ::-1].copy()
    predictor = YoloPredictor(
        args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    result = predictor.predict(image_rgb)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_output = args.output_dir / f"{args.input.stem}_annotated.jpg"
    json_output = args.output_dir / f"{args.input.stem}_prediction.json"
    write_image(image_output, result.annotated_bgr)
    payload = result.as_dict()
    payload.update(
        {
            "created_at_utc": utc_now_iso(),
            "input_name": args.input.name,
            "input_sha256": sha256_file(args.input),
            "annotated_image": safe_artifact_path(image_output),
        }
    )
    write_json(json_output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
