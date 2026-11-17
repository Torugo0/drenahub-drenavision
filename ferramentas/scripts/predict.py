"""Run one real custom-checkpoint image prediction and export evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.core.config import (  # noqa: E402
    Settings,
    validate_image_size,
)
from drenavision.inference.media_io import read_image_bgr, write_image  # noqa: E402
from drenavision.inference.runtime import create_predictor  # noqa: E402
from drenavision.inference.utils import (  # noqa: E402
    safe_artifact_path,
    sha256_file,
    utc_now_iso,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    runtime = Settings()
    parser = argparse.ArgumentParser(
        description="Executa inferencia real em uma imagem JPG/PNG."
    )
    parser.add_argument("--input", type=Path, required=True, help="Imagem JPG/PNG")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=Path,
        default=Path("resultados/local/models/best.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("resultados/local/predictions")
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--water-model", type=Path, default=runtime.water_model_path)
    parser.add_argument(
        "--water-pixel-confidence", type=float, default=runtime.water_pixel_confidence
    )
    parser.add_argument(
        "--water-min-area-fraction", type=float, default=runtime.water_min_area_fraction
    )
    parser.add_argument(
        "--water-mode", choices=("augment", "replace"), default=runtime.water_mode
    )
    parser.add_argument(
        "--animal-conf", type=float, default=runtime.animal_confidence_threshold
    )
    parser.add_argument(
        "--imgsz",
        type=validate_image_size,
        default=640,
        help="Lado de inferência, múltiplo de 32 entre 320 e 2048 (padrão: 640)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"imagem ausente: {args.input}")
    image_bgr = read_image_bgr(args.input)
    image_rgb = image_bgr[:, :, ::-1].copy()
    predictor = create_predictor(
        args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        image_size=args.imgsz,
        water_model_path=args.water_model,
        water_pixel_confidence=args.water_pixel_confidence,
        water_min_area_fraction=args.water_min_area_fraction,
        water_mode=args.water_mode,
        residue_confidence=args.conf,
        animal_confidence=args.animal_conf,
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
