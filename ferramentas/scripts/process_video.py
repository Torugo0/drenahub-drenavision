"""Annotate a recorded MP4 with optional ByteTrack identity counting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.core.config import Settings, validate_image_size  # noqa: E402
from drenavision.core.storage import EventRepository, EvidenceStore  # noqa: E402
from drenavision.inference.utils import write_json  # noqa: E402
from drenavision.inference.video import persist_video_event, process_video  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Processa video frame a frame com o checkpoint customizado."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=settings.model_path)
    parser.add_argument("--conf", type=float, default=settings.confidence_threshold)
    parser.add_argument("--iou", type=float, default=settings.iou_threshold)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument(
        "--water-model",
        type=Path,
        default=settings.water_model_path,
        help="Segmentador local de água ativo; falha se configurado e indisponível",
    )
    parser.add_argument(
        "--no-water",
        action="store_true",
        help="Desativa explicitamente o segmentador neste comando",
    )
    parser.add_argument(
        "--water-pixel-confidence", type=float, default=settings.water_pixel_confidence
    )
    parser.add_argument(
        "--water-min-area-fraction",
        type=float,
        default=settings.water_min_area_fraction,
    )
    parser.add_argument(
        "--water-mode", choices=("augment", "replace"), default=settings.water_mode
    )
    parser.add_argument(
        "--residue-conf", type=float, default=settings.confidence_threshold
    )
    parser.add_argument(
        "--animal-conf", type=float, default=settings.animal_confidence_threshold
    )
    parser.add_argument(
        "--water-confirmation-seconds",
        type=float,
        default=settings.video_water_confirmation_seconds,
        help="Tempo mínimo de presença consecutiva de água para confirmar o vídeo (0 desativa)",
    )
    parser.add_argument(
        "--imgsz",
        type=validate_image_size,
        default=settings.image_size,
        help="Lado de inferência, múltiplo de 32 entre 320 e 2048 (padrão definido na configuração)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=18_000,
        help="Limite operacional de quadros (padrao: 18000)",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Usa ByteTrack e conta IDs unicos, sem somar deteccoes entre frames",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite opcional para registrar um evento-resumo do vídeo",
    )
    parser.add_argument(
        "--point-id",
        help="Identificador opcional do ponto associado ao evento de vídeo",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("resultados/local/events/evidence"),
        help="Diretório da evidência JSON quando --database for usado",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = process_video(
        args.input,
        args.output,
        args.model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        image_size=args.imgsz,
        track=args.track,
        max_frames=args.max_frames,
        water_model_path=None if args.no_water else args.water_model,
        water_pixel_confidence=args.water_pixel_confidence,
        water_min_area_fraction=args.water_min_area_fraction,
        water_mode=args.water_mode,
        residue_confidence=args.residue_conf,
        animal_confidence=args.animal_conf,
        water_confirmation_seconds=args.water_confirmation_seconds,
    )
    if args.database is not None:
        event = persist_video_event(
            manifest,
            EventRepository(args.database),
            EvidenceStore(args.evidence_dir),
            point_id=args.point_id,
        )
        manifest["event"] = event.model_dump(mode="json")
        write_json(args.output.with_suffix(".json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
