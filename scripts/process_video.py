"""Annotate a recorded MP4 with optional ByteTrack identity counting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drenavision.core.storage import EventRepository, EvidenceStore  # noqa: E402
from drenavision.inference.utils import write_json  # noqa: E402
from drenavision.inference.video import persist_video_event, process_video  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa video frame a frame com o checkpoint customizado."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/best.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
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
        default=Path("artifacts/events/evidence"),
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
        track=args.track,
        max_frames=args.max_frames,
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
