"""Encode MP4 files that browsers can actually play.

OpenCV's ``mp4v`` writer creates MPEG-4 Part 2, which many HTML video players
cannot decode. All public video outputs pass through H.264/yuv420p with the
MP4 index at the beginning of the file (faststart).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from uuid import uuid4


class VideoEncodingError(RuntimeError):
    """The source could not be converted into a browser-playable video."""


def ffmpeg_executable() -> str:
    """Use the bundled binary; imageio also supports an explicit system binary."""

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as exc:
        raise VideoEncodingError(
            "O conversor de vídeo está indisponível. Reinstale as dependências "
            "do projeto para habilitar a reprodução no navegador."
        ) from exc


def prepare_browser_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    timeout_seconds: int = 600,
) -> dict[str, object]:
    """Create a silent H.264 MP4, preserving frame order and timing.

    The destination is replaced only after FFmpeg succeeds. An odd source size
    is padded by at most one pixel so yuv420p remains valid. This helper is also
    used for uploaded previews, including AVI/MOV/MKV and old MP4V demos.
    """

    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Vídeo de entrada não encontrado: {source.name}")
    if source == destination:
        raise ValueError("entrada e saída do conversor devem ser diferentes")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("a saída do conversor deve usar .mp4")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds deve ser positivo")

    executable = ffmpeg_executable()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.stem}-{uuid4().hex}.partial.mp4")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode != 0:
            raise VideoEncodingError(
                "Não foi possível preparar este vídeo para reprodução. "
                "Tente outro arquivo MP4, MOV, AVI ou MKV que abra normalmente "
                "no seu computador."
            )
        if not partial.is_file() or partial.stat().st_size < 32:
            raise VideoEncodingError("O conversor não produziu um vídeo válido.")
        partial.replace(destination)
    except subprocess.TimeoutExpired as exc:
        raise VideoEncodingError(
            "A preparação do vídeo demorou além do limite. Tente um trecho menor."
        ) from exc
    except OSError as exc:
        raise VideoEncodingError(
            "Não foi possível salvar o vídeo preparado. Verifique o espaço em disco."
        ) from exc
    finally:
        partial.unlink(missing_ok=True)
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "container": "mp4",
        "faststart": True,
        "audio_preserved": False,
    }
