"""Download attributed real footage and photographs for the public demo library.

All inputs are independent of model predictions. The library contains genuine
recordings, never animated photos presented as recordings. Original checksums
and the live Commons license metadata are checked before creating H.264 clips.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import cv2
import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "DrenaVision-academic-demo/0.2 (attributed public media preparation)"
SOURCES = (
    {
        "id": "rato_close",
        "title": "Rato em close",
        "description": "Filmagem real de um rato doméstico visto de perto. Serve para experimentar a marcação de animais; o ambiente difere de um bueiro.",
        "kind": "video",
        "commons_title": "File:Rat.ogv",
        "original_sha1": "dfed639da4eb3462aec3f0f9d6fcdb17aeae910e",
        "license": "CC BY-SA 4.0",
        "content_tags": ["animal_visivel"],
    },
    {
        "id": "rato_rua",
        "title": "Rato entre vegetação",
        "description": "Filmagem real de um rato parcialmente escondido entre vegetação e frutos em Ljubljana. Exemplo difícil para conferir possíveis falhas.",
        "kind": "video",
        "commons_title": "File:2022-08-02 LJUBLJANA Grudnovo nabrežje RODENTIA Rattus PODGANA.webm",
        "original_sha1": "86c42571332199b3ad9bed87e408efd6f9ee0ea2",
        "license": "CC BY 3.0",
        "content_tags": ["animal_visivel"],
    },
    {
        "id": "rato_residuos",
        "title": "Rato entre resíduos",
        "description": "Filmagem real de um rato se alimentando em Hanói, Vietnã.",
        "kind": "video",
        "commons_title": "File:Manĝanta rato en Hanojo.webm",
        "original_sha1": "6f8f7291b2f62255b975e8d07f9ca57dbd916575",
        "license": "CC BY-SA 4.0",
        "content_tags": ["animal_visivel", "obstrucao_visivel"],
        "start_seconds": 6,
    },
    {
        "id": "drenagem_chuva",
        "title": "Água na drenagem",
        "description": "Filmagem real de água transbordando em um ponto de drenagem de Werrington, Austrália.",
        "kind": "video",
        "commons_title": "File:Overflowing water drain Werrington March 2022.webm",
        "original_sha1": "1bf24f62111c0b46d8226dd026bf49a137249e1b",
        "license": "CC BY 3.0",
        "content_tags": ["agua_acumulada"],
    },
    {
        "id": "canal_residuos",
        "title": "Canal de drenagem",
        "description": "Filmagem real do canal de drenagem de Alajo, em Accra, Gana.",
        "kind": "video",
        "commons_title": "File:Alajo Storm Drain 011.webm",
        "original_sha1": "7eff318adde6de3d20548682e424fa00170adc69",
        "license": "CC BY-SA 4.0",
        "content_tags": ["agua_acumulada"],
    },
    {
        "id": "sarjeta_residuos",
        "title": "Sarjeta e resíduos",
        "description": "Filmagem real de uma sarjeta que conduz água ao canal de Alajo, Gana.",
        "kind": "video",
        "commons_title": "File:Gutter to the Alajo Storm Drain.webm",
        "original_sha1": "28f108c68813ffcdeaebdcc810b3baf08a91dd79",
        "license": "CC BY-SA 4.0",
        "content_tags": ["obstrucao_visivel", "agua_acumulada"],
    },
    {
        "id": "ralo_lixo",
        "title": "Lixo junto ao bueiro",
        "description": "Fotografia real de um bueiro com resíduos em Queens, Nova York.",
        "kind": "image",
        "commons_title": "File:Sutphin Bl Jamaica Av td (2019-08-18) 14.jpg",
        "original_sha1": "68cffaf67f98fc4a175be3802f6bb7ccb536d412",
        "license": "CC BY-SA 4.0",
        "content_tags": ["obstrucao_visivel"],
    },
    {
        "id": "bueiro_agua",
        "title": "Bueiro após chuva",
        "description": "Fotografia real de um ponto de drenagem em Nova Orleans após chuva forte.",
        "kind": "image",
        "commons_title": "File:Storm Drain Clogged - Mid-City New Orleans.jpg",
        "original_sha1": "6d267080161c22f02c96d8c28d994284c7c85eeb",
        "license": "CC BY 2.0",
        "content_tags": ["obstrucao_visivel", "agua_acumulada"],
    },
)


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def commons_info(title: str) -> dict:
    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|sha1",
            "titles": title,
        }
    )
    request = Request(
        f"https://commons.wikimedia.org/w/api.php?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    pages = list(payload["query"]["pages"].values())
    if len(pages) != 1 or not pages[0].get("imageinfo"):
        raise ValueError(f"Commons não retornou mídia para {title}")
    return pages[0]["imageinfo"][0]


def download_original(info: dict, path: Path, expected_sha1: str) -> None:
    if path.is_file() and checksum(path, "sha1") == expected_sha1:
        return
    if info["sha1"] != expected_sha1:
        raise ValueError(
            "A mídia original foi substituída; revise a origem antes de baixar"
        )
    if int(info["size"]) > 100 * 1024 * 1024:
        raise ValueError("A mídia excede o limite de 100 MiB da biblioteca")
    original_url = urlsplit(info["url"])
    url = urlunsplit(original_url._replace(query="", fragment=""))
    request = Request(url, headers={"User-Agent": USER_AGENT})
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        with urlopen(request, timeout=90) as response, partial.open("wb") as stream:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > 100 * 1024 * 1024:
                    raise ValueError("Download excedeu 100 MiB")
                stream.write(chunk)
        if checksum(partial, "sha1") != expected_sha1:
            raise ValueError("Checksum SHA-1 do original não confere")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def prepare_clip(
    source: Path, output: Path, seconds: float, start_seconds: float = 0
) -> dict:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(source),
        "-t",
        str(seconds),
        "-an",
        "-vf",
        "scale=w='min(960,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps=20",
        "-c:v",
        "libx264",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=180)
    capture = cv2.VideoCapture(str(output))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível validar {output.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_MSEC, min(4000, frames / fps * 500))
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frames < 1 or fps <= 0:
            raise RuntimeError("Vídeo preparado não contém quadros válidos")
        thumbnail = output.with_name(output.stem + "_capa.jpg")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            thumbnail, quality=88
        )
        return {
            "duration_seconds": round(frames / fps, 3),
            "frames": frames,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "thumbnail": thumbnail.name,
        }
    finally:
        capture.release()


def prepare_photo(source: Path, output: Path) -> dict:
    with Image.open(source) as photo:
        photo = photo.convert("RGB")
        photo.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        photo.save(output, quality=91)
        return {"width": photo.width, "height": photo.height, "thumbnail": output.name}


def prepare_library(output_dir: Path, cache_dir: Path, seconds: float = 12) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for source in SOURCES:
        info = commons_info(source["commons_title"])
        metadata = info["extmetadata"]
        license_name = metadata["LicenseShortName"]["value"]
        if license_name != source["license"]:
            raise ValueError(f"Licença mudou para {source['id']}: {license_name}")
        original_suffix = Path(urlsplit(info["url"]).path).suffix
        original_path = cache_dir / (source["id"] + original_suffix)
        download_original(info, original_path, source["original_sha1"])
        extension = ".mp4" if source["kind"] == "video" else ".jpg"
        output = output_dir / (source["id"] + extension)
        print(f"Preparando {source['title']}...", flush=True)
        if source["kind"] == "video":
            start_seconds = float(source.get("start_seconds", 0))
            media_info = prepare_clip(original_path, output, seconds, start_seconds)
            changes = f"Trecho de até {seconds:g} s a partir de {start_seconds:g} s; H.264; máximo 960x720; 20 FPS; áudio removido."
        else:
            media_info = prepare_photo(original_path, output)
            changes = "Redimensionamento para até 1600x1200 e conversão JPEG; metadados removidos."
        items.append(
            {
                **source,
                "path": output.name,
                **media_info,
                "source_url": info["descriptionurl"],
                "original_url": urlunsplit(
                    urlsplit(info["url"])._replace(query="", fragment="")
                ),
                "author": plain_text(metadata["Artist"]["value"]),
                "license_url": metadata["LicenseUrl"]["value"],
                "changes": changes,
                "is_recorded_video": source["kind"] == "video",
                "contains_model_predictions": False,
                "sha256": checksum(output),
                "bytes": output.stat().st_size,
                "evaluation_note": "Exemplo externo para uso manual; não integra treino, validação ou teste e não possui caixas de referência revisadas.",
            }
        )
    manifest = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Entradas reais, sem anotações do modelo, para experimentar a interface.",
        "content_tags_note": "As categorias descrevem o assunto do arquivo; não são resultados de detecção ou rótulos de avaliação.",
        "items": items,
    }
    (output_dir / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Biblioteca pronta: {len(items)} exemplos em {output_dir}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dados/examples")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "dados/cache/examples")
    parser.add_argument("--seconds", type=float, default=12)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds deve estar entre 1 e 30")
    prepare_library(args.output_dir, args.cache_dir, args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
