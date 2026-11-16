"""Unicode-safe OpenCV image I/O for Windows paths."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def read_image_bgr(path: str | Path) -> Any:
    """Decode an image from bytes instead of passing a Unicode path to OpenCV."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"imagem ausente: {source}")
    cv2 = importlib.import_module("cv2")
    numpy = importlib.import_module("numpy")
    encoded = numpy.frombuffer(source.read_bytes(), dtype=numpy.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"arquivo nao e uma imagem decodificavel: {source.name}")
    return image


def write_image(path: str | Path, image_bgr: Any, *, jpeg_quality: int = 95) -> None:
    """Encode in memory and write with ``Path`` so Unicode destinations work."""

    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("saida de imagem deve ser JPG ou PNG")
    cv2 = importlib.import_module("cv2")
    extension = ".jpg" if suffix in {".jpg", ".jpeg"} else ".png"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if extension == ".jpg" else []
    ok, encoded = cv2.imencode(extension, image_bgr, parameters)
    if not ok:
        raise RuntimeError(f"OpenCV nao codificou a imagem {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded.tobytes())
