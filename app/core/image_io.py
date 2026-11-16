"""Defensive image decoding and encoding."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when input bytes are not a supported, safe image."""


def decode_image(data: bytes, *, max_pixels: int) -> np.ndarray:
    if not data:
        raise InvalidImageError("Arquivo de imagem vazio.")

    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise InvalidImageError("Formato inválido; use JPEG ou PNG.")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise InvalidImageError("Dimensões da imagem excedem o limite.")
            source.load()
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            return np.asarray(rgb, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Não foi possível decodificar a imagem.") from exc


def encode_jpeg(image_rgb: np.ndarray, *, quality: int = 90) -> bytes:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected an RGB image with shape HxWx3")
    output = BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(
        output, format="JPEG", quality=quality, optimize=True
    )
    return output.getvalue()
