"""Small, testable CUDA/CPU device selection with lazy torch import."""

from __future__ import annotations

import importlib
from typing import Any


def _select_device_from_torch(torch_module: Any, requested: str = "auto") -> str | int:
    normalized = str(requested).strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())

    if normalized in {"", "auto"}:
        return 0 if cuda_available else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"cuda", "cuda:0", "0"}:
        if not cuda_available:
            raise RuntimeError(
                "CUDA foi solicitada, mas torch.cuda.is_available() e falso"
            )
        return 0
    if normalized.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(
                "CUDA foi solicitada, mas torch.cuda.is_available() e falso"
            )
        index = normalized.split(":", 1)[1]
        if not index.isdigit():
            raise ValueError(f"dispositivo CUDA invalido: {requested}")
        return int(index)
    if normalized.isdigit():
        if not cuda_available:
            raise RuntimeError(
                "CUDA foi solicitada, mas torch.cuda.is_available() e falso"
            )
        return int(normalized)
    raise ValueError("device deve ser auto, cpu, cuda, cuda:N ou N")


def select_device(requested: str = "auto") -> str | int:
    """Return the Ultralytics device value without importing torch at module load."""

    torch_module = importlib.import_module("torch")
    return _select_device_from_torch(torch_module, requested)


def describe_device(device: str | int) -> str:
    if device == "cpu":
        return "cpu"
    return f"cuda:{device}"
