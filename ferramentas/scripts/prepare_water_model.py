"""Reproduce the pinned local B1 water model without publishing pickle files.

Only the official, SHA-verified state_dict is loaded, with weights_only=True.
Versioned source-manifest/license bytes are copied unchanged to preserve the
same runtime provenance on fresh clones. Existing mismatched targets are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "nvidia/segformer-b1-finetuned-ade-512-512"
REVISION = "a775b4f5dcb2179e57006dca0e268281306412ef"
BINARY_SHA256 = "ac49bf62c8c0578dcf17ff048930b4f886ab0bc080d23f5591669afad5ddf7da"
SAFETENSORS_SHA256 = "e0802e077c195d9dbaf48c69b94946ef6a3ae5b6665176fedf363e5e63a1ec93"
CONFIG_SHA256 = "8f1b7bcd2e1835ef0ef3a04a0d4dfe87ff9a6374253e7a3ca1751de812a420ff"
PROCESSOR_SHA256 = "8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c"
MANIFEST_SHA256 = "d22e86d4ad23c1c65bf4cfebc730d69501e2db98a800d4ce047778ed222a9dc6"
LICENSE_SHA256 = "b5a6a3804b12c1b4d37e1e45bcc31761a2e69064fd75b48da13d62e30d69b406"
SEALED_SOURCE = ROOT / "resultados/metrics/v3/water_model_source"
RUNTIME_FILENAMES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "download_manifest.json",
    "LICENSE-NVIDIA-SegFormer.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_hashes() -> dict[str, str]:
    return dict(
        zip(
            RUNTIME_FILENAMES,
            (
                SAFETENSORS_SHA256,
                CONFIG_SHA256,
                PROCESSOR_SHA256,
                MANIFEST_SHA256,
                LICENSE_SHA256,
            ),
            strict=True,
        )
    )


def _sealed_manifest(source_dir: Path) -> dict:
    for name, expected in (
        ("download_manifest.json", MANIFEST_SHA256),
        ("LICENSE-NVIDIA-SegFormer.txt", LICENSE_SHA256),
    ):
        if not (source_dir / name).is_file() or _sha256(source_dir / name) != expected:
            raise ValueError(
                f"Sealed source hash mismatch: {name}; restore the versioned file"
            )
    manifest = json.loads(
        (source_dir / "download_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise ValueError(
            "Sealed manifest repository/revision differs from the pinned model"
        )
    hashes = {item["name"]: item["sha256"] for item in manifest["files"]}
    for name, expected in (
        ("config.json", CONFIG_SHA256),
        ("preprocessor_config.json", PROCESSOR_SHA256),
        ("pytorch_model.bin", BINARY_SHA256),
        ("model.safetensors", SAFETENSORS_SHA256),
    ):
        if hashes.get(name) != expected:
            raise ValueError(f"Sealed manifest artifact hash differs: {name}")
    return manifest


def _download_to_path(url: str, destination: Path) -> None:
    """Download only a pinned official Hugging Face source URL."""
    import requests

    with requests.get(url, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    stream.write(block)


def _cache_source(cache_dir: Path, name: str, expected: str) -> Path:
    destination = cache_dir / name
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected:
            raise ValueError(
                f"Cached source hash mismatch; preserved for inspection: {destination}"
            )
        return destination
    partial = cache_dir / f".{name}.{uuid4().hex}.partial"
    try:
        _download_to_path(
            f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", partial
        )
        if _sha256(partial) != expected:
            raise ValueError(f"Downloaded source hash mismatch: {name}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _convert_weights(binary: Path, destination: Path) -> None:
    """Convert verified tensor-only state, never load arbitrary model objects."""
    torch = importlib.import_module("torch")
    safetensors = importlib.import_module("safetensors.torch")
    state = torch.load(binary, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
    ):
        raise ValueError("The verified source must contain only a tensor state_dict")
    safetensors.save_file(state, str(destination))


def _cache_safetensors(cache_dir: Path, binary: Path) -> Path:
    destination = cache_dir / f"model-{SAFETENSORS_SHA256[:16]}.safetensors"
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != SAFETENSORS_SHA256:
            raise ValueError(f"Converted cache hash mismatch; preserved: {destination}")
        return destination
    partial = cache_dir / f".converted-{uuid4().hex}.safetensors"
    try:
        _convert_weights(binary, partial)
        if _sha256(partial) != SAFETENSORS_SHA256:
            raise ValueError("Converted safetensors hash differs from the tested model")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def prepare_water_model(
    output_dir: Path,
    cache_dir: Path,
    *,
    source_dir: Path = SEALED_SOURCE,
) -> dict:
    """Create exactly five runtime files, or verify an already-correct target."""
    output_dir, cache_dir = output_dir.resolve(), cache_dir.resolve()
    _sealed_manifest(source_dir)
    expected = _runtime_hashes()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(
                f"Model output is not a directory; preserved: {output_dir}"
            )
        if all(
            (output_dir / name).is_file() and _sha256(output_dir / name) == digest
            for name, digest in expected.items()
        ):
            return {
                "status": "already_present",
                "repository": REPOSITORY,
                "revision": REVISION,
                "runtime_sha256": expected,
                "output_dir": str(output_dir),
                "license": "NVIDIA non-commercial research/evaluation only",
            }
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Non-empty model output differs and was preserved: {output_dir}; choose a new --output-dir"
            )
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = _cache_source(cache_dir, "config.json", CONFIG_SHA256)
    processor = _cache_source(cache_dir, "preprocessor_config.json", PROCESSOR_SHA256)
    binary = _cache_source(cache_dir, "pytorch_model.bin", BINARY_SHA256)
    tensors = _cache_safetensors(cache_dir, binary)
    sources = {
        "model.safetensors": tensors,
        "config.json": config,
        "preprocessor_config.json": processor,
        "download_manifest.json": source_dir / "download_manifest.json",
        "LICENSE-NVIDIA-SegFormer.txt": source_dir / "LICENSE-NVIDIA-SegFormer.txt",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-staging-{uuid4().hex}"
    # Both move endpoints are explicit children of the requested output parent.
    if staging.resolve().parent != output_dir.parent.resolve():
        raise ValueError("Refusing staging path outside the requested output parent")
    staging.mkdir()
    try:
        for name, source in sources.items():
            shutil.copyfile(source, staging / name)
            if _sha256(staging / name) != expected[name]:
                raise ValueError(f"Runtime copy hash mismatch: {name}")
        if output_dir.exists():
            output_dir.rmdir()  # Fails safely if another writer populated it.
        staging.replace(output_dir)
    finally:
        if staging.exists():
            # Remove only the five files created here; no recursive deletion.
            for name in RUNTIME_FILENAMES:
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    return {
        "status": "prepared",
        "repository": REPOSITORY,
        "revision": REVISION,
        "runtime_sha256": expected,
        "output_dir": str(output_dir),
        "source_binary_cache_only": str(binary),
        "license": "NVIDIA non-commercial research/evaluation only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "resultados/local/models/water_segformer_b1_ade",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "resultados/local/cache/water_model_b1"
    )
    args = parser.parse_args(argv)
    print(
        "Modelo NVIDIA para pesquisa/avaliação não comercial; a licença acompanha o pacote.",
        file=sys.stderr,
    )
    result = prepare_water_model(args.output_dir, args.cache_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
