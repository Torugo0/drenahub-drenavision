"""Pinned SigLIP 2 FixRes crop-verifier assets; no ML imports or model execution."""

from __future__ import annotations

import json
from pathlib import Path

from .object_artifacts import sha256

REPOSITORY = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
LICENSE_NAME = "LICENSE-APACHE-2.0.txt"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
ARTIFACTS = {
    "config.json": (
        "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
        253,
    ),
    "model.safetensors": (
        "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
        1500800904,
    ),
    "preprocessor_config.json": (
        "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
        394,
    ),
    "special_tokens_map.json": (
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
        636,
    ),
    "tokenizer.json": (
        "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
        34363039,
    ),
    "tokenizer_config.json": (
        "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
        47164,
    ),
    LICENSE_NAME: (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        11358,
    ),
}


def source_url(name: str) -> str:
    if name not in ARTIFACTS:
        raise ValueError(f"Artifact is not allowlisted: {name}")
    return (
        LICENSE_URL
        if name == LICENSE_NAME
        else f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
    )


def manifest_bytes() -> bytes:
    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "model_card_url": f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md",
        "license": "Apache-2.0",
        "license_url": LICENSE_URL,
        "weights_format": "safetensors",
        "remote_code_required": False,
        "purpose": "Optional zero-shot crop verification; downloading does not activate or calibrate it",
        "processor": "AutoProcessor with use_fast=True; GemmaTokenizerFast uses tokenizer.json without SentencePiece",
        "inference_notes": "Fixed-resolution SigLIP 2 uses the Siglip architecture; lowercase text; padding=max_length,max_length=64; score semantics require validation",
        "files": [
            {
                "name": name,
                "sha256": digest,
                "bytes": size,
                "source_url": source_url(name),
            }
            for name, (digest, size) in ARTIFACTS.items()
        ],
    }
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def verify_local_model(model_dir: Path) -> dict[str, str]:
    if not model_dir.is_dir():
        raise ValueError("pasta local do verificador ausente")
    expected_names = {*ARTIFACTS, "download_manifest.json"}
    actual_names = {entry.name for entry in model_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("arquivos locais do verificador ausentes ou não permitidos")
    if (model_dir / "download_manifest.json").read_bytes() != manifest_bytes():
        raise ValueError("manifesto do verificador diverge do selo fixado")
    hashes = {}
    for name, (digest, size) in ARTIFACTS.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"hash/tamanho do verificador não confere: {name}")
        hashes[name] = sha256(path)
        if hashes[name] != digest:
            raise ValueError(f"hash do verificador não confere: {name}")
    hashes["download_manifest.json"] = sha256(model_dir / "download_manifest.json")
    return hashes
