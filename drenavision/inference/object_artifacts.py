"""Pinned, tensor-only object-model artifacts; standard-library imports only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
LICENSE_NAME = "LICENSE-APACHE-2.0.txt"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
ARTIFACTS = {
    "added_tokens.json": (
        "909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2",
        82,
    ),
    "config.json": (
        "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a",
        1644,
    ),
    "model.safetensors": (
        "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
        689359096,
    ),
    "preprocessor_config.json": (
        "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60",
        457,
    ),
    "special_tokens_map.json": (
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        125,
    ),
    "tokenizer.json": (
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        711396,
    ),
    "tokenizer_config.json": (
        "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56",
        1237,
    ),
    "vocab.txt": (
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        231508,
    ),
    LICENSE_NAME: (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        11358,
    ),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_local_model(model_dir: Path) -> dict[str, str]:
    """Check the seal, manifest and exact runtime allowlist before any ML import."""
    if not model_dir.is_dir():
        raise ValueError("pasta local do detector de objetos ausente")
    required = {*ARTIFACTS, "download_manifest.json"}
    actual = {entry.name for entry in model_dir.iterdir()}
    missing = required - actual
    if missing:
        raise ValueError(f"arquivos locais ausentes: {', '.join(sorted(missing))}")
    if actual - required:
        raise ValueError(
            f"arquivos não permitidos no modelo local: {', '.join(sorted(actual - required))}"
        )
    manifest = json.loads(
        (model_dir / "download_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise ValueError("repositório/revisão do detector divergem do modelo fixado")
    if (
        manifest.get("weights_format") != "safetensors"
        or manifest.get("remote_code_required") is not False
    ):
        raise ValueError("manifesto exige safetensors local sem código remoto")
    if manifest.get("license") != "Apache-2.0":
        raise ValueError("licença declarada diverge do modelo fixado")
    entries = manifest.get("files", [])
    if len(entries) != len(ARTIFACTS) or {item["name"] for item in entries} != set(
        ARTIFACTS
    ):
        raise ValueError("lista de arquivos do manifesto diverge do modelo fixado")
    hashes = {}
    for entry in entries:
        name = entry["name"]
        expected_hash, expected_size = ARTIFACTS[name]
        path = model_dir / name
        if entry.get("sha256") != expected_hash or entry.get("bytes") != expected_size:
            raise ValueError(f"hash/tamanho do manifesto diverge do selo: {name}")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"hash/tamanho do arquivo não confere: {name}")
        hashes[name] = sha256(path)
        if hashes[name] != expected_hash:
            raise ValueError(f"hash do arquivo não confere: {name}")
    hashes["download_manifest.json"] = sha256(model_dir / "download_manifest.json")
    return hashes
