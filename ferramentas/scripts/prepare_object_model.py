"""Prepare pinned Grounding DINO Tiny assets without pickle or remote code.

This downloads data only; it never imports Torch, instantiates a detector or
runs inference. Existing mismatched output/cache files are preserved. Runtime
consumers should use local_files_only=True, trust_remote_code=False and
use_safetensors=True. This optional model does not replace the active YOLO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from drenavision.inference.object_artifacts import (
    ARTIFACTS,
    LICENSE_NAME,
    LICENSE_URL,
    REPOSITORY,
    REVISION,
)

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(name: str) -> str:
    if name not in ARTIFACTS:
        raise ValueError(f"Artifact is not allowlisted: {name}")
    if name == LICENSE_NAME:
        return LICENSE_URL
    return f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"


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
        "download_scope": "weights, configuration, tokenizer, processor and official license; no pickle or executable code",
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


def verify_file(path: Path, digest: str, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def download_to_path(url: str, destination: Path) -> None:
    import requests

    received = 0
    next_report = 64 * 1024 * 1024
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    received += len(chunk)
                    if received >= next_report:
                        print(
                            f"Downloaded {received // (1024 * 1024)} MiB",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_report += 64 * 1024 * 1024


def cache_artifact(cache_dir: Path, name: str, digest: str, size: int) -> Path:
    target = cache_dir / name
    if target.exists():
        if not verify_file(target, digest, size):
            raise ValueError(f"Cached artifact hash/size mismatch; preserved: {target}")
        return target
    partial = cache_dir / f".{name}.{uuid4().hex}.partial"
    try:
        download_to_path(source_url(name), partial)
        if not verify_file(partial, digest, size):
            raise ValueError(f"Downloaded artifact hash/size mismatch: {name}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def prepare_object_model(output_dir: Path, cache_dir: Path) -> dict:
    output_dir, cache_dir = output_dir.resolve(), cache_dir.resolve()
    expected_names = {*ARTIFACTS, "download_manifest.json"}
    manifest = manifest_bytes()
    result = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0",
        "output_dir": str(output_dir),
        "weights_sha256": ARTIFACTS["model.safetensors"][0],
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output is not a directory; preserved: {output_dir}")
        present = {path.name for path in output_dir.iterdir()}
        if (
            present == expected_names
            and all(
                verify_file(output_dir / name, digest, size)
                for name, (digest, size) in ARTIFACTS.items()
            )
            and (output_dir / "download_manifest.json").read_bytes() == manifest
        ):
            return {"status": "already_present", **result}
        if present:
            raise FileExistsError(
                f"Non-empty output differs and was preserved: {output_dir}; choose a new --output-dir"
            )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {
        name: cache_artifact(cache_dir, name, digest, size)
        for name, (digest, size) in ARTIFACTS.items()
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-staging-{uuid4().hex}"
    # Both rename endpoints are resolved children of the explicitly named parent.
    if staging.resolve().parent != output_dir.parent or output_dir.parent == output_dir:
        raise ValueError("Refusing a staging path outside the requested output parent")
    staging.mkdir()
    try:
        for name, source in cached.items():
            shutil.copyfile(source, staging / name)
            if not verify_file(staging / name, *ARTIFACTS[name]):
                raise ValueError(f"Runtime copy hash/size mismatch: {name}")
        (staging / "download_manifest.json").write_bytes(manifest)
        if output_dir.exists():
            output_dir.rmdir()  # Empty only; another writer fails safely.
        staging.replace(output_dir)
    finally:
        if staging.exists():
            for name in expected_names:
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    return {"status": "prepared", **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "resultados/local/models/grounding_dino_tiny",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "resultados/local/cache/object_model_grounding_dino",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_object_model(args.output_dir, args.cache_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
