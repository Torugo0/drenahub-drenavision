"""Download only pinned official SigLIP 2 safetensors and fast-tokenizer assets.

No model instantiation/inference, Torch import, pickle, SentencePiece dependency,
remote code, quantization install or changes to active detector configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from drenavision.inference import verifier_artifacts as artifacts

ROOT = Path(__file__).resolve().parents[2]


def download_to_path(url: str, destination: Path) -> None:
    import requests

    received, next_report = 0, 64 * 1024 * 1024
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


def cache_artifact(cache_dir: Path, name: str) -> Path:
    digest, size = artifacts.ARTIFACTS[name]
    target = cache_dir / name
    if target.exists():
        if (
            not target.is_file()
            or target.stat().st_size != size
            or artifacts.sha256(target) != digest
        ):
            raise ValueError(f"Cached verifier hash/size mismatch; preserved: {target}")
        return target
    partial = cache_dir / f".{name}.{uuid4().hex}.partial"
    try:
        download_to_path(artifacts.source_url(name), partial)
        if (
            not partial.is_file()
            or partial.stat().st_size != size
            or artifacts.sha256(partial) != digest
        ):
            raise ValueError(f"Downloaded verifier hash/size mismatch: {name}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def prepare_verifier_model(output_dir: Path, cache_dir: Path) -> dict:
    output_dir, cache_dir = output_dir.resolve(), cache_dir.resolve()
    manifest = artifacts.manifest_bytes()
    result = {
        "repository": artifacts.REPOSITORY,
        "revision": artifacts.REVISION,
        "license": "Apache-2.0",
        "output_dir": str(output_dir),
        "weights_sha256": artifacts.ARTIFACTS["model.safetensors"][0],
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output is not a directory; preserved: {output_dir}")
        if any(output_dir.iterdir()):
            try:
                artifacts.verify_local_model(output_dir)
            except Exception as exc:
                raise FileExistsError(
                    f"Non-empty verifier output differs and was preserved: {output_dir}; choose a new --output-dir"
                ) from exc
            return {"status": "already_present", **result}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {name: cache_artifact(cache_dir, name) for name in artifacts.ARTIFACTS}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}-staging-{uuid4().hex}"
    # Both rename endpoints are resolved children of the explicitly named parent.
    if staging.resolve().parent != output_dir.parent or output_dir.parent == output_dir:
        raise ValueError("Refusing staging outside the requested output parent")
    staging.mkdir()
    try:
        for name, source in cached.items():
            shutil.copyfile(source, staging / name)
        (staging / "download_manifest.json").write_bytes(manifest)
        artifacts.verify_local_model(staging)
        if output_dir.exists():
            output_dir.rmdir()  # Empty only; fails safely if another writer populated it.
        staging.replace(output_dir)
    finally:
        if staging.exists():
            for name in {*artifacts.ARTIFACTS, "download_manifest.json"}:
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    return {"status": "prepared", **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "resultados/local/models/siglip2_base_patch16_224",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "resultados/local/cache/verifier_model_siglip2",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_verifier_model(args.output_dir, args.cache_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
