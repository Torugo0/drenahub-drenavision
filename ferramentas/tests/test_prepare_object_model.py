from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ferramentas.scripts import prepare_object_model as preparation


def fixture_assets(monkeypatch) -> dict[str, bytes]:
    payloads = {
        "config.json": b'{"model_type":"grounding-dino"}',
        "model.safetensors": b"safe-tensor-fixture",
        "tokenizer.json": b'{"fixture":"tokenizer"}',
        preparation.LICENSE_NAME: b"Apache License fixture\n",
    }
    monkeypatch.setattr(
        preparation,
        "ARTIFACTS",
        {
            name: (hashlib.sha256(content).hexdigest(), len(content))
            for name, content in payloads.items()
        },
    )
    return payloads


def test_verified_assets_only_and_idempotent_offline_target(tmp_path, monkeypatch):
    payloads = fixture_assets(monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, content in payloads.items():
        (cache / name).write_bytes(content)
    monkeypatch.setattr(
        preparation,
        "download_to_path",
        lambda *_: pytest.fail("Cache must avoid network"),
    )
    output = tmp_path / "runtime"
    assert preparation.prepare_object_model(output, cache)["status"] == "prepared"
    assert {p.name for p in output.iterdir()} == {*payloads, "download_manifest.json"}
    manifest = json.loads(
        (output / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["revision"] == preparation.REVISION
    assert manifest["weights_format"] == "safetensors"
    assert manifest["remote_code_required"] is False
    assert all("pytorch_model.bin" not in x["source_url"] for x in manifest["files"])
    assert (
        preparation.prepare_object_model(output, tmp_path / "unused")["status"]
        == "already_present"
    )
    assert not (tmp_path / "unused").exists()


def test_wrong_download_never_publishes_runtime_or_partial(tmp_path, monkeypatch):
    fixture_assets(monkeypatch)
    urls = []

    def wrong_content(url: str, destination: Path):
        urls.append(url)
        destination.write_bytes(b"wrong-content")

    monkeypatch.setattr(preparation, "download_to_path", wrong_content)
    with pytest.raises(ValueError, match="Downloaded artifact hash/size mismatch"):
        preparation.prepare_object_model(tmp_path / "runtime", tmp_path / "cache")
    assert (
        urls[0]
        == f"https://huggingface.co/{preparation.REPOSITORY}/resolve/{preparation.REVISION}/config.json"
    )
    assert not (tmp_path / "runtime").exists()
    assert not list((tmp_path / "cache").iterdir())


def test_existing_output_and_corrupt_cache_are_preserved(tmp_path, monkeypatch):
    fixture_assets(monkeypatch)
    output = tmp_path / "runtime"
    output.mkdir()
    (output / "user-file").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="preserved"):
        preparation.prepare_object_model(output, tmp_path / "unused")
    assert (output / "user-file").read_bytes() == b"keep"
    assert not (tmp_path / "unused").exists()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "config.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached artifact hash/size mismatch"):
        preparation.prepare_object_model(tmp_path / "new-output", cache)
    assert (cache / "config.json").read_bytes() == b"corrupt"


def test_source_allowlist_rejects_pickle_and_path_traversal():
    for name in ("pytorch_model.bin", "../model.safetensors", "model.py"):
        with pytest.raises(ValueError, match="not allowlisted"):
            preparation.source_url(name)
