from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from drenavision.inference import verifier_artifacts as artifacts
from ferramentas.scripts import prepare_verifier_model as preparation


def assets_fixture(monkeypatch):
    payloads = {
        "config.json": b'{"model_type":"siglip"}',
        "model.safetensors": b"safe-fixture-only",
        "tokenizer.json": b"fast-tokenizer-fixture",
        artifacts.LICENSE_NAME: b"Apache-2.0-fixture",
    }
    monkeypatch.setattr(
        artifacts,
        "ARTIFACTS",
        {
            name: (hashlib.sha256(value).hexdigest(), len(value))
            for name, value in payloads.items()
        },
    )
    return payloads


def test_verified_cache_creates_only_safe_allowlist_and_is_idempotent(
    tmp_path, monkeypatch
):
    payloads = assets_fixture(monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, value in payloads.items():
        (cache / name).write_bytes(value)
    monkeypatch.setattr(
        preparation,
        "download_to_path",
        lambda *_: pytest.fail("No network for verified cache"),
    )
    output = tmp_path / "runtime"
    assert preparation.prepare_verifier_model(output, cache)["status"] == "prepared"
    assert set(artifacts.verify_local_model(output)) == {
        *payloads,
        "download_manifest.json",
    }
    assert (
        preparation.prepare_verifier_model(output, tmp_path / "unused")["status"]
        == "already_present"
    )
    assert not (tmp_path / "unused").exists()


def test_bad_download_is_not_published_and_partial_is_removed(tmp_path, monkeypatch):
    assets_fixture(monkeypatch)
    monkeypatch.setattr(
        preparation,
        "download_to_path",
        lambda url, path: Path(path).write_bytes(b"wrong"),
    )
    with pytest.raises(ValueError, match="Downloaded verifier hash/size mismatch"):
        preparation.prepare_verifier_model(tmp_path / "runtime", tmp_path / "cache")
    assert not (tmp_path / "runtime").exists()
    assert not list((tmp_path / "cache").iterdir())


def test_existing_user_output_and_modified_manifest_are_preserved(
    tmp_path, monkeypatch
):
    payloads = assets_fixture(monkeypatch)
    output = tmp_path / "runtime"
    output.mkdir()
    for name, value in payloads.items():
        (output / name).write_bytes(value)
    (output / "download_manifest.json").write_bytes(artifacts.manifest_bytes() + b" ")
    with pytest.raises(FileExistsError, match="preserved"):
        preparation.prepare_verifier_model(output, tmp_path / "unused")
    assert (output / "download_manifest.json").read_bytes().endswith(b" ")
    assert not (tmp_path / "unused").exists()


def test_model_source_rejects_pickle_code_and_slow_tokenizer():
    for name in ("pytorch_model.bin", "model.py", "tokenizer.model", "../config.json"):
        with pytest.raises(ValueError, match="not allowlisted"):
            artifacts.source_url(name)
