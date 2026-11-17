from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ferramentas.scripts import prepare_water_model as preparation


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "sealed"
    source.mkdir()
    payloads = {
        "config.json": b'{"model_type":"segformer"}',
        "preprocessor_config.json": b'{"size":512}',
        "pytorch_model.bin": b"pinned-tensor-state-source",
        "model.safetensors": b"converted-safe-tensor-state",
    }
    constants = {
        "CONFIG_SHA256": _digest(payloads["config.json"]),
        "PROCESSOR_SHA256": _digest(payloads["preprocessor_config.json"]),
        "BINARY_SHA256": _digest(payloads["pytorch_model.bin"]),
        "SAFETENSORS_SHA256": _digest(payloads["model.safetensors"]),
    }
    manifest = {
        "repository": preparation.REPOSITORY,
        "revision": preparation.REVISION,
        "files": [
            {"name": name, "sha256": _digest(content)}
            for name, content in payloads.items()
        ],
    }
    # Deliberate CRLF verifies preservation of sealed bytes rather than re-JSON.
    manifest_bytes = json.dumps(manifest, indent=2).replace("\n", "\r\n").encode()
    license_bytes = b"NVIDIA Source Code License\r\nResearch/evaluation only\r\n"
    (source / "download_manifest.json").write_bytes(manifest_bytes)
    (source / "LICENSE-NVIDIA-SegFormer.txt").write_bytes(license_bytes)
    constants.update(
        MANIFEST_SHA256=_digest(manifest_bytes), LICENSE_SHA256=_digest(license_bytes)
    )
    for name, value in constants.items():
        monkeypatch.setattr(preparation, name, value)
    return source, payloads


def _fill_cache(cache: Path, payloads: dict[str, bytes]):
    cache.mkdir()
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        (cache / name).write_bytes(payloads[name])


def test_cached_reconstruction_has_exact_five_files_and_preserves_sealed_bytes(
    tmp_path, monkeypatch
):
    sealed, payloads = _fixture(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    _fill_cache(cache, payloads)
    monkeypatch.setattr(
        preparation,
        "_download_to_path",
        lambda *_: pytest.fail("Cached reconstruction must not download"),
    )
    calls = []

    def convert(binary, destination):
        calls.append(binary)
        assert binary.read_bytes() == payloads["pytorch_model.bin"]
        destination.write_bytes(payloads["model.safetensors"])

    monkeypatch.setattr(preparation, "_convert_weights", convert)
    output = tmp_path / "runtime"
    result = preparation.prepare_water_model(output, cache, source_dir=sealed)
    assert result["status"] == "prepared"
    assert len(calls) == 1
    assert {path.name for path in output.iterdir()} == set(
        preparation.RUNTIME_FILENAMES
    )
    assert not (output / "pytorch_model.bin").exists()
    for name in ("download_manifest.json", "LICENSE-NVIDIA-SegFormer.txt"):
        assert (output / name).read_bytes() == (sealed / name).read_bytes()
    assert (
        preparation.prepare_water_model(output, cache, source_dir=sealed)["status"]
        == "already_present"
    )
    assert len(calls) == 1


def test_download_hash_failure_never_publishes_model_or_partial_file(
    tmp_path, monkeypatch
):
    sealed, _ = _fixture(tmp_path, monkeypatch)
    urls = []

    def download(url, destination):
        urls.append(url)
        destination.write_bytes(b"unexpected upstream content")

    monkeypatch.setattr(preparation, "_download_to_path", download)
    with pytest.raises(ValueError, match="Downloaded source hash mismatch"):
        preparation.prepare_water_model(
            tmp_path / "runtime", tmp_path / "cache", source_dir=sealed
        )
    assert urls == [
        f"https://huggingface.co/{preparation.REPOSITORY}/resolve/{preparation.REVISION}/config.json"
    ]
    assert not (tmp_path / "runtime").exists()
    assert not list((tmp_path / "cache").iterdir())


def test_nonempty_mismatched_target_and_corrupt_cache_are_preserved(
    tmp_path, monkeypatch
):
    sealed, payloads = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime"
    output.mkdir()
    sentinel = output / "user-file.txt"
    sentinel.write_bytes(b"must remain")
    with pytest.raises(FileExistsError, match="preserved"):
        preparation.prepare_water_model(
            output, tmp_path / "unused-cache", source_dir=sealed
        )
    assert sentinel.read_bytes() == b"must remain"
    assert not (tmp_path / "unused-cache").exists()
    cache = tmp_path / "cache"
    _fill_cache(cache, payloads)
    (cache / "config.json").write_bytes(b"corrupt existing cache")
    with pytest.raises(ValueError, match="Cached source hash mismatch"):
        preparation.prepare_water_model(
            tmp_path / "new-output", cache, source_dir=sealed
        )
    assert (cache / "config.json").read_bytes() == b"corrupt existing cache"


def test_changed_seal_or_conversion_fails_without_creating_runtime(
    tmp_path, monkeypatch
):
    sealed, payloads = _fixture(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    _fill_cache(cache, payloads)
    monkeypatch.setattr(
        preparation,
        "_convert_weights",
        lambda binary, destination: destination.write_bytes(b"different serialization"),
    )
    with pytest.raises(ValueError, match="safetensors hash differs"):
        preparation.prepare_water_model(tmp_path / "runtime", cache, source_dir=sealed)
    assert not (tmp_path / "runtime").exists()
    assert not list(cache.glob(".converted-*"))
    (sealed / "download_manifest.json").write_bytes(b"modified sealed content")
    with pytest.raises(ValueError, match="Sealed source hash mismatch"):
        preparation.prepare_water_model(tmp_path / "runtime", cache, source_dir=sealed)


def test_converter_loads_only_tensors_with_weights_only_and_cpu(tmp_path, monkeypatch):
    class Tensor:
        pass

    state = {"parameter": Tensor()}
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return state

    def save_file(value, destination):
        assert value is state
        Path(destination).write_bytes(b"safe")

    modules = {
        "torch": SimpleNamespace(Tensor=Tensor, load=load),
        "safetensors.torch": SimpleNamespace(save_file=save_file),
    }
    monkeypatch.setattr(preparation.importlib, "import_module", modules.__getitem__)
    output = tmp_path / "converted"
    preparation._convert_weights(tmp_path / "verified.bin", output)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert output.read_bytes() == b"safe"
    state["arbitrary_object"] = object()
    with pytest.raises(ValueError, match="only a tensor state_dict"):
        preparation._convert_weights(tmp_path / "verified.bin", tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()
