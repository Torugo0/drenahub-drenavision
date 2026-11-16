import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from drenavision.data.constants import SourceSpec
from drenavision.data.downloader import (
    download_click_sample,
    download_resumable,
    safe_extract_zip,
)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), color).save(stream, format="JPEG")
    return stream.getvalue()


def test_click_partial_download_is_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads: dict[str, bytes] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        for index in range(3):
            name = f"trash ({index}).jpg"
            payloads[f"images/{split}/{name}"] = _jpeg_bytes((split_index, index, 5))
            payloads[f"single class/labels/{split}/trash ({index}).txt"] = (
                b"0 0.5 0.5 0.2 0.2\n"
            )

    openings = 0

    class FakeRemoteZip:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal openings
            openings += 1

        def __enter__(self) -> "FakeRemoteZip":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return list(payloads)

        def read(self, name: str) -> bytes:
            return payloads[name]

    monkeypatch.setitem(
        sys.modules, "remotezip", types.SimpleNamespace(RemoteZip=FakeRemoteZip)
    )
    quotas = {"train": 2, "val": 1, "test": 1}
    root = download_click_sample(
        tmp_path, quotas=quotas, seed=42, progress=lambda _message: None
    )
    manifest = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    first_selection = [item["image_remote"] for item in manifest["items"]]
    assert len(first_selection) == 4
    assert all((root / item["label_local"]).is_file() for item in manifest["items"])

    download_click_sample(
        tmp_path, quotas=quotas, seed=42, progress=lambda _message: None
    )
    assert openings == 1
    assert [item["image_remote"] for item in manifest["items"]] == first_selection


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "extract")


def test_resumable_download_uses_range_and_verifies_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "archive.bin"
    destination.with_name("archive.bin.part").write_bytes(b"abc")
    expected = b"abcdef"
    source = SourceSpec(
        key="fixture",
        title="fixture",
        landing_url="https://example.test",
        download_url="https://example.test/archive.bin",
        version="1",
        license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        expected_bytes=len(expected),
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(expected).hexdigest(),
    )
    captured_headers: dict[str, str] = {}

    class FakeResponse:
        status_code = 206

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, *, chunk_size: int) -> list[bytes]:
            assert chunk_size > 0
            return [b"def"]

    def fake_get(_url: str, **kwargs: object) -> FakeResponse:
        captured_headers.update(kwargs["headers"])  # type: ignore[arg-type]
        return FakeResponse()

    monkeypatch.setattr("drenavision.data.downloader.requests.get", fake_get)
    result = download_resumable(source, destination, progress=lambda _message: None)
    assert result.read_bytes() == expected
    assert captured_headers["Range"] == "bytes=3-"
