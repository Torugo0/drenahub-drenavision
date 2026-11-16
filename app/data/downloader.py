"""Reproducible downloads for the three approved DrenaVision sources."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import requests

from .constants import SOURCES, SPLITS, SourceSpec

Progress = Callable[[str], None]
DEFAULT_CLICK_QUOTAS = {"train": 100, "val": 20, "test": 20}


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_download(path: Path, source: SourceSpec) -> None:
    if (
        source.expected_bytes is not None
        and path.stat().st_size != source.expected_bytes
    ):
        raise RuntimeError(
            f"unexpected size for {source.key}: {path.stat().st_size}; "
            f"expected {source.expected_bytes}"
        )
    if source.checksum and source.checksum_algorithm:
        actual = file_checksum(path, source.checksum_algorithm)
        if actual.lower() != source.checksum.lower():
            raise RuntimeError(
                f"checksum mismatch for {source.key}: {actual} != {source.checksum}"
            )


def download_resumable(
    source: SourceSpec,
    destination: Path,
    *,
    force: bool = False,
    progress: Progress = print,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with HTTP Range resume and verify known metadata."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.exists()
        and not force
        and (
            source.expected_bytes is None
            or destination.stat().st_size == source.expected_bytes
        )
    ):
        try:
            _verify_download(destination, source)
        except RuntimeError:
            progress(f"discarding corrupt completed download: {destination}")
            destination.unlink()
        else:
            progress(f"already verified: {destination}")
            return destination

    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and not force and not partial.exists():
        destination.replace(partial)
    if force and partial.exists():
        partial.unlink()

    if partial.exists() and source.expected_bytes is not None:
        partial_size = partial.stat().st_size
        if partial_size == source.expected_bytes:
            try:
                _verify_download(partial, source)
            except RuntimeError:
                progress(f"discarding corrupt partial download: {partial}")
                partial.unlink()
            else:
                partial.replace(destination)
                progress(f"already verified: {destination}")
                return destination
        elif partial_size > source.expected_bytes:
            progress(f"discarding oversized partial download: {partial}")
            partial.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DrenaVision-Dataset-Builder/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        progress(f"resuming {source.key} at byte {offset}")

    with requests.get(
        source.download_url, headers=headers, stream=True, timeout=(30, 180)
    ) as response:
        response.raise_for_status()
        resumed = offset > 0 and response.status_code == 206
        if offset and not resumed:
            progress("server ignored Range; restarting download safely")
            offset = 0
        mode = "ab" if resumed else "wb"
        transferred = offset
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                stream.write(chunk)
                transferred += len(chunk)
                if transferred % (25 * 1024 * 1024) < chunk_size:
                    progress(f"{source.key}: {transferred / 1024**2:.1f} MiB")

    partial.replace(destination)
    _verify_download(destination, source)
    progress(f"downloaded and verified: {destination}")
    return destination


def _safe_member_destination(root: Path, member_name: str) -> Path:
    root_resolved = root.resolve()
    destination = (root / member_name).resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"unsafe archive member: {member_name}") from exc
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    marker = destination / ".extracted.ok"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            _safe_member_destination(destination, info.filename)
        bundle.extractall(destination)
    marker.write_text("verified extraction\n", encoding="utf-8")
    return destination


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    marker = destination / ".extracted.ok"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            _safe_member_destination(destination, member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not accepted: {member.name}")
        bundle.extractall(destination, members=members, filter="data")
    marker.write_text("verified extraction\n", encoding="utf-8")
    return destination


def download_stagnant_water(
    raw_root: Path, *, force: bool = False, progress: Progress = print
) -> Path:
    source = SOURCES["stagnant_water"]
    root = raw_root / source.raw_subdir
    archive = download_resumable(
        source, root / "stagnant_water.zip", force=force, progress=progress
    )
    return safe_extract_zip(archive, root / "extracted")


def download_irodent(
    raw_root: Path, *, force: bool = False, progress: Progress = print
) -> Path:
    source = SOURCES["irodent"]
    root = raw_root / source.raw_subdir
    archive = download_resumable(
        source, root / "irodent.tar.gz", force=force, progress=progress
    )
    return safe_extract_tar(archive, root / "extracted")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def download_click_sample(
    raw_root: Path,
    *,
    quotas: dict[str, int] | None = None,
    seed: int = 42,
    force: bool = False,
    progress: Progress = print,
) -> Path:
    """Read only selected CLICK files from its 5.57 GB ZIP using HTTP Range."""

    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError("remotezip is required: pip install remotezip") from exc

    quotas = dict(quotas or DEFAULT_CLICK_QUOTAS)
    unknown_splits = set(quotas) - set(SPLITS)
    if unknown_splits or any(value < 0 for value in quotas.values()):
        raise ValueError(f"invalid CLICK quotas: {quotas}")

    source = SOURCES["click_trash"]
    root = raw_root / source.raw_subdir / "sample"
    manifest_path = root / "selection.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = sum(quotas.values())
        if (
            manifest.get("seed") == seed
            and len(manifest.get("items", [])) == expected
            and all(
                (root / item["image_local"]).exists()
                and (root / item["label_local"]).exists()
                and file_checksum(root / item["image_local"]) == item.get("sha256")
                for item in manifest["items"]
            )
        ):
            progress(f"already downloaded: {expected} CLICK image/label pairs")
            return root

    rng = random.Random(seed)
    selected: list[tuple[str, str, str]] = []
    progress("opening CLICK central directory with HTTP Range")
    with RemoteZip(
        source.download_url,
        headers={"User-Agent": "DrenaVision-Dataset-Builder/1.0"},
    ) as archive:
        names = set(archive.namelist())
        for split in SPLITS:
            count = quotas.get(split, 0)
            pool = sorted(
                name
                for name in names
                if name.startswith(f"images/{split}/")
                and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if count > len(pool):
                raise RuntimeError(
                    f"CLICK {split} has {len(pool)} images, fewer than requested {count}"
                )
            for image_name in rng.sample(pool, count):
                label_name = (
                    f"single class/labels/{split}/{PurePosixPath(image_name).stem}.txt"
                )
                if label_name not in names:
                    raise RuntimeError(f"missing CLICK label: {label_name}")
                selected.append((split, image_name, label_name))

        items: list[dict[str, object]] = []
        for index, (split, image_name, label_name) in enumerate(selected, 1):
            image_local = Path("images") / split / PurePosixPath(image_name).name
            label_local = Path("labels") / split / PurePosixPath(label_name).name
            image_path = root / image_local
            label_path = root / label_local
            if force or not image_path.exists():
                _atomic_bytes(image_path, archive.read(image_name))
            if force or not label_path.exists():
                _atomic_bytes(label_path, archive.read(label_name))
            items.append(
                {
                    "split": split,
                    "image_remote": image_name,
                    "label_remote": label_name,
                    "image_local": image_local.as_posix(),
                    "label_local": label_local.as_posix(),
                    "sha256": file_checksum(image_path),
                }
            )
            progress(f"CLICK {index}/{len(selected)}: {image_name}")

    manifest = {
        "source": source.key,
        "source_version": source.version,
        "source_url": source.landing_url,
        "archive_url": source.download_url,
        "archive_sha256_published": source.checksum,
        "seed": seed,
        "quotas": quotas,
        "items": items,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    progress(f"CLICK sample ready: {root}")
    return root
