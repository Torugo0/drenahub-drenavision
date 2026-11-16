"""Stable labels and source metadata for the DrenaVision dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.constants import CLASS_NAMES

EXPECTED_CLASSES = tuple(CLASS_NAMES.values())
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class SourceSpec:
    key: str
    title: str
    landing_url: str
    download_url: str
    version: str
    license_id: str
    license_url: str
    expected_bytes: int | None = None
    checksum_algorithm: str | None = None
    checksum: str | None = None
    raw_subdir: Path = Path(".")
    notes: str = ""


SOURCES: dict[str, SourceSpec] = {
    "stagnant_water": SourceSpec(
        key="stagnant_water",
        title="Dataset of Stagnant Water and Wet Surface",
        landing_url="https://data.mendeley.com/datasets/y6zyrnxbfm/4",
        download_url=(
            "https://data.mendeley.com/public-files/datasets/y6zyrnxbfm/files/"
            "b1892735-9b83-4519-9088-1c50d982b14a/file_downloaded"
        ),
        version="4",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        expected_bytes=330_169_869,
        checksum_algorithm="sha256",
        checksum="427e336858a996ee967abdbfc94c5582b96f496ed85d3261632183d47dd39f40",
        raw_subdir=Path("stagnant_water"),
        notes=(
            "1976 pares anotados em YOLO; a classe source 0 (water) e mapeada "
            "para agua_acumulada. Raw data e wet surface isolada sao excluidos."
        ),
    ),
    "click_trash": SourceSpec(
        key="click_trash",
        title="CLICK Trash Dataset v2",
        landing_url="https://data.mendeley.com/datasets/htcvvz6fm2/2",
        download_url=(
            "https://data.mendeley.com/public-files/datasets/htcvvz6fm2/files/"
            "811798c3-3f08-4101-ad81-1d2564c91446/file_downloaded"
        ),
        version="2",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        checksum_algorithm="sha256",
        checksum="902fd9665e3ca84222540ed5071d6a6d114ec09e785c2f174509b28fbd2e2d42",
        raw_subdir=Path("click_trash"),
        notes=(
            "1126 fotografias reais; usa anotacoes YOLO single-class revisadas. "
            "O downloader usa HTTP Range e nao transfere o ZIP completo."
        ),
    ),
    "irodent": SourceSpec(
        key="irodent",
        title="iRodent: rodents in the wild",
        landing_url="https://doi.org/10.5281/zenodo.8250392",
        download_url="https://zenodo.org/api/records/8250392/files/irodent.tar.gz/content",
        version="1.0",
        license_id="Apache-2.0 record; source images CC-BY-NC",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        expected_bytes=511_341_662,
        checksum_algorithm="md5",
        checksum="e2d7ac2e4ddffe7c51bd97352f09f7f0",
        raw_subdir=Path("irodent"),
        notes=(
            "Metadados declaram 443 imagens, mas os JSONs COCO do arquivo indexam "
            "447 (357 train + 90 test). O registro Zenodo declara Apache-2.0, mas "
            "as imagens-fonte sao CC BY-NC; uso comercial requer revisao."
        ),
    ),
}
