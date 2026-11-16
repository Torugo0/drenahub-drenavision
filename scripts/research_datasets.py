"""Report the reproducible dataset selection and optionally probe source access."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drenavision.data.constants import SOURCES  # noqa: E402
from drenavision.data.downloader import file_checksum  # noqa: E402

MAPPINGS = {
    "stagnant_water": {
        "source_classes": {"0": "water", "1": "wet surface"},
        "target_mapping": {"0": 1},
        "excluded": ["Raw data directory", "source class 1 (wet surface)"],
    },
    "click_trash": {
        "source_classes": {"0": "trash"},
        "target_mapping": {"0": 0},
        "excluded": ["13-class label set; single-class reviewed labels are used"],
    },
    "irodent": {
        "source_classes": {"1": "rodent"},
        "target_mapping": {"1": 2},
        "excluded": ["keypoints and masks are not training targets in this detector"],
    },
}

INVESTIGATED_NOT_SELECTED = [
    {
        "title": "drain_data",
        "provider": "Roboflow Universe",
        "url": "https://universe.roboflow.com/-jd8tk/drain_data",
        "license": "Public Domain (as displayed by provider)",
        "observed": "139 preview images; classes debris and drain; zero dataset versions",
        "decision": "excluded",
        "reason": "no exportable dataset version was available on the accessed page",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "UncleanDrain search results",
        "provider": "Roboflow Universe",
        "url": "https://universe.roboflow.com/search?q=UncleanDrain",
        "license": "varies by result; must be checked per dataset version",
        "observed": "provider search returned drain-related projects",
        "decision": "excluded",
        "reason": "export flow required provider sign-in; no authentication was bypassed",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "rat detection search results",
        "provider": "Roboflow Universe",
        "url": "https://universe.roboflow.com/search?q=rat%20detection",
        "license": "varies by result; must be checked per dataset version",
        "observed": "multiple rat detection projects and versions were discoverable",
        "decision": "excluded",
        "reason": "export required sign-in; iRodent offered a fixed direct archive and provenance",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "water leakage search results",
        "provider": "Roboflow Universe",
        "url": "https://universe.roboflow.com/search?q=water%20leakage",
        "license": "varies by result; must be checked per dataset version",
        "observed": "water and wet-surface projects were discoverable",
        "decision": "excluded",
        "reason": "export required sign-in; the versioned Mendeley water archive was reproducible",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "Open Images V7 — Mouse (/m/04rmv)",
        "provider": "Google Open Images",
        "url": "https://storage.googleapis.com/openimages/web/download.html",
        "class_id": "/m/04rmv",
        "license": "annotations CC BY 4.0; image licenses are recorded per image",
        "observed": "bounding-box annotations can be filtered by the Mouse class ID",
        "decision": "excluded",
        "reason": "each selected image needs its own rights audit; iRodent was a bounded fixed source",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "DAMM mouse detection",
        "provider": "Hugging Face Datasets",
        "url": "https://huggingface.co/datasets/gauravkaul/DAMM_mouse_detection",
        "license": "CC BY 4.0",
        "observed": "COCO-style mouse boxes/masks with many laboratory and video-derived frames",
        "decision": "excluded",
        "reason": "correlated lab frames require sequence-aware splitting and are less representative of deployment",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "26 Class Object Detection Dataset (VIAD-derived)",
        "provider": "Kaggle",
        "url": "https://www.kaggle.com/datasets/mohamedgobara/26-class-object-detection-dataset",
        "license": "Kaggle metadata reports Apache-2.0; upstream VIAD/Roboflow rights were not reconciled",
        "observed": "includes a Rat class and detection annotations",
        "decision": "excluded",
        "reason": "license/provenance conflict, near-duplicate augmentations, and domain gap made reuse insufficiently auditable",
        "accessed_at": "2026-11-16",
    },
    {
        "title": "Floating Waste and Aquatic Vegetation Image Dataset from Turbid Rivers",
        "provider": "Mendeley Data",
        "url": "https://data.mendeley.com/datasets/j26w4m645z/1",
        "license": "CC BY 4.0",
        "observed": "YOLO boxes for floating_waste and river_vegetation; photographs and video-derived frames",
        "decision": "reserve alternative",
        "reason": "floating river waste is less aligned with solid drain obstruction; correlated frames need grouping",
        "accessed_at": "2026-11-16",
    },
]

LOCAL_ARCHIVES = {
    "stagnant_water": Path("stagnant_water/stagnant_water.zip"),
    "irodent": Path("irodent/irodent.tar.gz"),
}


def _probe(url: str) -> dict[str, object]:
    try:
        response = requests.get(
            url,
            headers={
                "Range": "bytes=0-0",
                "User-Agent": "DrenaVision-Dataset-Research/1.0",
            },
            allow_redirects=True,
            stream=True,
            timeout=(15, 30),
        )
        return {
            "status_code": response.status_code,
            "accessible": response.status_code in {200, 206},
            "resolved_url": response.url,
            "content_range": response.headers.get("Content-Range"),
            "content_length": response.headers.get("Content-Length"),
        }
    except requests.RequestException as exc:
        return {"accessible": False, "error": str(exc)}


def build_report(
    raw_root: Path, *, check_access: bool = False, verify_local: bool = False
) -> dict[str, object]:
    selected: list[dict[str, object]] = []
    for key, source in SOURCES.items():
        record: dict[str, object] = {
            **asdict(source),
            **MAPPINGS[key],
            "decision": "selected",
            "accessed_at": "2026-11-16",
        }
        record["raw_subdir"] = source.raw_subdir.as_posix()
        local_relative = LOCAL_ARCHIVES.get(key)
        if local_relative is not None:
            local_path = raw_root / local_relative
            local: dict[str, object] = {
                "path": local_path.as_posix(),
                "exists": local_path.is_file(),
            }
            if local_path.is_file():
                local["bytes"] = local_path.stat().st_size
                if verify_local and source.checksum and source.checksum_algorithm:
                    actual = file_checksum(local_path, source.checksum_algorithm)
                    local["actual_checksum"] = actual
                    local["checksum_matches"] = (
                        actual.lower() == source.checksum.lower()
                    )
            record["local_archive"] = local
        else:
            selection = raw_root / "click_trash" / "sample" / "selection.json"
            record["local_selection_manifest"] = {
                "path": selection.as_posix(),
                "exists": selection.is_file(),
            }
        if check_access:
            record["access_probe"] = _probe(source.download_url)
        selected.append(record)
    return {
        "generated_from": "versioned repository source catalog",
        "selected": selected,
        "investigated_not_selected": INVESTIGATED_NOT_SELECTED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--check-access", action="store_true")
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw_root,
        check_access=args.check_access,
        verify_local=args.verify_local,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
