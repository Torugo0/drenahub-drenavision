from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from drenavision.core.config import Settings
from drenavision.core.schemas import Detection
from drenavision.inference.contracts import PredictionResult
from ferramentas.scripts import validate_examples as validation


def _fixtures(tmp_path: Path):
    photo, video, checkpoint = (
        tmp_path / "photo.jpg",
        tmp_path / "clip.mp4",
        tmp_path / "model.pt",
    )
    Image.new("RGB", (12, 10), "black").save(photo)
    video.write_bytes(b"original-video-contract")
    checkpoint.write_bytes(b"checkpoint-contract")
    items = []
    for name, path, kind in (("photo", photo, "image"), ("clip", video, "video")):
        items.append(
            {
                "id": name,
                "path": path.name,
                "kind": kind,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "content_tags": ["agua_acumulada"],
                **({"frames": 2} if kind == "video" else {}),
            }
        )
    catalog = tmp_path / "sources.json"
    catalog.write_text(json.dumps({"items": items}), encoding="utf-8")
    return catalog, checkpoint, {photo: photo.read_bytes(), video: video.read_bytes()}


def _stubs(monkeypatch, *, suppressed=False):
    calls = {"factory": [], "video": []}
    detection = Detection(
        class_id=1,
        class_name="agua_acumulada",
        confidence=0.99,
        bounding_box=(1, 2, 8, 9),
    )

    def factory(checkpoint, **kwargs):
        calls["factory"].append(kwargs)
        return SimpleNamespace(
            model_version="pipeline-frozen",
            is_ready=True,
            detail="ready",
            parameters={
                "residue_confidence": kwargs["residue_confidence"],
                "animal_confidence": kwargs["animal_confidence"],
                "water_mode": kwargs["water_mode"],
                "imgsz": kwargs["image_size"],
                "confidence": kwargs["confidence_threshold"],
                "iou": kwargs["iou_threshold"],
            },
            water=None
            if kwargs["water_model_path"] is None
            else SimpleNamespace(provenance={"pinned_water": True}),
            predict=lambda rgb: PredictionResult(
                [detection],
                np.full_like(rgb, (20, 80, 150)),
                2.0,
                "pipeline-frozen",
                rgb.shape[1],
                rgb.shape[0],
                kwargs["image_size"],
            ),
        )

    def process(source, output, checkpoint, **kwargs):
        calls["video"].append(kwargs)
        output.write_bytes(b"encoded-video-contract")
        Image.new("RGB", (12, 10), "blue").save(output.with_suffix(".jpg"))
        params = {
            "residue_confidence": kwargs["residue_confidence"],
            "animal_confidence": kwargs["animal_confidence"],
            "water_mode": kwargs["water_mode"],
            "imgsz": kwargs["image_size"],
            "confidence": kwargs["confidence"],
            "iou": kwargs["iou"],
        }
        video_version = (
            validation.profile_version(
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                None if kwargs["water_model_path"] is None else {"pinned_water": True},
                {
                    **params,
                    "video_water_confirmation_seconds": float(
                        kwargs["water_confirmation_seconds"]
                    ),
                },
            )
            if kwargs["water_confirmation_seconds"] > 0
            else "pipeline-frozen"
        )
        return {
            "model_version": video_version,
            "base_pipeline_version": "pipeline-frozen",
            "pipeline": {"base_pipeline_version": "pipeline-frozen"},
            "video": {"processed_frames": 2},
            "counts": {
                "agua_acumulada": {
                    "count": 0 if suppressed else 1,
                    "method": "max_simultaneous_per_frame",
                }
            },
            "raw_counts": {
                "agua_acumulada": {"count": 1, "method": "max_simultaneous_per_frame"}
            },
            "water_temporal_policy": {
                "confirmed": not suppressed,
                "raw_water_frames": 1,
                "suppressed_frames": 1 if suppressed else 0,
            },
            "review_frames": [],
            "frames": [
                {
                    "detections": []
                    if suppressed
                    else [detection.model_dump(mode="json")],
                    "raw_water_detections": [detection.model_dump(mode="json")],
                    "water_temporally_suppressed": suppressed,
                    "water": {"mask_pixels": 25},
                },
                {"detections": [], "water": {"mask_pixels": 0}},
            ],
        }

    monkeypatch.setattr(validation, "create_predictor", factory)
    monkeypatch.setattr(validation, "process_video", process)
    monkeypatch.setattr(
        validation,
        "probe_browser_video",
        lambda path: {
            "codec": "h264",
            "faststart": True,
            "decoded_frames": 2,
            "decoded_seconds": 1,
        },
    )
    return calls


def test_examples_use_one_full_pipeline_and_save_the_actual_masked_image(
    tmp_path, monkeypatch
):
    catalog, checkpoint, originals = _fixtures(tmp_path)
    calls = _stubs(monkeypatch)
    output = tmp_path / "results"
    report = validation.validate_examples(
        catalog,
        checkpoint,
        output,
        confidence=0.35,
        iou=0.45,
        device="cpu",
        image_size=640,
        water_model_path=tmp_path / "segformer",
        water_pixel_confidence=0.95,
        water_min_area_fraction=0.05,
        water_mode="augment",
        residue_confidence=0.35,
        animal_confidence=0.6,
        expected_model_version="pipeline-frozen",
    )
    assert report["status"] == "completed"
    assert report["summary"]["all_outputs_match_pipeline"]
    assert report["summary"]["processed_video_frames"] == 2
    assert (
        calls["factory"][0]["water_pixel_confidence"]
        == calls["video"][0]["water_pixel_confidence"]
        == 0.95
    )
    assert (
        calls["factory"][0]["animal_confidence"]
        == calls["video"][0]["animal_confidence"]
        == 0.6
    )
    assert calls["video"][0]["water_min_area_fraction"] == 0.05
    assert calls["video"][0]["track"] is True
    assert (
        calls["video"][0]["water_confirmation_seconds"]
        == Settings().video_water_confirmation_seconds
    )
    assert report["items"][1]["model_version"] != report["items"][0]["model_version"]
    assert report["items"][1]["base_pipeline_version"] == "pipeline-frozen"
    marked = np.array(Image.open(output / "photo_annotated.jpg"))
    assert np.allclose(marked[0, 0], (20, 80, 150), atol=2)
    assert report["items"][1]["frames_with_retained_water_mask"] == 1
    for path, content in originals.items():
        assert path.read_bytes() == content


def test_historical_no_water_flag_reaches_image_and_video_paths(tmp_path, monkeypatch):
    catalog, checkpoint, _ = _fixtures(tmp_path)
    calls = _stubs(monkeypatch)
    validation.validate_examples(
        catalog, checkpoint, tmp_path / "out", no_water=True, animal_confidence=0.35
    )
    assert calls["factory"][0]["water_model_path"] is None
    assert calls["video"][0]["water_model_path"] is None
    assert (
        calls["factory"][0]["animal_confidence"]
        == calls["video"][0]["animal_confidence"]
        == 0.35
    )


def test_frozen_profile_mismatch_stops_before_evaluating_any_file(
    tmp_path, monkeypatch
):
    catalog, checkpoint, _ = _fixtures(tmp_path)
    _stubs(monkeypatch)
    with pytest.raises(ValueError, match="Perfil divergente"):
        validation.validate_examples(
            catalog,
            checkpoint,
            tmp_path / "out",
            expected_model_version="pipeline-other",
        )
    assert not (tmp_path / "out").exists()


def test_example_cli_defaults_follow_runtime_settings_without_loading_models():
    settings = Settings()
    args = validation.build_parser().parse_args([])
    assert args.model == settings.model_path
    assert args.water_model == settings.water_model_path
    assert args.water_pixel_confidence == settings.water_pixel_confidence
    assert args.water_min_area_fraction == settings.water_min_area_fraction
    assert args.animal_conf == settings.animal_confidence_threshold
    assert args.water_confirmation_seconds == settings.video_water_confirmation_seconds
    assert validation.build_parser().parse_args(["--no-water"]).no_water


def test_temporally_suppressed_raw_water_stays_in_audit_but_not_displayed_counts(
    tmp_path, monkeypatch
):
    catalog, checkpoint, _ = _fixtures(tmp_path)
    _stubs(monkeypatch, suppressed=True)
    report = validation.validate_examples(
        catalog, checkpoint, tmp_path / "out", water_confirmation_seconds=0.5
    )
    video = report["items"][1]
    assert report["status"] == "completed"
    assert video["counts"]["agua_acumulada"] == 0
    assert video["raw_counts"]["agua_acumulada"]["count"] == 1
    assert video["raw_water_detection_frames"] == 1
    assert video["frames_with_retained_water_mask"] == 0
    assert video["frames_with_raw_semantic_water_mask"] == 1
    assert video["water_temporal_policy"]["suppressed_frames"] == 1


def test_wrong_expected_temporal_profile_stops_before_outputs(tmp_path, monkeypatch):
    catalog, checkpoint, _ = _fixtures(tmp_path)
    _stubs(monkeypatch)
    with pytest.raises(ValueError, match="Perfil de vídeo divergente"):
        validation.validate_examples(
            catalog,
            checkpoint,
            tmp_path / "out",
            expected_model_version="pipeline-frozen",
            expected_video_model_version="wrong-video-version",
        )
    assert not (tmp_path / "out").exists()
