from types import SimpleNamespace

import numpy as np
import pytest

from drenavision.core.schemas import Detection
from drenavision.inference.contracts import PredictionResult
from drenavision.inference.errors import ModelUnavailableError
from drenavision.inference.runtime import (
    InspectionPredictor,
    filter_detections,
    merge_water_detections,
    overlay_water,
    profile_version,
)


def detection(class_id, confidence=0.8, box=(4, 4, 24, 24)):
    return Detection(
        class_id=class_id,
        class_name={0: "obstrucao_visivel", 1: "agua_acumulada", 2: "animal_visivel"}[
            class_id
        ],
        confidence=confidence,
        bounding_box=box,
    )


def test_operating_thresholds_preserve_real_coordinates_and_other_classes():
    low, high, litter, water = (
        detection(2, 0.59),
        detection(2, 0.6),
        detection(0, 0.35),
        detection(1, 0.8),
    )
    assert filter_detections([low, high, litter, water]) == [high, litter, water]
    assert high.bounding_box == (4, 4, 24, 24)
    with pytest.raises(ValueError):
        filter_detections([high], animal_confidence=float("nan"))


def test_water_fusion_uses_existing_boxes_without_cross_class_suppression():
    broad = detection(1, 0.8, (0, 0, 30, 30))
    contained = detection(1, 0.9, (4, 4, 24, 24))
    separate = detection(1, 0.7, (29, 29, 40, 40))
    animal = detection(2, 0.8, (4, 4, 24, 24))
    fused = merge_water_detections([broad, animal], [contained, separate])
    assert fused == [animal, broad, separate]
    assert all(item in [animal, broad, contained, separate] for item in fused)
    assert merge_water_detections([broad, animal], [contained], mode="replace") == [
        animal,
        contained,
    ]


def test_tint_preserves_dry_holes_and_original_pixels():
    image = np.full((32, 48, 3), [255, 100, 0], dtype=np.uint8)
    original = image.copy()
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[5:20, 5:30] = 255
    mask[10:15, 10:15] = 0
    result = overlay_water(image, mask)
    assert np.array_equal(image, original)
    assert np.array_equal(result[11, 11], image[11, 11])
    assert not np.array_equal(result[6, 6], image[6, 6])


def test_profile_is_canonical_and_changes_with_either_model_or_policy():
    left = profile_version("master-sha", {"weight": "water-sha"}, {"b": 2, "a": 1})
    assert left == profile_version(
        "master-sha", {"weight": "water-sha"}, {"a": 1, "b": 2}
    )
    assert left != profile_version("master-sha", {"weight": "other"}, {"b": 2, "a": 1})
    assert left != profile_version(
        "other-master", {"weight": "water-sha"}, {"b": 2, "a": 1}
    )
    assert left != profile_version(
        "master-sha", {"weight": "water-sha"}, {"b": 3, "a": 1}
    )


def master(tmp_path, detections):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    return SimpleNamespace(
        model_path=checkpoint,
        image_size=640,
        confidence_threshold=0.35,
        iou_threshold=0.45,
        is_ready=True,
        detail="ready",
        predict=lambda rgb: PredictionResult(
            detections,
            np.zeros_like(rgb),
            3,
            "fixture",
            rgb.shape[1],
            rgb.shape[0],
            640,
        ),
    )


def test_filtered_animal_box_cannot_survive_in_exported_image(tmp_path):
    image = np.full((64, 96, 3), [210, 80, 30], dtype=np.uint8)
    predictor = InspectionPredictor(
        master(tmp_path, [detection(2, 0.5)]), animal_confidence=0.6
    )
    result = predictor.predict(image)
    assert result.detections == []
    assert np.array_equal(result.annotated_image, image)


def test_missing_configured_water_is_unavailable_and_cannot_silently_fallback(tmp_path):
    def unavailable(_rgb):
        raise ModelUnavailableError("configured water unavailable")

    water = SimpleNamespace(
        is_ready=False, detail="missing", predict=unavailable, provenance={}
    )
    predictor = InspectionPredictor(master(tmp_path, []), water)
    assert not predictor.is_ready
    assert predictor.detail == "missing"
    with pytest.raises(ModelUnavailableError, match="configured water"):
        predictor.predict(np.zeros((64, 96, 3), dtype=np.uint8))


def test_api_receives_water_configuration_and_animal_operating_point(
    monkeypatch, tmp_path
):
    from drenavision.api.main import _build_predictor
    from drenavision.core.config import Settings
    from drenavision.inference import runtime

    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(is_ready=True)

    monkeypatch.setattr(runtime, "create_predictor", create)
    settings = Settings(
        _env_file=None,
        water_model_path=tmp_path / "water",
        water_pixel_confidence=0.9,
        water_min_area_fraction=0.003,
        water_mode="augment",
        animal_confidence_threshold=0.6,
    )
    assert _build_predictor(settings).is_ready
    assert received["water_model_path"] == tmp_path / "water"
    assert received["animal_confidence"] == 0.6
    assert received["water_pixel_confidence"] == 0.9


def test_interface_displays_exact_recorded_evidence_instead_of_redrawing_boxes(
    monkeypatch,
):
    from io import BytesIO

    from PIL import Image

    from drenavision.ui import streamlit_app

    source = BytesIO()
    Image.new("RGB", (40, 30), "red").save(source, format="PNG")
    marked = BytesIO()
    Image.new("RGB", (40, 30), "blue").save(marked, format="PNG")
    calls = []

    def request(method, path, **_kwargs):
        calls.append((method, path))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"event_id": "test-evidence", "detections": []},
            content=marked.getvalue(),
        )

    monkeypatch.setattr(streamlit_app, "api_request", request)
    output = streamlit_app.analyze_image(source.getvalue(), "photo.png", "")
    assert output["image"].getpixel((0, 0)) == (0, 0, 255)
    assert calls == [
        ("POST", "/vision/predict"),
        ("GET", "/vision/evidence/test-evidence"),
    ]
