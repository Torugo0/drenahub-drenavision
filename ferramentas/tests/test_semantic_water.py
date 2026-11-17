from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drenavision.inference import water
from drenavision.inference.water import (
    SemanticWaterPredictor,
    _regions_from_labels,
    draw_mask_on_rgb,
    regions_from_maps,
)


def _model_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    labels = {str(index): f"label_{index}" for index in range(150)}
    labels.update(
        {
            "21": "water",
            "26": "sea",
            "60": "river",
            "109": "swimming pool",
            "113": "waterfall",
            "128": "lake",
        }
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "segformer", "id2label": labels}), encoding="utf-8"
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({"do_normalize": True, "do_resize": True}), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(
        b"test-only-checkpoint; model loader is stubbed"
    )
    (root / "LICENSE-NVIDIA-SegFormer.txt").write_text(
        "NVIDIA Source Code License for SegFormer", encoding="utf-8"
    )
    _refresh_manifest(root)
    return root


def _refresh_manifest(root: Path) -> None:
    (root / "download_manifest.json").write_text(
        json.dumps(
            {
                "repository": "nvidia/segformer-b1-finetuned-ade-512-512",
                "revision": "a" * 40,
                "license": "NVIDIA non-commercial research/evaluation",
                "license_url": "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
                "files": [
                    {
                        "name": name,
                        "sha256": hashlib.sha256(
                            (root / name).read_bytes()
                        ).hexdigest(),
                    }
                    for name in (
                        "model.safetensors",
                        "config.json",
                        "preprocessor_config.json",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )


def test_missing_or_modified_weights_fail_before_loading_any_library(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        water.importlib,
        "import_module",
        lambda name: pytest.fail(
            "No runtime import expected for missing/changed artifacts"
        ),
    )
    missing = SemanticWaterPredictor(tmp_path / "missing")
    assert missing.is_ready is False
    assert "ausente" in missing.detail
    root = _model_fixture(tmp_path / "model")
    (root / "model.safetensors").write_bytes(b"changed")
    changed = SemanticWaterPredictor(root)
    assert changed.is_ready is False
    assert "hash" in changed.detail


def test_expected_water_classes_must_exist_by_name_not_assumed_number(tmp_path):
    root = _model_fixture(tmp_path / "model")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"]["21"] = "not water"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _refresh_manifest(root)
    predictor = SemanticWaterPredictor(root)
    assert predictor.is_ready is False
    assert "classes de água" in predictor.detail


def test_local_loader_disables_downloads_and_remote_code_and_reads_actual_ids(
    tmp_path, monkeypatch
):
    root = _model_fixture(tmp_path / "model")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"]["21"], config["id2label"]["42"] = (
        config["id2label"]["42"],
        config["id2label"]["21"],
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _refresh_manifest(root)
    calls = []

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(id2label=config["id2label"])

        def to(self, device):
            assert device == "cpu"

        def eval(self):
            pass

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(kwargs)
            assert path == root
            return Model()

    monkeypatch.setattr(water, "select_device", lambda _: "cpu")
    monkeypatch.setattr(
        water.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(
                SegformerImageProcessor=Loader, SegformerForSemanticSegmentation=Loader
            )
            if name == "transformers"
            else torch
        ),
    )
    predictor = SemanticWaterPredictor(root)
    assert predictor.is_ready
    assert predictor.provenance["policy"]["water_labels"]["42"] == "water"
    assert "21" not in predictor.provenance["policy"]["water_labels"]
    assert all(
        call["local_files_only"] and call["trust_remote_code"] is False
        for call in calls
    )
    assert calls[1]["use_safetensors"] is True


def test_policy_and_all_model_artifacts_participate_in_model_version(tmp_path):
    root = _model_fixture(tmp_path / "model")
    normal = SemanticWaterPredictor(root)
    changed_threshold = SemanticWaterPredictor(root, pixel_confidence=0.6)
    assert normal.model_version != changed_threshold.model_version
    assert normal.provenance["policy"]["water_labels"]["21"] == "water"
    assert "preprocessor_config.json" in normal.provenance["file_sha256"]
    original_version = normal.model_version
    processor = root / "preprocessor_config.json"
    processor.write_text(
        '{"do_normalize": true, "do_resize": true, "size": 512}', encoding="utf-8"
    )
    _refresh_manifest(root)
    assert SemanticWaterPredictor(root).model_version != original_version
    detached = normal.provenance
    detached["policy"]["pixel_confidence"] = 0
    assert normal.provenance["policy"]["pixel_confidence"] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"processing_size": 1024},
        {"processing_size": 513},
        {"processing_size": True},
        {"pixel_confidence": float("nan")},
        {"pixel_confidence": -0.1},
        {"min_area_fraction": 1.1},
    ],
)
def test_invalid_processing_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        SemanticWaterPredictor("unused", **kwargs)


def test_component_gate_uses_actual_water_pixels_instead_of_envelope_area():
    labels = np.zeros((10, 20), np.int16)
    labels[1:6, 1] = 21
    labels[5, 1:6] = 21  # 9-pixel L, with 25-pixel bounding rectangle.
    labels[1:3, 10:20] = 21  # 20-pixel rectangle.
    labels[8, 8] = 21
    probability = np.full(labels.shape, 0.8, np.float32)
    probability[8, 8] = 0.49
    detections, mask, regions = _regions_from_labels(
        labels, probability, [21], 0.5, 0.05
    )
    assert len(detections) == 1
    assert detections[0].bounding_box == (10, 1, 20, 3)
    assert detections[0].confidence == pytest.approx(0.8)
    assert detections[0].track_id is None
    assert np.count_nonzero(mask) == 20
    assert regions[0]["mask_pixels"] == 20
    assert not mask[1:6, 1:6].any()


def test_masks_keep_holes_and_other_semantic_labels_are_not_water():
    labels = np.zeros((7, 7), np.int16)
    labels[1:4, 1:4] = 21
    labels[2, 2] = 126  # An animal inside a water-mask hole remains untouched.
    detections, mask, _ = _regions_from_labels(
        labels, np.full(labels.shape, 0.9, np.float32), [21], 0.5, 0
    )
    assert len(detections) == 1
    assert np.count_nonzero(mask) == 8
    assert mask[2, 2] == 0


def test_bounded_logits_preserve_original_size_and_pixel_coordinates(
    tmp_path, monkeypatch
):
    root = _model_fixture(tmp_path / "model")
    predictor = SemanticWaterPredictor(root, device="cpu", processing_size=128)
    predictor._prepare_metadata()
    sizes = []

    def processor(**kwargs):
        sizes.append(kwargs["size"])
        return {"pixel_values": torch.zeros(1, 3, 128, 128)}

    class Model:
        def __call__(self, **kwargs):
            logits = torch.zeros(1, 150, 2, 2)
            logits[:, 0] = 8
            logits[:, 21, 1, 1] = 16
            return SimpleNamespace(logits=logits)

    predictor._model = Model()
    predictor._processor = processor
    predictor._torch = torch
    predictor._device = "cpu"
    calls = []
    original_interpolate = torch.nn.functional.interpolate

    def interpolate(*args, **kwargs):
        calls.append(kwargs["size"])
        return original_interpolate(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", interpolate)
    image = np.zeros((20, 30, 3), np.uint8)
    result = predictor.predict(image)
    assert sizes == [{"height": 128, "width": 128}]
    assert calls == [(128, 128)]  # No 150-channel upsample to original photos.
    assert result.mask.shape == (20, 30)
    assert result.mask.dtype == np.uint8
    assert not image.any()
    assert len(result.detections) == 1
    x1, y1, x2, y2 = result.detections[0].bounding_box
    assert 0 <= x1 < x2 <= 30 and 0 <= y1 < y2 <= 20
    assert result.metadata["processing_size"] == 128
    assert result.metadata["model_version"] == predictor.model_version


def test_drawing_tints_only_mask_pixels_and_keeps_original_untouched():
    image = np.full((4, 5, 3), (10, 20, 30), np.uint8)
    mask = np.zeros((4, 5), np.uint8)
    mask[2, 3] = 255
    marked = draw_mask_on_rgb(image, mask, alpha=0.5, color=(0, 158, 225))
    assert marked[2, 3].tolist() == [5, 89, 127]
    assert marked[0, 0].tolist() == [10, 20, 30]
    assert image[2, 3].tolist() == [10, 20, 30]


def test_cached_maps_can_apply_multiple_policies_without_modifying_neural_outputs():
    probability = np.array([[0.4, 0.6], [0.8, 0.95]], np.float32)
    semantic = np.ones((2, 2), bool)
    before = probability.copy()
    _, ordinary, _ = regions_from_maps(probability, semantic, 0.5, 0)
    _, strict, _ = regions_from_maps(probability, semantic, 0.9, 0)
    assert np.count_nonzero(ordinary) == 3
    assert np.count_nonzero(strict) == 1
    assert np.array_equal(probability, before)
    assert semantic.all()
