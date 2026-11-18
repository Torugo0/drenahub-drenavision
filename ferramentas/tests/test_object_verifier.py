from __future__ import annotations

import copy
import hashlib
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drenavision.inference import object_verifier as verifier
from drenavision.inference import verifier_artifacts


def model_fixture(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    payloads = {
        name: f"fixture-{name}".encode() for name in verifier_artifacts.ARTIFACTS
    }
    payloads["config.json"] = b'{"model_type":"siglip"}'
    monkeypatch.setattr(
        verifier_artifacts,
        "ARTIFACTS",
        {
            name: (hashlib.sha256(value).hexdigest(), len(value))
            for name, value in payloads.items()
        },
    )
    for name, value in payloads.items():
        (root / name).write_bytes(value)
    (root / "download_manifest.json").write_bytes(verifier_artifacts.manifest_bytes())
    return root


def fake_runtime(monkeypatch):
    calls, full_models = [], []

    class Processor:
        def __call__(self, **kwargs):
            if "text" in kwargs:
                calls.append(("text_input", kwargs))
                return {
                    "input_ids": torch.ones(
                        (len(kwargs["text"]), 64), dtype=torch.int64
                    )
                }
            calls.append(("image_input", kwargs))
            features = np.stack(
                [image.mean(axis=(0, 1)) / 255 for image in kwargs["images"]]
            )
            return {"pixel_values": torch.as_tensor(features, dtype=torch.float32)}

    class Vision:
        def to(self, **kwargs):
            calls.append(("vision_to", kwargs))
            assert all(reference() is None for reference in full_models)
            return self

        def eval(self):
            pass

        def __call__(self, **kwargs):
            return SimpleNamespace(pooler_output=kwargs["pixel_values"] + 1)

    class FullModel:
        config = SimpleNamespace(model_type="siglip")

        def __init__(self):
            self.vision_model = Vision()

        def to(self, **kwargs):
            calls.append(("full_to", kwargs))
            assert kwargs["device"] == "cpu"
            assert kwargs["dtype"] == torch.float32

        def eval(self):
            pass

        def get_text_features(self, **kwargs):
            calls.append(("encode_text", kwargs))
            assert kwargs["input_ids"].device.type == "cpu"
            count = kwargs["input_ids"].shape[0]
            return torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) + 1

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("processor_load", kwargs))
            return Processor()

    class ModelLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("model_load", kwargs))
            value = FullModel()
            full_models.append(weakref.ref(value))
            return value

    transformers = SimpleNamespace(AutoProcessor=ProcessorLoader, AutoModel=ModelLoader)
    monkeypatch.setattr(
        verifier.importlib,
        "import_module",
        lambda name: transformers if name == "transformers" else torch,
    )
    monkeypatch.setattr(verifier, "select_device", lambda _: "cpu")
    return calls


def test_crop_padding_only_affects_classifier_input_and_prefers_clipped_box():
    candidate = {
        "bounding_box": [-50, -50, 70, 70],
        "clipped_bounding_box": [10, 20, 40, 50],
    }
    original = copy.deepcopy(candidate)
    assert verifier.crop_bounds(candidate, 100, 90, 0.1) == (7, 17, 43, 53)
    assert candidate == original
    assert verifier.crop_bounds({"bounding_box": [0, 0, 100, 90]}, 100, 90, 0.5) == (
        0,
        0,
        100,
        90,
    )
    with pytest.raises(ValueError, match="área positiva"):
        verifier.crop_bounds({"bounding_box": [110, 0, 120, 20]}, 100, 90, 0.1)


def test_similarity_margin_is_positive_max_minus_negative_max_without_softmax():
    similarities = np.full(len(verifier.FIXED_TEXTS), -0.2, dtype=np.float32)
    similarities[verifier.FIXED_TEXTS.index("photograph of a mouse")] = 0.6
    similarities[verifier.FIXED_TEXTS.index("photograph of a rat")] = 0.4
    similarities[verifier.FIXED_TEXTS.index("photograph of a leaf")] = 0.58
    result = verifier.verification_from_similarities(similarities, 2)
    assert result["best_positive"] == "photograph of a mouse"
    assert result["best_negative"] == "photograph of a leaf"
    assert result["margin"] == pytest.approx(0.02, abs=1e-6)
    assert result["positive_similarity"] == pytest.approx(0.6)
    assert result["negative_similarity"] == pytest.approx(0.58)
    assert not any(
        key in result for key in ("probability", "risk", "accepted", "threshold")
    )


def test_missing_or_modified_assets_fail_before_ml_import(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verifier.importlib,
        "import_module",
        lambda _: pytest.fail("ML import must follow the seal check"),
    )
    missing = verifier.CropObjectVerifier(tmp_path / "missing")
    assert missing.is_ready is False
    assert "ausente" in missing.detail
    root = model_fixture(tmp_path, monkeypatch)
    (root / "model.safetensors").write_bytes(b"changed")
    assert verifier.CropObjectVerifier(root).is_ready is False


def test_verifier_keeps_candidate_order_original_boxes_and_all_fields(
    tmp_path, monkeypatch
):
    root = model_fixture(tmp_path, monkeypatch)
    calls = fake_runtime(monkeypatch)
    scorer = verifier.CropObjectVerifier(root, batch_size=1)
    image = np.arange(20 * 25 * 3, dtype=np.uint8).reshape(20, 25, 3)
    before = image.copy()
    candidates = [
        {
            "class_id": 2,
            "confidence": 0.4,
            "bounding_box": [2, 3, 10, 15],
            "source": {"name": "unchanged"},
        },
        {
            "class_id": 0,
            "confidence": 0.6,
            "bounding_box": [10, 4, 23, 18],
            "phrase": "litter",
        },
    ]
    original = copy.deepcopy(candidates)
    scored = scorer.score(image, candidates)
    assert candidates == original
    for expected, actual in zip(original, scored, strict=True):
        verification = actual.pop("verification")
        assert actual == expected
        assert set(verification) == {
            "margin",
            "positive_similarity",
            "negative_similarity",
            "best_positive",
            "best_negative",
        }
        assert np.isfinite(verification["margin"])
    np.testing.assert_array_equal(image, before)
    assert sum(name == "image_input" for name, _ in calls) == 2


def test_text_tower_cpu_once_then_released_only_vision_is_retained(
    tmp_path, monkeypatch
):
    root = model_fixture(tmp_path, monkeypatch)
    calls = fake_runtime(monkeypatch)
    scorer = verifier.CropObjectVerifier(root)
    candidates = [{"class_id": 2, "bounding_box": [1, 1, 9, 9]}]
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    scorer.score(image, candidates)
    scorer.score(image, candidates)
    assert sum(name == "model_load" for name, _ in calls) == 1
    assert sum(name == "encode_text" for name, _ in calls) == 1
    assert scorer._text_features.dtype == torch.float32
    assert scorer._text_features.device.type == "cpu"
    torch.testing.assert_close(
        scorer._text_features.norm(dim=1), torch.ones(len(verifier.FIXED_TEXTS))
    )
    assert not hasattr(scorer, "_model")
    loads = [kwargs for name, kwargs in calls if name.endswith("_load")]
    assert all(
        kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        for kwargs in loads
    )
    model_load = next(kwargs for name, kwargs in calls if name == "model_load")
    assert model_load["use_safetensors"] is True
    assert model_load["dtype"] == torch.float32
    text_input = next(kwargs for name, kwargs in calls if name == "text_input")
    assert text_input["max_length"] == 64
    assert text_input["padding"] == "max_length"
    assert text_input["text"] == list(verifier.FIXED_TEXTS)
    assert all(text == text.lower() for text in text_input["text"])


def test_crop_policy_changes_version_and_provenance_is_detached(tmp_path, monkeypatch):
    root = model_fixture(tmp_path, monkeypatch)
    first = verifier.CropObjectVerifier(root)
    second = verifier.CropObjectVerifier(root, crop_padding=0.2)
    assert first.model_version != second.model_version
    detached = first.provenance
    detached["policy"]["family_prompts"][2]["positive"] = ("modified",)
    assert (
        first.provenance["policy"]["family_prompts"][2]["positive"][0]
        == "photograph of a rat"
    )


def test_empty_candidates_skip_model_loading_and_invalid_input_fails(
    tmp_path, monkeypatch
):
    scorer = verifier.CropObjectVerifier(tmp_path / "missing")
    monkeypatch.setattr(
        verifier.importlib,
        "import_module",
        lambda _: pytest.fail("Empty proposals must not load a model"),
    )
    assert scorer.score(np.zeros((10, 10, 3), dtype=np.uint8), []) == []
    with pytest.raises(ValueError, match="RGB uint8"):
        scorer.score(np.zeros((10, 10, 3), dtype=np.float32), [])
    with pytest.raises(ValueError, match="class_id"):
        scorer.score(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [{"class_id": 1, "bounding_box": [0, 0, 2, 2]}],
        )


def test_retained_vision_pooler_matches_official_siglip_get_image_features_cpu():
    """Architecture parity using a tiny random CPU fixture, never downloaded weights."""
    from transformers import (
        SiglipConfig,
        SiglipModel,
        SiglipTextConfig,
        SiglipVisionConfig,
    )

    text = SiglipTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
    )
    model = (
        SiglipModel(
            SiglipConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
        )
        .to("cpu")
        .eval()
    )
    pixels = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.inference_mode():
        full_features = model.get_image_features(pixel_values=pixels)
        retained_features = model.vision_model(pixel_values=pixels).pooler_output
    torch.testing.assert_close(full_features, retained_features, rtol=0, atol=0)
