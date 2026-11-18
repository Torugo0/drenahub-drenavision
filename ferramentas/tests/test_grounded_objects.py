from __future__ import annotations

import contextlib
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from drenavision.inference import object_artifacts, objects
from drenavision.inference.errors import InferenceError
from ferramentas.scripts import prepare_object_model as preparation


def model_fixture(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    payloads = {name: f"fixture-{name}".encode() for name in object_artifacts.ARTIFACTS}
    payloads["config.json"] = b'{"model_type":"grounding-dino"}'
    pins = {
        name: (hashlib.sha256(content).hexdigest(), len(content))
        for name, content in payloads.items()
    }
    monkeypatch.setattr(object_artifacts, "ARTIFACTS", pins)
    monkeypatch.setattr(preparation, "ARTIFACTS", pins)
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    (root / "download_manifest.json").write_bytes(preparation.manifest_bytes())
    return root


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("a rat", 2),
        ("mouse", 2),
        ("mice", 2),
        ("a rodent", 2),
        ("plastic waste", 0),
        ("discarded bottle", 0),
        ("discarded plastic bag", 0),
        ("discarded cup", 0),
        ("discarded can", 0),
        ("a rat discarded bottle", None),
        ("water", None),
        ("pirate", None),
    ],
)
def test_phrase_family_map_avoids_substrings_and_cross_family(phrase, expected):
    assert objects.classify_phrase(phrase) == expected


def test_postprocessing_preserves_raw_and_clips_without_inventing_regions():
    detections, raw = objects.postprocess_candidates(
        [0.8, 0.7, 0.9, 0.2, 0.99, 0.99],
        [
            [-5, 2, 35, 70],
            [1, 2, 35, 65],
            [0, 0, 25, 25],
            [50, 20, 70, 30],
            [0, 0, 10, 10],
            [150, 10, 200, 20],
        ],
        [
            "a rat",
            "a mouse",
            "discarded bottle",
            "plastic waste",
            "a rat discarded cup",
            "discarded can",
        ],
        image_width=100,
        image_height=60,
        residue_confidence=0.3,
        animal_confidence=0.5,
    )
    assert len(raw) == 6
    assert raw[0]["bounding_box"] == [-5, 2, 35, 70]
    assert raw[0]["clipped_bounding_box"] == [0, 2, 35, 60]
    assert raw[4]["class_id"] is None
    assert [(d.class_id, d.confidence) for d in detections] == [(0, 0.9), (2, 0.8)]
    assert detections[1].bounding_box == (0, 2, 35, 60)
    assert all(d.track_id is None for d in detections)


def test_nms_does_not_merge_other_classes_or_low_overlap_objects():
    detections, _ = objects.postprocess_candidates(
        [0.8, 0.7, 0.6],
        [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]],
        ["a rat", "litter", "a rodent"],
        image_width=50,
        image_height=50,
    )
    assert [d.class_id for d in detections] == [2, 0, 2]


def test_invalid_or_missing_model_fails_before_any_ml_import(tmp_path, monkeypatch):
    monkeypatch.setattr(
        objects.importlib,
        "import_module",
        lambda _: pytest.fail("ML import before verified assets"),
    )
    missing = objects.GroundedObjectPredictor(tmp_path / "missing")
    assert missing.is_ready is False
    assert "ausente" in missing.detail
    root = model_fixture(tmp_path, monkeypatch)
    (root / "model.safetensors").write_bytes(b"tampered")
    changed = objects.GroundedObjectPredictor(root)
    assert changed.is_ready is False
    assert "hash" in changed.detail


def test_model_and_manifest_cannot_be_modified_together_to_bypass_pin(
    tmp_path, monkeypatch
):
    root = model_fixture(tmp_path, monkeypatch)
    (root / "model.safetensors").write_bytes(b"different-safe-weights")
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == "model.safetensors":
            entry["sha256"] = hashlib.sha256(b"different-safe-weights").hexdigest()
            entry["bytes"] = len(b"different-safe-weights")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selo"):
        object_artifacts.verify_local_model(root)


def test_extra_pickle_is_rejected_even_when_required_hashes_match(
    tmp_path, monkeypatch
):
    root = model_fixture(tmp_path, monkeypatch)
    (root / "pytorch_model.bin").write_bytes(b"not-allowed")
    with pytest.raises(ValueError, match="não permitidos"):
        object_artifacts.verify_local_model(root)


def mock_runtime(monkeypatch):
    calls = []

    class Tensor:
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __call__(self, **kwargs):
            calls.append(("processor_call", kwargs))
            return {"input_ids": Tensor(), "pixel_values": Tensor()}

        def post_process_grounded_object_detection(self, output, **kwargs):
            calls.append(("postprocess", kwargs))
            return [
                {
                    "scores": np.array([0.8, 0.7]),
                    "boxes": np.array([[1, 2, 8, 9], [5, 5, 19, 14]]),
                    "text_labels": ["a rat", "discarded bag"],
                }
            ]

    class Model:
        config = SimpleNamespace(
            model_type="grounding-dino", disable_custom_kernels=True
        )

        def to(self, device):
            assert device == "cpu"

        def eval(self):
            pass

        def __call__(self, **kwargs):
            return SimpleNamespace()

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("processor_load", kwargs))
            return Processor()

    class ModelLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("model_load", kwargs))
            return Model()

    fake_transformers = SimpleNamespace(
        AutoProcessor=ProcessorLoader, AutoModelForZeroShotObjectDetection=ModelLoader
    )
    monkeypatch.setattr(objects, "select_device", lambda _: "cpu")
    monkeypatch.setattr(
        objects.importlib,
        "import_module",
        lambda name: (
            fake_transformers
            if name == "transformers"
            else SimpleNamespace(inference_mode=contextlib.nullcontext)
        ),
    )
    return calls


def test_runtime_uses_only_verified_local_safetensors_and_fixed_prompt(
    tmp_path, monkeypatch
):
    root = model_fixture(tmp_path, monkeypatch)
    calls = mock_runtime(monkeypatch)
    predictor = objects.GroundedObjectPredictor(
        root, residue_confidence=0.4, animal_confidence=0.5
    )
    rgb = np.arange(15 * 20 * 3, dtype=np.uint8).reshape(15, 20, 3)
    before = rgb.copy()
    result = predictor.predict(rgb)
    assert [d.class_id for d in result.detections] == [2, 0]
    assert len(result.raw_candidates) == 2
    assert result.inference_ms >= 0
    np.testing.assert_array_equal(rgb, before)
    loads = [kwargs for name, kwargs in calls if name.endswith("_load")]
    assert all(
        k["local_files_only"] is True and k["trust_remote_code"] is False for k in loads
    )
    model_kwargs = next(k for name, k in calls if name == "model_load")
    assert model_kwargs["use_safetensors"] is True
    assert model_kwargs["disable_custom_kernels"] is True
    processor_kwargs = next(k for name, k in calls if name == "processor_call")
    assert processor_kwargs["text"] == objects.OBJECT_PROMPT
    assert processor_kwargs["size"] == {"shortest_edge": 800, "longest_edge": 1333}
    postprocess_kwargs = next(k for name, k in calls if name == "postprocess")
    assert postprocess_kwargs["threshold"] == 0.4
    assert "box_threshold" not in postprocess_kwargs
    assert postprocess_kwargs["target_sizes"] == [(15, 20)]


def test_lazy_loader_only_initializes_once_for_concurrent_readiness(
    tmp_path, monkeypatch
):
    root = model_fixture(tmp_path, monkeypatch)
    calls = mock_runtime(monkeypatch)
    predictor = objects.GroundedObjectPredictor(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda _: predictor.is_ready, range(8)))
    assert sum(name == "model_load" for name, _ in calls) == 1
    assert sum(name == "processor_load" for name, _ in calls) == 1


def test_policy_is_versioned_and_provenance_is_detached(tmp_path, monkeypatch):
    root = model_fixture(tmp_path, monkeypatch)
    predictor = objects.GroundedObjectPredictor(root)
    changed = objects.GroundedObjectPredictor(root, text_confidence=0.2)
    assert predictor.model_version != changed.model_version
    assert predictor.provenance["policy"]["prompt"] == objects.OBJECT_PROMPT
    detached = predictor.provenance
    detached["policy"]["prompt"] = "different"
    assert predictor.provenance["policy"]["prompt"] == objects.OBJECT_PROMPT


def test_bad_image_and_non_finite_neural_outputs_fail_explicitly(tmp_path, monkeypatch):
    predictor = objects.GroundedObjectPredictor(tmp_path / "missing")
    with pytest.raises(ValueError, match="RGB uint8"):
        predictor.predict(np.zeros((10, 20, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="confiança inválida"):
        objects.postprocess_candidates(
            [float("nan")], [[0, 0, 1, 1]], ["a rat"], image_width=10, image_height=10
        )
    root = model_fixture(tmp_path, monkeypatch)
    mock_runtime(monkeypatch)
    predictor = objects.GroundedObjectPredictor(root)
    assert predictor.is_ready
    monkeypatch.setattr(
        predictor._processor,
        "post_process_grounded_object_detection",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
    )
    with pytest.raises(InferenceError, match="falha na detecção"):
        predictor.predict(np.zeros((10, 20, 3), dtype=np.uint8))
