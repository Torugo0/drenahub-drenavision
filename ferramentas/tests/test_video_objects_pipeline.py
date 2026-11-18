from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drenavision.core.schemas import Detection
from drenavision.inference import video
from drenavision.inference.runtime import InspectionPredictor
from ferramentas.scripts import process_video as video_cli
from ferramentas.tests.test_video_water_pipeline import _fake_models, _recording


def _objects(monkeypatch, *, fail_at=None, ready=True):
    instances = []

    class Objects:
        is_ready = ready
        detail = "ready" if ready else "local specialist weights unavailable"
        model_version = "grounded-fixture"

        def __init__(self, model_dir, **kwargs):
            self.model_dir, self.arguments = model_dir, kwargs
            self.inputs = []
            self.provenance = {
                "model_version": self.model_version,
                "weights_sha256": "f" * 64,
                "policy": kwargs,
            }
            instances.append(self)

        def predict(self, rgb):
            self.inputs.append(rgb.copy())
            if len(self.inputs) == fail_at:
                raise RuntimeError("specialist current-frame forward failed")
            time.sleep(0.004)
            # The middle frame is genuinely empty. No previous box may leak in.
            detections = (
                [
                    Detection(
                        class_id=0,
                        class_name="obstrucao_visivel",
                        confidence=0.85,
                        bounding_box=(70, 95, 100, 115),
                        track_id=987,
                    ),
                    Detection(
                        class_id=2,
                        class_name="animal_visivel",
                        confidence=0.87,
                        bounding_box=(10, 10, 40, 35),
                        track_id=988,
                    ),
                ]
                if len(self.inputs) != 2
                else []
            )
            return SimpleNamespace(
                detections=detections,
                raw_candidates=[{"frame_forward": len(self.inputs)}],
                inference_ms=4.0,
            )

    monkeypatch.setattr(video, "GroundedObjectPredictor", Objects)
    return instances


@pytest.mark.parametrize("mode", ["augment", "replace", "animal_replace"])
def test_objects_run_every_original_frame_without_propagation_or_fictitious_ids(
    tmp_path, monkeypatch, mode
):
    source, output, checkpoint = _recording(tmp_path)
    master_calls, _, master_frames, Water = _fake_models(monkeypatch)
    objects = _objects(monkeypatch)
    manifest = video.process_video(
        source,
        output,
        checkpoint,
        device="cpu",
        track=True,
        water_model_path=tmp_path / "water",
        water_mode="replace",
        water_confirmation_seconds=0.5,
        animal_confidence=0.6,
        object_model_path=tmp_path / "objects",
        object_mode=mode,
        object_residue_confidence=0.45,
        object_animal_confidence=0.4,
        object_text_confidence=0.2,
        object_shortest_edge=640,
        object_longest_edge=960,
        object_verifier_path=tmp_path / "verifier",
        object_residue_min_margin=0.08,
        object_animal_min_margin=0.02,
    )
    assert len(master_calls) == len(objects[0].inputs) == 3
    assert all(
        call["persist"] and call["tracker"] == "bytetrack.yaml" for call in master_calls
    )
    assert objects[0].arguments == {
        "device": "cpu",
        "residue_confidence": 0.45,
        "animal_confidence": 0.4,
        "text_confidence": 0.2,
        "shortest_edge": 640,
        "longest_edge": 960,
        "verifier_model_path": tmp_path / "verifier",
        "residue_min_margin": 0.08,
        "animal_min_margin": 0.02,
    }
    for rgb, original in zip(objects[0].inputs, master_frames, strict=True):
        assert np.array_equal(rgb, original[:, :, ::-1])
    assert manifest["pipeline"]["objects"] == objects[0].provenance
    assert (
        manifest["pipeline"]["operating_parameters"]["objects"] == objects[0].provenance
    )
    assert manifest["pipeline"]["operating_parameters"]["object_mode"] == mode
    for index, frame in enumerate(manifest["frames"]):
        assert frame["objects"]["raw_candidates"] == [{"frame_forward": index + 1}]
        assert frame["objects"]["processing"] == "forward_on_current_original_frame"
        assert frame["objects"]["inference_ms"] == 4.0
        assert frame["objects"]["forward_wall_ms"] >= 3
        assert frame["inference_ms"] >= 10  # Water and object work both included.
        assert all(item["track_id"] is None for item in frame["objects"]["detections"])
        assert all(item["track_id"] not in {987, 988} for item in frame["detections"])
    middle = manifest["frames"][1]
    assert middle["objects"]["detections"] == []
    if mode in {"replace", "animal_replace"}:
        assert not any(item["class_id"] == 2 for item in middle["detections"])
        assert manifest["counts"]["animal_visivel"] == {
            "count": 1,
            "method": "max_simultaneous_per_frame",
        }
    if mode == "replace":
        assert not any(item["class_id"] == 0 for item in middle["detections"])
        assert (
            manifest["counts"]["obstrucao_visivel"]["method"]
            == "max_simultaneous_per_frame"
        )
    assert manifest["counts"]["agua_acumulada"]["count"] == 1
    assert manifest["water_temporal_policy"]["confirmed"]
    master = SimpleNamespace(
        model_path=checkpoint,
        image_size=640,
        confidence_threshold=0.35,
        iou_threshold=0.45,
    )
    image_pipeline = InspectionPredictor(
        master,
        Water(tmp_path / "water", pixel_confidence=0.5, min_area_fraction=0.001),
        residue_confidence=0.35,
        animal_confidence=0.6,
        water_mode="replace",
        objects=objects[0],
        object_mode=mode,
    )
    assert image_pipeline.model_version == manifest["base_pipeline_version"]


def test_unavailable_configured_objects_fail_before_creating_outputs(
    tmp_path, monkeypatch
):
    source, _, checkpoint = _recording(tmp_path)
    destination = tmp_path / "not-created/output.mp4"
    _objects(monkeypatch, ready=False)
    with pytest.raises(RuntimeError, match="especialista de objetos.*indisponível"):
        video.process_video(
            source, destination, checkpoint, object_model_path=tmp_path / "missing"
        )
    assert not destination.parent.exists()


def test_object_forward_failure_preserves_previous_successful_export(
    tmp_path, monkeypatch
):
    source, output, checkpoint = _recording(tmp_path)
    output.write_bytes(b"previous successful browser video")
    output.with_suffix(".json").write_bytes(b"previous evidence")
    _fake_models(monkeypatch)
    _objects(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="specialist current-frame forward failed"):
        video.process_video(
            source, output, checkpoint, object_model_path=tmp_path / "objects"
        )
    assert output.read_bytes() == b"previous successful browser video"
    assert output.with_suffix(".json").read_bytes() == b"previous evidence"


def test_disabled_objects_preserve_v3_profile_and_configured_policy_changes_version(
    tmp_path, monkeypatch
):
    source, output, checkpoint = _recording(tmp_path)
    _fake_models(monkeypatch)
    objects = _objects(monkeypatch)
    legacy = video.process_video(source, output, checkpoint, device="cpu")
    still_disabled = video.process_video(
        source,
        tmp_path / "disabled.mp4",
        checkpoint,
        device="cpu",
        object_model_path=None,
        object_mode="replace",
        object_animal_confidence=0.9,
        object_verifier_path=tmp_path / "unavailable-but-disabled",
        object_residue_min_margin=0.1,
        object_animal_min_margin=0.1,
    )
    assert legacy["model_version"] == still_disabled["model_version"]
    assert "objects" not in legacy["pipeline"]["operating_parameters"]
    assert not objects
    configured = video.process_video(
        source,
        tmp_path / "configured.mp4",
        checkpoint,
        device="cpu",
        object_model_path=tmp_path / "objects",
    )
    changed = video.process_video(
        source,
        tmp_path / "changed.mp4",
        checkpoint,
        device="cpu",
        object_model_path=tmp_path / "objects",
        object_animal_confidence=0.45,
    )
    assert (
        len(
            {
                legacy["model_version"],
                configured["model_version"],
                changed["model_version"],
            }
        )
        == 3
    )


def test_cli_forwards_every_object_control_and_explicit_disable(monkeypatch, capsys):
    received = {}

    def process(*args, **kwargs):
        received.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(video_cli, "process_video", process)
    flags = [
        "--input",
        "source.mp4",
        "--output",
        "output.mp4",
        "--object-model",
        "objects",
        "--object-residue-confidence",
        ".4",
        "--object-animal-confidence",
        ".45",
        "--object-text-confidence",
        ".25",
        "--object-mode",
        "replace",
        "--object-shortest-edge",
        "640",
        "--object-longest-edge",
        "960",
        "--object-verifier",
        "verifier",
        "--object-residue-min-margin",
        ".08",
        "--object-animal-min-margin",
        ".02",
    ]
    assert video_cli.main(flags) == 0
    assert received["object_model_path"] == Path("objects")
    assert received["object_residue_confidence"] == 0.4
    assert received["object_animal_confidence"] == 0.45
    assert received["object_text_confidence"] == 0.25
    assert received["object_mode"] == "replace"
    assert received["object_shortest_edge"] == 640
    assert received["object_longest_edge"] == 960
    assert received["object_verifier_path"] == Path("verifier")
    assert received["object_residue_min_margin"] == 0.08
    assert received["object_animal_min_margin"] == 0.02
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert video_cli.main([*flags, "--no-objects"]) == 0
    assert received["object_model_path"] is None
    assert received["object_verifier_path"] is None


def test_cli_object_defaults_follow_current_settings(tmp_path, monkeypatch):
    for name, value in {
        "DRENAVISION_OBJECT_MODEL_PATH": str(tmp_path / "objects"),
        "DRENAVISION_OBJECT_RESIDUE_CONFIDENCE": ".44",
        "DRENAVISION_OBJECT_ANIMAL_CONFIDENCE": ".46",
        "DRENAVISION_OBJECT_TEXT_CONFIDENCE": ".23",
        "DRENAVISION_OBJECT_MODE": "augment",
        "DRENAVISION_OBJECT_SHORTEST_EDGE": "640",
        "DRENAVISION_OBJECT_LONGEST_EDGE": "960",
        "DRENAVISION_OBJECT_VERIFIER_PATH": str(tmp_path / "verifier"),
        "DRENAVISION_OBJECT_RESIDUE_MIN_MARGIN": ".08",
        "DRENAVISION_OBJECT_ANIMAL_MIN_MARGIN": ".02",
    }.items():
        monkeypatch.setenv(name, value)
    args = video_cli.build_parser().parse_args(
        ["--input", "source.mp4", "--output", "result.mp4"]
    )
    assert args.object_model == tmp_path / "objects"
    assert args.object_residue_confidence == 0.44
    assert args.object_animal_confidence == 0.46
    assert args.object_text_confidence == 0.23
    assert args.object_mode == "augment"
    assert args.object_shortest_edge == 640
    assert args.object_longest_edge == 960
    assert args.object_verifier == tmp_path / "verifier"
    assert args.object_residue_min_margin == 0.08
    assert args.object_animal_min_margin == 0.02


def _shared_pipeline(tmp_path, monkeypatch):
    from drenavision.inference.contracts import PredictionResult

    source, output, checkpoint = _recording(tmp_path)
    master_calls, water_args, _, Water = _fake_models(monkeypatch)
    instances = _objects(monkeypatch)
    objects = video.GroundedObjectPredictor(
        tmp_path / "objects",
        device="cpu",
        residue_confidence=0.35,
        animal_confidence=0.35,
        text_confidence=0.2,
        shortest_edge=800,
        longest_edge=1333,
        verifier_model_path=None,
        residue_min_margin=0.0,
        animal_min_margin=0.0,
    )
    water = Water(tmp_path / "water", pixel_confidence=0.5, min_area_fraction=0.001)
    master = SimpleNamespace(
        model_path=checkpoint,
        image_size=640,
        confidence_threshold=0.35,
        iou_threshold=0.45,
        is_ready=True,
        predict=lambda rgb: PredictionResult(
            [], rgb.copy(), 1.0, "master", rgb.shape[1], rgb.shape[0]
        ),
    )
    shared = InspectionPredictor(master, water, animal_confidence=0.6, objects=objects)
    return (
        source,
        output,
        checkpoint,
        shared,
        objects,
        water,
        instances,
        water_args,
        master_calls,
    )


def test_shared_image_video_models_are_borrowed_once_with_real_forward_per_frame(
    tmp_path, monkeypatch
):
    (
        source,
        output,
        checkpoint,
        shared,
        objects,
        water,
        instances,
        water_args,
        master_calls,
    ) = _shared_pipeline(tmp_path, monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Large model constructors must not run again for API video")

    monkeypatch.setattr(video, "GroundedObjectPredictor", forbidden)
    monkeypatch.setattr(video, "SemanticWaterPredictor", forbidden)
    shared.predict(np.full((120, 160, 3), 130, np.uint8))
    manifest = video.process_video(
        source,
        output,
        checkpoint,
        device="cpu",
        track=True,
        animal_confidence=0.6,
        water_model_path=tmp_path / "water",
        object_model_path=tmp_path / "objects",
        shared_predictor=shared,
    )
    assert len(instances) == len(water_args) == 1
    assert len(objects.inputs) == water.calls == 4  # One photo, then all3 video frames.
    assert len(master_calls) == 3
    assert manifest["base_pipeline_version"] == shared.model_version
    assert "borrowed API" in manifest["pipeline"]["model_ownership"]
    assert (
        manifest["frames"][0]["objects"]["detections"] == []
    )  # No photo-result propagation.
    assert not any(
        item["class_id"] == 2 for item in manifest["frames"][0]["detections"]
    )


def test_shared_profile_mismatch_fails_without_model_reloading_or_new_output(
    tmp_path, monkeypatch
):
    source, _, checkpoint, shared, objects, water, _, _, _ = _shared_pipeline(
        tmp_path, monkeypatch
    )
    destination = tmp_path / "not-created/out.mp4"
    with pytest.raises(RuntimeError, match="parâmetros.*divergem"):
        video.process_video(
            source, destination, checkpoint, confidence=0.5, shared_predictor=shared
        )
    assert not destination.parent.exists()
    assert not objects.inputs and water.calls == 0
