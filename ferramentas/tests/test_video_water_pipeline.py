from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from drenavision.core.constants import CLASS_NAMES
from drenavision.core.schemas import Detection
from drenavision.inference import video
from drenavision.inference.runtime import InspectionPredictor
from ferramentas.scripts import process_video as video_cli


def _recording(
    tmp_path: Path, *, frames: int = 3, fps: float = 5
) -> tuple[Path, Path, Path]:
    source, output, checkpoint = (
        tmp_path / "source.mp4",
        tmp_path / "output.mp4",
        tmp_path / "master.pt",
    )
    checkpoint.write_bytes(b"verified master checkpoint")
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120)
    )
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((120, 160, 3), 40, np.uint8))
    writer.release()
    return source, output, checkpoint


class _Boxes:
    xyxy = [
        [15, 75, 30, 110],
        [50, 40, 120, 95],
        [130, 10, 155, 30],
        [125, 70, 150, 110],
    ]
    cls = [0, 1, 2, 2]
    conf = [0.8, 0.9, 0.5, 0.8]

    def __init__(self, index: int) -> None:
        self.id = [7, 100 + index, 9, 10]

    def __len__(self) -> int:
        return 4


def _fake_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: int | None = None,
    water_frames: set[int] | None = None,
):
    calls, water_arguments, input_frames = [], [], []

    class Master:
        names = CLASS_NAMES

        def track(self, **kwargs):
            calls.append(kwargs)
            input_frames.append(kwargs["source"].copy())
            return [SimpleNamespace(boxes=_Boxes(len(calls)), speed={"inference": 2.0})]

        predict = track

    class Water:
        is_ready = True
        detail = "ready"

        def __init__(self, model_dir, **kwargs):
            water_arguments.append((model_dir, kwargs))
            self.calls = 0
            self.provenance = {
                "model_version": "water-abc",
                "file_sha256": {"weights": "a" * 64},
                "policy": {
                    "pixel_confidence": kwargs["pixel_confidence"],
                    "min_area_fraction": kwargs["min_area_fraction"],
                    "processing_size": 512,
                },
            }

        def predict(self, rgb):
            self.calls += 1
            if self.calls == fail_at:
                raise RuntimeError("semantic frame failed")
            time.sleep(0.008)
            mask = np.zeros(rgb.shape[:2], np.uint8)
            mask[45:90, 55:115] = 255
            mask[60:65, 70:75] = 0  # A real mask hole must stay untouched.
            visible = water_frames is None or self.calls in water_frames
            if not visible:
                mask[:] = 0
            return SimpleNamespace(
                detections=[
                    Detection(
                        class_id=1,
                        class_name="agua_acumulada",
                        confidence=0.85,
                        bounding_box=(55, 45, 115, 90),
                    )
                ]
                if visible
                else [],
                mask=mask,
                inference_ms=8.0,
                metadata={
                    "model_version": "water-abc",
                    "mask_pixels": int(np.count_nonzero(mask)),
                },
            )

    real_import = video.importlib.import_module

    def imports(name):
        if name == "ultralytics":
            return SimpleNamespace(YOLO=lambda _: Master())
        return real_import(name)

    def encode(source, destination):
        shutil.copyfile(source, destination)
        return {"codec": "fixture-mp4v"}

    monkeypatch.setattr(video.importlib, "import_module", imports)
    monkeypatch.setattr(video, "select_device", lambda _: "cpu")
    monkeypatch.setattr(video, "SemanticWaterPredictor", Water)
    monkeypatch.setattr(video, "prepare_browser_video", encode)
    return calls, water_arguments, input_frames, Water


@pytest.mark.parametrize("mode", ["augment", "replace"])
def test_video_water_uses_original_pixels_shared_profile_and_region_peak_counts(
    tmp_path, monkeypatch, mode
):
    source, output, checkpoint = _recording(tmp_path)
    calls, water_args, originals, Water = _fake_models(monkeypatch)
    displays = []
    original_annotation = video.annotate_video_frame

    def annotation(frame, detections):
        displays.append(frame.copy())
        return original_annotation(frame, detections)

    monkeypatch.setattr(video, "annotate_video_frame", annotation)
    manifest = video.process_video(
        source,
        output,
        checkpoint,
        device="cpu",
        track=True,
        water_model_path=tmp_path / "water",
        water_pixel_confidence=0.55,
        water_min_area_fraction=0.002,
        water_mode=mode,
        animal_confidence=0.6,
    )
    assert len(calls) == 3
    assert all(
        call["classes"] == ([0, 2] if mode == "replace" else [0, 1, 2])
        for call in calls
    )
    assert water_args[0][1] == {
        "device": "cpu",
        "pixel_confidence": 0.55,
        "min_area_fraction": 0.002,
        "processing_size": 512,
    }
    assert manifest["counts"]["agua_acumulada"] == {
        "count": 1,
        "method": "max_simultaneous_per_frame",
    }
    assert manifest["counts"]["animal_visivel"]["count"] == 1
    assert manifest["counts"]["obstrucao_visivel"]["count"] == 1
    for frame in manifest["frames"]:
        water = next(item for item in frame["detections"] if item["class_id"] == 1)
        assert water["track_id"] is None
        assert water["bounding_box"] == (
            [55.0, 45.0, 115.0, 90.0]
            if mode == "replace"
            else [50.0, 40.0, 120.0, 95.0]
        )
        assert all(item["track_id"] != 9 for item in frame["detections"])
        assert (
            frame["inference_ms"] >= 7
        )  # Includes semantic work, not the master's 2ms statistic.
        assert frame["water"]["model_version"] == "water-abc"
    for original, display in zip(originals, displays, strict=True):
        assert np.array_equal(
            original[62, 72], display[62, 72]
        )  # Mask hole stays clear.
        assert not np.array_equal(original[80, 100], display[80, 100])
        assert np.array_equal(
            original[20, 140], display[20, 140]
        )  # No rejected-animal pixels leak.
    master = SimpleNamespace(
        model_path=checkpoint,
        image_size=640,
        confidence_threshold=0.35,
        iou_threshold=0.45,
    )
    image_pipeline = InspectionPredictor(
        master,
        Water(tmp_path / "water", pixel_confidence=0.55, min_area_fraction=0.002),
        residue_confidence=0.35,
        animal_confidence=0.6,
        water_mode=mode,
    )
    assert manifest["model_version"] == image_pipeline.model_version
    assert (
        manifest["pipeline"]["master_model_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert (
        json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))[
            "model_version"
        ]
        == manifest["model_version"]
    )


def test_missing_configured_water_fails_before_outputs_are_created(
    tmp_path, monkeypatch
):
    source, _, checkpoint = _recording(tmp_path)
    destination = tmp_path / "new" / "output.mp4"

    class Missing:
        is_ready = False
        detail = "weights missing or invalid"

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(video, "SemanticWaterPredictor", Missing)
    with pytest.raises(RuntimeError, match="segmentador.*indisponível"):
        video.process_video(
            source, destination, checkpoint, water_model_path=tmp_path / "missing"
        )
    assert not destination.parent.exists()


def test_semantic_failure_preserves_previous_output_and_cleans_temporary_video(
    tmp_path, monkeypatch
):
    source, output, checkpoint = _recording(tmp_path)
    output.write_bytes(b"previous successful export")
    output.with_suffix(".json").write_bytes(b"previous successful evidence")
    _fake_models(monkeypatch, fail_at=2)
    scratch_paths = []
    temporary_directory = video.tempfile.TemporaryDirectory

    def scratch(**kwargs):
        handle = temporary_directory(**kwargs)
        scratch_paths.append(Path(handle.name))
        return handle

    monkeypatch.setattr(video.tempfile, "TemporaryDirectory", scratch)
    with pytest.raises(RuntimeError, match="semantic frame failed"):
        video.process_video(
            source, output, checkpoint, water_model_path=tmp_path / "water"
        )
    assert output.read_bytes() == b"previous successful export"
    assert output.with_suffix(".json").read_bytes() == b"previous successful evidence"
    assert scratch_paths and all(not path.exists() for path in scratch_paths)


def test_video_cli_forwards_all_pipeline_controls(monkeypatch, capsys):
    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(video_cli, "process_video", run)
    assert (
        video_cli.main(
            [
                "--input",
                "source.mp4",
                "--output",
                "result.mp4",
                "--water-model",
                "water-dir",
                "--water-pixel-confidence",
                ".55",
                "--water-min-area-fraction",
                ".002",
                "--water-mode",
                "replace",
                "--residue-conf",
                ".35",
                "--animal-conf",
                ".6",
                "--water-confirmation-seconds",
                ".75",
            ]
        )
        == 0
    )
    assert received["water_model_path"] == Path("water-dir")
    assert received["water_pixel_confidence"] == 0.55
    assert received["water_min_area_fraction"] == 0.002
    assert received["water_mode"] == "replace"
    assert received["animal_confidence"] == 0.6
    assert received["residue_confidence"] == 0.35
    assert received["water_confirmation_seconds"] == 0.75
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_video_cli_defaults_follow_application_settings_and_allow_explicit_water_disable(
    tmp_path, monkeypatch, capsys
):
    from drenavision.core.config import Settings

    for name, value in {
        "DRENAVISION_WATER_MODEL_PATH": str(tmp_path / "water"),
        "DRENAVISION_WATER_PIXEL_CONFIDENCE": ".96",
        "DRENAVISION_WATER_MIN_AREA_FRACTION": ".04",
        "DRENAVISION_ANIMAL_CONFIDENCE_THRESHOLD": ".65",
        "DRENAVISION_IMAGE_SIZE": "960",
        "DRENAVISION_VIDEO_WATER_CONFIRMATION_SECONDS": ".75",
    }.items():
        monkeypatch.setenv(name, value)
    settings = Settings()
    args = video_cli.build_parser().parse_args(
        ["--input", "source.mp4", "--output", "result.mp4"]
    )
    assert args.model == settings.model_path
    assert args.water_model == settings.water_model_path
    assert args.water_pixel_confidence == settings.water_pixel_confidence
    assert args.water_min_area_fraction == settings.water_min_area_fraction
    assert args.animal_conf == settings.animal_confidence_threshold
    assert args.imgsz == settings.image_size
    assert args.water_confirmation_seconds == settings.video_water_confirmation_seconds
    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(video_cli, "process_video", run)
    assert (
        video_cli.main(
            ["--input", "source.mp4", "--output", "result.mp4", "--no-water"]
        )
        == 0
    )
    assert received["water_model_path"] is None
    assert received["animal_confidence"] == 0.65
    assert received["water_confirmation_seconds"] == 0.75
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


@pytest.mark.parametrize(
    "fps,frame_count,water_frames,required,suppressed",
    [
        (
            5,
            5,
            {1, 2, 4},
            3,
            3,
        ),  # Three flashes do not form the required three-frame run.
        (4, 4, {1, 2}, 2, 0),  # Half a second at 4 FPS is two consecutive frames.
        (10, 4, {1, 2}, 5, 2),  # The same two frames are too brief at 10 FPS.
    ],
)
def test_video_water_confirmation_is_fps_normalized_and_rerenders_without_ghosts(
    tmp_path, monkeypatch, fps, frame_count, water_frames, required, suppressed
):
    from drenavision.inference.media_io import read_image_bgr

    source, output, checkpoint = _recording(tmp_path, frames=frame_count, fps=fps)
    calls, _, _, _ = _fake_models(monkeypatch, water_frames=water_frames)
    controls = {
        "device": "cpu",
        "track": True,
        "water_model_path": tmp_path / "water",
        "water_mode": "replace",
        "animal_confidence": 0.6,
    }
    raw = video.process_video(
        source,
        tmp_path / "raw.mp4",
        checkpoint,
        water_confirmation_seconds=0,
        **controls,
    )
    confirmed = video.process_video(
        source, output, checkpoint, water_confirmation_seconds=0.5, **controls
    )
    policy = confirmed["water_temporal_policy"]
    assert policy["required_consecutive_frames"] == required
    assert policy["longest_consecutive_water_run"] == 2
    assert policy["raw_water_frames"] == len(water_frames)
    assert policy["suppressed_frames"] == suppressed
    assert confirmed["base_pipeline_version"] == raw["model_version"]
    assert confirmed["model_version"] != raw["model_version"]
    assert confirmed["raw_counts"] == raw["counts"]
    assert (
        len(calls) == 2 * frame_count
    )  # Re-rendering does not run either model again.
    for original, current in zip(raw["frames"], confirmed["frames"], strict=True):
        assert [d for d in original["detections"] if d["class_id"] != 1] == [
            d for d in current["detections"] if d["class_id"] != 1
        ]
        assert current["raw_water_detections"] == [
            d for d in original["detections"] if d["class_id"] == 1
        ]
        assert current["water"] == original["water"]
    assert confirmed["counts"]["animal_visivel"] == raw["counts"]["animal_visivel"]
    assert (
        confirmed["counts"]["obstrucao_visivel"] == raw["counts"]["obstrucao_visivel"]
    )
    poster = read_image_bgr(output.with_suffix(".jpg"))
    if suppressed:
        assert confirmed["counts"]["agua_acumulada"]["count"] == 0
        assert all(
            d["class_id"] != 1
            for frame in confirmed["frames"]
            for d in frame["detections"]
        )
        assert all(d["class_id"] != 1 for d in confirmed["representative_detections"])
        assert all(
            item["class_name"] != "agua_acumulada"
            for item in confirmed["review_frames"]
        )
        assert not output.with_name("output_agua_acumulada.jpg").exists()
        assert int(poster[80, 100].max()) - int(poster[80, 100].min()) <= 3
    else:
        assert confirmed["counts"]["agua_acumulada"] == raw["counts"]["agua_acumulada"]
        assert [frame["detections"] for frame in confirmed["frames"]] == [
            frame["detections"] for frame in raw["frames"]
        ]
        assert int(poster[80, 100].max()) - int(poster[80, 100].min()) > 15
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == frame_count == confirmed["video"]["processed_frames"]
