from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from drenavision.api.main import _build_predictor
from drenavision.core.config import Settings
from drenavision.core.constants import CLASS_NAMES
from drenavision.inference import predictor, video
from ferramentas.scripts import predict, process_video


@pytest.mark.parametrize("size", [0, 288, 321, 2050, -640, True, 960.0, "960.0"])
def test_invalid_image_sizes_fail_in_settings_predictor_and_video_before_work(size):
    with pytest.raises(ValueError, match="image_size/imgsz"):
        Settings(_env_file=None, image_size=size)
    with pytest.raises(ValueError, match="image_size/imgsz"):
        predictor.YoloPredictor("not-loaded.pt", image_size=size)
    with pytest.raises(ValueError, match="image_size/imgsz"):
        video.process_video(
            "missing.mp4", "missing_output.mp4", "not-loaded.pt", image_size=size
        )


def test_size_environment_and_api_predictor_are_consistent(monkeypatch):
    monkeypatch.setenv("DRENAVISION_IMAGE_SIZE", "960")
    settings = Settings(_env_file=None)
    assert settings.image_size == 960
    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(is_ready=True)

    from drenavision.inference import runtime

    monkeypatch.setattr(runtime, "create_predictor", factory)
    assert _build_predictor(settings).is_ready
    assert received["image_size"] == 960


@pytest.mark.parametrize("script", [predict, process_video])
def test_cli_rejects_non_stride_size_instead_of_allowing_silent_rounding(script):
    argv = ["--input", "input.mp4"]
    if script is process_video:
        argv += ["--output", "output.mp4"]
    with pytest.raises(SystemExit):
        script.build_parser().parse_args(argv + ["--imgsz", "641"])
    assert script.build_parser().parse_args(argv).imgsz == 640
    assert script.build_parser().parse_args(argv + ["--imgsz", "960"]).imgsz == 960


def test_video_cli_forwards_explicit_size(monkeypatch, capsys):
    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        return {"status": "completed", "inference": {"imgsz": kwargs["image_size"]}}

    monkeypatch.setattr(process_video, "process_video", run)
    assert (
        process_video.main(
            ["--input", "input.mp4", "--output", "output.mp4", "--imgsz", "960"]
        )
        == 0
    )
    assert received["image_size"] == 960
    assert json.loads(capsys.readouterr().out)["inference"]["imgsz"] == 960


@pytest.mark.parametrize("tracking", [False, True])
def test_video_size_reaches_both_inference_modes_and_manifest(
    tmp_path: Path, monkeypatch, tracking
):
    source, output, checkpoint = (
        tmp_path / "input.mp4",
        tmp_path / "output.mp4",
        tmp_path / "model.pt",
    )
    checkpoint.write_bytes(b"checkpoint-test")
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120)
    )
    assert writer.isOpened()
    writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.write(np.full((120, 160, 3), 40, dtype=np.uint8))
    writer.release()
    calls = []

    class Boxes:
        xyxy = [[10, 20, 50, 60]]
        cls = [1]
        conf = [0.8]
        id = None

        def __len__(self):
            return 1

    class Model:
        names = CLASS_NAMES

        def predict(self, **kwargs):
            calls.append(("predict", kwargs))
            return [SimpleNamespace(boxes=Boxes(), speed={"inference": 2})]

        def track(self, **kwargs):
            calls.append(("track", kwargs))
            return [SimpleNamespace(boxes=Boxes(), speed={"inference": 2})]

    def imports(name):
        if name == "cv2":
            return cv2
        assert name == "ultralytics"
        return SimpleNamespace(YOLO=lambda _: Model())

    def export(source_path, destination):
        shutil.copyfile(source_path, destination)
        return {"codec": "test-codec"}

    monkeypatch.setattr(video.importlib, "import_module", imports)
    monkeypatch.setattr(video, "select_device", lambda _: "cpu")
    monkeypatch.setattr(video, "prepare_browser_video", export)
    manifest = video.process_video(
        source, output, checkpoint, image_size=960, track=tracking
    )
    assert len(calls) == 2
    assert all(mode == ("track" if tracking else "predict") for mode, _ in calls)
    assert all(kwargs["imgsz"] == 960 for _, kwargs in calls)
    assert all(kwargs["source"].shape == (120, 160, 3) for _, kwargs in calls)
    assert manifest["frames"][0]["detections"][0]["bounding_box"] == [
        10.0,
        20.0,
        50.0,
        60.0,
    ]
    assert (manifest["video"]["width"], manifest["video"]["height"]) == (160, 120)
    assert (
        json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))[
            "inference"
        ]["imgsz"]
        == 960
    )
