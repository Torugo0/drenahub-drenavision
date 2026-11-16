import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from drenavision.ui import streamlit_app
from drenavision.ui.streamlit_app import annotate, result_counts


def test_phone_photo_boxes_use_the_same_exif_orientation_as_inference():
    image = Image.new("RGB", (160, 90), "white")
    exif = Image.Exif()
    exif[274] = 6  # Portrait photograph stored sideways by a phone.
    encoded = BytesIO()
    image.save(encoded, format="JPEG", exif=exif)
    marked = annotate(
        encoded.getvalue(),
        [{"class_name": "animal_visivel", "bounding_box": [10, 40, 70, 120]}],
    )
    assert marked.size == (90, 160)
    assert marked.getpixel((10, 100)) == (124, 58, 237)


def test_video_counts_do_not_sum_repeated_frame_detections():
    payload = {
        "counts": {"animal_visivel": {"count": 1}},
        "frames": [{"detections": [{"class_name": "animal_visivel"}]}] * 50,
    }
    assert result_counts(payload, "video") == {
        "obstrucao_visivel": 0,
        "agua_acumulada": 0,
        "animal_visivel": 1,
    }


def test_image_counts_include_all_detections_and_missing_classes():
    payload = {
        "detections": [
            {"class_name": "obstrucao_visivel"},
            {"class_name": "obstrucao_visivel"},
            {"class_name": "animal_visivel"},
        ]
    }
    assert result_counts(payload, "image") == {
        "obstrucao_visivel": 2,
        "agua_acumulada": 0,
        "animal_visivel": 1,
    }


def test_video_job_uses_current_settings_instead_of_stale_streamlit_state(
    tmp_path, monkeypatch
):
    model = tmp_path / "current/best.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint-contract")
    monkeypatch.setenv("DRENAVISION_MODEL_PATH", str(model))
    monkeypatch.setenv("DRENAVISION_DATABASE_PATH", str(tmp_path / "current/events.db"))
    monkeypatch.setenv("DRENAVISION_EVIDENCE_DIR", str(tmp_path / "current/evidence"))
    monkeypatch.setattr(
        streamlit_app,
        "SETTINGS",
        SimpleNamespace(model_path=tmp_path / "artifacts/models/best.pt"),
    )
    monkeypatch.setattr(streamlit_app, "RESULTS_DIR", tmp_path / "results")

    def fake_process(command, **kwargs):
        assert Path(command[command.index("--model") + 1]) == model
        assert (
            Path(command[command.index("--database") + 1])
            == tmp_path / "current/events.db"
        )
        assert kwargs["encoding"] == "utf-8"
        destination = Path(command[command.index("--output") + 1])
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"encoded-video-contract")
        destination.with_suffix(".json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(streamlit_app.subprocess, "run", fake_process)
    result = streamlit_app.analyze_video(tmp_path / "uploaded.mp4", "garagem")
    assert result["payload"]["status"] == "completed"


def test_missing_model_has_specific_error_before_processing(tmp_path, monkeypatch):
    monkeypatch.setenv("DRENAVISION_MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(
        streamlit_app.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "must fail before launching video inference"
        ),
    )
    with pytest.raises(RuntimeError, match="modelo de análise não foi encontrado"):
        streamlit_app.analyze_video(tmp_path / "uploaded.mp4", "")
