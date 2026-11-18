import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from PIL import Image

from drenavision.ui import streamlit_app
from drenavision.ui.streamlit_app import annotate, result_counts


def test_phone_photo_boxes_use_the_same_exif_orientation_as_inference():
    image = Image.new("RGB", (160, 90), "white")
    exif = Image.Exif()
    exif[274] = 6
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


def test_video_upload_and_download_use_api_shared_models_without_subprocess(
    tmp_path, monkeypatch
):
    source = tmp_path / "uploaded.mp4"
    source.write_bytes(b"original recorded video")
    identifier = str(uuid4())
    payload = {
        "status": "completed",
        "event_id": identifier,
        "video_url": f"/vision/videos/{identifier}",
        "event": {"event_id": identifier},
    }
    calls, handles = [], []
    monkeypatch.setattr(streamlit_app, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        streamlit_app.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "UI must not launch a second model process"
        ),
    )

    def api(method, path, **kwargs):
        calls.append((method, path))
        assert kwargs["timeout"] == 1800
        if method == "POST":
            name, handle, mime = kwargs["files"]["file"]
            assert name == "uploaded.mp4" and mime == "video/mp4"
            assert handle.read() == source.read_bytes()
            assert kwargs["data"] == {"point_id": "garagem"}
            handles.append(handle)
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200, content=b"browser MP4 bytes", headers={"content-type": "video/mp4"}
        )

    monkeypatch.setattr(streamlit_app, "api_request", api)
    result = streamlit_app.analyze_video(source, "garagem")
    assert calls == [
        ("POST", "/vision/predict/video"),
        ("GET", f"/vision/videos/{identifier}"),
    ]
    assert handles[0].closed
    assert Path(result["path"]).read_bytes() == b"browser MP4 bytes"
    assert (
        json.loads(
            Path(result["path"]).with_suffix(".json").read_text(encoding="utf-8")
        )
        == payload
    )
    assert source.read_bytes() == b"original recorded video"


@pytest.mark.parametrize(
    "status, message",
    [
        (503, "não está pronto"),
        (413, "excede o limite"),
        (400, "Não foi possível analisar"),
    ],
)
def test_video_api_failure_does_not_create_a_local_result(
    tmp_path, monkeypatch, status, message
):
    source = tmp_path / "uploaded.mp4"
    source.write_bytes(b"original")
    monkeypatch.setattr(streamlit_app, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        streamlit_app,
        "api_request",
        lambda *_a, **_kw: httpx.Response(status, json={"detail": "failure"}),
    )
    with pytest.raises(RuntimeError, match=message):
        streamlit_app.analyze_video(source, "")
    assert not (tmp_path / "results").exists()
    assert source.read_bytes() == b"original"


def test_api_request_allows_video_timeout_override_without_duplicate_keyword(
    monkeypatch,
):
    timeouts = []

    def request(method, url, *, timeout, **kwargs):
        timeouts.append(timeout)
        return httpx.Response(200)

    monkeypatch.setattr(streamlit_app.httpx, "request", request)
    streamlit_app.api_request("GET", "/health")
    streamlit_app.api_request("POST", "/vision/predict/video", timeout=1800)
    assert timeouts == [90, 1800]


def test_video_download_failure_keeps_upload_and_does_not_create_broken_player_file(
    tmp_path, monkeypatch
):
    source = tmp_path / "uploaded.mp4"
    source.write_bytes(b"original")
    identifier = str(uuid4())
    monkeypatch.setattr(streamlit_app, "RESULTS_DIR", tmp_path / "results")

    def api(method, *_args, **_kwargs):
        return (
            httpx.Response(
                200,
                json={
                    "status": "completed",
                    "event_id": identifier,
                    "video_url": f"/vision/videos/{identifier}",
                },
            )
            if method == "POST"
            else httpx.Response(404)
        )

    monkeypatch.setattr(streamlit_app, "api_request", api)
    with pytest.raises(RuntimeError, match="não pôde ser carregado"):
        streamlit_app.analyze_video(source, "")
    assert not (tmp_path / "results").exists()
    assert source.read_bytes() == b"original"
