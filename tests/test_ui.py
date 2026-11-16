from __future__ import annotations

from io import BytesIO

from drenavision.ui import streamlit_app


def test_video_preview_passes_file_like_object_to_streamlit(monkeypatch) -> None:
    upload = BytesIO(b"not-a-real-video-but-valid-for-the-render-contract")
    captured: dict[str, object] = {}

    def fake_video(data, **kwargs) -> None:
        captured["data"] = data
        captured.update(kwargs)
        assert data.tell() == 0
        data.read()

    monkeypatch.setattr(streamlit_app.st, "video", fake_video)

    streamlit_app.render_video_preview(upload, "video/mp4")

    assert captured["data"] is upload
    assert not isinstance(captured["data"], memoryview)
    assert captured["format"] == "video/mp4"
    assert captured["alt"] == "Prévia do vídeo enviado para análise"
    assert upload.tell() == 0


def test_video_preview_uses_mp4_for_unknown_mime(monkeypatch) -> None:
    upload = BytesIO(b"video")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        streamlit_app.st,
        "video",
        lambda _data, **kwargs: captured.update(kwargs),
    )

    streamlit_app.render_video_preview(upload, "application/octet-stream")

    assert captured["format"] == "video/mp4"
