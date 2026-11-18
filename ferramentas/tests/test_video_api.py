from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from drenavision.api import main as api
from drenavision.core.config import Settings
from drenavision.core.storage import EventRepository
from ferramentas.tests.test_api import FakePredictor, NotReadyPredictor
from ferramentas.tests.test_video_objects_pipeline import _shared_pipeline


def settings(tmp_path, *, limit=1024 * 1024):
    model = tmp_path / "best.pt"
    model.write_bytes(b"master checkpoint")
    return Settings(
        model_path=model,
        database_path=tmp_path / "events.db",
        evidence_dir=tmp_path / "evidence",
        max_video_bytes=limit,
        water_model_path=None,
        object_model_path=None,
        object_verifier_path=None,
    )


def processor(source, destination, checkpoint, **kwargs):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"0123456789abcdefghijklmnopqrstuvwxyz")
    destination.with_suffix(".jpg").write_bytes(b"poster")
    return {
        "status": "completed",
        "model_version": "shared-profile",
        "representative_detections": [],
        "frames": [],
        "counts": {},
        "pipeline": {},
        "video": {"processed_frames": 3},
    }


def jpeg():
    data = BytesIO()
    Image.new("RGB", (16, 12), "white").save(data, format="JPEG")
    return data.getvalue()


def test_video_api_shares_one_predictor_with_images_and_serves_range_and_json_evidence(
    tmp_path, monkeypatch
):
    runtime = settings(tmp_path)
    loads, borrowed, upload_paths = [], [], []
    predictor = FakePredictor()

    def build(_settings):
        loads.append(True)
        return predictor

    monkeypatch.setattr(api, "_build_predictor", build)
    application = api.create_app(settings=runtime)

    def inspect(source, destination, checkpoint, **kwargs):
        assert application.state.inference_semaphore.locked()
        assert kwargs["shared_predictor"] is predictor
        assert kwargs["water_model_path"] is None
        assert kwargs["object_verifier_path"] is None
        assert (
            kwargs["water_confirmation_seconds"]
            == runtime.video_water_confirmation_seconds
        )
        borrowed.append(kwargs["shared_predictor"])
        upload_paths.append(Path(source))
        assert Path(source).read_bytes() == b"original video"
        return processor(source, destination, checkpoint, **kwargs)

    monkeypatch.setattr(api, "process_video", inspect)
    with TestClient(application) as client:
        image = client.post(
            "/vision/predict", files={"file": ("scene.jpg", jpeg(), "image/jpeg")}
        )
        assert image.status_code == 200
        response = client.post(
            "/vision/predict/video",
            files={"file": ("scene.mp4", b"original video", "video/mp4")},
            data={"point_id": "  garagem  "},
        )
        assert response.status_code == 200
        payload = response.json()
        event_id = payload["event_id"]
        assert payload["event"]["event_id"] == event_id
        assert payload["event"]["point_id"] == "garagem"
        assert payload["event"]["input_type"] == "video"
        assert client.get("/vision/events").json()["total"] == 2
        assert client.get(f"/vision/events/{event_id}").status_code == 200
        media = client.get(payload["video_url"])
        assert media.status_code == 200 and media.headers["content-type"] == "video/mp4"
        part = client.get(payload["video_url"], headers={"Range": "bytes=0-9"})
        assert part.status_code == 206 and part.content == b"0123456789"
        evidence = client.get(f"/vision/evidence/{event_id}")
        assert (
            evidence.status_code == 200
            and evidence.headers["content-type"] == "application/json"
        )
        assert evidence.json()["status"] == "completed"
        assert (
            client.get(f"/vision/videos/{image.json()['event_id']}").status_code == 404
        )
        for path in (
            "/vision/videos/not-a-uuid",
            f"/vision/videos/{uuid4()}",
            "/vision/videos/..%2Fsecret.mp4",
        ):
            assert client.get(path).status_code == 404
    assert len(loads) == len(borrowed) == 1
    assert all(not path.exists() and not path.parent.exists() for path in upload_paths)


@pytest.mark.parametrize(
    "filename,mime,status,data",
    [
        ("oversize.mp4", "video/mp4", 413, b"x" * 1025),
        ("empty.mp4", "video/mp4", 400, b""),
        ("not-video.jpg", "video/mp4", 415, b"data"),
        ("mismatch.mp4", "video/quicktime", 415, b"data"),
        ("payload.mp4", "text/plain", 415, b"data"),
    ],
)
def test_video_upload_limits_media_validation_and_cleanup(
    tmp_path, monkeypatch, filename, mime, status, data
):
    runtime = settings(tmp_path, limit=1024)
    monkeypatch.setattr(
        api,
        "process_video",
        lambda *_a, **_kw: pytest.fail("Invalid upload must not reach models"),
    )
    scratch_paths = []
    real_temporary = api.tempfile.TemporaryDirectory

    def temporary(**kwargs):
        handle = real_temporary(**kwargs)
        scratch_paths.append(Path(handle.name))
        return handle

    monkeypatch.setattr(api.tempfile, "TemporaryDirectory", temporary)
    with TestClient(
        api.create_app(settings=runtime, predictor=FakePredictor())
    ) as client:
        response = client.post(
            "/vision/predict/video", files={"file": (filename, data, mime)}
        )
        assert response.status_code == status
        assert client.get("/vision/events").json()["total"] == 0
    assert all(not path.exists() for path in scratch_paths)
    assert not list(runtime.evidence_dir.rglob("*.mp4"))


@pytest.mark.parametrize("failure_stage", ["processing", "after_commit"])
def test_video_failure_rolls_back_only_new_uuid_files_and_event(
    tmp_path, monkeypatch, failure_stage
):
    runtime = settings(tmp_path)
    protected = runtime.evidence_dir / "videos/keep-original.mp4"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"existing permanent artifact")
    upload_paths = []

    def inspect(source, destination, checkpoint, **kwargs):
        upload_paths.append(Path(source))
        manifest = processor(source, destination, checkpoint, **kwargs)
        destination.with_suffix(".json").write_text("partial")
        if failure_stage == "processing":
            raise RuntimeError("invalid source or failed forward")
        return manifest

    monkeypatch.setattr(api, "process_video", inspect)
    if failure_stage == "after_commit":
        persist = api.persist_video_event

        def fail_after_commit(*args, **kwargs):
            persist(*args, **kwargs)
            raise OSError("simulated adapter error after committing")

        monkeypatch.setattr(api, "persist_video_event", fail_after_commit)
    with TestClient(
        api.create_app(settings=runtime, predictor=FakePredictor())
    ) as client:
        response = client.post(
            "/vision/predict/video",
            files={"file": ("scene.mp4", b"original", "video/mp4")},
        )
        assert response.status_code in {400, 500}
        assert client.get("/vision/events").json()["total"] == 0
    assert protected.read_bytes() == b"existing permanent artifact"
    assert list(runtime.evidence_dir.rglob("*.*")) == [protected]
    assert upload_paths and all(not path.exists() for path in upload_paths)
    assert EventRepository(runtime.database_path).list().total == 0


def test_valid_video_mime_does_not_bypass_real_container_decoder(tmp_path, monkeypatch):
    _, _, checkpoint, shared, _, _, _, _, _ = _shared_pipeline(tmp_path, monkeypatch)
    runtime = Settings(
        model_path=checkpoint,
        database_path=tmp_path / "events.db",
        evidence_dir=tmp_path / "evidence",
        device="cpu",
        animal_confidence_threshold=0.6,
        water_model_path=tmp_path / "water",
        water_pixel_confidence=0.5,
        water_min_area_fraction=0.001,
        object_model_path=tmp_path / "objects",
        object_verifier_path=None,
        object_residue_confidence=0.35,
        object_animal_confidence=0.35,
        object_text_confidence=0.2,
        object_shortest_edge=800,
        object_longest_edge=1333,
    )
    with TestClient(api.create_app(settings=runtime, predictor=shared)) as client:
        response = client.post(
            "/vision/predict/video",
            files={"file": ("broken.mp4", b"not a video container", "video/mp4")},
        )
        assert response.status_code == 400
        assert client.get("/vision/events").json()["total"] == 0
    assert not list(runtime.evidence_dir.rglob("*.mp4"))


def test_unavailable_shared_video_model_is_503_and_does_not_create_output(tmp_path):
    runtime = settings(tmp_path)
    with TestClient(
        api.create_app(settings=runtime, predictor=NotReadyPredictor())
    ) as client:
        response = client.post(
            "/vision/predict/video",
            files={"file": ("scene.mp4", b"original", "video/mp4")},
        )
        assert response.status_code == 503
        assert client.get("/vision/events").json()["total"] == 0
    assert not list(runtime.evidence_dir.rglob("*.mp4"))
