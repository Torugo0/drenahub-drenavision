from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from drenavision.api.main import create_app
from drenavision.core.config import Settings
from drenavision.core.schemas import Detection


@dataclass
class FakeResult:
    detections: list[Detection]
    annotated_image: np.ndarray
    inference_ms: float = 4.2
    model_version: str = "fake-test-model"


class FakePredictor:
    is_ready = True

    def predict(self, image_rgb: np.ndarray) -> FakeResult:
        return FakeResult(
            detections=[
                Detection(
                    class_id=1,
                    class_name="agua_acumulada",
                    confidence=0.87,
                    bounding_box=(1, 1, 8, 8),
                )
            ],
            annotated_image=image_rgb,
        )


class NotReadyPredictor:
    is_ready = False

    def predict(self, image_rgb: np.ndarray) -> None:  # pragma: no cover
        raise AssertionError("must not run")


class EmptyPredictor:
    is_ready = True

    def predict(self, image_rgb: np.ndarray) -> FakeResult:
        return FakeResult(detections=[], annotated_image=image_rgb)


def _settings(tmp_path) -> Settings:
    return Settings(
        model_path=tmp_path / "missing.pt",
        database_path=tmp_path / "events.db",
        evidence_dir=tmp_path / "evidence",
        max_image_bytes=1024 * 1024,
        max_image_pixels=1_000_000,
    )


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), (30, 90, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_prediction_and_event_history(tmp_path) -> None:
    app = create_app(settings=_settings(tmp_path), predictor=FakePredictor())
    with TestClient(app) as client:
        response = client.post(
            "/vision/predict",
            files={"file": ("scene.jpg", _jpeg(), "image/jpeg")},
            data={"point_id": "galeria-norte"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["detections"][0]["class_name"] == "agua_acumulada"
        assert payload["inspection_recommended"] is True
        assert payload["evidence_uri"].endswith(".jpg")

        history = client.get("/vision/events").json()
        assert history["total"] == 1
        detail = client.get(f"/vision/events/{payload['event_id']}")
        assert detail.status_code == 200


def test_unavailable_model_is_explicit(tmp_path) -> None:
    app = create_app(settings=_settings(tmp_path), predictor=NotReadyPredictor())
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        response = client.post(
            "/vision/predict",
            files={"file": ("scene.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 503
        assert "Modelo treinado indisponível" in response.json()["detail"]


def test_no_detections_is_a_successful_event(tmp_path) -> None:
    app = create_app(settings=_settings(tmp_path), predictor=EmptyPredictor())
    with TestClient(app) as client:
        response = client.post(
            "/vision/predict",
            files={"file": ("scene.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["detections"] == []
        assert payload["inspection_recommended"] is False
        assert payload["status"] == "sem_deteccao"


def test_rejects_bad_media_and_reports_classes(tmp_path) -> None:
    app = create_app(settings=_settings(tmp_path), predictor=FakePredictor())
    with TestClient(app) as client:
        response = client.post(
            "/vision/predict",
            files={"file": ("payload.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 415
        classes = client.get("/vision/classes").json()["classes"]
        assert [item["name"] for item in classes] == [
            "obstrucao_visivel",
            "agua_acumulada",
            "animal_visivel",
        ]
