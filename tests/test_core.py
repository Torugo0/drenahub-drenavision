from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.core.image_io import InvalidImageError, decode_image, encode_jpeg
from app.core.schemas import Detection, PredictionEvent
from app.core.storage import EventRepository, EvidenceStore


def test_decode_and_encode_image_round_trip() -> None:
    source = np.zeros((20, 30, 3), dtype=np.uint8)
    source[:, :, 1] = 180
    encoded = encode_jpeg(source)
    decoded = decode_image(encoded, max_pixels=10_000)
    assert decoded.shape == (20, 30, 3)
    assert decoded.dtype == np.uint8


def test_decode_rejects_non_image() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image", max_pixels=10_000)


def test_decode_rejects_unsupported_gif() -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="GIF")
    with pytest.raises(InvalidImageError, match="JPEG ou PNG"):
        decode_image(buffer.getvalue(), max_pixels=10_000)


def test_event_repository_round_trip(tmp_path) -> None:
    repository = EventRepository(tmp_path / "events.db")
    repository.initialize()
    event = PredictionEvent(
        event_id="6ac60e0e-0a96-4210-b559-65f12f27f4b8",
        point_id="bueiro-001",
        timestamp=datetime.now(UTC),
        input_type="image",
        model_version="test-model",
        detections=[
            Detection(
                class_id=0,
                class_name="obstrucao_visivel",
                confidence=0.9,
                bounding_box=(1, 2, 10, 12),
            )
        ],
        inspection_recommended=True,
        status="revisao_recomendada",
        inference_ms=12.5,
    )
    repository.save(event)
    restored = repository.get(event.event_id)
    assert restored == event
    listed = repository.list(point_id="bueiro-001")
    assert listed.total == 1
    assert listed.items[0].detection_count == 1


def test_evidence_store_blocks_traversal(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence")
    with pytest.raises(ValueError):
        store.resolve("../../secret.jpg")


def test_detection_rejects_inconsistent_class_and_box() -> None:
    with pytest.raises(ValueError, match="incompatíveis"):
        Detection(
            class_id=0,
            class_name="animal_visivel",
            confidence=0.8,
            bounding_box=(1, 2, 10, 12),
        )
    with pytest.raises(ValueError, match="área positiva"):
        Detection(
            class_id=0,
            class_name="obstrucao_visivel",
            confidence=0.8,
            bounding_box=(10, 2, 1, 12),
        )
