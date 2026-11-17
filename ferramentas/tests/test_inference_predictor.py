from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drenavision.core.schemas import Detection
from drenavision.inference.errors import ModelUnavailableError
from drenavision.inference.predictor import YoloPredictor


class _Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Boxes:
    def __init__(self, detections: bool = True) -> None:
        if detections:
            self.xyxy = _Tensor([[1.0, 2.0, 9.0, 8.0]])
            self.conf = _Tensor([0.75])
            self.cls = _Tensor([2.0])
            self.id = None
            self._length = 1
        else:
            self.xyxy = _Tensor([])
            self.conf = _Tensor([])
            self.cls = _Tensor([])
            self.id = None
            self._length = 0

    def __len__(self):
        return self._length


class _Result:
    def __init__(self, detections: bool = True) -> None:
        self.boxes = _Boxes(detections)
        self.speed = {"inference": 3.25}

    def plot(self):
        # Ultralytics plot() returns BGR.
        return np.full((10, 12, 3), [10, 20, 30], dtype=np.uint8)


class _Model:
    def __init__(self, detections: bool = True) -> None:
        self.detections = detections
        self.last_source = None
        self.last_kwargs = None

    def predict(self, **kwargs):
        self.last_source = kwargs["source"]
        self.last_kwargs = kwargs
        return [_Result(self.detections)]


def _stub_predictor(model: _Model) -> YoloPredictor:
    predictor = YoloPredictor(Path("unused.pt"))
    predictor._model = model
    predictor._device = "cpu"
    predictor._model_version = "test-model"
    return predictor


def test_missing_checkpoint_is_explicit_and_does_not_import_model(
    tmp_path: Path,
) -> None:
    predictor = YoloPredictor(tmp_path / "missing.pt")
    assert predictor.is_ready is False
    assert "checkpoint customizado ausente" in predictor.detail
    with pytest.raises(ModelUnavailableError):
        predictor.predict(np.zeros((8, 8, 3), dtype=np.uint8))


def test_prediction_returns_canonical_schema_and_rgb_annotation() -> None:
    model = _Model()
    predictor = _stub_predictor(model)
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # red in RGB
    result = predictor.predict(image)

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert isinstance(detection, Detection)
    assert detection.class_name == "animal_visivel"
    assert detection.bounding_box == (1.0, 2.0, 9.0, 8.0)
    assert result.inference_ms == pytest.approx(3.25)
    assert result.annotated_image[0, 0].tolist() == [30, 20, 10]
    assert model.last_source[0, 0].tolist() == [0, 0, 255]
    assert model.last_kwargs["imgsz"] == 640


def test_explicit_inference_size_is_forwarded_without_scaling_original_pixel_boxes():
    model = _Model()
    predictor = _stub_predictor(model)
    predictor.image_size = 960
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    result = predictor.predict(image)
    assert model.last_kwargs["imgsz"] == 960
    assert model.last_source.shape == image.shape
    assert result.detections[0].bounding_box == (1.0, 2.0, 9.0, 8.0)
    assert (result.image_width, result.image_height) == (12, 10)
    assert result.as_dict()["inference"] == {"imgsz": 960}


def test_zero_detections_remains_a_successful_real_result() -> None:
    predictor = _stub_predictor(_Model(detections=False))
    result = predictor.predict(np.zeros((10, 12, 3), dtype=np.uint8))
    assert result.detections == []


def test_non_uint8_input_is_rejected() -> None:
    predictor = _stub_predictor(_Model())
    with pytest.raises(ValueError, match="uint8"):
        predictor.predict(np.zeros((10, 12, 3), dtype=np.float32))
