"""Real YOLO inference, training, evaluation, and video processing.

Heavy dependencies (PyTorch and Ultralytics) are loaded only when an operation
actually needs a model. Importing this package is therefore safe on API hosts
where the custom checkpoint has not been provisioned yet.
"""

from drenavision.core.schemas import Detection

from .contracts import PredictionResult
from .errors import (
    ClassMapMismatchError,
    DatasetUnavailableError,
    InferenceError,
    ModelUnavailableError,
)
from .predictor import YOLOPredictor, YoloPredictor

__all__ = [
    "ClassMapMismatchError",
    "DatasetUnavailableError",
    "Detection",
    "InferenceError",
    "ModelUnavailableError",
    "PredictionResult",
    "YOLOPredictor",
    "YoloPredictor",
]
