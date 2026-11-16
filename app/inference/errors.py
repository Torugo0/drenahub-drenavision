"""Explicit operational errors used instead of silent model fallbacks."""


class InferenceError(RuntimeError):
    """Base class for actionable DrenaVision inference failures."""


class ModelUnavailableError(InferenceError):
    """Raised when a required custom checkpoint cannot be used."""


class ClassMapMismatchError(InferenceError):
    """Raised when a checkpoint does not expose the three project classes."""


class DatasetUnavailableError(RuntimeError):
    """Raised when training/evaluation data is absent or structurally invalid."""
