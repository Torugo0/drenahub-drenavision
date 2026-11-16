from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drenavision.core.storage import EventRepository, EvidenceStore
from drenavision.inference.device import _select_device_from_torch
from drenavision.inference.errors import ClassMapMismatchError
from drenavision.inference.evaluation import extract_detection_metrics
from drenavision.inference.media_io import read_image_bgr, write_image
from drenavision.inference.training import snapshot_interrupted_run
from drenavision.inference.utils import (
    box_iou,
    greedy_class_matches,
    validate_class_map,
)
from drenavision.inference.video import persist_video_event


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_device_auto_prefers_cuda_and_falls_back_to_cpu() -> None:
    assert _select_device_from_torch(SimpleNamespace(cuda=_Cuda(True)), "auto") == 0
    assert (
        _select_device_from_torch(SimpleNamespace(cuda=_Cuda(False)), "auto") == "cpu"
    )


def test_explicit_unavailable_cuda_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        _select_device_from_torch(SimpleNamespace(cuda=_Cuda(False)), "cuda")


def test_class_map_must_match_exactly() -> None:
    expected = {
        0: "obstrucao_visivel",
        1: "agua_acumulada",
        2: "animal_visivel",
    }
    assert validate_class_map(expected) == expected
    with pytest.raises(ClassMapMismatchError):
        validate_class_map({0: "person", 1: "car", 2: "dog"})


def test_iou_and_class_aware_matching() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    truth = [(0, (0, 0, 10, 10)), (1, (20, 20, 30, 30))]
    predictions = [
        (0, 0.9, (0, 0, 10, 10)),
        (2, 0.8, (20, 20, 30, 30)),
    ]
    matched_truth, matched_predictions = greedy_class_matches(truth, predictions)
    assert matched_truth == {0}
    assert matched_predictions == {0}


def test_metric_export_has_aggregate_and_per_class_values() -> None:
    box = SimpleNamespace(
        p=[0.7, 0.6, 0.5],
        r=[0.6, 0.5, 0.4],
        ap50=[0.8, 0.7, 0.6],
        maps=[0.5, 0.4, 0.3],
        mp=0.6,
        mr=0.5,
        map50=0.7,
        map=0.4,
    )
    exported = extract_detection_metrics(
        SimpleNamespace(box=box, speed={"inference": 4.2})
    )
    assert exported["aggregate"]["map50_95"] == pytest.approx(0.4)
    assert exported["per_class"]["animal_visivel"]["precision"] == pytest.approx(0.5)


def test_metric_export_maps_sparse_class_indexes() -> None:
    box = SimpleNamespace(
        p=[0.8, 0.4],
        r=[0.7, 0.3],
        ap50=[0.9, 0.5],
        maps=[0.6, 0.4, 0.2],
        ap_class_index=[0, 2],
        mp=0.6,
        mr=0.5,
        map50=0.7,
        map=0.4,
    )
    exported = extract_detection_metrics(SimpleNamespace(box=box, speed={}))
    assert exported["per_class"]["agua_acumulada"]["precision"] is None
    assert exported["per_class"]["animal_visivel"]["precision"] == pytest.approx(0.4)


def test_resume_snapshot_preserves_interrupted_run(tmp_path: Path) -> None:
    run = tmp_path / "old_run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"real checkpoint bytes")
    (run / "results.csv").write_text("epoch,metric\n1,0.1\n", encoding="utf-8")

    snapshot = snapshot_interrupted_run(
        last, tmp_path / "artifacts" / "training", completed_epoch=1
    )
    manifest = json.loads((snapshot / "interrupted_manifest.json").read_text())
    assert (snapshot / "weights" / "last.pt").read_bytes() == last.read_bytes()
    assert manifest["status"] == "interrupted"
    assert manifest["last_completed_epoch"] == 2
    assert str(tmp_path) not in manifest["source_run"]


def test_image_io_supports_unicode_windows_style_paths(tmp_path: Path) -> None:
    destination = tmp_path / "Área de Trabalho" / "evidência.jpg"
    original = np.full((12, 16, 3), (20, 80, 140), dtype=np.uint8)
    write_image(destination, original)
    decoded = read_image_bgr(destination)
    assert decoded.shape == original.shape
    assert abs(float(decoded.mean()) - float(original.mean())) < 3


def test_video_manifest_can_persist_summary_event(tmp_path: Path) -> None:
    manifest = {
        "status": "completed",
        "model_version": "best-abc123",
        "representative_detections": [
            {
                "class_id": 2,
                "class_name": "animal_visivel",
                "confidence": 0.81,
                "bounding_box": [1, 2, 30, 40],
                "bounding_box_format": "xyxy_pixels",
                "track_id": 7,
            }
        ],
        "frames": [{"inference_ms": 4.0}, {"inference_ms": 6.0}],
    }
    repository = EventRepository(tmp_path / "events.db")
    evidence_store = EvidenceStore(tmp_path / "evidence")
    event = persist_video_event(
        manifest,
        repository,
        evidence_store,
        point_id="  ponto-01  ",
    )
    stored = repository.get(event.event_id)

    assert stored is not None
    assert stored.input_type == "video"
    assert stored.point_id == "ponto-01"
    assert stored.inference_ms == pytest.approx(5.0)
    assert stored.detections[0].track_id == 7
    assert stored.evidence_uri is not None
    assert evidence_store.resolve(stored.evidence_uri).is_file()
