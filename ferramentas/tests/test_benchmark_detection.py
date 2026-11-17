from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ferramentas.scripts.benchmark_detection import (
    frame_indices,
    merge_predictions,
    predict,
    predict_pipeline,
    score,
    stratified_paths,
    tile_starts,
    totals,
)


def test_stratified_selection_is_stable_and_includes_negative_images(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images" / "val"
    labels = tmp_path / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    paths = []
    for name, contents in {
        "mixed": "0 .5 .5 .2 .2\n2 .5 .5 .2 .2\n",
        "water_a": "1 .5 .5 .2 .2\n",
        "water_b": "1 .5 .5 .2 .2\n",
        "negative": "",
    }.items():
        path = images / f"{name}.jpg"
        path.touch()
        (labels / f"{name}.txt").write_text(contents)
        paths.append(path)
    selected = stratified_paths(paths, 1, 42)
    assert selected == stratified_paths(list(reversed(paths)), 1, 42)
    assert {path.stem for path in selected} >= {"mixed", "negative"}
    assert len(selected) == 3
    (labels / "negative.txt").unlink()
    with pytest.raises(FileNotFoundError):
        stratified_paths(paths, 1, 42)


def test_frame_sampling_and_tiles_cover_edges_without_repeating_short_inputs() -> None:
    assert frame_indices(240, 6) == [0, 48, 96, 143, 191, 239]
    assert frame_indices(2, 6) == [0, 1]
    starts = tile_starts(1024, 640, 0.2)
    assert starts == [0, 384]
    assert tile_starts(240, 640, 0.2) == [0]
    assert all(
        starts[index + 1] <= value + 640 for index, value in enumerate(starts[:-1])
    )


def test_nms_preserves_distinct_classes_and_highest_confidence_coordinates() -> None:
    predictions = [
        {"class_id": 2, "confidence": 0.7, "bounding_box": [10, 10, 30, 30]},
        {"class_id": 2, "confidence": 0.9, "bounding_box": [11, 11, 31, 31]},
        {"class_id": 0, "confidence": 0.6, "bounding_box": [10, 10, 30, 30]},
    ]
    merged = merge_predictions(predictions, 0.45)
    assert len(merged) == 2
    assert merged[0]["bounding_box"] == [11, 11, 31, 31]
    assert {item["class_id"] for item in merged} == {0, 2}


class _Boxes(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.xyxy)


def test_tiled_prediction_returns_original_pixel_coordinates() -> None:
    class Model:
        calls = 0

        def predict(self, image: np.ndarray, **kwargs: object) -> list:
            self.calls += 1
            assert kwargs["imgsz"] == 640
            # Full frame then four overlapping crops; only bottom-right has a detection.
            detected = self.calls == 5
            boxes = _Boxes(
                xyxy=[[1, 2, 3, 5]] if detected else [],
                cls=[2] if detected else [],
                conf=[0.8] if detected else [],
                id=None,
            )
            return [SimpleNamespace(boxes=boxes)]

    model = Model()
    predicted = predict(
        model, np.zeros((10, 10, 3), dtype=np.uint8), "cpu", 640, 0.35, 0.45, 6, 0.2
    )
    assert model.calls == 5
    assert predicted[0]["bounding_box"] == [5, 6, 7, 9]


def test_scores_count_duplicate_and_wrong_class_predictions_as_false_positives() -> (
    None
):
    truth = [
        {"class_id": 2, "bounding_box": [0, 0, 10, 10]},
        {"class_id": 1, "bounding_box": [20, 20, 30, 30]},
    ]
    predictions = [
        {"class_id": 2, "confidence": 0.9, "bounding_box": [0, 0, 10, 10]},
        {"class_id": 2, "confidence": 0.7, "bounding_box": [0, 0, 10, 10]},
        {"class_id": 0, "confidence": 0.8, "bounding_box": [20, 20, 30, 30]},
    ]
    result = score(truth, predictions, 0.5)
    counts = totals([{"score": result}, {"score": None}])
    assert counts["animal_visivel"] == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "precision": 0.5,
        "recall": 1,
        "f1": pytest.approx(2 / 3),
    }
    assert counts["agua_acumulada"]["fn"] == 1
    assert counts["agua_acumulada"]["precision"] is None
    assert counts["obstrucao_visivel"]["fp"] == 1
    assert counts["obstrucao_visivel"]["recall"] is None


def test_shared_pipeline_benchmark_preserves_rgb_input_and_original_box_coordinates() -> (
    None
):
    from drenavision.core.schemas import Detection

    image = np.full((20, 30, 3), [1, 2, 3], dtype=np.uint8)

    class Pipeline:
        def predict(self, rgb):
            assert rgb.flags.c_contiguous
            assert rgb[0, 0].tolist() == [3, 2, 1]
            assert rgb.shape == image.shape
            return SimpleNamespace(
                detections=[
                    Detection(
                        class_id=1,
                        class_name="agua_acumulada",
                        confidence=0.97,
                        bounding_box=(2.5, 3.5, 14.5, 16.5),
                    )
                ],
                annotated_image=np.full((20, 30, 3), [7, 8, 9], dtype=np.uint8),
            )

    predictions, annotated = predict_pipeline(Pipeline(), image)
    assert predictions == [
        {"class_id": 1, "confidence": 0.97, "bounding_box": [2.5, 3.5, 14.5, 16.5]}
    ]
    assert annotated[0, 0].tolist() == [9, 8, 7]
    assert image[0, 0].tolist() == [1, 2, 3]
