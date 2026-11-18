from types import SimpleNamespace

import numpy as np
import pytest

from drenavision.core.schemas import Detection
from drenavision.inference.contracts import PredictionResult
from drenavision.inference.errors import ModelUnavailableError
from drenavision.inference.runtime import InspectionPredictor, merge_object_detections


def detection(class_id, box, confidence=0.8):
    return Detection(
        class_id=class_id,
        class_name={0: "obstrucao_visivel", 1: "agua_acumulada", 2: "animal_visivel"}[class_id],
        confidence=confidence,
        bounding_box=box,
    )


def test_animal_replace_retains_whole_learned_box_and_distinct_litter():
    partial = detection(2, (20, 30, 40, 40))
    whole = detection(2, (15, 10, 45, 45))
    master_litter = detection(0, (60, 20, 70, 40))
    other_litter = detection(0, (80, 20, 90, 40))
    water = detection(1, (0, 50, 100, 60))
    result = merge_object_detections(
        [partial, master_litter, water], [whole, other_litter], mode="animal_replace"
    )
    assert result == [water, whole, other_litter, master_litter]
    assert all(item in [whole, other_litter, master_litter, water] for item in result)


def test_fusion_does_not_merge_different_classes_or_invent_union_boxes():
    litter = detection(0, (10, 10, 30, 30))
    duplicate = detection(0, (11, 10, 30, 30), 0.4)
    animal = detection(2, (11, 10, 30, 30))
    result = merge_object_detections([litter, animal], [duplicate], mode="augment")
    assert result == [duplicate, animal]
    assert duplicate.bounding_box == (11, 10, 30, 30)
    assert merge_object_detections([litter, animal], [], mode="replace") == []
    with pytest.raises(ValueError):
        merge_object_detections([], [], mode="unknown")


def test_missing_configured_object_specialist_cannot_silently_use_old_model(tmp_path):
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    checkpoint = tmp_path / "master.pt"
    checkpoint.write_bytes(b"fixture")
    master = SimpleNamespace(
        is_ready=True, detail="ready", model_path=checkpoint,
        image_size=640, confidence_threshold=.35, iou_threshold=.45,
        predict=lambda _: PredictionResult([], image, 1, "master", 96, 64, 640),
    )

    def unavailable(_):
        raise ModelUnavailableError("specialist missing")

    objects = SimpleNamespace(
        is_ready=False, detail="specialist missing", predict=unavailable, provenance={}
    )
    predictor = InspectionPredictor(master, objects=objects)
    assert not predictor.is_ready
    assert predictor.detail == "specialist missing"
    with pytest.raises(ModelUnavailableError):
        predictor.predict(image)


def test_profile_changes_with_specialist_and_fusion_policy(tmp_path):
    checkpoint = tmp_path / "master.pt"
    checkpoint.write_bytes(b"fixture")
    master = SimpleNamespace(
        model_path=checkpoint, image_size=640,
        confidence_threshold=.35, iou_threshold=.45,
    )
    objects = SimpleNamespace(provenance={"weights": "specialist", "threshold": .35})
    baseline = InspectionPredictor(master).model_version
    first = InspectionPredictor(master, objects=objects).model_version
    second = InspectionPredictor(master, objects=objects, object_mode="replace").model_version
    assert len({baseline, first, second}) == 3
