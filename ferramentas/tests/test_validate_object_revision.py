from __future__ import annotations

import pytest

from ferramentas.scripts.validate_object_revision import (
    assert_water_equal,
    presence_metrics,
)


def test_unknown_presence_is_excluded_rather_than_counted_as_negative():
    animal = {"class_id": 2}
    rows = [
        {
            "presence": {"2": True},
            "baseline": {"detections": [animal]},
            "candidate": {"detections": []},
        },
        {
            "presence": {"2": False},
            "baseline": {"detections": [animal]},
            "candidate": {"detections": []},
        },
        {
            "presence": {"2": None},
            "baseline": {"detections": [animal]},
            "candidate": {"detections": [animal]},
        },
    ]
    baseline, candidate = (
        presence_metrics(rows, name, 2) for name in ("baseline", "candidate")
    )
    assert {key: baseline[key] for key in ("tp", "fp", "tn", "fn", "unknown")} == {
        "tp": 1,
        "fp": 1,
        "tn": 0,
        "fn": 0,
        "unknown": 1,
    }
    assert {key: candidate[key] for key in ("tp", "fp", "tn", "fn", "unknown")} == {
        "tp": 0,
        "fp": 0,
        "tn": 1,
        "fn": 1,
        "unknown": 1,
    }
    assert candidate["negative_images"] == 1
    assert candidate["positive_images"] == 1


def test_water_invariance_ignores_object_changes_but_rejects_water_drift():
    water = {"class_id": 1, "confidence": 0.98, "bounding_box": [0, 2, 20, 30]}
    baseline = [water, {"class_id": 0}]
    assert_water_equal(baseline, [{"class_id": 2}, water])
    assert_water_equal([water], [{**water, "confidence": 0.980005}])
    with pytest.raises(AssertionError, match="Water detections changed"):
        assert_water_equal([water], [{**water, "confidence": 0.98002}])
    with pytest.raises(AssertionError, match="Water detections changed"):
        assert_water_equal([water], [])
