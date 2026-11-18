from __future__ import annotations

import copy
import hashlib
from uuid import uuid4

import httpx
import pytest

from ferramentas.scripts.validate_running_product import (
    EXPECTED_IMAGE_VERSION,
    download_video,
    frame_selection,
    validate_video_manifest,
    video_presence,
)


def test_manifest_requires_shared_instances_frozen_models_and_all_frame_forwards():
    event_id = str(uuid4())
    frozen = {
        "objects": {"model_version": "objects-fixed"},
        "water": {},
        "frozen_at_utc": "2026-11-18T00:00:00Z",
    }
    frames = [
        {
            "frame_index": index,
            "objects": {
                "model_version": "objects-fixed",
                "processing": "forward_on_current_original_frame",
            },
        }
        for index in range(2)
    ]
    manifest = {
        "status": "completed",
        "base_pipeline_version": EXPECTED_IMAGE_VERSION,
        "input_sha256": "source-hash",
        "created_at_utc": "2026-11-18T01:00:00Z",
        "event_id": event_id,
        "video_url": f"/vision/videos/{event_id}",
        "pipeline": {
            "model_ownership": "borrowed API water/object/verifier instances",
            "objects": frozen["objects"],
            "water": {},
        },
        "water_temporal_policy": {"confirmation_seconds": 0.5},
        "inference": {"video_water_confirmation_seconds": 0.5},
        "frames": frames,
        "video": {"processed_frames": 2},
    }
    entry = {"sha256": "source-hash", "frames": 2}
    assert validate_video_manifest(manifest, entry, frozen) == event_id
    changed = copy.deepcopy(manifest)
    changed["pipeline"].pop("model_ownership")
    with pytest.raises(ValueError, match="shared API"):
        validate_video_manifest(changed, entry, frozen)
    changed = copy.deepcopy(manifest)
    changed["frames"][1]["objects"]["processing"] = "cached_previous_frame"
    with pytest.raises(ValueError, match="current-frame"):
        validate_video_manifest(changed, entry, frozen)


def test_video_download_checks_real_range_bytes_and_output_digest(tmp_path):
    payload = b"actual-video-fixture-bytes"
    calls = []

    def handler(request):
        calls.append(request)
        if "range" in request.headers:
            return httpx.Response(
                206,
                content=payload,
                headers={"content-range": f"bytes 0-{len(payload) - 1}/{len(payload)}"},
            )
        return httpx.Response(
            200, content=payload, headers={"content-type": "video/mp4"}
        )

    manifest = {"output_sha256": hashlib.sha256(payload).hexdigest()}
    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = download_video(client, "uuid", manifest, tmp_path / "result.mp4")
    assert result["status_code"] == 206
    assert result["verified_bytes"] == len(payload)
    assert len(calls) == 2


def test_review_frames_prefer_detected_examples_but_empty_frame_counts_stay_complete():
    records = [
        {"frame_index": i, "detections": [{"class_id": 2}] if i in {1, 3, 5} else []}
        for i in range(6)
    ]
    assert frame_selection(records) == [1, 3, 5]
    presence = video_presence(records)
    assert presence["frames_with_detection_by_class"]["animal_visivel"] == 3
    assert presence["empty_frames"] == 3
    assert presence["total_frames"] == 6
