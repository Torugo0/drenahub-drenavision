from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from drenavision.core.schemas import Detection
from drenavision.inference.video import annotate_video_frame
from drenavision.inference.video_codec import VideoEncodingError, prepare_browser_video


def _top_level_atoms(path: Path) -> list[bytes]:
    atoms = []
    with path.open("rb") as stream:
        while header := stream.read(8):
            size, tag = struct.unpack(">I4s", header)
            atoms.append(tag)
            assert size >= 8
            stream.seek(size - 8, 1)
    return atoms


def test_mp4v_is_reencoded_to_h264_faststart_without_losing_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "opencv.mp4"
    output = tmp_path / "browser.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (160, 120)
    )
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((120, 160, 3), index * 15, np.uint8))
    writer.release()

    metadata = prepare_browser_video(source, output)

    assert metadata["codec"] == "h264"
    assert metadata["pixel_format"] == "yuv420p"
    atoms = _top_level_atoms(output)
    assert atoms.index(b"moov") < atoms.index(b"mdat")
    assert b"avcC" in output.read_bytes()
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(8.0)
    first_ok, first_frame = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 11)
    last_ok, last_frame = capture.read()
    capture.release()
    assert first_ok and last_ok
    assert last_frame.mean() > first_frame.mean() + 100


def test_invalid_input_leaves_existing_destination_intact(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp4"
    output = tmp_path / "previous.mp4"
    source.write_bytes(b"not a video")
    output.write_bytes(b"previous successful video")

    with pytest.raises(VideoEncodingError, match="preparar este vídeo"):
        prepare_browser_video(source, output)

    assert output.read_bytes() == b"previous successful video"
    assert not list(tmp_path.glob("*.partial.mp4"))


def test_annotations_are_visible_and_source_frame_stays_unmodified() -> None:
    frame = np.zeros((200, 300, 3), np.uint8)
    detection = Detection(
        class_id=2,
        class_name="animal_visivel",
        confidence=0.8,
        bounding_box=(30, 70, 200, 180),
    )

    annotated = annotate_video_frame(frame, [detection])

    assert not frame.any()
    assert annotated[70, 30].any()
    assert annotated[180, 100].any()
    assert annotated[:70].any()  # A readable label appears above the object.
