from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from ferramentas.scripts.montar_demonstracao import (
    CapturedFrame,
    chapter_timeline,
    load_frames,
    select_timeline,
    validate_chapters,
    validate_segments,
    write_narration_timeline,
)


def test_real_timestamps_keep_processing_gap_and_inclusive_bounds() -> None:
    frames = [
        CapturedFrame(i, Path(f"{i}.jpg"), t)
        for i, t in enumerate([0, 125, 250, 1500, 1625])
    ]
    timeline = select_timeline(frames, 125, 1625)
    assert [frame.index for frame, _ in timeline] == [1, 2, 3, 4]
    assert [seconds for _, seconds in timeline] == [0.125, 1.25, 0.125, 0]
    assert sum(seconds for _, seconds in timeline) == pytest.approx(1.5)


def test_chapter_may_start_between_frames_without_speeding_up() -> None:
    frames = [
        CapturedFrame(i, Path(f"{i}.jpg"), t) for i, t in enumerate([0, 125, 250, 375])
    ]
    timeline = select_timeline(frames, 100, 350)
    assert sum(seconds for _, seconds in timeline) == pytest.approx(0.25)
    assert timeline[0][0].t_ms == 125


def test_capture_cannot_extend_past_available_frames() -> None:
    frames = [CapturedFrame(0, Path("0.jpg"), 0), CapturedFrame(1, Path("1.jpg"), 125)]
    with pytest.raises(ValueError, match="ultrapassa"):
        select_timeline(frames, 0, 5000)


def test_timestamp_order_and_files_are_checked(tmp_path: Path) -> None:
    (tmp_path / "frame.jpg").write_bytes(b"JPEG fixture existence only")
    rows = [
        {"index": 1, "path": "frame.jpg", "t_ms": 500},
        {"index": 2, "path": "frame.jpg", "t_ms": 200},
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="crescentes"):
        load_frames(index)


def test_cuts_only_between_chapters_no_overlapping_replay() -> None:
    validate_chapters(
        [
            {"id": "a", "start_ms": 0, "end_ms": 1000},
            {"id": "b", "start_ms": 5000, "end_ms": 6000},
        ]
    )
    with pytest.raises(ValueError, match="sobrepostos"):
        validate_chapters(
            [
                {"id": "a", "start_ms": 0, "end_ms": 1000},
                {"id": "b", "start_ms": 900, "end_ms": 2000},
            ]
        )


def test_explicit_blocks_remove_only_outside_gaps_and_keep_processing() -> None:
    timestamps = [0, 250, 1500, 1750, 5000, 5250, 6000, 6250]
    frames = [CapturedFrame(i, Path(f"{i}.jpg"), t) for i, t in enumerate(timestamps)]
    chapter = {
        "id": "video_agua",
        "start_ms": 100,
        "end_ms": 6500,
        "process_start_ms": 300,
        "result_ms": 1800,
    }
    blocks = [{"start_ms": 0, "end_ms": 2000}, {"start_ms": 5000, "end_ms": 7000}]
    timeline, selected_blocks, origin = chapter_timeline(frames, chapter, blocks)
    assert origin == "global_explicit"
    assert selected_blocks == [
        {"start_ms": 100, "end_ms": 2000},
        {"start_ms": 5000, "end_ms": 6500},
    ]
    assert sum(seconds for _, seconds in timeline) == pytest.approx(3.4)
    # Mantém os 1,25s sem frame novo e os 0,15s até a primeira amostra.
    assert timeline[0][1] == pytest.approx(1.4)


def test_block_cut_cannot_remove_processing_wait() -> None:
    frames = [
        CapturedFrame(i, Path(f"{i}.jpg"), t)
        for i, t in enumerate([0, 250, 5000, 5250])
    ]
    chapter = {
        "id": "video_rato",
        "start_ms": 0,
        "end_ms": 5500,
        "process_start_ms": 200,
        "result_ms": 5100,
        "segments": [
            {"start_ms": 0, "end_ms": 500},
            {"start_ms": 5000, "end_ms": 5500},
        ],
    }
    with pytest.raises(ValueError, match="espera do processamento"):
        chapter_timeline(frames, chapter, None)


def test_gap_inference_is_limited_to_audited_legacy_chapters() -> None:
    frames = [
        CapturedFrame(i, Path(f"{i}.jpg"), t)
        for i, t in enumerate([0, 125, 250, 5000, 5125, 5250])
    ]
    legacy = {"id": "foto_bueiro", "start_ms": 0, "end_ms": 5375}
    timeline, blocks, origin = chapter_timeline(frames, legacy, [])
    assert len(blocks) == 2
    assert sum(seconds for _, seconds in timeline) == pytest.approx(0.75)
    assert "legacy_inference" in origin
    later = {**legacy, "id": "video_rato"}
    with pytest.raises(ValueError, match="nenhum bloco explícito"):
        chapter_timeline(frames, later, [])
    continuous, _, origin = chapter_timeline(frames, later, None)
    assert origin == "continuous_chapter_range"
    assert sum(seconds for _, seconds in continuous) == pytest.approx(5.375)


def test_capture_blocks_must_be_ordered_and_nonoverlapping() -> None:
    with pytest.raises(ValueError, match="sobrepostos"):
        validate_segments(
            [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 900, "end_ms": 2000}]
        )


def test_final_audio_matches_cumulative_video_timeline_without_segment_padding(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "fala.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(b"\x01\x00" * 220)
    output = tmp_path / "narracao_completa.wav"
    durations = [1 / 24] * 29
    write_narration_timeline(output, [audio] * len(durations), durations)
    with wave.open(str(output), "rb") as stream:
        # Arredondamento cumulativo, sem somar padding/arredondamento por trecho.
        assert stream.getnframes() == round(sum(durations) * 22050)
        samples = stream.readframes(stream.getnframes())
    second_start = round(durations[0] * 22050)
    assert samples[:440] == b"\x01\x00" * 220
    assert samples[440 : second_start * 2] == b"\x00" * (second_start - 220) * 2
    assert samples[second_start * 2 : second_start * 2 + 440] == b"\x01\x00" * 220
