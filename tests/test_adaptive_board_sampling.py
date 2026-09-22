from pathlib import Path

from PIL import Image, ImageDraw

from automatic_lecture_tex.frame_selection import (
    board_state_change_score,
    select_board_state_frames,
)
from automatic_lecture_tex.schemas import ExtractedFrame, LectureChunk
from automatic_lecture_tex.vision import board_change_probe_times


def _image(path: Path, *, changed: bool) -> None:
    image = Image.new("RGB", (160, 90), "black")
    if changed:
        draw = ImageDraw.Draw(image)
        draw.line((20, 20, 140, 70), fill="white", width=4)
        draw.rectangle((60, 30, 110, 55), outline="white", width=3)
    image.save(path)


def test_board_change_probe_times_cover_interval_without_llm_confidence():
    chunk = LectureChunk(
        id="chunk",
        start=10.0,
        end=50.0,
        segment_ids=["s"],
        text="x",
    )
    times = board_change_probe_times(chunk, probe_seconds=8.0, max_probe_frames=48)
    assert times[0] == 10.0
    assert times[-1] == 50.0
    assert 2 < len(times) <= 48
    assert times == sorted(times)


def test_board_state_selector_keeps_only_real_visual_change(tmp_path):
    first_path = tmp_path / "first.png"
    same_path = tmp_path / "same.png"
    changed_path = tmp_path / "changed.png"
    _image(first_path, changed=False)
    _image(same_path, changed=False)
    _image(changed_path, changed=True)

    assert board_state_change_score(first_path, same_path) == 0.0
    assert board_state_change_score(first_path, changed_path) > 0.01

    frames = [
        ExtractedFrame(timestamp=0.0, path=first_path),
        ExtractedFrame(timestamp=8.0, path=same_path),
        ExtractedFrame(timestamp=16.0, path=changed_path),
    ]
    selected = select_board_state_frames(
        frames,
        max_states=5,
        change_threshold=0.01,
        min_gap_seconds=4.0,
    )

    assert [item.timestamp for item in selected] == [0.0, 16.0]


def test_board_state_selector_does_not_add_unchanged_last_probe(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"same_{index}.png"
        _image(path, changed=False)
        paths.append(path)

    selected = select_board_state_frames(
        [
            ExtractedFrame(timestamp=0.0, path=paths[0]),
            ExtractedFrame(timestamp=8.0, path=paths[1]),
            ExtractedFrame(timestamp=16.0, path=paths[2]),
        ],
        max_states=5,
        change_threshold=0.01,
        min_gap_seconds=4.0,
    )

    assert [item.timestamp for item in selected] == [0.0]



def test_board_state_selector_spreads_budget_across_all_detected_changes(tmp_path):
    frames = []
    for index in range(9):
        path = tmp_path / f"state_{index}.png"
        image = Image.new("RGB", (160, 90), "black")
        draw = ImageDraw.Draw(image)
        for line_index in range(index + 1):
            y = 8 + 8 * line_index
            draw.line((15, y, 145, y), fill="white", width=2)
        image.save(path)
        frames.append(ExtractedFrame(timestamp=float(index * 10), path=path))

    selected = select_board_state_frames(
        frames,
        max_states=5,
        change_threshold=0.001,
        min_gap_seconds=1.0,
    )

    timestamps = [item.timestamp for item in selected]
    assert len(timestamps) == 5
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 80.0
    assert timestamps == sorted(timestamps)


def test_board_state_selector_keeps_sparse_new_formula_on_latest_probe(tmp_path):
    first_path = tmp_path / "board_before.png"
    latest_path = tmp_path / "board_after.png"

    Image.new("RGB", (320, 180), "black").save(first_path)
    latest = Image.new("RGB", (320, 180), "black")
    draw = ImageDraw.Draw(latest)
    # A small newly written relation occupies only a tiny fraction of the whole board.
    draw.line((230, 145, 300, 145), fill="white", width=2)
    draw.line((250, 137, 250, 153), fill="white", width=2)
    latest.save(latest_path)

    score = board_state_change_score(first_path, latest_path)
    assert score > 0.0

    selected = select_board_state_frames(
        [
            ExtractedFrame(timestamp=0.0, path=first_path),
            ExtractedFrame(timestamp=20.0, path=latest_path),
        ],
        max_states=5,
        # The ordinary threshold is deliberately above the measured sparse change. The latest-state
        # safeguard should still retain it at the half-threshold writing floor.
        change_threshold=score * 1.5,
        min_gap_seconds=1.0,
    )

    assert [frame.timestamp for frame in selected] == [0.0, 20.0]
