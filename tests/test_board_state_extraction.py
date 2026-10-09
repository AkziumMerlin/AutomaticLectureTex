from __future__ import annotations

import pytest

from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.knowledge import (
    GeneratedBoardStateLine,
    GeneratedBoardStateWindow,
    board_state_delta,
    board_state_delta_to_observations,
)
from automatic_lecture_tex.schemas import LectureChunk, ObservationKind


def _chunk() -> LectureChunk:
    return LectureChunk(
        id="chunk_0001",
        start=30.0,
        end=60.0,
        segment_ids=["s1"],
        text="",
    )


def test_board_state_delta_emits_only_new_completed_entries() -> None:
    previous = GeneratedBoardStateWindow(
        lines=[
            GeneratedBoardStateLine(
                region="left",
                latex=r"x=1",
                complete=True,
            ),
            GeneratedBoardStateLine(
                region="center",
                latex=r"y=",
                complete=False,
            ),
        ]
    )
    current = GeneratedBoardStateWindow(
        lines=[
            GeneratedBoardStateLine(
                region="left",
                latex=r"x = 1",
                complete=True,
            ),
            GeneratedBoardStateLine(
                region="center",
                latex=r"y=2",
                complete=True,
            ),
            GeneratedBoardStateLine(
                region="right",
                literal_text="Теорема Хана--Банаха",
                complete=True,
            ),
        ]
    )

    batch, removed = board_state_delta_to_observations(
        _chunk(),
        previous=previous,
        current=current,
    )

    assert len(batch.observations) == 2
    assert batch.observations[0].kind == ObservationKind.EQUATION
    assert batch.observations[0].latex == r"y=2"
    assert batch.observations[1].kind == ObservationKind.REMARK
    assert batch.observations[1].text == "Теорема Хана--Банаха"
    assert [item.id for item in batch.observations] == [
        "obs_chunk_0001_000",
        "obs_chunk_0001_001",
    ]
    assert len(removed) == 1
    assert removed[0].latex == r"y="


def test_incomplete_board_entry_stays_in_snapshot_but_not_semantic_delta() -> None:
    current = GeneratedBoardStateWindow(
        lines=[
            GeneratedBoardStateLine(
                region="center",
                latex=r"V(x,\varphi,\varepsilon)=",
                complete=False,
                legibility="clear",
            )
        ]
    )

    batch, removed = board_state_delta_to_observations(
        _chunk(),
        previous=None,
        current=current,
    )

    assert batch.observations == []
    assert removed == []
    assert len(current.lines) == 1


def test_board_state_diff_ignores_latex_whitespace() -> None:
    previous = GeneratedBoardStateWindow(
        lines=[GeneratedBoardStateLine(region="left", latex=r"x + y = z")]
    )
    current = GeneratedBoardStateWindow(
        lines=[GeneratedBoardStateLine(region="left", latex=r"x+y=z")]
    )

    added, removed = board_state_delta(previous, current)

    assert added == []
    assert removed == []


def test_board_state_backend_requires_mutable_graph_state_mode() -> None:
    config = NotesConfig(
        architecture="state",
        state_semantic_backend="mutable_graph",
        window_evidence_backend="native_video_board_state",
    )
    assert config.native_video_board_chunk_seconds == 30.0

    with pytest.raises(ValueError, match="requires architecture=state"):
        NotesConfig(
            architecture="knowledge",
            window_evidence_backend="native_video_board_state",
        )
