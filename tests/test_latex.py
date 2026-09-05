import pytest
from pydantic import ValidationError

from automatic_lecture_tex.latex import render_lecture
from automatic_lecture_tex.schemas import (
    BlockType,
    ChunkNotes,
    CorrectionRecord,
    LectureIR,
    NoteBlock,
)


def test_render_lecture_uses_deterministic_environments():
    ir = LectureIR(
        lecture_id="l1",
        title="Лекция 1",
        chunks=[
            ChunkNotes(
                chunk_id="c1",
                start=0,
                end=10,
                section_title="Гильбертовы пространства",
                blocks=[
                    NoteBlock(type=BlockType.DEFINITION, latex=r"Пусть $H$ --- пространство."),
                    NoteBlock(type=BlockType.EQUATION, latex=r"\langle x,y\rangle=0"),
                ],
            )
        ],
    )
    text = render_lecture(ir)
    assert r"\begin{definition}" in text
    assert r"\section{Гильбертовы пространства}" in text
    assert r"\langle x,y\rangle=0" in text


def test_render_lecture_includes_correction_audit_comments():
    ir = LectureIR(
        lecture_id="l1",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="Section",
                blocks=[],
                corrections=[
                    CorrectionRecord(
                        original="икс це",
                        corrected=r"X \to \mathbb{C}",
                        reason="Visible on the board.",
                        basis="visual",
                        confidence=0.9,
                    )
                ],
            )
        ],
    )

    text = render_lecture(ir)

    assert "% Reconstruction corrections:" in text
    assert "икс це" in text
    assert "confidence=0.90" in text


def test_note_block_rejects_raw_latex_environment():
    with pytest.raises(ValidationError):
        NoteBlock(type=BlockType.PARAGRAPH, latex=r"\begin{cases}x=1\end{cases}")
