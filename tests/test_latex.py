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


def test_render_lecture_keeps_correction_audit_out_of_tex():
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

    assert "% Reconstruction corrections:" not in text
    assert "икс це" not in text
    assert "confidence=0.90" not in text


def test_note_block_allows_internal_math_environment():
    block = NoteBlock(
        type=BlockType.PARAGRAPH,
        latex=r"Пусть $f(x)=\begin{cases}x,&x\ge0,\\-x,&x<0.\end{cases}$",
    )

    assert r"\begin{cases}" in block.latex


def test_equation_block_allows_aligned_fragment():
    block = NoteBlock(
        type=BlockType.EQUATION,
        latex=r"\begin{aligned}x&=y\\&=z\end{aligned}",
    )

    assert r"\begin{aligned}" in block.latex


def test_note_block_rejects_renderer_owned_environment():
    with pytest.raises(ValidationError, match="renderer owns document/block wrappers"):
        NoteBlock(
            type=BlockType.PARAGRAPH,
            latex=r"\begin{theorem}T\end{theorem}",
        )


def test_equation_block_rejects_outer_display_environment():
    with pytest.raises(ValidationError, match="equation note blocks"):
        NoteBlock(
            type=BlockType.EQUATION,
            latex=r"\begin{equation}x=y\end{equation}",
        )


def test_note_block_requires_latex_in_structured_schema():
    schema = NoteBlock.model_json_schema()

    assert "latex" in schema["required"]


def test_note_block_rejects_empty_renderable_content():
    with pytest.raises(ValidationError, match="non-figure note block"):
        NoteBlock(type=BlockType.THEOREM, latex="   ")


def test_figure_block_may_use_asset_instead_of_latex():
    block = NoteBlock(
        type=BlockType.FIGURE,
        latex="",
        asset_path="figures/lecture_01/board.png",
    )

    assert block.asset_path == "figures/lecture_01/board.png"


def test_empty_figure_block_is_invalid():
    with pytest.raises(ValidationError, match="figure block"):
        NoteBlock(type=BlockType.FIGURE, latex="")
