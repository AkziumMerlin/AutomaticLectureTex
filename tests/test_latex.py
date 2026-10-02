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


def test_render_block_title_wraps_bare_latin_math_identifiers():
    ir = LectureIR(
        lecture_id="l1",
        title="Лекция",
        chunks=[
            ChunkNotes(
                section_title="Раздел",
                blocks=[
                    NoteBlock(
                        type=BlockType.REMARK,
                        title=(
                            "Связь комплексно-линейного функционала f "
                            "с действительными составляющими u и v"
                        ),
                        latex="Содержательное замечание.",
                    )
                ],
            )
        ],
    )

    text = render_lecture(ir)

    assert (
        r"\begin{remark}[Связь комплексно-линейного функционала $f$ "
        r"с действительными составляющими $u$ и $v$]"
        in text
    )


def test_heading_math_wraps_decorated_latin_identifiers_without_touching_words():
    ir = LectureIR(
        lecture_id="l1",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="Топология на X* и пространство H",
                blocks=[
                    NoteBlock(
                        type=BlockType.PARAGRAPH,
                        title="Сходимость f_n к f",
                        latex="Текст.",
                    )
                ],
            )
        ],
    )

    text = render_lecture(ir)

    assert r"\section{Топология на $X^*$ и пространство $H$}" in text
    assert r"\paragraph{Сходимость $f_n$ к $f$}" in text
    assert r"\chapter{Lecture}" in text


def test_long_semicolon_display_is_broken_into_aligned_lines():
    ir = LectureIR(
        lecture_id="l1",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="Section",
                blocks=[
                    NoteBlock(
                        type=BlockType.PARAGRAPH,
                        latex=(
                            r"\["
                            r"u=\operatorname{Re}h,\ u:L\to\mathbb{R},\ \|u\|=\|h\|;"
                            r"\ \exists w:X\to\mathbb{R},\ w|_L=u,\ \|w\|=\|u\|;"
                            r"\ f:X\to\mathbb{C},\ f(x)=w(x)-iw(ix)"
                            r"\]"
                        ),
                    )
                ],
            )
        ],
    )

    text = render_lecture(ir)

    assert r"\begin{aligned}" in text
    assert r"u=\operatorname{Re}h" in text
    assert r"\exists w:X\to\mathbb{R}" in text
    assert r"f:X\to\mathbb{C}" in text
    assert text.count(r"\\") >= 2


def test_long_comma_display_without_alignment_uses_multlined():
    ir = LectureIR(
        lecture_id="l1",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="Section",
                blocks=[
                    NoteBlock(
                        type=BlockType.EQUATION,
                        latex=(
                            r"A_1,A_2,A_3,A_4,A_5,A_6,A_7,A_8,A_9,A_{10},"
                            r"A_{11},A_{12},A_{13},A_{14},A_{15},A_{16}"
                        ),
                    )
                ],
            )
        ],
    )

    text = render_lecture(ir)

    assert r"\begin{multlined}" in text
    assert r"\end{multlined}" in text


def test_display_layout_does_not_split_inside_fraction_or_existing_aligned():
    ir = LectureIR(
        lecture_id="l1",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="Section",
                blocks=[
                    NoteBlock(
                        type=BlockType.PARAGRAPH,
                        latex=(
                            r"\[\frac{a;b}{c;d}+\frac{e;f}{g;h}\]"
                            "\n"
                            r"\[\begin{aligned}x&=y\\&=z\end{aligned}\]"
                        ),
                    )
                ],
            )
        ],
    )

    text = render_lecture(ir)

    assert r"\frac{a;b}{c;d}+\frac{e;f}{g;h}" in text
    assert text.count(r"\begin{aligned}") == 1
