from automatic_lecture_tex.latex import render_lecture
from automatic_lecture_tex.schemas import BlockType, ChunkNotes, LectureIR, NoteBlock


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
