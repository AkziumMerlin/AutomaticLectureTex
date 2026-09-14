import pytest
from pydantic import ValidationError

from automatic_lecture_tex.generated_notes import GeneratedChunkNotes, GeneratedNoteBlock


def test_generated_note_schema_requires_non_empty_latex():
    schema = GeneratedNoteBlock.model_json_schema()

    assert schema["properties"]["latex"]["minLength"] == 1


def test_generated_note_block_rejects_empty_and_whitespace_latex():
    with pytest.raises(ValidationError):
        GeneratedNoteBlock(type="definition", latex="")
    with pytest.raises(ValidationError):
        GeneratedNoteBlock(type="definition", latex="   ")


def test_generated_note_block_does_not_allow_figure_type():
    with pytest.raises(ValidationError):
        GeneratedNoteBlock(type="figure", latex="figure body")


def test_generated_equation_normalizes_unicode_math():
    equation = GeneratedNoteBlock(type="equation", latex="x ∈ X")

    assert equation.type == "equation"
    assert equation.latex == r"x \in  X"


def test_generated_prose_mislabeled_as_equation_becomes_paragraph():
    block = GeneratedNoteBlock(type="equation", latex="Пусть $x=1$.")

    assert block.type == "paragraph"
    assert block.latex == "Пусть $x=1$."


def test_generated_chunk_converts_to_final_ir():
    generated = GeneratedChunkNotes(
        section_title="Section",
        blocks=[
            GeneratedNoteBlock(
                type="definition",
                title="Definition",
                latex="Полное тело определения.",
                source_evidence_ids=["obs_001"],
            ),
            GeneratedNoteBlock(
                type="equation",
                latex=r"\|f\|=\|u\|",
                source_evidence_ids=["obs_002"],
            ),
        ],
    )

    notes = generated.to_chunk_notes()

    assert notes.blocks[0].latex == "Полное тело определения."
    assert notes.blocks[0].source_evidence_ids == ["obs_001"]
    assert notes.blocks[1].type == "equation"
