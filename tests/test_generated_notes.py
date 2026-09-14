import pytest
from pydantic import ValidationError

from automatic_lecture_tex.generated_notes import (
    GeneratedChunkNotes,
    GeneratedEquationNoteBlock,
    GeneratedTextNoteBlock,
)


def test_generated_text_schema_requires_non_empty_latex():
    schema = GeneratedTextNoteBlock.model_json_schema()

    assert schema["properties"]["latex"]["minLength"] == 1


def test_generated_text_block_rejects_empty_and_whitespace_latex():
    with pytest.raises(ValidationError):
        GeneratedTextNoteBlock(type="definition", latex="")
    with pytest.raises(ValidationError):
        GeneratedTextNoteBlock(type="definition", latex="   ")


def test_generated_text_block_does_not_allow_figure_type():
    with pytest.raises(ValidationError):
        GeneratedTextNoteBlock(type="figure", latex="figure body")


def test_generated_equation_requires_bare_math_fragment():
    equation = GeneratedEquationNoteBlock(type="equation", latex="x ∈ X")
    assert equation.latex == r"x \in  X"

    with pytest.raises(ValidationError):
        GeneratedEquationNoteBlock(type="equation", latex="$x=1$")
    with pytest.raises(ValidationError):
        GeneratedEquationNoteBlock(type="equation", latex="Пусть $x=1$.")


def test_generated_chunk_converts_typed_union_to_final_ir():
    generated = GeneratedChunkNotes(
        section_title="Section",
        blocks=[
            {
                "type": "definition",
                "title": "Definition",
                "latex": "Полное тело определения.",
                "source_evidence_ids": ["obs_001"],
            },
            {
                "type": "equation",
                "latex": r"\|f\|=\|u\|",
                "source_evidence_ids": ["obs_002"],
            },
        ],
    )

    notes = generated.to_chunk_notes()

    assert notes.blocks[0].latex == "Полное тело определения."
    assert notes.blocks[0].source_evidence_ids == ["obs_001"]
    assert notes.blocks[1].type == "equation"
