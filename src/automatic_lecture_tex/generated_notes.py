from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from .schemas import (
    BlockType,
    ChunkNotes,
    CorrectionRecord,
    NotationItem,
    NoteBlock,
    _DISPLAY_MATH_ENVIRONMENTS,
    _RENDERER_BLOCK_ENVIRONMENTS,
    _reject_environments,
)
from .tex_safety import looks_like_math_fragment, normalize_math_unicode, strip_control_chars

GeneratedTextBlockType = Literal[
    BlockType.PARAGRAPH,
    BlockType.DEFINITION,
    BlockType.THEOREM,
    BlockType.LEMMA,
    BlockType.PROPOSITION,
    BlockType.COROLLARY,
    BlockType.PROOF,
    BlockType.EXAMPLE,
    BlockType.REMARK,
    BlockType.EXERCISE,
]


class _GeneratedBlockBase(BaseModel):
    title: str | None = None
    source_claim_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, value: str | None) -> str | None:
        return strip_control_chars(value) if value is not None else None


class GeneratedTextNoteBlock(_GeneratedBlockBase):
    """LLM-facing prose/theorem block. Math may appear inline inside the body."""

    type: GeneratedTextBlockType
    latex: str = Field(min_length=1)

    @field_validator("latex")
    @classmethod
    def reject_blank_or_renderer_owned_latex(cls, value: str) -> str:
        value = strip_control_chars(value)
        if not value.strip():
            raise ValueError("generated note block latex must contain non-whitespace content")
        return _reject_environments(
            value,
            _RENDERER_BLOCK_ENVIRONMENTS,
            context="generated note blocks",
        )

    def to_note_block(self) -> NoteBlock:
        return NoteBlock(
            type=self.type,
            title=self.title,
            latex=self.latex,
            source_claim_ids=list(self.source_claim_ids),
            source_evidence_ids=list(self.source_evidence_ids),
        )


class GeneratedEquationNoteBlock(_GeneratedBlockBase):
    """LLM-facing display equation.

    The renderer owns display delimiters, so the model must return a bare mathematical fragment.
    Prose belongs in a paragraph/theorem/proof block instead of an equation block.
    """

    type: Literal[BlockType.EQUATION]
    latex: str = Field(min_length=1)

    @field_validator("latex")
    @classmethod
    def require_bare_math_fragment(cls, value: str) -> str:
        value = normalize_math_unicode(value)
        if not value.strip():
            raise ValueError("generated equation must contain non-whitespace mathematics")
        _reject_environments(
            value,
            _RENDERER_BLOCK_ENVIRONMENTS | _DISPLAY_MATH_ENVIRONMENTS,
            context="generated equation blocks",
        )
        if not looks_like_math_fragment(value):
            raise ValueError(
                "generated equation must be bare math without prose or $/\\[ display delimiters"
            )
        return value

    def to_note_block(self) -> NoteBlock:
        return NoteBlock(
            type=BlockType.EQUATION,
            title=self.title,
            latex=self.latex,
            source_claim_ids=list(self.source_claim_ids),
            source_evidence_ids=list(self.source_evidence_ids),
        )


GeneratedNoteBlock = Annotated[
    GeneratedTextNoteBlock | GeneratedEquationNoteBlock,
    Field(discriminator="type"),
]


class GeneratedChunkNotes(BaseModel):
    """Structured-output schema used by the episode writer only."""

    chunk_id: str = ""
    start: float = 0.0
    end: float = 0.0
    section_title: str
    blocks: list[GeneratedNoteBlock]
    notation: list[NotationItem] = Field(default_factory=list)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    def to_chunk_notes(self) -> ChunkNotes:
        return ChunkNotes(
            chunk_id=self.chunk_id,
            start=self.start,
            end=self.end,
            section_title=strip_control_chars(self.section_title),
            blocks=[block.to_note_block() for block in self.blocks],
            notation=list(self.notation),
            corrections=list(self.corrections),
            unresolved=[strip_control_chars(item) for item in self.unresolved],
        )
