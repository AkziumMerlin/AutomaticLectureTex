from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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

GeneratedBlockType = Literal[
    BlockType.PARAGRAPH,
    BlockType.DEFINITION,
    BlockType.THEOREM,
    BlockType.LEMMA,
    BlockType.PROPOSITION,
    BlockType.COROLLARY,
    BlockType.PROOF,
    BlockType.EXAMPLE,
    BlockType.REMARK,
    BlockType.EQUATION,
    BlockType.EXERCISE,
]


class GeneratedNoteBlock(BaseModel):
    """LLM-facing block schema.

    A generated block is always substantive text/math. Asset-backed figures are host-owned and are
    represented only in the final ``NoteBlock`` IR, so the generation schema can make non-empty
    ``latex`` a JSON-Schema-level invariant instead of relying on a post-hoc model validator.
    """

    type: GeneratedBlockType
    title: str | None = None
    latex: str = Field(min_length=1)
    source_claim_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("latex")
    @classmethod
    def reject_blank_or_renderer_owned_latex(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("generated note block latex must contain non-whitespace content")
        return _reject_environments(
            value,
            _RENDERER_BLOCK_ENVIRONMENTS,
            context="generated note blocks",
        )

    @model_validator(mode="after")
    def reject_outer_equation_environment(self) -> GeneratedNoteBlock:
        if self.type == BlockType.EQUATION:
            _reject_environments(
                self.latex,
                _DISPLAY_MATH_ENVIRONMENTS,
                context="generated equation blocks",
            )
        return self

    def to_note_block(self) -> NoteBlock:
        return NoteBlock(
            type=self.type,
            title=self.title,
            latex=self.latex,
            source_claim_ids=list(self.source_claim_ids),
            source_evidence_ids=list(self.source_evidence_ids),
        )


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
            section_title=self.section_title,
            blocks=[block.to_note_block() for block in self.blocks],
            notation=list(self.notation),
            corrections=list(self.corrections),
            unresolved=list(self.unresolved),
        )
