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
from .tex_safety import looks_like_math_fragment, normalize_math_unicode, strip_control_chars

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
    """Flat LLM-facing block schema with host-side equation classification."""

    type: GeneratedBlockType
    title: str | None = None
    latex: str = Field(min_length=1)
    source_claim_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, value: str | None) -> str | None:
        return strip_control_chars(value) if value is not None else None

    @field_validator("latex")
    @classmethod
    def sanitize_latex(cls, value: str) -> str:
        value = strip_control_chars(value)
        if not value.strip():
            raise ValueError("generated note block latex must contain non-whitespace content")
        return _reject_environments(
            value,
            _RENDERER_BLOCK_ENVIRONMENTS,
            context="generated note blocks",
        )

    @model_validator(mode="after")
    def normalize_equation_type(self) -> GeneratedNoteBlock:
        if self.type != BlockType.EQUATION:
            return self
        _reject_environments(
            self.latex,
            _DISPLAY_MATH_ENVIRONMENTS,
            context="generated equation blocks",
        )
        normalized = normalize_math_unicode(self.latex)
        if looks_like_math_fragment(normalized):
            self.latex = normalized
            return self
        # A model occasionally labels prose containing inline math as an equation. Keep the content,
        # but make its renderable type honest instead of wrapping prose in another display environment.
        self.type = BlockType.PARAGRAPH
        return self

    def to_note_block(self) -> NoteBlock:
        return NoteBlock(
            type=self.type,
            title=self.title,
            latex=self.latex,
            source_claim_ids=list(self.source_claim_ids),
            source_evidence_ids=list(self.source_evidence_ids),
        )


class GeneratedObservationResolution(BaseModel):
    """Final value for exactly one chronological lecture observation.

    The text field is required. A correction record is audit metadata and must never be the sole
    carrier of the resolved state.
    """

    text: str = Field(min_length=1)
    latex: str | None = None
    correction: CorrectionRecord | None = None
    unresolved: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def sanitize_text(cls, value: str) -> str:
        value = strip_control_chars(value).strip()
        if not value:
            raise ValueError("resolved observation text must be non-empty")
        return value

    @field_validator("latex")
    @classmethod
    def sanitize_resolution_latex(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = strip_control_chars(value).strip()
        return value or None



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
