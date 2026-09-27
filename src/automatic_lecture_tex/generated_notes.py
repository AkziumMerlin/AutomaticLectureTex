from __future__ import annotations

import re
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


ObservationStateAction = Literal["keep", "replace", "reject"]


class GeneratedStateSectionBlock(BaseModel):
    """Minimal final-writer block: prose/LaTeX only, with provenance assigned host-side."""

    type: GeneratedBlockType
    title: str | None = None
    latex: str = Field(min_length=1)

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, value: str | None) -> str | None:
        return strip_control_chars(value) if value is not None else None

    @field_validator("latex")
    @classmethod
    def sanitize_latex(cls, value: str) -> str:
        value = strip_control_chars(value)
        if not value.strip():
            raise ValueError("generated state-section block must contain non-whitespace content")
        return _reject_environments(
            value,
            _RENDERER_BLOCK_ENVIRONMENTS,
            context="generated state-section blocks",
        )

    @model_validator(mode="after")
    def normalize_equation_type(self) -> GeneratedStateSectionBlock:
        if self.type != BlockType.EQUATION:
            return self
        _reject_environments(
            self.latex,
            _DISPLAY_MATH_ENVIRONMENTS,
            context="generated state-section equation blocks",
        )
        normalized = normalize_math_unicode(self.latex)
        if looks_like_math_fragment(normalized):
            self.latex = normalized
            return self
        self.type = BlockType.PARAGRAPH
        return self

    def to_note_block(self, *, source_evidence_ids: list[str]) -> NoteBlock:
        return NoteBlock(
            type=self.type,
            title=self.title,
            latex=self.latex,
            source_claim_ids=[],
            source_evidence_ids=list(source_evidence_ids),
        )


class GeneratedStateSectionNotes(BaseModel):
    """Minimal structured schema for synthesis from already-canonical state."""

    blocks: list[GeneratedStateSectionBlock] = Field(default_factory=list)


class GeneratedObservationStatePatch(BaseModel):
    """One transactional update to the CURRENT observation.

    The model never rewrites accepted history. keep preserves the mathematical event while
    providing clean semantic prose; replace provides corrected semantic prose/LaTeX; reject
    removes CURRENT from canonical synthesis when local evidence cannot support it.
    """

    action: ObservationStateAction
    semantic_text: str | None = None
    replacement_latex: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = ""
    unresolved: list[str] = Field(default_factory=list)

    @field_validator("semantic_text")
    @classmethod
    def sanitize_semantic_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = strip_control_chars(value).strip()
        if not value:
            return None
        if "$" in value or re.search(r"\\(?:[A-Za-z]+|[()[\]{}|])", value):
            raise ValueError(
                "semantic_text must be plain prose without LaTeX commands or math delimiters"
            )
        return value

    @field_validator("replacement_latex")
    @classmethod
    def sanitize_replacement_latex(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = strip_control_chars(value).strip()
        return value or None

    @field_validator("evidence_refs")
    @classmethod
    def clean_evidence_refs(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item and item.strip()))

    @model_validator(mode="after")
    def validate_transaction(self) -> GeneratedObservationStatePatch:
        if self.action in {"keep", "replace"} and self.semantic_text is None:
            raise ValueError(f"{self.action} requires semantic_text")
        if self.action == "keep":
            if self.replacement_latex is not None:
                raise ValueError("keep must not carry replacement_latex")
            return self
        if not self.reason.strip():
            raise ValueError("replace/reject must explain the local evidence conflict")
        if not self.evidence_refs:
            raise ValueError("replace/reject must cite local evidence refs")
        if self.action == "reject" and (
            self.semantic_text is not None or self.replacement_latex is not None
        ):
            raise ValueError("reject must not carry semantic/replacement fields")
        return self

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



class GeneratedFormulaObservationResolution(GeneratedObservationResolution):
    """Resolved observation whose mathematical formula must remain explicit."""

    latex: str = Field(min_length=1)

    @field_validator("latex")
    @classmethod
    def require_resolution_latex(cls, value: str) -> str:
        value = strip_control_chars(value).strip()
        if not value:
            raise ValueError("resolved formula observation latex must be non-empty")
        return value


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
