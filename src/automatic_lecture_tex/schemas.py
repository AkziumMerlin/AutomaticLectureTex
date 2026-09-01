from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    id: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[TranscriptWord] = Field(default_factory=list)


class Transcript(BaseModel):
    lecture_id: str
    language: str | None = None
    segments: list[TranscriptSegment]


class LectureChunk(BaseModel):
    id: str
    start: float
    end: float
    segment_ids: list[str]
    text: str


class VisualRequest(BaseModel):
    id: str
    timestamp: float
    reason: str
    question: str
    priority: int = Field(default=1, ge=1, le=5)


class ChunkAnalysis(BaseModel):
    visual_requests: list[VisualRequest] = Field(default_factory=list)


class VisualKind(StrEnum):
    EQUATION = "equation"
    NOTATION = "notation"
    DIAGRAM = "diagram"
    SLIDE_TEXT = "slide_text"
    NONE = "none"


class VisualEvidence(BaseModel):
    request_id: str = ""
    kind: VisualKind = VisualKind.NONE
    latex: str | None = None
    description: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_figure_in_notes: bool = False
    best_frame_index: int | None = None
    asset_path: str | None = None


class BlockType(StrEnum):
    PARAGRAPH = "paragraph"
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    PROOF = "proof"
    EXAMPLE = "example"
    REMARK = "remark"
    EQUATION = "equation"
    FIGURE = "figure"
    EXERCISE = "exercise"


class NoteBlock(BaseModel):
    type: BlockType
    title: str | None = None
    latex: str = ""
    asset_path: str | None = None
    caption: str | None = None


class NotationItem(BaseModel):
    latex: str
    meaning: str


class ChunkNotes(BaseModel):
    chunk_id: str = ""
    start: float = 0.0
    end: float = 0.0
    section_title: str
    blocks: list[NoteBlock]
    notation: list[NotationItem] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class LectureIR(BaseModel):
    lecture_id: str
    title: str
    chunks: list[ChunkNotes]


class BlockReviewDecision(BaseModel):
    status: str
    issue: str | None = None
    suggested_patch: str | None = None
    source_excerpt_ids: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    block_index: int
    status: str
    issue: str | None = None
    suggested_patch: str | None = None
    source_excerpt_ids: list[str] = Field(default_factory=list)


class ReviewReport(BaseModel):
    lecture_id: str
    findings: list[ReviewFinding]


class ExtractedFrame(BaseModel):
    timestamp: float
    path: Path
