from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


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
    timestamped_text: str = ""
    asr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    low_confidence_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


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
    BOARD_SCAN = "board_scan"
    AUDIO_VIDEO = "audio_video"
    NONE = "none"


class CorrectionBasis(StrEnum):
    VISUAL = "visual"
    AUDIO_CONTEXT = "audio_context"
    MULTIMODAL = "multimodal"
    MATHEMATICAL_CONSISTENCY = "mathematical_consistency"
    NOTATION_REGISTRY = "notation_registry"


class CorrectionRecord(BaseModel):
    original: str
    corrected: str
    reason: str
    basis: CorrectionBasis
    confidence: float = Field(ge=0.0, le=1.0)


class MathOCRCandidate(BaseModel):
    backend: str
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class VisualEvidence(BaseModel):
    request_id: str = ""
    kind: VisualKind = VisualKind.NONE
    raw_latex: str | None = None
    latex: str | None = None
    description: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    math_ocr_candidates: list[MathOCRCandidate] = Field(default_factory=list)
    requires_figure_in_notes: bool = False
    best_frame_index: int | None = None
    asset_path: str | None = None
    # Raw/cropped sensor frames retained for direct multimodal reconstruction. These are working
    # artifact paths, not user-facing figure references.
    frame_paths: list[str] = Field(default_factory=list)
    frame_timestamps: list[float] = Field(default_factory=list)


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


_RENDERER_BLOCK_ENVIRONMENTS = frozenset(
    {
        "definition",
        "theorem",
        "lemma",
        "proposition",
        "corollary",
        "proof",
        "example",
        "remark",
        "exercise",
        "figure",
        "document",
    }
)
_DISPLAY_MATH_ENVIRONMENTS = frozenset(
    {
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "displaymath",
    }
)


def _reject_environments(value: str, environments: frozenset[str], *, context: str) -> str:
    for environment in environments:
        if f"\\begin{{{environment}}}" in value or f"\\end{{{environment}}}" in value:
            raise ValueError(
                f"{context} must not contain outer LaTeX environment {environment!r}; "
                "the renderer owns document/block wrappers"
            )
    return value


class NoteBlock(BaseModel):
    """A renderable node in the final lecture IR.

    Missing or ambiguous lecture content is represented by ``ChunkNotes.unresolved`` instead of an
    empty placeholder block. This keeps the IR invariant explicit: every non-figure NoteBlock has
    content that can actually be rendered.

    Mathematical fragment environments such as ``aligned``, ``cases`` and matrix environments are
    valid block contents. Only outer document/block wrappers owned by the deterministic renderer are
    forbidden.
    """

    type: BlockType
    title: str | None = None
    latex: str
    asset_path: str | None = None
    caption: str | None = None
    source_claim_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("latex")
    @classmethod
    def reject_renderer_owned_environments(cls, value: str) -> str:
        return _reject_environments(value, _RENDERER_BLOCK_ENVIRONMENTS, context="note blocks")

    @model_validator(mode="after")
    def require_renderable_content(self) -> NoteBlock:
        if self.type == BlockType.FIGURE:
            if not self.asset_path and not self.latex.strip():
                raise ValueError("figure block must contain asset_path or latex")
            return self
        if not self.latex.strip():
            raise ValueError(
                "non-figure note block must contain non-empty latex; "
                "use ChunkNotes.unresolved for unreconstructed content"
            )
        if self.type == BlockType.EQUATION:
            _reject_environments(
                self.latex,
                _DISPLAY_MATH_ENVIRONMENTS,
                context="equation note blocks",
            )
        return self


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
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class MathAuditCorrection(BaseModel):
    block_index: int = Field(ge=0)
    corrected_latex: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("corrected_latex")
    @classmethod
    def reject_renderer_owned_environments(cls, value: str) -> str:
        return _reject_environments(value, _RENDERER_BLOCK_ENVIRONMENTS, context="audit corrections")


class MathAudit(BaseModel):
    corrections: list[MathAuditCorrection] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class ObservationKind(StrEnum):
    DEFINITION = "definition"
    CLAIM = "claim"
    EQUATION = "equation"
    PROOF_STEP = "proof_step"
    EXAMPLE = "example"
    NOTATION = "notation"
    CORRECTION = "correction"
    RETRACTION = "retraction"
    TRANSITION = "transition"
    REMARK = "remark"
    UNRESOLVED = "unresolved"


class SourceStatus(StrEnum):
    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    INFERRED = "inferred"


class ClaimStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    UNRESOLVED = "unresolved"


class MathStatus(StrEnum):
    UNCHECKED = "unchecked"
    CONSISTENT = "consistent"
    SUSPICIOUS = "suspicious"
    INCORRECT = "incorrect"


class AnchorKind(StrEnum):
    TOPIC = "topic"
    DEFINITION = "definition"
    THEOREM = "theorem"
    PROOF = "proof"
    EXAMPLE = "example"
    NOTATION = "notation"
    CORRECTION = "correction"


class EpisodeKind(StrEnum):
    TOPIC = "topic"
    DEFINITION = "definition"
    THEOREM = "theorem"
    PROOF = "proof"
    EXAMPLE = "example"
    DERIVATION = "derivation"
    NOTATION = "notation"
    REMARK = "remark"


class EpisodeStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class HierarchyLevel(StrEnum):
    TOPIC = "topic"
    SUBTOPIC = "subtopic"


class LectureObservation(BaseModel):
    id: str = ""
    window_id: str = ""
    window_ids: list[str] = Field(default_factory=list)
    episode_id: str = ""
    start: float
    end: float
    kind: ObservationKind
    text: str = ""
    latex: str | None = None
    target_observation_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_status: SourceStatus = SourceStatus.OBSERVED
    evidence_refs: list[str] = Field(default_factory=list)


class WindowObservations(BaseModel):
    window_id: str = ""
    start: float = 0.0
    end: float = 0.0
    observations: list[LectureObservation] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class KnowledgeClaim(BaseModel):
    id: str = ""
    kind: ObservationKind = ObservationKind.CLAIM
    content: str
    latex: str | None = None
    scope: str = ""
    episode_id: str = ""
    status: ClaimStatus = ClaimStatus.ACTIVE
    math_status: MathStatus = MathStatus.UNCHECKED
    source_status: SourceStatus = SourceStatus.OBSERVED
    evidence_ids: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    introduced_at: float = 0.0


class SymbolRecord(BaseModel):
    id: str = ""
    symbol: str
    meaning: str
    type_hint: str = ""
    scope: str = ""
    episode_id: str = ""
    introduced_at: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    active: bool = True


class SemanticAnchor(BaseModel):
    id: str = ""
    timestamp: float
    title: str
    kind: AnchorKind = AnchorKind.TOPIC
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class SemanticEpisode(BaseModel):
    id: str = ""
    title: str
    kind: EpisodeKind = EpisodeKind.TOPIC
    start: float
    end: float
    status: EpisodeStatus = EpisodeStatus.OPEN
    observation_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    symbol_ids: list[str] = Field(default_factory=list)
    window_ids: list[str] = Field(default_factory=list)

    @property
    def anchor_kind(self) -> AnchorKind:
        mapping = {
            EpisodeKind.DEFINITION: AnchorKind.DEFINITION,
            EpisodeKind.THEOREM: AnchorKind.THEOREM,
            EpisodeKind.PROOF: AnchorKind.PROOF,
            EpisodeKind.EXAMPLE: AnchorKind.EXAMPLE,
            EpisodeKind.NOTATION: AnchorKind.NOTATION,
        }
        return mapping.get(self.kind, AnchorKind.TOPIC)


class EpisodeBoundary(BaseModel):
    before_observation_id: str
    kind: EpisodeKind = EpisodeKind.TOPIC
    title: str


class EpisodeTrackingUpdate(BaseModel):
    boundaries: list[EpisodeBoundary] = Field(default_factory=list)
    close_after_observation_ids: list[str] = Field(default_factory=list)
    symbols: list[SymbolRecord] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class HierarchyBoundary(BaseModel):
    before_episode_id: str
    level: HierarchyLevel = HierarchyLevel.TOPIC
    title: str


class EpisodeHierarchyPlan(BaseModel):
    boundaries: list[HierarchyBoundary] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class KnowledgeUpdate(BaseModel):
    """Legacy delta schema retained for cached/legacy tooling; the knowledge pipeline no longer
    lets an LLM create canonical claims or independent anchors."""

    claims: list[KnowledgeClaim] = Field(default_factory=list)
    symbols: list[SymbolRecord] = Field(default_factory=list)
    anchors: list[SemanticAnchor] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class LectureKnowledgeBase(BaseModel):
    lecture_id: str
    title: str
    observations: list[LectureObservation] = Field(default_factory=list)
    observation_aliases: dict[str, str] = Field(default_factory=dict)
    claims: list[KnowledgeClaim] = Field(default_factory=list)
    symbols: list[SymbolRecord] = Field(default_factory=list)
    episodes: list[SemanticEpisode] = Field(default_factory=list)
    # Compatibility view only: anchors are deterministically derived from episodes.
    anchors: list[SemanticAnchor] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class OutlineSubsection(BaseModel):
    title: str
    start: float
    end: float
    episode_ids: list[str] = Field(default_factory=list)


class OutlineSection(BaseModel):
    id: str = ""
    title: str
    start: float
    end: float
    episode_ids: list[str] = Field(default_factory=list)
    anchor_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    subsections: list[OutlineSubsection] = Field(default_factory=list)


class LectureOutline(BaseModel):
    sections: list[OutlineSection] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class LectureState(BaseModel):
    """Persistent semantic state assembled before any final note blocks are written."""

    lecture_id: str
    title: str
    observations: list[LectureObservation] = Field(default_factory=list)
    symbols: list[SymbolRecord] = Field(default_factory=list)
    episodes: list[SemanticEpisode] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    outline: LectureOutline | None = None


class GlobalBlockCorrection(BaseModel):
    section_index: int = Field(ge=0)
    block_index: int = Field(ge=0)
    corrected_latex: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("corrected_latex")
    @classmethod
    def reject_renderer_owned_environments(cls, value: str) -> str:
        return _reject_environments(value, _RENDERER_BLOCK_ENVIRONMENTS, context="global corrections")


class GlobalValidation(BaseModel):
    corrections: list[GlobalBlockCorrection] = Field(default_factory=list)
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
