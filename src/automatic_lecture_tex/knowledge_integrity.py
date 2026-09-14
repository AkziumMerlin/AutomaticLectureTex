from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator, model_validator

from .knowledge import KnowledgeOrchestrator, _merge_unique
from .schemas import (
    EpisodeStatus,
    LectureChunk,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    SourceStatus,
    Transcript,
    VisualEvidence,
    WindowObservations,
)


class GeneratedLectureObservation(BaseModel):
    """LLM-facing evidence event.

    Time bounds are intentionally absent. The model may only identify exact ASR segment ids; the
    host derives numeric timestamps from the transcript so malformed MM:SS conversions cannot enter
    the canonical knowledge graph.
    """

    id: str = ""
    kind: ObservationKind
    text: str = Field(min_length=1)
    latex: str | None = None
    target_observation_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_status: SourceStatus
    source_segment_ids: list[str] = Field(min_length=1)
    visual_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("generated lecture observation text must be non-empty")
        return value

    @field_validator("source_segment_ids")
    @classmethod
    def require_nonblank_segment_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("every generated observation must reference at least one ASR segment")
        return cleaned

    @model_validator(mode="after")
    def require_equation_latex(self) -> GeneratedLectureObservation:
        if self.kind == ObservationKind.EQUATION and not (self.latex or "").strip():
            raise ValueError("equation observations must include exact latex in addition to prose")
        return self


class GeneratedWindowObservations(BaseModel):
    observations: list[GeneratedLectureObservation]
    unresolved: list[str] = Field(default_factory=list)


@dataclass
class IntegrityKnowledgeOrchestrator(KnowledgeOrchestrator):
    transcript: Transcript

    def _segment_payload(self, chunk: LectureChunk) -> list[dict]:
        allowed = set(chunk.segment_ids)
        return [
            {
                "id": segment.id,
                "start_seconds": segment.start,
                "end_seconds": segment.end,
                "text": segment.text,
                "confidence": segment.confidence,
            }
            for segment in self.transcript.segments
            if segment.id in allowed
        ]

    def extract_observations(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        kb: LectureKnowledgeBase,
    ) -> WindowObservations:
        segment_payload = self._segment_payload(chunk)
        segment_map = {item["id"]: item for item in segment_payload}
        visual_ids = {item.request_id for item in evidence if item.request_id}
        visual_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        symbols = [item.model_dump(mode="json") for item in kb.symbols if item.active]
        recent_observations = [
            item.model_dump(mode="json") for item in kb.observations[-20:]
        ]
        open_episodes = [
            item.model_dump(mode="json")
            for item in kb.episodes
            if item.status == EpisodeStatus.OPEN
        ]

        base_prompt = f"""Extract evidence events from one OVERLAPPING technical window of a
university lecture. Do not write lecture notes and do not decide final document sections.

Window id: {chunk.id}
ASR segments. `source_segment_ids` MUST contain only exact ids from this list. Never convert the
printed times into numbers yourself; the host derives all observation timestamps from these ids:
{json.dumps(segment_payload, ensure_ascii=False, separators=(",", ":"))}

Visual evidence. `visual_evidence_ids` may contain exact request_id values from this list, but every
observation must still cite at least one ASR segment id for temporal grounding:
{visual_json}

Known symbol registry:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Recent canonical observations from the previous overlap/context:
{json.dumps(recent_observations, ensure_ascii=False, separators=(",", ":"))}

Currently open semantic episodes:
{json.dumps(open_episodes, ensure_ascii=False, separators=(",", ":"))}

Return observations in temporal order. Use only these event meanings:
- definition/claim/equation/proof_step/example/notation/remark: something actually asserted or
  written in this window;
- correction: the lecturer explicitly corrects a previous statement, sign, symbol, derivation, or
  board entry. If it targets a recent canonical observation shown above, use that exact id in
  target_observation_id; if the target is earlier in THIS window, use the local observation id;
- retraction: the lecturer explicitly withdraws a statement, with target_observation_id when known;
- transition: a real topic/proof/example transition, not a technical window edge;
- unresolved: evidence is too ambiguous to reconstruct safely.

The lecturer may make mistakes and then fix them. Preserve both the mistaken event and the later
correction as evidence. Do not replace either with textbook knowledge. Technical window overlap is
not semantic structure: repeated material is expected and will be reconciled by the host.

Every observation MUST contain non-empty prose in `text`, even when it is primarily a formula.
Put the exact formula additionally in `latex`. `source_status=observed` means directly supported by
audio/visible board. Use `reconstructed` only for a local reconstruction strongly forced by the
evidence; use `inferred` sparingly and never for new mathematical content. `confidence` is required.
Write descriptive strings in language code `{self.output_language}`.
"""

        result = None
        invalid_refs: list[str] = []
        prompt = base_prompt
        for _attempt in range(2):
            result = self._structured(
                prompt,
                GeneratedWindowObservations,
                operation="knowledge_extract",
                max_tokens=4096,
            )
            invalid_refs = sorted(
                {
                    ref
                    for observation in result.observations
                    for ref in observation.source_segment_ids
                    if ref not in segment_map
                }
            )
            invalid_visuals = sorted(
                {
                    ref
                    for observation in result.observations
                    for ref in observation.visual_evidence_ids
                    if ref not in visual_ids
                }
            )
            if not invalid_refs and not invalid_visuals:
                break
            prompt = (
                base_prompt
                + "\nYour previous response used unknown provenance ids. "
                + "Regenerate the full object. "
                + f"Unknown ASR ids: {invalid_refs}; unknown visual ids: {invalid_visuals}.\n"
            )

        assert result is not None
        observations: list[LectureObservation] = []
        unresolved = list(result.unresolved)
        for index, item in enumerate(result.observations):
            valid_segment_ids = [ref for ref in item.source_segment_ids if ref in segment_map]
            if not valid_segment_ids:
                unresolved.append(
                    "Dropped generated observation "
                    f"{item.id or index}: no valid source segment ids."
                )
                continue
            segments = [segment_map[ref] for ref in valid_segment_ids]
            valid_visual_ids = [ref for ref in item.visual_evidence_ids if ref in visual_ids]
            observation_id = item.id or f"obs_{chunk.id}_{index:03d}"
            observations.append(
                LectureObservation(
                    id=observation_id,
                    window_id=chunk.id,
                    window_ids=[chunk.id],
                    start=min(segment["start_seconds"] for segment in segments),
                    end=max(segment["end_seconds"] for segment in segments),
                    kind=item.kind,
                    text=item.text,
                    latex=item.latex,
                    target_observation_id=item.target_observation_id,
                    confidence=item.confidence,
                    source_status=item.source_status,
                    evidence_refs=[*valid_segment_ids, *valid_visual_ids],
                )
            )

        return WindowObservations(
            window_id=chunk.id,
            start=chunk.start,
            end=chunk.end,
            observations=observations,
            unresolved=_merge_unique([], unresolved),
        )
