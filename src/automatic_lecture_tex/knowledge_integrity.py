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
    """LLM-facing semantic event with host-derived temporal provenance."""

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
                "raw_asr": segment.text,
                "asr_confidence": segment.confidence,
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
        recent_limit = min(30, self.config.knowledge_recent_observations)
        recent_observations = [
            item.model_dump(mode="json") for item in kb.observations[-recent_limit:]
        ]
        open_episodes = [
            item.model_dump(mode="json")
            for item in kb.episodes
            if item.status == EpisodeStatus.OPEN
        ]

        base_prompt = f"""Reconstruct the CANONICAL MATHEMATICAL EVENTS in one bounded overlapping
window of a university lecture. This is semantic reconstruction from noisy evidence, not literal
ASR cleanup and not lecture-note writing.

Window id: {chunk.id}
Raw timestamped ASR segments:
{json.dumps(segment_payload, ensure_ascii=False, separators=(",", ":"))}

Visual/board evidence aligned with this window:
{visual_json}

Known notation established earlier in the lecture:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Recent canonical mathematical events from the preceding context:
{json.dumps(recent_observations, ensure_ascii=False, separators=(",", ":"))}

Currently open semantic episodes:
{json.dumps(open_episodes, ensure_ascii=False, separators=(",", ":"))}

Your task is to recover what mathematical content was actually communicated in this window.
The ASR is noisy and may contain phonetic nonsense, broken technical terms, lost punctuation,
misheard variable names, or malformed formulas. You MAY repair those errors using ALL LOCAL evidence:
neighboring utterances, mathematical consistency, already-established notation, and board/visual
evidence. For example, if a formula or technical term is acoustically corrupted but its intended
reading is strongly determined by the surrounding derivation, reconstruct the intended reading.

Visual evidence can contain both VLM OCR (`raw_latex`/`latex`) and independent
`math_ocr_candidates`. These are FALLIBLE SENSOR HYPOTHESES, not ground truth. The first attached
board image may be a temporal-median composite built from nearby frames: it is useful for recovering
writing hidden by a moving lecturer, but it can combine content that existed at slightly different
moments. Do not infer temporal order from the composite itself. Never copy a specialized OCR
candidate merely because it is more explicit than the speech. Compare OCR channels against the raw
visible transcription, established notation, neighboring equations, and local mathematical
consistency. If exact signs/variables remain materially inconsistent across sensors, report the
content as unresolved instead of choosing the most convenient formula. A formula is not
`source_status=observed` merely because one OCR backend proposed it.

However, mathematical knowledge is a disambiguation tool, not a license to complete the lecture.
Do NOT add a theorem, hypothesis, proof step, definition, formula, or conclusion merely because it
would be standard textbook material. Do NOT silently fix a genuine mistake made by the lecturer.
If the lecturer makes an error and later corrects it, preserve the erroneous event and the explicit
correction/retraction as separate evidence events. If two materially different interpretations are
plausible from the local evidence, return an `unresolved` event rather than guessing.

`source_segment_ids` MUST contain only exact ids from the raw ASR list above. The host derives all
numeric timestamps from those ids; never invent timestamps. `visual_evidence_ids` may contain only
exact request ids from the supplied visual evidence. Every event needs transcript provenance even
when the board is decisive.

Use event kinds as follows:
- definition/claim/equation/proof_step/example/notation/remark: mathematical content actually
  communicated in this window;
- correction: an explicit correction of an earlier statement/sign/symbol/derivation;
- retraction: an explicit withdrawal of earlier content;
- transition: a genuine semantic transition in the lecture;
- unresolved: locally ambiguous content that cannot be reconstructed safely.

`text` must be clean, coherent prose expressing the reconstructed mathematical event, NOT a quote of
broken ASR. For equations also put the canonical formula in `latex`. Use
`source_status=observed` when the mathematical content is directly clear from speech/board and
`source_status=reconstructed` when you had to repair ASR or reconcile sensors using local context.
Use `inferred` only for a weak local inference and never for new mathematical content. `confidence`
is confidence that THIS semantic reconstruction matches the lecture, not ASR token confidence and
not confidence that the mathematical statement is true.

Return events in temporal order. Write descriptive strings in language code
`{self.output_language}`.
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
                + "\nYour previous response used unknown provenance ids. Regenerate the complete "
                + f"object. Unknown ASR ids: {invalid_refs}; unknown visual ids: {invalid_visuals}.\n"
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
            valid_visual_ids = [ref for ref in item.visual_evidence_ids if ref in visual_ids]
            if item.kind == ObservationKind.UNRESOLVED:
                unresolved.append(f"{item.text} [segments={','.join(valid_segment_ids)}]")
                continue

            segments = [segment_map[ref] for ref in valid_segment_ids]
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
