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
from .visual_formula_gate import find_formula_gate_violations


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

Your task is to recover the intended mathematical content communicated in this window, while
preserving genuine lecturer corrections and mistakes when they are actually evidenced.

Treat ASR as PHONETIC EVIDENCE, not authoritative wording. It can contain severe substitutions,
word fragments, invented-looking proper names, missing negations, broken mathematical terminology,
and verbalized formulas with lost symbols. Do not preserve a nonsensical literal ASR reading merely
because it is the only transcript string available.

Resolve corrupted ASR using ALL LOCAL evidence together:
- neighboring utterances before and after the phrase;
- the current theorem/definition/proof role;
- mathematical type/notation consistency;
- already-established symbols and terminology;
- board/visual evidence when available;
- standard mathematical knowledge as a DISAMBIGUATION PRIOR.

Standard mathematics may be used to choose the canonical reading of a locally evidenced term,
formula, theorem name, or short derivation when the surrounding lecture context strongly determines
it. This includes canonicalizing a phonetically corrupted technical term or proper name and repairing
an ASR-damaged sign/variable when only one reading is compatible with the local derivation. This is
reconstruction of supplied evidence, not addition of textbook material.

A garbled proper name must NEVER be expanded into an unrelated specific theorem/person by free
association. Use a canonical name only when the mathematical statement, role, or surrounding
discussion identifies it strongly; otherwise describe the result without the name or mark it
unresolved.

Distinguish ASR corruption from a genuine lecturer mistake. A lone garbled phrase that would make an
otherwise coherent local argument mathematically nonsensical is NOT sufficient evidence that the
lecturer made that error. Preserve a lecturer mistake only when the erroneous content itself is
supported coherently by speech, board evidence, repetition, or an explicit later correction.

Visual evidence can contain both VLM OCR (`raw_latex`/`latex`) and independent
`math_ocr_candidates`. These are FALLIBLE SENSOR HYPOTHESES, not ground truth. The first attached
image is a selected low-occlusion raw frame; a temporal-median composite may follow as secondary
context and can combine writing from slightly different moments. Do not infer temporal order from
the composite itself. Compare OCR channels against established notation, neighboring equations, and
local mathematical consistency. If exact signs/variables remain materially inconsistent across
sensors, report the content as unresolved instead of choosing the most convenient formula.

When you cite high-confidence visual evidence for a formula, preserve its literal variable names,
operators, signs, roots, subscripts, and constants unless another supplied local source explicitly
contradicts it. Do not silently turn `v` into `u`, `\\sqrt{{2}}` into `2`, or `\\varepsilon_i` into
`f_i`. If an event materially states a formula/relation, put that relation in `latex` even when the
event kind is claim or proof_step; this lets the host verify symbol preservation.

Mathematical knowledge is a disambiguation tool, not a license to complete the lecture. Do NOT add
a theorem, hypothesis, proof step, definition, formula, or conclusion merely because it would be
standard textbook material. Do not infer material that has no local speech/board support.

When a short algebraic or logical relation is central to the event, check that the reconstructed
formula is internally consistent with the immediately surrounding derivation instead of copying a
broken ASR token sequence. Conversely, do NOT silently repair a genuine lecturer mistake that is
actually supported by the evidence. If the lecturer makes an error and later corrects it, preserve
the erroneous event and the explicit correction/retraction as separate evidence events.

If two materially different interpretations remain plausible after using local context, notation,
visual evidence, and mathematical consistency, return an `unresolved` event rather than guessing.

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
        invalid_visuals: list[str] = []
        violations = []
        prompt = base_prompt
        for attempt in range(2):
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
            violations = find_formula_gate_violations(result.observations, evidence)
            if not invalid_refs and not invalid_visuals and not violations:
                break
            if attempt == 0:
                feedback = []
                if invalid_refs or invalid_visuals:
                    feedback.append(
                        f"Unknown ASR ids: {invalid_refs}; unknown visual ids: {invalid_visuals}."
                    )
                for violation in violations:
                    feedback.append(
                        "Host formula-preservation check rejected observation "
                        f"{violation.observation_index} using {violation.evidence_id}: "
                        f"visual={violation.visual_formula!r}, generated={violation.generated_formula!r}; "
                        f"{violation.reason}. Preserve the visible formula literally or mark the "
                        "event unresolved if local evidence conflicts."
                    )
                prompt = base_prompt + "\n\nHOST VALIDATION FEEDBACK:\n" + "\n".join(feedback)

        assert result is not None
        blocked_indices = {item.observation_index for item in violations}
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
            if index in blocked_indices:
                matching = [v for v in violations if v.observation_index == index]
                details = "; ".join(
                    f"{v.evidence_id}: {v.visual_formula!r} -> {v.generated_formula!r}"
                    for v in matching
                )
                unresolved.append(
                    "Host visual-formula gate suppressed observation "
                    f"{item.id or index} after retry: {details}"
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
