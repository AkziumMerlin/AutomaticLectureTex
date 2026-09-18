from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, model_validator
from pydantic_core import ValidationError

from .llm_robust import LectureModelClient as RobustLectureModelClient
from .schemas import ChunkNotes, CorrectionRecord, LectureChunk

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Included in linear cache identity by pipeline_robust so prompt-policy changes cannot silently reuse
# older chunk artifacts.
LINEAR_SOURCE_POLICY_VERSION = 4

_FINALIZE_SOURCE_POLICY = r"""

STRICT SOURCE-FAITHFULNESS POLICY FOR THE LINEAR LECTURE PIPELINE:
- The current transcript and current visual evidence are the only evidence for NEW mathematical
  content in this chunk. Known notation and preceding notes are continuity context, not evidence for
  adding a theorem, construction, assumption, example, or proof step.
- You may repair an ASR/OCR error only when the corrected reading is locally forced by the current
  speech/board evidence, or when established notation makes the intended symbol unambiguous.
- You may perform only an immediate local algebraic normalization that is directly forced by an
  explicitly spoken/written formula. Do NOT invent or complete a multi-step derivation.
- In particular, do NOT introduce a new sequence, vector construction, auxiliary object, theorem,
  standard argument, proof strategy, or textbook completion that is absent from the current source.
- Mathematical plausibility is not evidence. If the lecturer and board consistently state something
  mathematically suspicious, preserve what the lecture says and put the concern in unresolved rather
  than silently replacing it with the textbook result. A later explicit lecturer correction is
  handled by the separate correction pass.
- Preserve exact source roles and notation: signs, constants, indices, quantifiers, and which object
  has which property (for example z_f versus y_f, uniqueness versus up-to-scalar).
- `unresolved` is only for ambiguities that materially affect a retained note block or force omission
  of unique substantive mathematical content. Ignore filler, false starts, repetitions, and isolated
  garbled ASR fragments that do not carry recoverable mathematical content.
- If a faithful statement cannot be reconstructed from current evidence, OMIT the unsupported claim
  from note blocks. Only when this omission loses unique substantive mathematical content, add one
  unresolved entry prefixed exactly with `[omitted-math] `. Never use that marker for filler, names,
  historical remarks, generic ASR noise, or an ambiguity that did not force mathematical omission.
"""

_AUDIT_SOURCE_POLICY = r"""

SOURCE-FAITHFUL AUDIT POLICY:
- Produce exactly one keep/replace/suppress verdict for every retained draft block.
- Every replace/suppress verdict must identify its exact block with a verbatim target_excerpt copied
  from that block, and must cite verbatim CURRENT transcript/visual evidence.
- Never use a target_excerpt from one block to justify an action on another block.
- Use suppress only for writer-added source drift when no complete source-backed replacement is safe.
- Never suppress a statement that the lecturer/current board literally states merely because a
  textbook theorem disagrees with it.
- Use replace only when the complete replacement is directly forced by cited current evidence.
- Check exact signs, constants, object identity (especially z_f versus y_f), uniqueness versus
  up-to-scalar, indices/quantifiers/domains, and topology/type labels.
- Mathematical plausibility is not evidence. Preceding notes are not correction evidence.
- If uncertain, keep. Do not rewrite for style or expand the lecture.
"""

# The historical pre-PR2 prompt contained two permissions that encouraged textbook completion. The
# base implementation remains untouched, but the effective linear prompt removes them before the
# request reaches the model.
_PERMISSIVE_FINALIZE = (
    (
        "You may actively correct ASR/OCR errors, normalize terminology, reconstruct formulas from combined\n"
        "audio and video evidence, and complete a short derivation when its mathematical conclusion is\n"
        "reliable. Do not add unrelated textbook exposition.",
        "You may correct ASR/OCR errors, normalize terminology, and reconstruct formulas from combined\n"
        "audio and video evidence only when the intended reading is locally forced by those sources.\n"
        "Do not complete a derivation or add mathematical steps that are absent from the current evidence.",
    ),
    (
        "Use the speech, known\nnotation, and mathematical consistency here to make any further correction or inference, and record\n"
        "every such content-changing step in `corrections`.",
        "Use speech and known notation only to resolve local ASR/OCR ambiguity. Do not use mathematical\n"
        "consistency to add an inference that is not supported by the current transcript or visual evidence.\n"
        "Record every source-supported content-changing correction in `corrections`.",
    ),
    (
        "For figure blocks, asset_path must be copied exactly from visual evidence. Record unresolved\n"
        "ambiguities in `unresolved`.",
        "For figure blocks, asset_path must be copied exactly from visual evidence. Record in `unresolved`\n"
        "only ambiguities that materially affect a retained block or omit unique substantive content.",
    ),
)

# Deliberately conservative lexical trigger. False negatives merely postpone a rare correction to
# manual review; false positives cost a full catalog+LLM scan on every ordinary chunk.
_CORRECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bоговор(?:ил(?:ся|ась)?|илась|ка|ки)\b",
        r"\b(?:поправ|исправ)\w*\b",
        r"\bошиб(?:ся|лась|ка|очно|очн\w*)\b",
        r"\bневерн\w*\b",
        r"\b(?:точнее|вернее)\b",
        r"\b(?:здесь|тут|там|выше|раньше)\b.{0,80}\bдолжн\w*\s+быть\b",
        r"\b(?:выше|раньше)\b.{0,80}\b(?:ошиб|неверн|оговор|исправ)\w*\b",
        r"\b(?:плюс|минус|знак|индекс|букв\w*|символ)\b.{0,50}\bа\s+не\b",
        (
            r"\b(?:нет|не)\s*,?\s*(?:здесь|тут|там)\b.{0,80}"
            r"\b(?:плюс|минус|знак|индекс|букв\w*|должн\w*)\b"
        ),
        r"\b(?:i\s+misspoke|correction|correct that|should be|rather than)\b",
    )
)


class AuditEvidence(BaseModel):
    source: Literal["transcript", "visual"]
    quote: str = Field(min_length=4)


class SourceGroundedAuditVerdict(BaseModel):
    block_index: int = Field(ge=0)
    action: Literal["keep", "replace", "suppress"]
    target_excerpt: str | None = None
    replacement_latex: str | None = None
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[AuditEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_payload(self) -> SourceGroundedAuditVerdict:
        if self.action == "keep":
            return self
        if not self.target_excerpt or len(self.target_excerpt.strip()) < 4:
            raise ValueError("replace/suppress verdict requires a concrete target_excerpt")
        if not self.evidence:
            raise ValueError("replace/suppress verdict requires current-source evidence")
        if self.action == "replace" and not (self.replacement_latex or "").strip():
            raise ValueError("replace verdict requires replacement_latex")
        return self


class SourceGroundedMathAudit(BaseModel):
    verdicts: list[SourceGroundedAuditVerdict] = Field(default_factory=list)


def _current_correction_transcript(prompt: str) -> str:
    """Extract only the CURRENT transcript from the correction-scan prompt."""

    start_marker = "Current timestamped transcript:\n"
    end_marker = "\n\nCurrent visual evidence:"
    start = prompt.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = prompt.find(end_marker, start)
    return prompt[start:] if end < 0 else prompt[start:end]


def has_explicit_correction_signal(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _CORRECTION_PATTERNS)


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _visual_source_text(evidence_json: str) -> str:
    try:
        payload = json.loads(evidence_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    parts: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        for key in ("raw_latex", "latex", "description"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return "\n".join(parts)


def audit_evidence_supported(
    evidence: list[AuditEvidence], *, chunk: LectureChunk, evidence_json: str
) -> bool:
    """Host-side check that every claimed audit citation literally exists in current evidence."""

    if not evidence:
        return False
    transcript = _normalize_quote(chunk.timestamped_text or chunk.text)
    visual = _normalize_quote(_visual_source_text(evidence_json))
    for item in evidence:
        quote = _normalize_quote(item.quote)
        if len(quote) < 4:
            return False
        haystack = transcript if item.source == "transcript" else visual
        if quote not in haystack:
            return False
    return True


def _visual_request_ids_for_evidence(
    evidence: list[AuditEvidence], *, evidence_json: str
) -> list[str]:
    """Map verified visual quotes back to the concrete current visual request ids."""

    try:
        payload = json.loads(evidence_json)
    except (json.JSONDecodeError, TypeError):
        return []
    result: list[str] = []
    for cited in evidence:
        if cited.source != "visual":
            continue
        quote = _normalize_quote(cited.quote)
        for item in payload if isinstance(payload, list) else []:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                continue
            text = "\n".join(
                value
                for key in ("raw_latex", "latex", "description")
                if isinstance((value := item.get(key)), str) and value.strip()
            )
            if quote and quote in _normalize_quote(text) and request_id not in result:
                result.append(request_id)
    return result


def _target_excerpt_matches_block(target_excerpt: str | None, block_latex: str) -> bool:
    if not target_excerpt:
        return False
    excerpt = _normalize_quote(target_excerpt)
    return len(excerpt) >= 4 and excerpt in _normalize_quote(block_latex)


def _target_excerpt_is_literal_source(
    target_excerpt: str | None, *, chunk: LectureChunk, evidence_json: str
) -> bool:
    """Protect lecturer/source statements from suppress verdicts.

    If the allegedly unsupported draft excerpt is itself literally present in the current lecture
    evidence, suppression is not source-faithful; preserve it and let a later explicit correction
    pass handle lecturer retractions.
    """

    if not target_excerpt:
        return False
    excerpt = _normalize_quote(target_excerpt)
    if len(excerpt) < 4:
        return False
    transcript = _normalize_quote(chunk.timestamped_text or chunk.text)
    visual = _normalize_quote(_visual_source_text(evidence_json))
    return excerpt in transcript or excerpt in visual


def _basis_from_evidence(evidence: list[AuditEvidence]) -> str:
    sources = {item.source for item in evidence}
    if sources == {"visual"}:
        return "visual"
    if sources == {"transcript"}:
        return "audio_context"
    return "mathematical_consistency"


class LectureModelClient(RobustLectureModelClient):
    """Robust client with narrow policies specific to the simple linear production path."""

    def _structured(
        self,
        prompt: str,
        schema: type[T],
        images: list[Path] | None = None,
        max_tokens: int | None = None,
        *,
        guided_json: bool = True,
        operation: str = "structured",
    ) -> T:
        if operation == "linear_correction_scan":
            transcript = _current_correction_transcript(prompt)
            if not has_explicit_correction_signal(transcript):
                return schema.model_validate({"patches": [], "unresolved": []})
        elif operation == "finalize_chunk":
            for old, new in _PERMISSIVE_FINALIZE:
                prompt = prompt.replace(old, new)
            prompt += _FINALIZE_SOURCE_POLICY
        elif operation == "math_audit":
            prompt += _AUDIT_SOURCE_POLICY

        return super()._structured(
            prompt,
            schema,
            images=images,
            max_tokens=max_tokens,
            guided_json=guided_json,
            operation=operation,
        )

    def _audit_math(
        self,
        notes: ChunkNotes,
        *,
        chunk: LectureChunk,
        evidence_json: str,
        previous_context: dict | None,
    ) -> ChunkNotes:
        if not self.config.math_audit or not notes.blocks:
            return notes

        indexed_draft = "\n\n".join(
            f"BLOCK_INDEX={index}\n"
            + json.dumps(block.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for index, block in enumerate(notes.blocks)
        )
        prompt = f"""Check every retained lecture-note block only for fidelity to the supplied CURRENT
lecture source. This is not a mathematical correctness solver and not a textbook editor.

Return EXACTLY ONE verdict for EVERY draft block, in ascending block_index order:
- keep: the block is source-faithful enough to retain;
- replace: the block contains a concrete source-fidelity error and CURRENT source directly supports
  the complete replacement_latex;
- suppress: the block contains a concrete unsupported/source-drift claim, but no safe complete
  source-backed replacement can be written.

For every replace/suppress verdict:
1. block_index MUST identify the block that actually contains the bad claim;
2. target_excerpt MUST be a VERBATIM contiguous excerpt copied from that exact draft block and must
   identify the problematic claim. Never put an excerpt from another block here;
3. evidence MUST contain one or more VERBATIM contiguous quotes copied from CURRENT transcript or
   CURRENT visual evidence. Preceding notes and textbook knowledge are never evidence;
4. confidence must describe source-fidelity confidence, not mathematical plausibility.

Use suppress for writer-added claims that are absent from the source when the cited current-source
passage establishes what was actually said but does not justify the extra draft assertion. Do NOT
suppress a statement merely because a standard theorem says it is false. If the lecturer/source
literally states the target claim, keep it unless the current source itself explicitly corrects it.
Use replace only when the cited source forces the complete replacement. If uncertain, choose keep.

Pay particular attention to exact object roles (for example z_f versus y_f), uniqueness versus
up-to-scalar, signs/constants/indices/quantifiers, and topology/type labels. Do not report raw ASR
garbage or rewrite correct blocks for style.

Preceding context is continuity context only, not correction evidence:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

CURRENT transcript:
{chunk.timestamped_text or chunk.text}

CURRENT visual evidence:
{evidence_json}

DRAFT BLOCKS:
{indexed_draft}

Write reasons in language code `{self.config.output_language}`.
"""
        try:
            audit = self._structured(
                prompt,
                SourceGroundedMathAudit,
                max_tokens=2048,
                operation="math_audit",
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("[%s] source-grounded audit skipped: %s", chunk.id, exc)
            return notes

        verdicts = audit.verdicts
        expected_indices = list(range(len(notes.blocks)))
        indices = [item.block_index for item in verdicts]
        if len(verdicts) != len(notes.blocks) or sorted(indices) != expected_indices:
            logger.warning(
                "[%s] source-grounded audit contract rejected: expected one verdict per block; got %s",
                chunk.id,
                indices,
            )
            return notes

        audit_findings: list[str] = []
        for verdict in verdicts:
            block = notes.blocks[verdict.block_index]
            if verdict.action == "keep":
                continue
            if verdict.confidence < 0.8:
                logger.info(
                    "[%s] audit %s for block %d ignored at confidence %.3f",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                    verdict.confidence,
                )
                continue
            if not _target_excerpt_matches_block(verdict.target_excerpt, block.latex):
                logger.warning(
                    "[%s] audit %s rejected for block %d: target_excerpt does not belong to block",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                )
                continue
            if not audit_evidence_supported(
                verdict.evidence, chunk=chunk, evidence_json=evidence_json
            ):
                logger.warning(
                    "[%s] audit %s rejected for block %d: current-source citation not verified",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                )
                continue

            if verdict.action == "replace":
                replacement = (verdict.replacement_latex or "").strip()
                if replacement == block.latex.strip():
                    continue
                original = block.latex
                block.latex = replacement
                notes.corrections.append(
                    CorrectionRecord(
                        original=original,
                        corrected=replacement,
                        reason=verdict.reason,
                        basis=_basis_from_evidence(verdict.evidence),
                        confidence=verdict.confidence,
                    )
                )
                continue

            # suppress: never erase a statement that is itself literally present in the current
            # lecture source. This protects lecturer mistakes from textbook-driven "correction".
            if _target_excerpt_is_literal_source(
                verdict.target_excerpt, chunk=chunk, evidence_json=evidence_json
            ):
                logger.warning(
                    "[%s] suppress rejected for block %d: target excerpt is literal current source",
                    chunk.id,
                    verdict.block_index,
                )
                continue

            marker = f"audit-suppress:{verdict.confidence:.3f}"
            if marker not in block.source_evidence_ids:
                block.source_evidence_ids.append(marker)
            for request_id in _visual_request_ids_for_evidence(
                verdict.evidence, evidence_json=evidence_json
            ):
                visual_marker = f"audit-visual:{request_id}"
                if visual_marker not in block.source_evidence_ids:
                    block.source_evidence_ids.append(visual_marker)
            audit_findings.append(
                f"Audit block {verdict.block_index}: suppressed source-drift claim: {verdict.reason}"
            )

        notes.unresolved.extend(audit_findings[:3])
        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        return notes
