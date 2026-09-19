from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from .llm_robust import LectureModelClient as RobustLectureModelClient
from .schemas import ChunkNotes, CorrectionRecord, LectureChunk

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Included in linear cache identity by pipeline_robust so prompt-policy changes cannot silently reuse
# older chunk artifacts.
LINEAR_SOURCE_POLICY_VERSION = 6

_FINALIZE_SOURCE_POLICY = r"""

STRICT MULTIMODAL SOURCE-FIDELITY POLICY FOR THE LINEAR LECTURE PIPELINE:
- The current ASR transcript and attached current board frames are synchronized noisy observations.
  Neither is ground truth by itself. Reconstruct NEW mathematical content from their agreement,
  temporal development, and locally forced cross-channel consistency.
- Known notation and preceding notes are continuity context, not evidence for adding a theorem,
  construction, assumption, example, proof step, or named result.
- You may repair an ASR/OCR error only when the intended reading is locally forced by the current
  speech/board evidence, or when established notation makes the intended symbol unambiguous.
- Never guess the name of a theorem, person, or named construction from mathematical plausibility.
  If the identity itself is not recoverable, omit the name and use a source-supported descriptive
  formulation instead.
- You may perform only immediate local algebraic normalization directly forced by current evidence.
  Do NOT invent or complete a multi-step derivation, proof strategy, or textbook completion.
- If the lecturer and board consistently state something mathematically suspicious, preserve what
  the lecture says and put the concern in unresolved rather than replacing it with textbook truth.
- Preserve signs, constants, indices, quantifiers, object roles, and uniqueness/up-to-scalar claims.
- Ignore filler, false starts, repetitions, and isolated garbled ASR fragments that do not carry
  recoverable mathematical content.
- If a faithful statement cannot be reconstructed from the synchronized observations, OMIT it from
  note blocks. Only when this loses unique substantive mathematical content, add one unresolved entry
  prefixed exactly with `[omitted-math] `.
"""

_AUDIT_SOURCE_POLICY = r"""

MULTIMODAL VERIFIER POLICY:
- Produce exactly one keep/replace/suppress verdict for every retained draft block.
- The transcript is noisy ASR, not authoritative text. Attached board images are an independent
  synchronized source. Judge fidelity from the pair, not from ASR alone.
- Every replace/suppress verdict must identify its exact block with a verbatim target_excerpt.
- Set support to transcript, visual, combined, or uncertain. Transcript support must cite verbatim
  transcript evidence. Visual support may rely directly on attached images because no intermediate
  OCR is required. Combined support may use both.
- Never guess a theorem/person/name from topic knowledge. If a draft inserted a name that is not
  recoverable from the observations, prefer a source-supported descriptive replacement or suppress
  only the unsupported name/claim.
- Use replace only when the complete replacement is directly supported by current observations.
  Use suppress for writer-added source drift when no safe complete replacement is available.
- Mathematical plausibility and textbook knowledge are not evidence. Preceding notes are not
  correction evidence.
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
    """LLM-facing verdict.

    Keep the schema structurally permissive enough that one malformed action cannot invalidate the
    entire per-block audit batch. Safety-critical requirements for replace/suppress are enforced
    independently by the host below, so a bad verdict is rejected locally while valid sibling
    verdicts remain usable.
    """

    block_index: int = Field(ge=0)
    action: Literal["keep", "replace", "suppress"]
    target_excerpt: str | None = None
    replacement_latex: str | None = None
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    support: Literal["transcript", "visual", "combined", "uncertain"] = "uncertain"
    evidence: list[AuditEvidence] = Field(default_factory=list)


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


def _effective_support(verdict: SourceGroundedAuditVerdict) -> str:
    if verdict.support != "uncertain":
        return verdict.support
    sources = {item.source for item in verdict.evidence}
    if sources == {"visual"}:
        return "visual"
    if sources == {"transcript"}:
        return "transcript"
    if sources == {"visual", "transcript"}:
        return "combined"
    return "uncertain"


def _basis_from_verdict(verdict: SourceGroundedAuditVerdict) -> str:
    support = _effective_support(verdict)
    if support == "visual":
        return "visual"
    if support == "transcript":
        return "audio_context"
    if support == "combined":
        return "multimodal"
    return "mathematical_consistency"


def _board_scan_request_ids(evidence_json: str) -> list[str]:
    try:
        payload = json.loads(evidence_json)
    except (json.JSONDecodeError, TypeError):
        return []
    result: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict) or item.get("kind") != "board_scan":
            continue
        request_id = item.get("request_id")
        if isinstance(request_id, str) and request_id and request_id not in result:
            result.append(request_id)
    return result


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
        images: list[Path] | None = None,
    ) -> ChunkNotes:
        if not self.config.math_audit or not notes.blocks:
            return notes

        indexed_draft = "\n\n".join(
            f"BLOCK_INDEX={index}\n"
            + json.dumps(block.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for index, block in enumerate(notes.blocks)
        )
        prompt = f"""Verify every retained lecture-note block against the synchronized CURRENT
observations. This is a fidelity verifier, not a mathematical correctness solver.

The timestamped transcript below is noisy ASR. The attached images are the uniformly sampled board
states from the same interval. Use both channels directly. A board formula may resolve garbled ASR;
clear speech may resolve unreadable writing. If they still conflict, do not invent a resolution.

Return EXACTLY ONE verdict for EVERY draft block, in ascending block_index order:
- keep: faithful enough to retain;
- replace: a concrete source-fidelity error exists and the current observations directly support the
  complete replacement_latex;
- suppress: the draft contains unsupported/source-drift content and no safe complete replacement can
  be written.

For every replace/suppress verdict:
1. target_excerpt MUST be a verbatim contiguous excerpt from that exact draft block;
2. set support to transcript, visual, combined, or uncertain;
3. if support uses transcript as the decisive source, include verbatim transcript evidence quotes;
4. visual support may rely directly on attached board images and therefore needs no OCR quote;
5. confidence describes source-fidelity confidence, not mathematical plausibility.

Never infer a theorem/person/name because it would be the standard theorem in this context. If a
name is not recoverable, a descriptive source-supported formulation is safer than a guessed name.
Do not use textbook knowledge to replace a lecturer statement. If uncertain, choose keep.

Preceding context (continuity only, never correction evidence):
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

CURRENT noisy transcript:
{chunk.timestamped_text or chunk.text}

CURRENT auxiliary visual metadata/OCR:
{evidence_json}

DRAFT BLOCKS:
{indexed_draft}

Keep reasons under 12 words and replace/suppress reasons under 30 words.
Write reasons in language code `{self.config.output_language}`.
"""
        try:
            audit = self._structured(
                prompt,
                SourceGroundedMathAudit,
                images=images,
                max_tokens=max(4096, self.config.max_tokens),
                guided_json=not bool(images),
                operation="math_audit",
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("[%s] multimodal verifier skipped: %s", chunk.id, exc)
            return notes

        verdicts = audit.verdicts
        expected_indices = list(range(len(notes.blocks)))
        indices = [item.block_index for item in verdicts]
        if len(verdicts) != len(notes.blocks) or sorted(indices) != expected_indices:
            logger.warning(
                "[%s] verifier contract rejected: expected one verdict per block; got %s",
                chunk.id,
                indices,
            )
            return notes

        board_request_ids = _board_scan_request_ids(evidence_json)
        audit_findings: list[str] = []
        for verdict in verdicts:
            block = notes.blocks[verdict.block_index]
            if verdict.action == "keep":
                continue
            if verdict.confidence < 0.8:
                logger.info(
                    "[%s] verifier %s for block %d ignored at confidence %.3f",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                    verdict.confidence,
                )
                continue
            if not _target_excerpt_matches_block(verdict.target_excerpt, block.latex):
                logger.warning(
                    "[%s] verifier %s rejected for block %d: target excerpt mismatch",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                )
                continue

            support = _effective_support(verdict)
            if support == "uncertain":
                logger.warning(
                    "[%s] verifier %s rejected for block %d: support is uncertain",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                )
                continue

            if support == "transcript":
                if not audit_evidence_supported(
                    verdict.evidence, chunk=chunk, evidence_json=evidence_json
                ):
                    logger.warning(
                        "[%s] verifier %s rejected for block %d: transcript citation not verified",
                        chunk.id,
                        verdict.action,
                        verdict.block_index,
                    )
                    continue
            elif support in {"visual", "combined"}:
                # Direct board images are sufficient visual evidence without OCR. Legacy/supplemental
                # visual requests can still support a verdict through host-verified OCR quotes.
                if images:
                    if verdict.evidence and not audit_evidence_supported(
                        verdict.evidence, chunk=chunk, evidence_json=evidence_json
                    ):
                        logger.warning(
                            "[%s] verifier %s rejected for block %d: supplied citation not verified",
                            chunk.id,
                            verdict.action,
                            verdict.block_index,
                        )
                        continue
                elif not verdict.evidence or not audit_evidence_supported(
                    verdict.evidence, chunk=chunk, evidence_json=evidence_json
                ):
                    logger.warning(
                        "[%s] verifier %s rejected for block %d: no visual support available",
                        chunk.id,
                        verdict.action,
                        verdict.block_index,
                    )
                    continue

            if verdict.action == "replace":
                replacement = (verdict.replacement_latex or "").strip()
                if not replacement:
                    logger.warning(
                        "[%s] verifier replace rejected for block %d: replacement is empty",
                        chunk.id,
                        verdict.block_index,
                    )
                    continue
                if replacement == block.latex.strip():
                    continue
                original = block.latex
                block.latex = replacement
                notes.corrections.append(
                    CorrectionRecord(
                        original=original,
                        corrected=replacement,
                        reason=verdict.reason,
                        basis=_basis_from_verdict(verdict),
                        confidence=verdict.confidence,
                    )
                )
                continue

            # Preserve literal current speech only for transcript-only verdicts. In multimodal mode
            # the ASR string itself may be the corrupted channel that the board resolves.
            if support == "transcript" and _target_excerpt_is_literal_source(
                verdict.target_excerpt, chunk=chunk, evidence_json=evidence_json
            ):
                logger.warning(
                    "[%s] suppress rejected for block %d: literal transcript statement",
                    chunk.id,
                    verdict.block_index,
                )
                continue

            marker = f"audit-suppress:{verdict.confidence:.3f}"
            if marker not in block.source_evidence_ids:
                block.source_evidence_ids.append(marker)

            request_ids = _visual_request_ids_for_evidence(
                verdict.evidence, evidence_json=evidence_json
            )
            if support in {"visual", "combined"}:
                request_ids.extend(
                    request_id
                    for request_id in board_request_ids
                    if request_id not in request_ids
                )
            for request_id in request_ids:
                visual_marker = f"audit-visual:{request_id}"
                if visual_marker not in block.source_evidence_ids:
                    block.source_evidence_ids.append(visual_marker)

            audit_findings.append(
                f"Verifier block {verdict.block_index}: suppressed source-drift claim: {verdict.reason}"
            )

        notes.unresolved.extend(audit_findings[:3])
        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        return notes

