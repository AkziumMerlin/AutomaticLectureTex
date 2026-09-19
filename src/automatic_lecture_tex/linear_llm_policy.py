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
LINEAR_SOURCE_POLICY_VERSION = 7

_FINALIZE_SOURCE_POLICY = r"""

CONTEXTUAL MULTIMODAL EDITORIAL POLICY:
- The goal is a mathematically correct, readable lecture note, not a literal transcript.
- Treat ASR, board frames, previous notes, established notation, and standard mathematics as
  complementary evidence about the lecturer's intended mathematical content.
- ASR and board OCR are noisy observations. Repair obvious recognition errors and lecturer slips when
  the intended statement is clear from the surrounding argument.
- You MAY use standard mathematical knowledge to disambiguate or normalize an intended result,
  theorem name, formula, or proof step when the local context determines it with high confidence.
  Example: extension of a bounded linear functional from a subspace to the whole normed space with
  preservation of norm identifies the Hahn--Banach theorem even if its spoken name is garbled.
- Do not invent unrelated textbook material. Add only definitions, intermediate algebraic steps, or
  short connective explanations needed to make the lecture's own argument correct and readable.
- Prefer a standard canonical formulation over preserving malformed speech. If several mathematical
  interpretations remain plausible, use a descriptive formulation or record the ambiguity.
- Correct internal contradictions in the generated notes, including object identity, signs,
  quantifiers, uniqueness claims, topology names, and hypotheses.
- Section titles should describe the mathematical topic cleanly. Never propagate a garbled proper
  name merely because it appeared in a previous generated title.
- Preserve the scope and progression of the lecture: improve the exposition without turning the
  chunk into an independent textbook chapter.
"""

_AUDIT_SOURCE_POLICY = r"""

CONTEXTUAL MULTIMODAL EDITOR POLICY:
- The target is a mathematically correct and useful lecture note, not verbatim fidelity.
- Produce exactly one keep/replace/suppress verdict for every retained draft block.
- Use the current ASR, attached board images, preceding reconstructed context, established notation,
  internal consistency, and standard mathematical knowledge together to infer the lecturer's intent.
- Set support to transcript, visual, combined, contextual, or uncertain.
- contextual means that the literal channels are noisy/incomplete but the intended mathematical
  statement is determined with high confidence by the surrounding argument and standard mathematics.
- Contextual repair is explicitly allowed for garbled theorem names, standard definitions, signs,
  object roles, omitted hypotheses, and short proof steps when the intended result is essentially
  unique. Do not require a verbatim source quote in that case.
- For ambiguous cases with multiple plausible reconstructions, prefer a neutral descriptive wording
  or keep the uncertainty rather than hallucinating a specific detail.
- Correct mathematical errors and internal contradictions even when they were literally spoken or
  written, if the intended correct statement is clear from context. Record the correction rather than
  preserving the error in final notes.
- Check consistency across all blocks in the chunk and with the preceding context, not block-by-block
  in isolation.
- Section titles are part of the final note and must also be corrected contextually. Prefer canonical
  mathematical terminology and stable descriptive titles.
- Do not add unrelated exposition, stronger theorems, new examples, or proof strategies that the
  lecture was not developing.
"""

# Rewrite legacy finalize wording so the writer performs bounded contextual reconstruction rather
# than either literal ASR copying or unconstrained textbook completion.
_PERMISSIVE_FINALIZE = (
    (
        "You may actively correct ASR/OCR errors, normalize terminology, reconstruct formulas from combined\n"
        "audio and video evidence, and complete a short derivation when its mathematical conclusion is\n"
        "reliable. Do not add unrelated textbook exposition.",
        "You may correct ASR/OCR errors, normalize terminology, reconstruct formulas from combined\n"
        "audio/video/context evidence, and complete short missing steps when the lecturer's intended\n"
        "mathematics is clear. Use standard mathematics as a bounded disambiguation prior, not as a\n"
        "license to add unrelated exposition.",
    ),
    (
        "Use the speech, known\nnotation, and mathematical consistency here to make any further correction or inference, and record\n"
        "every such content-changing step in `corrections`.",
        "Use speech, board evidence, preceding context, established notation, and mathematical\n"
        "consistency together to reconstruct the intended lecture content. Record substantive\n"
        "content-changing repairs in `corrections`.",
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
    support: Literal["transcript", "visual", "combined", "contextual", "uncertain"] = "uncertain"
    evidence: list[AuditEvidence] = Field(default_factory=list)


class SourceGroundedMathAudit(BaseModel):
    section_title: str | None = None
    section_title_reason: str = ""
    section_title_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
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
    if support == "contextual":
        return "mathematical_consistency"
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
        prompt = f"""Edit this reconstructed lecture chunk into a mathematically correct, concise and
useful set of lecture notes. Do not optimize for verbatim transcription.

The timestamped transcript is noisy ASR. The attached images are synchronized board states. The
preceding context is reconstructed notes from the immediately previous interval. Use all of them,
together with standard mathematical knowledge, to infer the lecturer's intended argument.

Return a corrected concise section_title and EXACTLY ONE verdict for EVERY draft block:
- keep: already correct and useful;
- replace: the block should be rewritten to express the intended mathematics correctly;
- suppress: the block is noise/repetition or cannot be reconstructed into useful content.

For replace/suppress:
1. target_excerpt must be a verbatim excerpt from the draft block;
2. support is transcript, visual, combined, contextual, or uncertain;
3. contextual is allowed when noisy literal observations do not state the answer cleanly but the
   surrounding mathematical argument determines the intended statement with high confidence;
4. transcript/visual quotes are useful provenance when available, but are not mandatory for
   contextual reconstruction;
5. replacement_latex must be the complete final block, not a commentary about the error.

Editorial rules:
- Correct obvious lecturer slips and ASR/OCR corruption instead of preserving them.
- Use canonical theorem/definition names when the mathematical context identifies them essentially
  uniquely. For example, extension of a bounded linear functional from a subspace with preservation
  of norm is Hahn--Banach. If a name is not uniquely identifiable, use a descriptive phrase.
- Repair short omitted steps needed for a coherent proof, but do not import unrelated textbook
  exposition.
- Check blocks jointly for contradictions: uniqueness vs. up-to-scalar, z_f vs. y_f, signs, domains,
  hypotheses, topology names, and implications.
- Prefer mathematically standard formulations that make the notes convenient to study from.
- Preserve the lecture's topic and level; do not strengthen results or introduce new material.

Preceding reconstructed context:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

CURRENT noisy transcript:
{chunk.timestamped_text or chunk.text}

CURRENT auxiliary visual metadata/OCR:
{evidence_json}

CURRENT section title:
{notes.section_title}

DRAFT BLOCKS:
{indexed_draft}

Write the final section title, reasons, and prose in language code `{self.config.output_language}`.
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
            logger.warning("[%s] contextual editor skipped: %s", chunk.id, exc)
            return notes

        if (
            audit.section_title
            and audit.section_title.strip()
            and audit.section_title_confidence >= 0.80
        ):
            notes.section_title = audit.section_title.strip().replace("$", "")

        verdicts = audit.verdicts
        expected_indices = list(range(len(notes.blocks)))
        indices = [item.block_index for item in verdicts]
        if len(verdicts) != len(notes.blocks) or sorted(indices) != expected_indices:
            logger.warning(
                "[%s] contextual editor contract rejected: expected one verdict per block; got %s",
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

            support = _effective_support(verdict)
            threshold = 0.90 if support == "contextual" else 0.80
            if verdict.confidence < threshold:
                logger.info(
                    "[%s] editor %s for block %d ignored at confidence %.3f",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                    verdict.confidence,
                )
                continue
            if not _target_excerpt_matches_block(verdict.target_excerpt, block.latex):
                logger.warning(
                    "[%s] editor %s rejected for block %d: target excerpt mismatch",
                    chunk.id,
                    verdict.action,
                    verdict.block_index,
                )
                continue
            if support == "uncertain":
                continue

            if support == "transcript":
                if not audit_evidence_supported(
                    verdict.evidence, chunk=chunk, evidence_json=evidence_json
                ):
                    logger.warning(
                        "[%s] editor %s rejected for block %d: transcript citation not verified",
                        chunk.id,
                        verdict.action,
                        verdict.block_index,
                    )
                    continue
            elif support in {"visual", "combined"}:
                if images:
                    if verdict.evidence and not audit_evidence_supported(
                        verdict.evidence, chunk=chunk, evidence_json=evidence_json
                    ):
                        logger.warning(
                            "[%s] editor %s rejected for block %d: supplied citation not verified",
                            chunk.id,
                            verdict.action,
                            verdict.block_index,
                        )
                        continue
                elif not verdict.evidence or not audit_evidence_supported(
                    verdict.evidence, chunk=chunk, evidence_json=evidence_json
                ):
                    continue
            # contextual support is intentionally accepted without a literal source quote. It is
            # reserved for high-confidence editorial reconstruction of the intended mathematics.

            if verdict.action == "replace":
                replacement = (verdict.replacement_latex or "").strip()
                if not replacement or replacement == block.latex.strip():
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

            # Literal transcript support cannot justify erasing a lecturer statement merely
            # because the editor dislikes it. A genuine correction of a lecturer slip must be
            # represented explicitly as high-confidence contextual reconstruction.
            if support == "transcript" and _target_excerpt_is_literal_source(
                verdict.target_excerpt, chunk=chunk, evidence_json=evidence_json
            ):
                logger.warning(
                    "[%s] suppress rejected for block %d: literal transcript statement needs "
                    "contextual correction, not transcript-only suppression",
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
                f"Editor block {verdict.block_index}: suppressed unusable content: {verdict.reason}"
            )

        notes.unresolved.extend(audit_findings[:3])
        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        return notes

