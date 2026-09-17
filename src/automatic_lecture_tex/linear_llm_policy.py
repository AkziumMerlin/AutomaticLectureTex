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
LINEAR_SOURCE_POLICY_VERSION = 2

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
- If a faithful statement cannot be reconstructed from current evidence, put that material ambiguity
  in unresolved and OMIT the unsupported claim from note blocks.
"""

_AUDIT_SOURCE_POLICY = r"""

SOURCE-FAITHFUL AUDIT POLICY:
- This is a source-fidelity checker, not a second mathematical author and not a textbook solver.
- First compare each retained draft block literally with the CURRENT transcript and visual evidence.
- Check exact signs, constants, variable names, indices, quantifiers, domains/codomains, and object
  identity. Explicitly check distinctions such as z_f versus y_f and unique versus up-to-scalar.
- A correction MUST cite one or more verbatim source excerpts in `evidence`. Each excerpt must be
  copied from the current transcript or current visual evidence. Do not cite preceding notes,
  textbook knowledge, or your own derivation as evidence.
- Mathematical consistency may help detect a likely draft error, but mathematical plausibility is
  not evidence. Never change a lecturer statement solely because a standard theorem says otherwise.
- A short algebraic correction is allowed only when it is directly forced by the cited local source
  formulas. Do not introduce a new construction, theorem, assumption, sequence, or multi-step proof.
- If a retained block appears to misrepresent the source but no source-backed replacement is safe,
  return an `issue` for that block instead of a correction.
- Do NOT report raw ASR garbage, filler, false starts, or unrelated source ambiguities. An issue must
  concern an existing retained draft block and materially affect what the notes say.
- Do not rewrite for style and do not expand the lecture.
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


class SourceGroundedAuditCorrection(BaseModel):
    block_index: int = Field(ge=0)
    corrected_latex: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[AuditEvidence] = Field(default_factory=list)


class SourceGroundedAuditIssue(BaseModel):
    block_index: int = Field(ge=0)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class SourceGroundedMathAudit(BaseModel):
    corrections: list[SourceGroundedAuditCorrection] = Field(default_factory=list)
    issues: list[SourceGroundedAuditIssue] = Field(default_factory=list)


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

        draft = json.dumps(
            [block.model_dump(mode="json") for block in notes.blocks],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = f"""Check a draft lecture-note chunk only for fidelity to the supplied lecture source.
Do not solve the mathematics from general knowledge and do not replace the lecturer with a textbook.

For a concrete source-backed error in a retained block, return a correction with:
- the zero-based block_index;
- the complete replacement block content;
- a short reason;
- confidence;
- one or more `evidence` items, each containing `source` (`transcript` or `visual`) and a VERBATIM
  contiguous `quote` copied from that current source. The host will reject a correction if a quote
  cannot be found literally in the declared source.

Return at most four corrections. If a retained block materially misrepresents the current source but
there is no safe source-backed replacement, return an issue for that block instead. Return at most
three issues. Do not report raw ASR garbage, filler, false starts, or ambiguities that did not affect
an existing retained block. Do not rewrite correct blocks for style.

Preceding context is continuity context only, not evidence for a correction:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

CURRENT transcript:
{chunk.timestamped_text or chunk.text}

CURRENT visual evidence:
{evidence_json}

Draft blocks:
{draft}

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

        audit_findings: list[str] = []
        for item in audit.corrections:
            if item.block_index >= len(notes.blocks):
                logger.warning("[%s] audit returned invalid block index %d", chunk.id, item.block_index)
                continue
            block = notes.blocks[item.block_index]
            if block.latex.strip() == item.corrected_latex.strip():
                continue
            if item.confidence < 0.8:
                audit_findings.append(
                    f"Audit block {item.block_index}: low-confidence finding not applied: {item.reason}"
                )
                continue
            if not audit_evidence_supported(item.evidence, chunk=chunk, evidence_json=evidence_json):
                audit_findings.append(
                    f"Audit block {item.block_index}: source support not verified; not applied: {item.reason}"
                )
                continue

            original = block.latex
            block.latex = item.corrected_latex
            notes.corrections.append(
                CorrectionRecord(
                    original=original,
                    corrected=item.corrected_latex,
                    reason=item.reason,
                    basis=_basis_from_evidence(item.evidence),
                    confidence=item.confidence,
                )
            )

        for issue in audit.issues:
            if issue.block_index >= len(notes.blocks) or issue.confidence < 0.6:
                continue
            audit_findings.append(f"Audit block {issue.block_index}: {issue.reason}")

        # Keep audit diagnostics small and block-linked. They remain in the audit sidecar through
        # ChunkNotes.unresolved, but raw ASR noise can no longer flood it.
        notes.unresolved.extend(audit_findings[:4])
        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        return notes
