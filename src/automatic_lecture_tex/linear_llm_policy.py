from __future__ import annotations

import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .llm_robust import LectureModelClient as RobustLectureModelClient
from .schemas import ChunkNotes

T = TypeVar("T", bound=BaseModel)

# Included in linear cache identity by pipeline_robust so prompt-policy changes cannot silently reuse
# older chunk artifacts.
LINEAR_SOURCE_POLICY_VERSION = 1

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
- If a faithful statement cannot be reconstructed from current evidence, put the ambiguity in
  unresolved and OMIT the unsupported claim from note blocks.
"""

_AUDIT_SOURCE_POLICY = r"""

SOURCE-FAITHFUL AUDIT POLICY:
- First check the draft literally against the transcript and visual evidence. Mathematical
  plausibility by itself is NOT evidence.
- Check exact signs, constants, variable names, indices, quantifiers, domains/codomains, and object
  identity. Explicitly check distinctions such as z_f versus y_f and unique versus up-to-scalar.
- Detect source drift: if a draft block introduces a sequence, construction, theorem, assumption,
  example, or multi-step argument not actually present in the current transcript/board evidence, do
  not justify it using standard mathematics. Replace it with the minimal source-supported content
  when possible; otherwise report it in unresolved.
- Use mathematical consistency only to identify likely reconstruction/transcription mistakes. Apply
  a correction only when the corrected version is supported by the local lecture evidence or
  unambiguous established notation.
- Never change a lecturer statement solely because a standard textbook would state something else.
  Explicit later lecturer corrections belong to the separate cross-chunk correction pass.
- Do not rewrite for style and do not expand the lecture.
"""

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
        r"\b(?:нет|не)\s*,?\s*(?:здесь|тут|там)\b.{0,80}\b(?:плюс|минус|знак|индекс|букв\w*|должн\w*)\b",
        r"\b(?:i\s+misspoke|correction|correct that|should be|rather than)\b",
    )
)


def _current_correction_transcript(prompt: str) -> str:
    """Extract only the CURRENT transcript from the correction-scan prompt.

    Searching the whole prompt would spuriously trigger on words such as "ошибка" inside the
    catalog of earlier note blocks or in the instructions themselves.
    """

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


class LectureModelClient(RobustLectureModelClient):
    """Robust client with narrow policies specific to the simple linear production path.

    The underlying pre-PR2 ``finalize_chunk`` implementation is intentionally left untouched. This
    subclass only adds source-boundary instructions at the structured-call boundary, makes the local
    audit run on every non-empty chunk, and skips the expensive distant-correction scan unless the
    current transcript contains an explicit correction cue.
    """

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

    def _audit_math(self, notes: ChunkNotes, **kwargs) -> ChunkNotes:
        if not self.config.math_audit or not notes.blocks:
            return notes

        # The historical writer gated audit by the number of '=' characters, which misses prose
        # claims such as uniqueness/up-to-scalar and object-identity mistakes. Linear mode audits
        # every non-empty chunk. Temporarily lowering the legacy threshold lets us reuse the exact
        # old audit implementation rather than forking it.
        previous_threshold = self.config.math_audit_min_equals
        object.__setattr__(self.config, "math_audit_min_equals", 0)
        try:
            return super()._audit_math(notes, **kwargs)
        finally:
            object.__setattr__(self.config, "math_audit_min_equals", previous_threshold)
