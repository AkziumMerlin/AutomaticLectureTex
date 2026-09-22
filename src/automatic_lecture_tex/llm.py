from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import threading
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_core import ValidationError

from .config import LLMConfig
from .schemas import (
    BlockReviewDecision,
    ChunkAnalysis,
    ChunkNotes,
    CorrectionRecord,
    LectureChunk,
    MathAudit,
    NoteBlock,
    VisualEvidence,
    VisualKind,
    VisualRequest,
)
from .util import strip_thinking_and_fences

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

_SPECULATIVE_BLOCK = re.compile(
    r"\b(ASR|вероятн\w*|или аналогич\w*|контекст\w* неоднознач\w*|может означать|"
    r"не подтвержден\w*|по-видимому|probably|or similar|unclear|may mean)\b",
    re.IGNORECASE,
)


def _restore_json_escaped_latex(value: Any) -> Any:
    if isinstance(value, str):
        restored = value.translate(
            {
                ord("\b"): r"\b",
                ord("\f"): r"\f",
                ord("\r"): r"\r",
                ord("\t"): r"\t",
                # A standalone LaTeX variable v is sometimes serialized as a vertical-tab escape.
                ord("\v"): "v",
            }
        )
        # A model occasionally emits JSON with a single slash before commands. The JSON parser
        # turns LaTeX n-commands into a newline plus the command suffix. Restore only known
        # suffixes so genuine prose line breaks remain untouched.
        return re.sub(r"\n(?=(?:abla|eq|otin|ot)\b)", r"\\n", restored)
    if isinstance(value, list):
        return [_restore_json_escaped_latex(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_json_escaped_latex(item) for key, item in value.items()}
    return value


SYSTEM = """You are reconstructing faithful university lecture notes from evidence.
Never silently add facts that are not supported by the lecture evidence. Preserve the lecturer's
notation whenever it can be determined. If something remains ambiguous, record that ambiguity
rather than inventing a correction. Return strict JSON only when a JSON schema is supplied."""


class StructuredTaskTooLargeError(RuntimeError):
    """A structured task cannot fit in one backend request and must be split upstream."""


class LectureModelClient:
    _USAGE_KEYS = (
        "requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
        "reasoning_tokens",
    )

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._usage_lock = threading.Lock()
        self._usage: dict[str, Any] = {}
        self.reset_usage()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("LLM/VLM backend requires the `openai` Python package") from exc
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    def _empty_usage() -> dict[str, Any]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
            "by_operation": {},
        }

    @classmethod
    def combine_usage(cls, usages: list[dict[str, Any]]) -> dict[str, Any]:
        combined = cls._empty_usage()
        for usage in usages:
            for key in cls._USAGE_KEYS:
                combined[key] += int(usage.get(key, 0) or 0)
            for operation, values in usage.get("by_operation", {}).items():
                target = combined["by_operation"].setdefault(
                    operation,
                    {key: 0 for key in cls._USAGE_KEYS},
                )
                for key in cls._USAGE_KEYS:
                    target[key] += int(values.get(key, 0) or 0)
        return combined

    @classmethod
    def usage_delta(cls, after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
        delta = cls._empty_usage()
        for key in cls._USAGE_KEYS:
            delta[key] = max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0))
        operations = set(after.get("by_operation", {})) | set(before.get("by_operation", {}))
        for operation in operations:
            after_values = after.get("by_operation", {}).get(operation, {})
            before_values = before.get("by_operation", {}).get(operation, {})
            values = {
                key: max(
                    0,
                    int(after_values.get(key, 0) or 0) - int(before_values.get(key, 0) or 0),
                )
                for key in cls._USAGE_KEYS
            }
            if any(values.values()):
                delta["by_operation"][operation] = values
        return delta

    def reset_usage(self) -> None:
        with self._usage_lock:
            self._usage = self._empty_usage()

    def usage_snapshot(self) -> dict[str, Any]:
        with self._usage_lock:
            return json.loads(json.dumps(self._usage))

    def _record_usage(self, operation: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        values = {
            "requests": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        with self._usage_lock:
            per_operation = self._usage["by_operation"].setdefault(
                operation,
                {key: 0 for key in values},
            )
            for key, value in values.items():
                self._usage[key] += value
                per_operation[key] += value

    def _extra_body(self) -> dict:
        return {
            "chat_template_kwargs": {
                "enable_thinking": self.config.thinking,
                "preserve_thinking": False,
            }
        }

    def _response_format(self, schema: type[T]) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }

    def _parse_json(self, raw: str, schema: type[T]) -> T:
        text = strip_thinking_and_fences(raw)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(text[start : end + 1])
        return schema.model_validate(_restore_json_escaped_latex(value))

    def _demote_speculative_blocks(self, notes: ChunkNotes) -> ChunkNotes:
        kept: list[NoteBlock] = []
        for block in notes.blocks:
            candidate = " ".join(part for part in (block.title, block.latex) if part)
            if _SPECULATIVE_BLOCK.search(candidate):
                notes.unresolved.append(f"Неподтверждённый блок исключён из TeX: {candidate}")
            else:
                kept.append(block)
        notes.blocks = kept
        notes.corrections = [
            item for item in notes.corrections if item.original.strip() != item.corrected.strip()
        ]
        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        return notes

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
        schema_instruction = ""
        if not guided_json:
            schema_instruction = "\nJSON schema:\n" + json.dumps(
                schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
            )
        base_instruction = (
            f"{prompt}{schema_instruction}\n\n"
            "Return only the JSON object requested by the response schema."
        )
        content: list[dict] = [{"type": "text", "text": base_instruction}]
        for image in images or []:
            mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )

        current_max_tokens = max_tokens or self.config.max_tokens
        # Retries are also the bounded growth budget. A call starting at 2048 with two retries can
        # therefore grow to at most 8192; a 4096-token call can grow to at most 16384. We only grow
        # when the backend explicitly reports output truncation, never for ordinary schema errors.
        max_retry_tokens = current_max_tokens * (2**self.config.max_retries)
        parse_error: json.JSONDecodeError | ValidationError | None = None
        previous_truncated = False

        for attempt in range(self.config.max_retries + 1):
            if attempt and parse_error is not None:
                if previous_truncated:
                    failure = (
                        f"The previous response was truncated by the output-token limit. "
                        f"The new output budget is {current_max_tokens} tokens."
                    )
                else:
                    failure = f"The previous response was invalid ({parse_error})."
                content[0]["text"] = (
                    f"{base_instruction}\n\n{failure} "
                    "Regenerate the complete object from the beginning; do not continue the "
                    "previous partial JSON."
                )

            request_kwargs = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content},
                ],
                "temperature": self.config.temperature,
                "max_tokens": current_max_tokens,
                "extra_body": self._extra_body(),
            }
            if guided_json:
                request_kwargs["response_format"] = self._response_format(schema)
            response = self.client.chat.completions.create(**request_kwargs)
            self._record_usage(operation, response)

            choice = response.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            truncated = finish_reason == "length" or (
                finish_reason is None
                and current_max_tokens > 0
                and completion_tokens >= current_max_tokens
            )

            try:
                return self._parse_json(raw, schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                parse_error = exc
                previous_truncated = truncated
                if truncated and current_max_tokens < max_retry_tokens:
                    next_max_tokens = min(max_retry_tokens, current_max_tokens * 2)
                    logger.warning(
                        "[%s] structured output truncated at max_tokens=%d; retrying with %d",
                        operation,
                        current_max_tokens,
                        next_max_tokens,
                    )
                    current_max_tokens = next_max_tokens

        assert parse_error is not None
        raise parse_error

    def analyze_chunk(self, chunk: LectureChunk, known_notation: dict[str, str]) -> ChunkAnalysis:
        prompt = f"""Analyze the following transcript interval solely to decide whether video frames
are needed to recover information that speech alone does not determine. Typical reasons include
exact notation, an undefined symbol, a referenced drawing, a graph, arrows, a slide, or an ASR
ambiguity. Do not request frames merely because they might be interesting. Request timestamp must
be within [{chunk.start:.3f}, {chunk.end:.3f}].

Known course notation:
{json.dumps(known_notation, ensure_ascii=False, separators=(",", ":"))}

Timestamped transcript:
{chunk.timestamped_text or chunk.text}

Mean ASR confidence (when available): {chunk.asr_confidence}
"""
        analysis = self._structured(
            prompt,
            ChunkAnalysis,
            max_tokens=1024,
            operation="visual_selector",
        )
        analysis.visual_requests = [
            request
            for request in analysis.visual_requests
            if chunk.start <= request.timestamp <= chunk.end
        ]
        return analysis

    def resolve_visual_request(
        self,
        request: VisualRequest,
        chunk: LectureChunk,
        frame_paths: list[Path],
        frame_timestamps: list[float] | None = None,
    ) -> VisualEvidence:
        frame_timestamps = frame_timestamps or []
        frame_index = "\n".join(
            f"Frame {index}: {timestamp:.3f}s" for index, timestamp in enumerate(frame_timestamps)
        )
        prompt = f"""Perform literal OCR of the relevant blackboard or slide content. This stage is
deliberately isolated from the speech transcript so that spoken context cannot leak into claims
about what is visible.

Set `raw_latex` to a literal transcription of relevant writing actually visible in the frames.
Preserve the lecturer's symbols and mark an unreadable character with `?`. Set `latex` to the same
content with unambiguous LaTeX typography normalization only (for example
Re -> \\operatorname{{Re}}), without changing variable names, signs, or completing a derivation.
If that normalization changes content rather than syntax, report it in `corrections`. Confidence
describes legibility.

Request id: {request.id}
Reason: {request.reason}
Question: {request.question}
Target timestamp: {request.timestamp:.3f}s

Attached frame timestamps:
{frame_index}

Use the request question only to locate the relevant region, never as evidence for its contents.
Use exact LaTeX for mathematical notation/equations. Set best_frame_index to the zero-based index of
the most useful frame. Set requires_figure_in_notes=true only when retaining the visual itself is
materially useful (e.g. a nontrivial diagram/graph), not for ordinary equations. If interpretation
is not reliable, preserve the ambiguity and lower confidence instead of pretending certainty.

Write descriptions in language code `{self.config.output_language}`.
"""
        # Current vLLM guided-decoding backends can collapse a valid multimodal answer to schema
        # defaults (kind=none, confidence=0). Vision extraction is therefore prompted as strict
        # JSON and validated locally; text-only calls retain faster server-side guided decoding.
        evidence = self._structured(
            prompt,
            VisualEvidence,
            images=frame_paths,
            max_tokens=2048 if request.reason == "chunk_board_scan" else 1024,
            guided_json=False,
            operation="visual_ocr",
        )
        evidence.request_id = request.id
        return evidence

    def finalize_chunk(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        known_notation: dict[str, str],
        previous_notes: ChunkNotes | None = None,
    ) -> ChunkNotes:
        evidence_json = json.dumps(
            [item.model_dump(mode="json", exclude={"frame_paths"}) for item in evidence],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # The mandatory board scan remains an image channel all the way into reconstruction.
        # Do not force the writer to trust an intermediate OCR transcription.
        multimodal_images: list[Path] = []
        multimodal_frame_labels: list[str] = []
        for item in evidence:
            if item.kind != VisualKind.BOARD_SCAN:
                continue
            for index, raw_path in enumerate(item.frame_paths):
                path = Path(raw_path)
                if not path.is_file():
                    continue
                timestamp = (
                    item.frame_timestamps[index]
                    if index < len(item.frame_timestamps)
                    else None
                )
                multimodal_frame_labels.append(
                    f"Image {len(multimodal_images)}: "
                    + (f"{timestamp:.3f}s" if timestamp is not None else "timestamp unavailable")
                )
                multimodal_images.append(path)
                if len(multimodal_images) >= 5:
                    break
            if len(multimodal_images) >= 5:
                break
        multimodal_frame_index = "\n".join(multimodal_frame_labels)

        previous_context = None
        if previous_notes is not None:
            previous_context = {
                "section_title": previous_notes.section_title,
                "blocks": [block.model_dump(mode="json") for block in previous_notes.blocks[-6:]],
                "unresolved": previous_notes.unresolved,
            }
        prompt = f"""Create concise, mathematically coherent notes from this lecture interval.
You may actively correct ASR/OCR errors, normalize terminology, reconstruct formulas from combined
audio and video evidence, and complete a short derivation when its mathematical conclusion is
reliable. Do not add unrelated textbook exposition. Every content-changing correction must be
reported in `corrections` with the source fragment, replacement, reason, basis, and confidence;
punctuation, whitespace, and purely syntactic LaTeX normalization need not be logged.
Put mathematical content directly in LaTeX.
Do not emit section commands or environment commands: choose block types and let the renderer do it.
Use formal block types only for statements presented as such in the evidence; ordinary setup or
explanation must remain a paragraph.
For figure blocks, asset_path must be copied exactly from visual evidence. Record unresolved
ambiguities in `unresolved`.

Known notation from earlier in the course:
{json.dumps(known_notation, ensure_ascii=False, separators=(",", ":"))}

Immediately preceding reconstructed context:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

Timestamped transcript interval [{chunk.start:.3f}, {chunk.end:.3f}]:
{chunk.timestamped_text or chunk.text}

Mean ASR confidence (when available): {chunk.asr_confidence}.
Fraction of low-confidence ASR segments: {chunk.low_confidence_fraction}.

The transcript is a noisy observation, not ground truth. The attached board images are a second
synchronized observation channel, and the preceding reconstructed notes provide mathematical
continuity. Reconstruct the lecturer's intended mathematics from all three together. You may use
standard mathematical knowledge as a bounded repair prior: correct garbled terminology, theorem
names, obvious lecturer slips, signs, hypotheses, and short missing steps when the surrounding
argument makes the intended result clear. For example, extension of a bounded linear functional from
a subspace with preservation of norm is the Hahn--Banach theorem even if ASR mangles the name.
If several interpretations remain genuinely plausible, prefer a descriptive formulation or record
the ambiguity rather than inventing a specific fact.

Write prose, titles, and ambiguity descriptions in language code `{self.config.output_language}`.
Do not translate established mathematical notation.
Never put unresolved guesses, alternatives, or ASR commentary into note blocks. Put genuine
ambiguities in `unresolved`. The final blocks should be mathematically correct and convenient to
study from, even when this requires contextual repair beyond the literal ASR wording. Do not add
unrelated textbook exposition or material the lecture was not developing.

Attached board-frame index (same order as attached images):
{multimodal_frame_index or "No direct board frames available."}

Auxiliary visual evidence (supplemental OCR/local requests only):
{evidence_json}

For `kind=board_scan`, the attached image itself is the evidence; `raw_latex` may intentionally
be empty because no intermediate VLM OCR pass was run. Other visual evidence entries remain
supplemental observations. Write every user-facing string in language code
`{self.config.output_language}`, including correction reasons and unresolved items. The preceding
context is for continuity and must not be repeated unless the current interval develops it.
"""
        notes = self._demote_speculative_blocks(
            self._structured(
                prompt,
                ChunkNotes,
                images=multimodal_images or None,
                guided_json=not bool(multimodal_images),
                operation="finalize_chunk",
            )
        )
        visual_corrections = [correction for item in evidence for correction in item.corrections]
        known = {
            (item.original, item.corrected, item.reason, item.basis, item.confidence)
            for item in notes.corrections
        }
        for correction in visual_corrections:
            if correction.original.strip() == correction.corrected.strip():
                continue
            key = (
                correction.original,
                correction.corrected,
                correction.reason,
                correction.basis,
                correction.confidence,
            )
            if key not in known:
                notes.corrections.append(correction)
                known.add(key)
        notes = self._audit_math(
            notes,
            chunk=chunk,
            evidence_json=evidence_json,
            previous_context=previous_context,
            images=multimodal_images or None,
        )
        notes = self._demote_speculative_blocks(notes)
        notes.chunk_id = chunk.id
        notes.start = chunk.start
        notes.end = chunk.end
        notes.section_title = notes.section_title.replace("$", "")
        return notes

    def _audit_math(
        self,
        notes: ChunkNotes,
        *,
        chunk: LectureChunk,
        evidence_json: str,
        previous_context: dict[str, Any] | None,
        images: list[Path] | None = None,
    ) -> ChunkNotes:
        equals_count = sum(block.latex.count("=") for block in notes.blocks)
        if not self.config.math_audit or equals_count < self.config.math_audit_min_equals:
            return notes

        draft = json.dumps(
            [block.model_dump(mode="json") for block in notes.blocks],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt = f"""Audit the mathematical consistency of a draft lecture-note chunk line by line.
Recompute algebraic transformations, signs, scalar factors, domains, and implications. Compare the
draft with transcript and visual evidence, but allow a correct and clearly identified mathematical
reconstruction. Return corrections only for blocks that contain a concrete error. Do not rewrite
correct blocks for style, and do not add unrelated exposition.

For each error, return its zero-based block_index and the complete corrected LaTeX/prose content of
that block. Return at most six corrections. Keep each reason under 50 words. Confidence below 0.8
means the issue is uncertain and will be reported but not applied. Do not emit raw LaTeX environment
commands. Write reasons and unresolved items in language code `{self.config.output_language}`.

Preceding context:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

Transcript:
{chunk.timestamped_text or chunk.text}

Visual evidence:
{evidence_json}

Draft blocks:
{draft}
"""
        try:
            audit = self._structured(
                prompt,
                MathAudit,
                images=images,
                max_tokens=4096,
                guided_json=not bool(images),
                operation="math_audit",
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("[%s] math audit skipped after invalid responses: %s", chunk.id, exc)
            notes.unresolved.append(
                "Математический audit не удалось разобрать после повторных попыток; "
                "основная реконструкция сохранена без его автоматических правок."
            )
            return notes
        for item in audit.corrections:
            if item.block_index >= len(notes.blocks):
                notes.unresolved.append(
                    f"Math audit returned invalid block index {item.block_index}: {item.reason}"
                )
                continue
            block = notes.blocks[item.block_index]
            if item.confidence < 0.8:
                notes.unresolved.append(
                    f"Неприменённая математическая правка (confidence={item.confidence:.2f}): "
                    f"{item.reason}"
                )
                continue
            if block.latex.strip() == item.corrected_latex.strip():
                continue
            original = block.latex
            block.latex = item.corrected_latex
            notes.corrections.append(
                CorrectionRecord(
                    original=original,
                    corrected=item.corrected_latex,
                    reason=item.reason,
                    basis="mathematical_consistency",
                    confidence=item.confidence,
                )
            )
        notes.unresolved.extend(audit.unresolved)
        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        return notes

    def review_block(
        self,
        block: NoteBlock,
        source_excerpts: list[dict[str, str]],
    ) -> BlockReviewDecision:
        prompt = f"""Review one reconstructed lecture-note block against retrieved literature.
The literature is supporting evidence, not an authority that automatically overrides the lecturer.
A missing match is not an error. Report `problem` only when the excerpts provide concrete evidence
of a factual error, a missing mathematical assumption, a terminology conflict, or a likely
transcription/reconstruction error. Use `ok` when no supported problem is found, and `uncertain`
when the evidence is insufficient. Suggested patches must be minimal and must not silently replace
the lecture with a textbook exposition.

Lecture block:
{json.dumps(block.model_dump(mode="json"), ensure_ascii=False, indent=2)}

Retrieved source excerpts:
{json.dumps(source_excerpts, ensure_ascii=False, indent=2)}

Write the review in language code `{self.config.output_language}`.
"""
        return self._structured(
            prompt,
            BlockReviewDecision,
            operation="literature_review",
        )