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
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"{prompt}{schema_instruction}\n\n"
                    "Return only the JSON object requested by the response schema."
                ),
            }
        ]
        for image in images or []:
            mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        parse_error: json.JSONDecodeError | ValidationError | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt and parse_error is not None:
                content[0]["text"] = (
                    f"{prompt}\n\nThe previous response was invalid ({parse_error}). "
                    "Return only a valid JSON object matching the response schema."
                )
            request_kwargs = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content},
                ],
                "temperature": self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "extra_body": self._extra_body(),
            }
            if guided_json:
                request_kwargs["response_format"] = self._response_format(schema)
            response = self.client.chat.completions.create(
                **request_kwargs,
            )
            self._record_usage(operation, response)
            raw = response.choices[0].message.content or ""
            try:
                return self._parse_json(raw, schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                parse_error = exc
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
            max_tokens=1024,
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
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        previous_context = None
        if previous_notes is not None:
            previous_context = {
                "section_title": previous_notes.section_title,
                "blocks": [block.model_dump(mode="json") for block in previous_notes.blocks[-3:]],
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

Mean ASR confidence (when available): {chunk.asr_confidence}. Treat low-confidence wording as
uncertain evidence: use visual evidence to resolve it, or record the ambiguity instead of turning it
into a mathematical claim.
Fraction of low-confidence ASR segments: {chunk.low_confidence_fraction}.

Write prose, titles, and ambiguity descriptions in language code `{self.config.output_language}`.
Do not translate established mathematical notation.
Never put guesses, alternatives, ASR commentary, or words such as "probably"/"вероятно" into note
blocks. Put them only in `unresolved`. Omit a mathematical statement unless the transcript or visual
evidence supports it.

Visual evidence:
{evidence_json}

In visual evidence, `raw_latex` is literal image-only OCR and has priority when you state what was
physically written. The `latex` field contains typography normalization only. Use the speech, known
notation, and mathematical consistency here to make any further correction or inference, and record
every such content-changing step in `corrections`. Write every user-facing string in language code
`{self.config.output_language}`, including correction reasons and unresolved items.
Treat visual confidence below 0.75 as weak evidence: it may suggest a reconstruction but must not
override clearer audio, preceding context, established notation, or a consistency check. The
preceding context is for continuity and must not be repeated unless the current interval develops
it.
"""
        notes = self._demote_speculative_blocks(
            self._structured(prompt, ChunkNotes, operation="finalize_chunk")
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
                max_tokens=4096,
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
