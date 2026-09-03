from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import LLMConfig
from .schemas import (
    BlockReviewDecision,
    ChunkAnalysis,
    ChunkNotes,
    LectureChunk,
    NoteBlock,
    VisualEvidence,
    VisualRequest,
)
from .util import strip_thinking_and_fences

T = TypeVar("T", bound=BaseModel)


SYSTEM = """You are reconstructing faithful university lecture notes from evidence.
Never silently add facts that are not supported by the lecture evidence. Preserve the lecturer's
notation whenever it can be determined. If something remains ambiguous, record that ambiguity
rather than inventing a correction. Return strict JSON only when a JSON schema is supplied."""


class LectureModelClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("LLM/VLM backend requires the `openai` Python package") from exc
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

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
        return schema.model_validate(value)

    def _structured(self, prompt: str, schema: type[T], images: list[Path] | None = None) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        content: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\nReturn JSON matching this schema exactly:\n{schema_json}",
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
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format=self._response_format(schema),
            extra_body=self._extra_body(),
        )
        raw = response.choices[0].message.content or ""
        return self._parse_json(raw, schema)

    def analyze_chunk(
        self, chunk: LectureChunk, known_notation: dict[str, str]
    ) -> ChunkAnalysis:
        prompt = f"""Analyze the following transcript interval solely to decide whether video frames
are needed to recover information that speech alone does not determine. Typical reasons include
exact notation, an undefined symbol, a referenced drawing, a graph, arrows, a slide, or an ASR
ambiguity. Do not request frames merely because they might be interesting. Request timestamp must
be within [{chunk.start:.3f}, {chunk.end:.3f}].

Known course notation:
{json.dumps(known_notation, ensure_ascii=False, indent=2)}

Transcript:
{chunk.text}
"""
        return self._structured(prompt, ChunkAnalysis)

    def resolve_visual_request(
        self,
        request: VisualRequest,
        chunk: LectureChunk,
        frame_paths: list[Path],
    ) -> VisualEvidence:
        prompt = f"""Resolve one visual ambiguity in a lecture.
Request id: {request.id}
Reason: {request.reason}
Question: {request.question}
Target timestamp: {request.timestamp:.3f}s

Transcript context:
{chunk.text}

The attached frames are chronological. Extract only information visible in them that resolves the
request. Use exact LaTeX for mathematical notation/equations. Set best_frame_index to the zero-based
index of the most useful frame. Set requires_figure_in_notes=true only when retaining the visual
itself is materially useful (e.g. a nontrivial diagram/graph), not for ordinary equations.
"""
        evidence = self._structured(prompt, VisualEvidence, images=frame_paths)
        evidence.request_id = request.id
        return evidence

    def finalize_chunk(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        known_notation: dict[str, str],
    ) -> ChunkNotes:
        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence], ensure_ascii=False, indent=2
        )
        prompt = f"""Turn this lecture interval into concise but faithful mathematical lecture notes.
Do not add textbook material that was not stated or clearly implied in the lecture. Correct ASR only
when context or visual evidence supports the correction. Put mathematical content directly in LaTeX.
Do not emit section commands or environment commands: choose block types and let the renderer do it.
For figure blocks, asset_path must be copied exactly from visual evidence. Record unresolved
ambiguities in `unresolved`.

Known notation from earlier in the course:
{json.dumps(known_notation, ensure_ascii=False, indent=2)}

Transcript interval [{chunk.start:.3f}, {chunk.end:.3f}]:
{chunk.text}

Visual evidence:
{evidence_json}
"""
        notes = self._structured(prompt, ChunkNotes)
        notes.chunk_id = chunk.id
        notes.start = chunk.start
        notes.end = chunk.end
        return notes

    def review_block(
        self,
        block: NoteBlock,
        source_excerpts: list[dict[str, str]],
    ) -> BlockReviewDecision:
        prompt = f"""Review one reconstructed lecture-note block against retrieved literature.
The literature is supporting evidence, not an authority that automatically overrides the lecturer.
A missing match is not an error. Report `problem` only when the excerpts provide concrete evidence of
a factual error, a missing mathematical assumption, a terminology conflict, or a likely
transcription/reconstruction error. Use `ok` when no supported problem is found, and `uncertain`
when the evidence is insufficient. Suggested patches must be minimal and must not silently replace
the lecture with a textbook exposition.

Lecture block:
{json.dumps(block.model_dump(mode="json"), ensure_ascii=False, indent=2)}

Retrieved source excerpts:
{json.dumps(source_excerpts, ensure_ascii=False, indent=2)}
"""
        return self._structured(prompt, BlockReviewDecision)
