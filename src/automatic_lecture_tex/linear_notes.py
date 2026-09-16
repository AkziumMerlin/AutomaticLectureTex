from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .schemas import BlockType, LectureChunk, NotationItem, NoteBlock, VisualEvidence
from .tex_safety import strip_control_chars

_BLOCK_ID_PREFIX = "block:"


def block_id(block: NoteBlock) -> str:
    for item in block.source_claim_ids:
        if item.startswith(_BLOCK_ID_PREFIX):
            return item[len(_BLOCK_ID_PREFIX) :]
    return ""


def block_segment_ids(block: NoteBlock) -> list[str]:
    return [item for item in block.source_claim_ids if not item.startswith(_BLOCK_ID_PREFIX)]


def block_visual_ids(block: NoteBlock) -> list[str]:
    return list(block.source_evidence_ids)


def provenance_claim_ids(stable_block_id: str, segment_ids: list[str]) -> list[str]:
    return [f"{_BLOCK_ID_PREFIX}{stable_block_id}", *segment_ids]


class GeneratedLinearBlock(BaseModel):
    type: BlockType
    title: str | None = None
    latex: str = Field(min_length=1)
    source_segment_ids: list[str] = Field(min_length=1)
    visual_evidence_ids: list[str] = Field(default_factory=list)
    asset_path: str | None = None
    caption: str | None = None

    @field_validator("latex")
    @classmethod
    def nonblank_latex(cls, value: str) -> str:
        value = strip_control_chars(value)
        if not value.strip():
            raise ValueError("linear note block must have non-empty latex")
        return value

    @field_validator("source_segment_ids")
    @classmethod
    def nonblank_sources(cls, value: list[str]) -> list[str]:
        result = list(dict.fromkeys(item.strip() for item in value if item and item.strip()))
        if not result:
            raise ValueError("linear note block must cite at least one transcript segment")
        return result


class LinearPatch(BaseModel):
    target_block_id: str = Field(min_length=1)
    action: Literal["replace", "retract"] = "replace"
    replacement_latex: str | None = None
    evidence_segment_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_replacement(self) -> LinearPatch:
        if self.action == "replace" and not (self.replacement_latex or "").strip():
            raise ValueError("replace patch requires replacement_latex")
        return self


class LinearChunkDraft(BaseModel):
    section_title: str = Field(min_length=1)
    blocks: list[GeneratedLinearBlock] = Field(default_factory=list)
    notation: list[NotationItem] = Field(default_factory=list)
    recent_patches: list[LinearPatch] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class LinearCorrectionScan(BaseModel):
    patches: list[LinearPatch] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


def _recent_payload(blocks: list[NoteBlock]) -> list[dict]:
    return [
        {
            "id": block_id(block),
            "type": block.type,
            "title": block.title,
            "latex": block.latex,
            "source_segment_ids": block_segment_ids(block),
        }
        for block in blocks
    ]


def draft_linear_chunk(
    llm,
    *,
    chunk: LectureChunk,
    evidence: list[VisualEvidence],
    known_notation: dict[str, str],
    recent_blocks: list[NoteBlock],
    previous_transcript_tail: list[dict],
    output_language: str,
) -> LinearChunkDraft:
    prompt = f"""Write the next chronological chunk of faithful university lecture notes.
This is the main and only note-writing pass. Do not build a knowledge graph, semantic episode,
outline tree, or textbook reconstruction.

CURRENT transcript segment ids are authoritative provenance ids:
{json.dumps(chunk.segment_ids, ensure_ascii=False)}

Previous transcript tail (context only; do not create new blocks sourced only from it):
{json.dumps(previous_transcript_tail, ensure_ascii=False, separators=(",", ":"))}

Recent already-written blocks (context only, except for an explicit lecturer correction):
{json.dumps(_recent_payload(recent_blocks), ensure_ascii=False, separators=(",", ":"))}

Known notation:
{json.dumps(known_notation, ensure_ascii=False, separators=(",", ":"))}

Current timestamped transcript:
{chunk.timestamped_text or chunk.text}

Visual evidence for the CURRENT chunk:
{json.dumps([item.model_dump(mode="json") for item in evidence], ensure_ascii=False, separators=(",", ":"))}

Rules:
- Produce concise NoteBlocks in chronological order. Do not repeat recent blocks merely for context.
- Every new block MUST cite one or more exact CURRENT transcript ids in source_segment_ids.
- visual_evidence_ids may cite only request ids supplied above.
- You MAY repair an ASR/OCR mistake inside the current chunk when the intended reading is locally
  unambiguous from speech, board evidence, neighboring steps, or already-established notation.
- Mathematical knowledge may help disambiguate local evidence, but MUST NOT be used to silently
  replace a lecturer statement merely because the textbook answer would be different.
- If the lecturer explicitly corrects/retracts one of the supplied recent blocks, emit a
  recent_patches entry targeting its exact id. A patch must cite CURRENT transcript ids containing
  the lecturer's correction. Do not patch a block merely because you believe it is mathematically
  wrong.
- If the lecturer's intended content remains ambiguous, put it in unresolved and omit the claim.
- Do not emit TeX section/environment wrappers. Use block types; the deterministic renderer owns
  wrappers.
- Use formal theorem/definition/proposition/proof block types only when the lecture presents the
  content that way. Ordinary commentary is paragraph/remark.
- For a figure, copy asset_path exactly from supplied visual evidence.

Write prose/titles/reasons in language code `{output_language}` and preserve established notation.
"""
    return llm._structured(
        prompt,
        LinearChunkDraft,
        operation="linear_write",
        max_tokens=4096,
    )


def compact_block_catalog(blocks: list[NoteBlock], max_chars: int) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        text = block.latex.strip()
        if len(text) > max_chars:
            half = max(1, (max_chars - 5) // 2)
            text = text[:half] + " ... " + text[-half:]
        result.append(
            {
                "id": block_id(block),
                "type": block.type,
                "title": block.title,
                "latex_excerpt": text,
            }
        )
    return result


def scan_linear_corrections(
    llm,
    *,
    chunk: LectureChunk,
    evidence: list[VisualEvidence],
    earlier_blocks: list[NoteBlock],
    output_language: str,
    catalog_chars: int,
) -> LinearCorrectionScan:
    prompt = f"""Scan ONE chronological lecture interval only for explicit corrections or retractions
of material stated EARLIER in the lecture. This is not a mathematical audit and not note writing.

Current transcript segment ids:
{json.dumps(chunk.segment_ids, ensure_ascii=False)}

Current timestamped transcript:
{chunk.timestamped_text or chunk.text}

Current visual evidence:
{json.dumps([item.model_dump(mode="json") for item in evidence], ensure_ascii=False, separators=(",", ":"))}

Compact catalog of earlier note blocks:
{json.dumps(compact_block_catalog(earlier_blocks, catalog_chars), ensure_ascii=False, separators=(",", ":"))}

Return a patch ONLY when the CURRENT lecture interval itself explicitly indicates that an earlier
statement/sign/symbol/formula was wrong, misspoken, or withdrawn. Typical evidence is language such
as "нет, здесь минус", "поправлю", "выше должно быть", "я оговорился", followed by the corrected
content. A board change can support such a correction when the speech identifies it as a correction.

Do NOT patch because standard mathematics says the earlier block is wrong. Do NOT infer a correction
from mere inconsistency. Do NOT rewrite for style. When the target block cannot be identified
unambiguously from the catalog, report unresolved instead of guessing.

Each patch must target an exact catalog block id and cite exact CURRENT evidence_segment_ids where
the lecturer makes the correction. replacement_latex must be the complete replacement block content;
use action=retract only for an explicit withdrawal with no replacement.
Write reasons/unresolved text in language code `{output_language}`.
"""
    return llm._structured(
        prompt,
        LinearCorrectionScan,
        operation="linear_correction_scan",
        max_tokens=2048,
    )
