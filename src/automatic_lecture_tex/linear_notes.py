from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .schemas import BlockType, LectureChunk, LectureIR, NotationItem, NoteBlock, VisualEvidence
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


class GlobalBlockEdit(BaseModel):
    target_block_id: str = Field(min_length=1)
    action: Literal["replace", "drop"]
    replacement_latex: str | None = None
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action(self) -> GlobalBlockEdit:
        if self.action == "replace" and not (self.replacement_latex or "").strip():
            raise ValueError("global replace edit requires replacement_latex")
        return self


class GlobalSectionPlan(BaseModel):
    title: str = Field(min_length=1)
    block_ids: list[str] = Field(min_length=1)


class GlobalLectureEditPlan(BaseModel):
    sections: list[GlobalSectionPlan] = Field(min_length=1)
    patches: list[GlobalBlockEdit] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class GlobalSectionBoundary(BaseModel):
    title: str = Field(min_length=1)
    first_block_id: str = Field(min_length=1)


class GlobalLectureStructurePlan(BaseModel):
    sections: list[GlobalSectionBoundary] = Field(min_length=1)
    drops: list[GlobalBlockEdit] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def drops_only(self) -> GlobalLectureStructurePlan:
        if any(item.action != "drop" for item in self.drops):
            raise ValueError("global structure pass may emit drop edits only")
        return self


class GlobalLecturePatchBatch(BaseModel):
    patches: list[GlobalBlockEdit] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def replacements_only(self) -> GlobalLecturePatchBatch:
        if any(item.action != "replace" for item in self.patches):
            raise ValueError("global math batch may emit replace edits only")
        return self


def _ordered_global_blocks(ir: LectureIR) -> list[tuple[str, NoteBlock, str, list[str]]]:
    result: list[tuple[str, NoteBlock, str, list[str]]] = []
    for chunk in ir.chunks:
        for block in chunk.blocks:
            stable_id = block_id(block)
            if not stable_id:
                raise ValueError("global editor requires stable ids on every draft block")
            result.append((stable_id, block, chunk.section_title, list(chunk.unresolved)))
    return result


def _shorten_global_text(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 7:
        return text[:max_chars]
    half = (max_chars - 5) // 2
    return text[:half] + " ... " + text[-half:]


def _global_catalog_payload(ir: LectureIR, excerpt_chars: int) -> list[dict]:
    payload: list[dict] = []
    for chunk_index, chunk in enumerate(ir.chunks):
        payload.append(
            {
                "chunk_index": chunk_index,
                "section_title": chunk.section_title,
                "unresolved": [
                    _shorten_global_text(item, 180) for item in chunk.unresolved[:4]
                ],
                "blocks": [
                    {
                        "id": block_id(block),
                        "type": block.type,
                        "title": block.title,
                        "excerpt": _shorten_global_text(block.latex, excerpt_chars),
                    }
                    for block in chunk.blocks
                ],
            }
        )
    return payload


def _expand_global_structure(
    draft_ir: LectureIR,
    structure: GlobalLectureStructurePlan,
    *,
    apply_threshold: float,
) -> tuple[list[GlobalSectionPlan], list[GlobalBlockEdit]]:
    ordered = _ordered_global_blocks(draft_ir)
    ordered_ids = [item[0] for item in ordered]
    known = set(ordered_ids)

    effective_drops: list[GlobalBlockEdit] = []
    dropped: set[str] = set()
    for patch in structure.drops:
        if patch.target_block_id not in known:
            raise ValueError(f"global structure references unknown block {patch.target_block_id!r}")
        if patch.confidence < apply_threshold:
            continue
        if patch.target_block_id in dropped:
            raise ValueError(f"duplicate global drop for {patch.target_block_id!r}")
        dropped.add(patch.target_block_id)
        effective_drops.append(patch)

    retained_ids = [stable_id for stable_id in ordered_ids if stable_id not in dropped]
    if not retained_ids:
        raise ValueError("global structure dropped every lecture block")

    boundaries = structure.sections
    boundary_ids = [item.first_block_id for item in boundaries]
    if len(boundary_ids) != len(set(boundary_ids)):
        raise ValueError("global structure contains duplicate section boundaries")
    if any(stable_id not in retained_ids for stable_id in boundary_ids):
        raise ValueError("global section boundary must reference a retained block")
    if boundary_ids[0] != retained_ids[0]:
        raise ValueError("first global section must start at the first retained block")

    position = {stable_id: index for index, stable_id in enumerate(retained_ids)}
    boundary_positions = [position[stable_id] for stable_id in boundary_ids]
    if boundary_positions != sorted(boundary_positions):
        raise ValueError("global section boundaries must preserve lecture order")

    sections: list[GlobalSectionPlan] = []
    for index, boundary in enumerate(boundaries):
        start = boundary_positions[index]
        end = (
            boundary_positions[index + 1]
            if index + 1 < len(boundary_positions)
            else len(retained_ids)
        )
        ids = retained_ids[start:end]
        if not ids:
            raise ValueError("global structure produced an empty section")
        sections.append(GlobalSectionPlan(title=boundary.title.strip(), block_ids=ids))
    return sections, effective_drops


def _global_batch_payload(
    ordered: list[tuple[str, NoteBlock, str, list[str]]],
    ids: list[str],
) -> list[dict]:
    wanted = set(ids)
    return [
        {
            "id": stable_id,
            "source_section": section_title,
            "type": block.type,
            "title": block.title,
            "latex": block.latex,
            "unresolved": unresolved[:3],
        }
        for stable_id, block, section_title, unresolved in ordered
        if stable_id in wanted
    ]


def _iter_global_batches(
    ordered: list[tuple[str, NoteBlock, str, list[str]]],
    retained_ids: set[str],
    max_chars: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for stable_id, block, section_title, unresolved in ordered:
        if stable_id not in retained_ids:
            continue
        item_chars = (
            len(stable_id)
            + len(section_title)
            + len(block.title or "")
            + len(block.latex)
            + sum(len(item) for item in unresolved[:3])
            + 160
        )
        if current and current_chars + item_chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(stable_id)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


def plan_global_lecture_edit(
    llm,
    *,
    draft_ir: LectureIR,
    output_language: str,
    apply_threshold: float = 0.85,
    batch_chars: int = 16000,
    catalog_excerpt_chars: int = 140,
) -> GlobalLectureEditPlan:
    """Build a lecture-wide edit plan without putting the full lecture into one context window."""

    catalog = _global_catalog_payload(draft_ir, catalog_excerpt_chars)
    catalog_json = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))

    structure_prompt = f"""Plan the GLOBAL structure of a reconstructed university lecture.
You see a compact catalog of the complete lecture, not the full block text. Use it to merge local
3-minute chunk headings into coherent study-note sections and to identify clear duplicate/redundant
blocks. Do not perform detailed mathematical rewrites in this pass.

COMPLETE COMPACT LECTURE CATALOG:
{catalog_json}

Return section BOUNDARIES only:
- sections are chronological;
- first_block_id is the exact id of the first retained block of that section;
- the first section must begin at the first non-dropped lecture block;
- use substantially fewer meaningful sections than local chunks;
- do not reorder blocks.

Return drops only for high-confidence duplication, redundant transitions, superseded text, or
unrecoverable noise visible from the catalog. Do not drop a substantive mathematical block just
because its excerpt looks suspicious; detailed mathematics is handled later.
Use canonical mathematical names in section titles. Write text in language code
`{output_language}`.
"""
    structure = llm._structured(
        structure_prompt,
        GlobalLectureStructurePlan,
        operation="global_lecture_structure",
        max_tokens=3072,
    )
    sections, effective_drops = _expand_global_structure(
        draft_ir,
        structure,
        apply_threshold=apply_threshold,
    )

    ordered = _ordered_global_blocks(draft_ir)
    dropped_ids = {item.target_block_id for item in effective_drops}
    retained_ids = {stable_id for stable_id, *_ in ordered if stable_id not in dropped_ids}
    batches = _iter_global_batches(ordered, retained_ids, batch_chars)

    section_context = [
        {
            "title": section.title,
            "first_block_id": section.block_ids[0],
            "last_block_id": section.block_ids[-1],
            "count": len(section.block_ids),
        }
        for section in sections
    ]
    section_json = json.dumps(section_context, ensure_ascii=False, separators=(",", ":"))

    patches: list[GlobalBlockEdit] = list(effective_drops)
    unresolved = list(structure.unresolved)
    patched_ids = set(dropped_ids)

    for batch_index, batch_ids in enumerate(batches):
        full_batch = _global_batch_payload(ordered, batch_ids)
        batch_prompt = f"""Mathematically edit ONE bounded batch of blocks from a complete reconstructed
lecture. The compact catalog and final section layout provide GLOBAL context; only blocks in
CURRENT FULL-TEXT BATCH may be rewritten.

COMPLETE COMPACT CATALOG:
{catalog_json}

FINAL SECTION LAYOUT:
{section_json}

CURRENT FULL-TEXT BATCH {batch_index + 1}/{len(batches)}:
{json.dumps(full_batch, ensure_ascii=False, separators=(",", ":"))}

Return replacement patches only when a current block contains a real mathematical/content error:
wrong theorem name, missing/incorrect hypothesis, wrong inequality direction, confused object role,
sign/domain/codomain/topology/compactness error, or contradiction with the global lecture context.
Use standard mathematics to recover the intended lecture statement when it is clear.

Rules:
- target_block_id must be from CURRENT FULL-TEXT BATCH only;
- action must be replace;
- replacement_latex is the complete corrected content of that ONE block;
- do not drop blocks here; deduplication was handled by the global structure pass;
- do not rewrite merely for style;
- preserve useful lecture-specific derivations/examples and established notation;
- do not add unrelated textbook exposition;
- if global context reveals an ambiguity that cannot be resolved confidently, report unresolved
  instead of guessing.
Write reasons/unresolved text in language code `{output_language}`.
"""
        batch_result = llm._structured(
            batch_prompt,
            GlobalLecturePatchBatch,
            operation="global_lecture_math_batch",
            max_tokens=4096,
        )
        allowed = set(batch_ids)
        for patch in batch_result.patches:
            if patch.target_block_id not in allowed:
                raise ValueError(
                    f"global math batch references out-of-batch block {patch.target_block_id!r}"
                )
            if patch.target_block_id in patched_ids:
                raise ValueError(f"multiple global edits target {patch.target_block_id!r}")
            patched_ids.add(patch.target_block_id)
            patches.append(patch)
        unresolved.extend(batch_result.unresolved)

    return GlobalLectureEditPlan(
        sections=sections,
        patches=patches,
        unresolved=list(dict.fromkeys(unresolved)),
    )

