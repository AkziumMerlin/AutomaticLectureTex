from __future__ import annotations

import json
import re
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
    merge_into_block_id: str | None = None
    merge_kind: Literal["exact_dedup", "reconciliation"] | None = None
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action(self) -> GlobalBlockEdit:
        if self.action == "replace":
            if not (self.replacement_latex or "").strip():
                raise ValueError("global replace edit requires replacement_latex")
            if self.merge_into_block_id is not None or self.merge_kind is not None:
                raise ValueError("replace edit cannot merge provenance into another block")
        else:
            if self.merge_into_block_id == self.target_block_id:
                raise ValueError("drop edit cannot merge provenance into itself")
            if (self.merge_into_block_id is None) != (self.merge_kind is None):
                raise ValueError("drop provenance merge requires both merge target and merge kind")
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
        if self.drops:
            raise ValueError(
                "global structure pass must not drop content before section-level reconciliation"
            )
        return self


class GlobalSectionReconciliation(BaseModel):
    target_block_id: str = Field(min_length=1)
    source_block_ids: list[str] = Field(min_length=1)
    replacement_latex: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ids(self) -> GlobalSectionReconciliation:
        sources = list(dict.fromkeys(self.source_block_ids))
        if len(sources) != len(self.source_block_ids):
            raise ValueError("reconciliation source ids must be unique")
        if self.target_block_id in sources:
            raise ValueError("reconciliation target cannot also be a source")
        return self


class GlobalLectureSectionEdit(BaseModel):
    patches: list[GlobalBlockEdit] = Field(default_factory=list)
    reconciliations: list[GlobalSectionReconciliation] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def replacements_only(self) -> GlobalLectureSectionEdit:
        if any(item.action != "replace" for item in self.patches):
            raise ValueError("section editor patches may only replace blocks")
        if any(
            item.merge_into_block_id is not None or item.merge_kind is not None
            for item in self.patches
        ):
            raise ValueError("section editor replacement patches cannot request provenance merges")
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


_DUPLICATE_SPACE = re.compile(r"\s+")
_EXACT_DEDUP_MIN_CHARS = 80


def _exact_duplicate_key(block: NoteBlock) -> tuple:
    """Canonical key for deterministic deduplication of genuinely identical note blocks."""

    return (
        block.type,
        _DUPLICATE_SPACE.sub(" ", (block.title or "").strip()),
        _DUPLICATE_SPACE.sub(" ", block.latex.strip()),
        block.asset_path or "",
        _DUPLICATE_SPACE.sub(" ", (block.caption or "").strip()),
    )


def _deduplicate_exact_within_sections(
    draft_ir: LectureIR,
    sections: list[GlobalSectionPlan],
    *,
    protected_ids: set[str] | None = None,
) -> tuple[list[GlobalSectionPlan], list[GlobalBlockEdit]]:
    """Drop exact repeats only inside one final semantic section.

    Keep the first occurrence. The drop records a merge target so the final host-side application
    can preserve transcript/visual provenance from a lecturer's later recap.
    """

    block_map = {
        stable_id: block
        for stable_id, block, _section_title, _unresolved in _ordered_global_blocks(draft_ir)
    }
    result_sections: list[GlobalSectionPlan] = []
    drops: list[GlobalBlockEdit] = []
    protected = set(protected_ids or set())

    for section in sections:
        seen: dict[tuple, str] = {}
        kept_ids: list[str] = []
        for stable_id in section.block_ids:
            block = block_map[stable_id]
            # Figures can legitimately reuse captions/assets and should never be collapsed here.
            if block.asset_path:
                kept_ids.append(stable_id)
                continue
            normalized_latex = _DUPLICATE_SPACE.sub(" ", block.latex.strip())
            if len(normalized_latex) < _EXACT_DEDUP_MIN_CHARS:
                kept_ids.append(stable_id)
                continue
            key = _exact_duplicate_key(block)
            first_id = seen.get(key)
            if first_id is None:
                seen[key] = stable_id
                kept_ids.append(stable_id)
                continue
            if stable_id in protected:
                # Never drop a reconciliation target into an unrelated earlier duplicate.
                kept_ids.append(stable_id)
                seen[key] = stable_id
                continue
            # If the already-kept first block is a reconciliation target, it is safe to merge
            # later exact duplicates directly into that protected canonical target.
            drops.append(
                GlobalBlockEdit(
                    target_block_id=stable_id,
                    action="drop",
                    merge_into_block_id=first_id,
                    merge_kind="exact_dedup",
                    reason="Точный повтор уже сохранённого блока в том же итоговом разделе.",
                    confidence=1.0,
                )
            )
        if not kept_ids:
            raise ValueError("exact deduplication emptied a global section")
        result_sections.append(
            GlobalSectionPlan(title=section.title, block_ids=kept_ids)
        )
    return result_sections, drops


def _remove_reconciliation_sources(
    sections: list[GlobalSectionPlan],
    source_ids: set[str],
) -> list[GlobalSectionPlan]:
    result: list[GlobalSectionPlan] = []
    for section in sections:
        kept = [stable_id for stable_id in section.block_ids if stable_id not in source_ids]
        if not kept:
            raise ValueError("section reconciliation removed every block from a section")
        result.append(GlobalSectionPlan(title=section.title, block_ids=kept))
    return result


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


def _block_payload_chars(
    block: NoteBlock,
    section_title: str,
    unresolved: list[str],
) -> int:
    return (
        len(section_title)
        + len(block.title or "")
        + len(block.latex)
        + sum(len(item) for item in unresolved[:3])
        + 160
    )


def _iter_section_editor_units(
    ordered: list[tuple[str, NoteBlock, str, list[str]]],
    sections: list[GlobalSectionPlan],
    max_chars: int,
) -> list[tuple[str, int, int, list[str]]]:
    """Prefer one full semantic section per editor call; split only oversized sections."""

    block_info = {
        stable_id: (block, source_title, unresolved)
        for stable_id, block, source_title, unresolved in ordered
    }
    units: list[tuple[str, int, int, list[str]]] = []
    for section in sections:
        parts: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for stable_id in section.block_ids:
            block, _source_title, unresolved = block_info[stable_id]
            item_chars = _block_payload_chars(block, section.title, unresolved)
            if current and current_chars + item_chars > max_chars:
                parts.append(current)
                current = []
                current_chars = 0
            current.append(stable_id)
            current_chars += item_chars
        if current:
            parts.append(current)
        for part_index, ids in enumerate(parts):
            units.append((section.title, part_index, len(parts), ids))
    return units


def plan_global_lecture_edit(
    llm,
    *,
    draft_ir: LectureIR,
    output_language: str,
    apply_threshold: float = 0.85,
    batch_chars: int = 16000,
    catalog_excerpt_chars: int = 140,
    course_conventions: list[str] | None = None,
) -> GlobalLectureEditPlan:
    """Build a lecture-wide edit plan without putting the full lecture into one context window."""

    catalog = _global_catalog_payload(draft_ir, catalog_excerpt_chars)
    catalog_json = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    conventions_json = json.dumps(
        list(course_conventions or []),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    structure_prompt = f"""Plan the GLOBAL structure of a reconstructed university lecture.
You see a compact catalog of the complete lecture, not the full block text. Use it to merge local
3-minute chunk headings into coherent study-note sections and to identify clear duplicate/redundant
blocks. Do not perform detailed mathematical rewrites in this pass.

COMPLETE COMPACT LECTURE CATALOG:
{catalog_json}

COURSE-SPECIFIC CONVENTIONS (authoritative; preserve them rather than "correcting" them):
{conventions_json}

Return section BOUNDARIES only:
- sections are chronological;
- first_block_id is the exact id of the first retained block of that section;
- the first section must begin at the first non-dropped lecture block;
- use substantially fewer meaningful sections than local chunks;
- do not reorder blocks.

Do NOT drop any blocks in this pass: return drops=[].
Repeated material can contain a lecturer correction or clarification, so every version must remain
visible to the section-level mathematical editor. Deduplication happens only after mathematical
reconciliation.
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
    units = _iter_section_editor_units(ordered, sections, batch_chars)

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
    patched_ids = {item.target_block_id for item in effective_drops}

    for unit_index, (section_title, part_index, part_count, batch_ids) in enumerate(units):
        full_batch = _global_batch_payload(ordered, batch_ids)
        batch_prompt = f"""Mathematically edit ONE semantic section of a reconstructed university
lecture. Normally this call sees the whole final section; only an oversized section is split into
multiple parts. This is deliberate: proofs, notation roles, and theorem hypotheses should be checked
together rather than in arbitrary character windows.

COMPLETE COMPACT CATALOG:
{catalog_json}

COURSE-SPECIFIC CONVENTIONS (authoritative; do NOT "correct" these):
{conventions_json}

FINAL SECTION LAYOUT:
{section_json}

CURRENT SECTION: {section_title}
SECTION PART: {part_index + 1}/{part_count}
EDITOR UNIT: {unit_index + 1}/{len(units)}
CURRENT FULL-TEXT SECTION BLOCKS:
{json.dumps(full_batch, ensure_ascii=False, separators=(",", ":"))}

First compare repeated or near-repeated versions of the SAME mathematical step/proof inside
this section. A later repetition may be a lecturer clarification or correction rather than noise.

Return:
1. ordinary replacement patches for isolated mathematical/content errors;
2. reconciliations when two or more blocks are alternative versions of the same semantic step and
   one canonical block should replace them all.

A reconciliation means:
- target_block_id: the EARLIEST logical occurrence among the versions being unified;
- source_block_ids: later/alternative versions that become redundant after reconciliation;
- replacement_latex: one complete mathematically correct canonical version, preserving all useful
  detail and using an explicit/later lecturer correction when the versions conflict;
- do not reconcile blocks that merely discuss related material or where a later repetition adds a
  genuinely distinct argument/example;
- if a repeated proof spans several blocks, emit multiple reconciliations for corresponding steps,
  or one reconciliation with several source blocks only when they jointly represent the same
  semantic unit as the target.

Ordinary error examples include wrong theorem name, missing/incorrect hypothesis, wrong inequality
direction, confused object role, sign/domain/codomain/topology/compactness error, or contradiction
with another version in this section. Independently recompute short algebraic/sign steps rather than
trusting fluent prose. Check quantifiers and dimension assumptions, topology/compactness hypotheses,
neighborhood centers and epsilon margins in convergence proofs, and whether a claimed separating
family really separates points. Use standard mathematics to recover the intended lecture statement
when it is clear.

Rules:
- every target/source id must be from CURRENT FULL-TEXT SECTION BLOCKS only;
- ordinary patches use action=replace and replace exactly one complete block;
- reconciliations are the ONLY way the model may request removal of a non-identical repeated version;
- do not rewrite merely for style;
- preserve useful lecture-specific derivations/examples, chronology, and established notation;
- preserve COURSE-SPECIFIC CONVENTIONS exactly;
- do not add unrelated textbook exposition;
- if competing versions cannot be reconciled confidently, report unresolved instead of guessing.
Write reasons/unresolved text in language code `{output_language}`.
"""
        batch_result = llm._structured(
            batch_prompt,
            GlobalLectureSectionEdit,
            operation="global_lecture_section_edit",
            max_tokens=4096,
        )
        allowed = set(batch_ids)
        position = {stable_id: index for index, stable_id in enumerate(batch_ids)}
        unit_claimed_ids: set[str] = set()

        for reconciliation in batch_result.reconciliations:
            ids = [reconciliation.target_block_id, *reconciliation.source_block_ids]
            unknown_ids = [stable_id for stable_id in ids if stable_id not in allowed]
            if unknown_ids:
                raise ValueError(
                    f"section reconciliation references out-of-batch blocks {unknown_ids!r}"
                )
            if reconciliation.confidence < apply_threshold:
                continue
            if reconciliation.target_block_id != min(ids, key=position.__getitem__):
                raise ValueError("reconciliation target must be the earliest unified block")
            overlap = unit_claimed_ids.intersection(ids)
            if overlap:
                raise ValueError(
                    f"section reconciliation reuses blocks already reconciled: {sorted(overlap)!r}"
                )
            if patched_ids.intersection(ids):
                raise ValueError("section reconciliation conflicts with another global edit")

            unit_claimed_ids.update(ids)
            patched_ids.update(ids)
            patches.append(
                GlobalBlockEdit(
                    target_block_id=reconciliation.target_block_id,
                    action="replace",
                    replacement_latex=reconciliation.replacement_latex,
                    reason=reconciliation.reason,
                    confidence=reconciliation.confidence,
                )
            )
            for source_id in reconciliation.source_block_ids:
                patches.append(
                    GlobalBlockEdit(
                        target_block_id=source_id,
                        action="drop",
                        merge_into_block_id=reconciliation.target_block_id,
                        merge_kind="reconciliation",
                        reason=reconciliation.reason,
                        confidence=reconciliation.confidence,
                    )
                )

        for patch in batch_result.patches:
            if patch.target_block_id not in allowed:
                raise ValueError(
                    f"global math batch references out-of-batch block {patch.target_block_id!r}"
                )
            if patch.target_block_id in unit_claimed_ids or patch.target_block_id in patched_ids:
                raise ValueError(f"multiple global edits target {patch.target_block_id!r}")
            patched_ids.add(patch.target_block_id)
            patches.append(patch)
        unresolved.extend(batch_result.unresolved)

    reconciliation_source_ids = {
        patch.target_block_id
        for patch in patches
        if patch.action == "drop" and patch.merge_kind == "reconciliation"
    }
    sections_after_reconciliation = _remove_reconciliation_sources(
        sections,
        reconciliation_source_ids,
    )

    # Apply high-confidence canonical replacements in a temporary copy BEFORE exact dedup.
    # Explicit reconciliations have already removed their alternative source blocks from the
    # section plan, but those sources remain in draft_ir for provenance merging during final apply.
    reconciled = draft_ir.model_copy(deep=True)
    reconciled_map = {
        stable_id: block
        for stable_id, block, _section_title, _unresolved in _ordered_global_blocks(reconciled)
    }
    for patch in patches:
        if (
            patch.action == "replace"
            and patch.confidence >= apply_threshold
            and patch.target_block_id in reconciled_map
        ):
            reconciled_map[patch.target_block_id].latex = (patch.replacement_latex or "").strip()

    reconciliation_targets = {
        patch.merge_into_block_id
        for patch in patches
        if patch.action == "drop"
        and patch.merge_kind == "reconciliation"
        and patch.merge_into_block_id is not None
    }
    dedup_sections, exact_drops = _deduplicate_exact_within_sections(
        reconciled,
        sections_after_reconciliation,
        protected_ids=reconciliation_targets,
    )
    exact_drop_ids = {item.target_block_id for item in exact_drops}
    # A block that is removed after reconciliation does not need its own replacement patch in the
    # final plan. Its corrected semantics are represented by the retained equivalent block, while
    # its source provenance is merged into that block by the host.
    patches = [
        patch
        for patch in patches
        if not (patch.action == "replace" and patch.target_block_id in exact_drop_ids)
    ]
    patches.extend(exact_drops)

    return GlobalLectureEditPlan(
        sections=dedup_sections,
        patches=patches,
        unresolved=list(dict.fromkeys(unresolved)),
    )

