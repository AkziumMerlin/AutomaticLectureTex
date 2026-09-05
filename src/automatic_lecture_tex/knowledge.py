from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import NotesConfig
from .schemas import (
    AnchorKind,
    ChunkNotes,
    ClaimStatus,
    CorrectionRecord,
    GlobalValidation,
    KnowledgeClaim,
    KnowledgeUpdate,
    LectureChunk,
    LectureIR,
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    NoteBlock,
    ObservationKind,
    OutlineSection,
    SemanticAnchor,
    SourceStatus,
    SymbolRecord,
    Transcript,
    VisualEvidence,
    WindowObservations,
)

if TYPE_CHECKING:
    from .llm import LectureModelClient


_WS = re.compile(r"\s+")


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return _WS.sub("", value).lower()


def _merge_unique(left: list[str], right: Iterable[str]) -> list[str]:
    seen = set(left)
    result = list(left)
    for item in right:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _observation_key(obs: LectureObservation) -> tuple[str, str, str]:
    return obs.kind.value, _norm(obs.text), _norm(obs.latex)


def merge_window_observations(
    kb: LectureKnowledgeBase,
    batch: WindowObservations,
) -> list[str]:
    """Merge one overlapping evidence window into the event-sourced lecture KB.

    Exact content repeated by overlapping windows is collapsed when the timestamp intervals overlap.
    The function never decides that two mathematically similar but textually different statements are
    equivalent; that semantic decision belongs to the LLM knowledge-update pass.
    """
    existing_by_key: dict[tuple[str, str, str], list[LectureObservation]] = {}
    for item in kb.observations:
        existing_by_key.setdefault(_observation_key(item), []).append(item)

    added_ids: list[str] = []
    for index, raw in enumerate(batch.observations):
        obs = raw.model_copy(deep=True)
        obs.window_id = batch.window_id
        if not obs.id:
            obs.id = f"obs_{batch.window_id}_{index:03d}"
        if not obs.evidence_refs:
            obs.evidence_refs = [batch.window_id]

        duplicate = None
        for candidate in existing_by_key.get(_observation_key(obs), []):
            if min(candidate.end, obs.end) >= max(candidate.start, obs.start):
                duplicate = candidate
                break
        if duplicate is not None:
            duplicate.start = min(duplicate.start, obs.start)
            duplicate.end = max(duplicate.end, obs.end)
            duplicate.confidence = max(duplicate.confidence, obs.confidence)
            duplicate.evidence_refs = _merge_unique(duplicate.evidence_refs, obs.evidence_refs)
            continue

        kb.observations.append(obs)
        existing_by_key.setdefault(_observation_key(obs), []).append(obs)
        added_ids.append(obs.id)

    kb.unresolved = _merge_unique(kb.unresolved, batch.unresolved)
    return added_ids


def _unique_id(prefix: str, existing: set[str], seed: int) -> str:
    index = seed
    while f"{prefix}_{index:04d}" in existing:
        index += 1
    value = f"{prefix}_{index:04d}"
    existing.add(value)
    return value


def apply_knowledge_update(
    kb: LectureKnowledgeBase,
    update: KnowledgeUpdate,
    *,
    window_id: str,
) -> None:
    claim_ids = {item.id for item in kb.claims if item.id}
    symbol_ids = {item.id for item in kb.symbols if item.id}
    anchor_ids = {item.id for item in kb.anchors if item.id}

    claim_by_id = {item.id: item for item in kb.claims if item.id}
    for index, raw in enumerate(update.claims):
        claim = raw.model_copy(deep=True)
        for old_id in claim.supersedes:
            old = claim_by_id.get(old_id)
            if old is not None and old.status == ClaimStatus.ACTIVE:
                old.status = ClaimStatus.SUPERSEDED

        if not claim.id:
            claim.id = _unique_id(f"claim_{window_id}", claim_ids, index)
        elif claim.id in claim_by_id:
            existing = claim_by_id[claim.id]
            existing.content = claim.content
            existing.latex = claim.latex
            existing.kind = claim.kind
            existing.scope = claim.scope
            existing.status = claim.status
            existing.math_status = claim.math_status
            existing.source_status = claim.source_status
            existing.evidence_ids = _merge_unique(existing.evidence_ids, claim.evidence_ids)
            existing.supersedes = _merge_unique(existing.supersedes, claim.supersedes)
            if claim.introduced_at:
                existing.introduced_at = min(existing.introduced_at, claim.introduced_at)
            continue
        else:
            claim_ids.add(claim.id)

        kb.claims.append(claim)
        claim_by_id[claim.id] = claim

    symbol_by_key = {(item.symbol, item.scope): item for item in kb.symbols if item.active}
    for index, raw in enumerate(update.symbols):
        symbol = raw.model_copy(deep=True)
        key = (symbol.symbol, symbol.scope)
        existing = symbol_by_key.get(key)
        if existing is not None:
            if symbol.meaning:
                existing.meaning = symbol.meaning
            if symbol.type_hint:
                existing.type_hint = symbol.type_hint
            existing.evidence_ids = _merge_unique(existing.evidence_ids, symbol.evidence_ids)
            existing.introduced_at = min(existing.introduced_at, symbol.introduced_at)
            continue
        if not symbol.id:
            symbol.id = _unique_id(f"sym_{window_id}", symbol_ids, index)
        elif symbol.id in symbol_ids:
            symbol.id = _unique_id(f"sym_{window_id}", symbol_ids, index)
        else:
            symbol_ids.add(symbol.id)
        kb.symbols.append(symbol)
        symbol_by_key[key] = symbol

    for index, raw in enumerate(update.anchors):
        anchor = raw.model_copy(deep=True)
        duplicate = next(
            (
                item
                for item in kb.anchors
                if item.title.strip().lower() == anchor.title.strip().lower()
                and abs(item.timestamp - anchor.timestamp) <= 30.0
            ),
            None,
        )
        if duplicate is not None:
            duplicate.claim_ids = _merge_unique(duplicate.claim_ids, anchor.claim_ids)
            duplicate.evidence_ids = _merge_unique(duplicate.evidence_ids, anchor.evidence_ids)
            duplicate.timestamp = min(duplicate.timestamp, anchor.timestamp)
            continue
        if not anchor.id:
            anchor.id = _unique_id(f"anchor_{window_id}", anchor_ids, index)
        elif anchor.id in anchor_ids:
            anchor.id = _unique_id(f"anchor_{window_id}", anchor_ids, index)
        else:
            anchor_ids.add(anchor.id)
        kb.anchors.append(anchor)

    kb.unresolved = _merge_unique(kb.unresolved, update.unresolved)


def compact_knowledge_state(kb: LectureKnowledgeBase, config: NotesConfig) -> dict[str, Any]:
    active_claims = [item for item in kb.claims if item.status == ClaimStatus.ACTIVE]
    claims = active_claims[-config.knowledge_max_active_claims :]
    observations = kb.observations[-config.knowledge_recent_observations :]
    return {
        "lecture_id": kb.lecture_id,
        "title": kb.title,
        "active_claims": [item.model_dump(mode="json") for item in claims],
        "symbols": [
            item.model_dump(mode="json")
            for item in kb.symbols
            if item.active
        ],
        "anchors": [item.model_dump(mode="json") for item in kb.anchors],
        "recent_observations": [item.model_dump(mode="json") for item in observations],
        "unresolved": kb.unresolved[-50:],
    }


def transcript_context(
    transcript: Transcript,
    *,
    center: float,
    radius: float,
) -> str:
    selected = [
        segment
        for segment in transcript.segments
        if segment.end >= center - radius and segment.start <= center + radius
    ]
    return "\n".join(
        f"[{segment.start:.3f}-{segment.end:.3f}] {segment.text}" for segment in selected
    )


def anchor_contexts(
    kb: LectureKnowledgeBase,
    transcript: Transcript,
    config: NotesConfig,
) -> list[dict[str, Any]]:
    return [
        {
            "anchor": anchor.model_dump(mode="json"),
            "context": transcript_context(
                transcript,
                center=anchor.timestamp,
                radius=config.boundary_context_seconds,
            ),
        }
        for anchor in kb.anchors
    ]


def evidence_for_section(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    transcript: Transcript,
    config: NotesConfig,
) -> dict[str, Any]:
    claim_ids = set(section.claim_ids)
    evidence_ids = set(section.evidence_ids)
    claims = [
        item
        for item in kb.claims
        if item.id in claim_ids and item.status == ClaimStatus.ACTIVE
    ]
    for claim in claims:
        evidence_ids.update(claim.evidence_ids)

    observations = [
        item
        for item in kb.observations
        if item.id in evidence_ids
        or (item.end >= section.start and item.start <= section.end)
    ]
    symbols = [
        item
        for item in kb.symbols
        if item.active and item.introduced_at <= section.end
    ]
    transcript_text = "\n".join(
        f"[{segment.start:.3f}-{segment.end:.3f}] {segment.text}"
        for segment in transcript.segments
        if segment.end >= section.start - config.boundary_context_seconds
        and segment.start <= section.end + config.boundary_context_seconds
    )
    return {
        "section": section.model_dump(mode="json"),
        "claims": [item.model_dump(mode="json") for item in claims],
        "observations": [item.model_dump(mode="json") for item in observations],
        "symbols": [item.model_dump(mode="json") for item in symbols],
        "transcript": transcript_text,
    }


def apply_global_validation(
    ir: LectureIR,
    validation: GlobalValidation,
    *,
    threshold: float,
) -> None:
    for item in validation.corrections:
        if item.confidence < threshold:
            if ir.chunks:
                ir.chunks[-1].unresolved.append(
                    f"Неприменённая глобальная правка "
                    f"(confidence={item.confidence:.2f}): {item.reason}"
                )
            continue
        if item.section_index >= len(ir.chunks):
            if ir.chunks:
                ir.chunks[-1].unresolved.append(
                    f"Global validation returned invalid section index {item.section_index}: "
                    f"{item.reason}"
                )
            continue
        section = ir.chunks[item.section_index]
        if item.block_index >= len(section.blocks):
            section.unresolved.append(
                f"Global validation returned invalid block index {item.block_index}: {item.reason}"
            )
            continue
        block = section.blocks[item.block_index]
        if block.latex.strip() == item.corrected_latex.strip():
            continue
        original = block.latex
        block.latex = item.corrected_latex
        section.corrections.append(
            CorrectionRecord(
                original=original,
                corrected=item.corrected_latex,
                reason=item.reason,
                basis="mathematical_consistency",
                confidence=item.confidence,
            )
        )
    if validation.unresolved and ir.chunks:
        ir.chunks[-1].unresolved = _merge_unique(
            ir.chunks[-1].unresolved,
            validation.unresolved,
        )


@dataclass
class KnowledgeOrchestrator:
    llm: LectureModelClient
    config: NotesConfig
    output_language: str

    def _structured(self, prompt: str, schema, *, operation: str, max_tokens: int | None = None):
        # Kept in one adapter so LectureModelClient can later expose this as a public method without
        # spreading the private-call dependency through the pipeline.
        return self.llm._structured(  # noqa: SLF001
            prompt,
            schema,
            operation=operation,
            max_tokens=max_tokens,
        )

    def extract_observations(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        kb: LectureKnowledgeBase,
    ) -> WindowObservations:
        visual_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        symbols = [
            item.model_dump(mode="json")
            for item in kb.symbols
            if item.active
        ]
        prompt = f"""Extract evidence events from one OVERLAPPING technical window of a university
lecture. Do not write lecture notes yet. A later pass will merge windows and generate prose.

Window id: {chunk.id}
Window bounds: [{chunk.start:.3f}, {chunk.end:.3f}]
Timestamped transcript:
{chunk.timestamped_text or chunk.text}

Visual evidence:
{visual_json}

Known symbol registry:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Return observations in temporal order. Use only these event meanings:
- definition/claim/equation/proof_step/example/notation/remark: something actually asserted or
  written in this window;
- correction: the lecturer explicitly corrects a previous statement, sign, symbol, derivation, or
  board entry. Point target_observation_id at an observation from THIS window when possible;
- retraction: the lecturer explicitly withdraws a statement;
- transition: a topic/proof/example boundary that can later become a semantic anchor;
- unresolved: evidence is too ambiguous to reconstruct safely.

The lecturer may make mistakes and then fix them. Preserve BOTH the mistaken event and the later
correction as evidence. Do not silently replace the first by textbook knowledge. Conversely, do not
label a statement wrong merely because it conflicts with textbook knowledge: source faithfulness is
separate from mathematical validation.

`source_status=observed` means directly supported by audio/visible board. Use `reconstructed` only
for a local reconstruction strongly forced by the evidence; use `inferred` sparingly and never for
new mathematical content. Put exact formulas in `latex`, prose in `text`. Every observation must use
timestamps inside the window and evidence_refs should name transcript/visual ids where possible.
Write descriptive strings in language code `{self.output_language}`.
"""
        result = self._structured(
            prompt,
            WindowObservations,
            operation="knowledge_extract",
            max_tokens=4096,
        )
        result.window_id = chunk.id
        result.start = chunk.start
        result.end = chunk.end
        for index, item in enumerate(result.observations):
            item.window_id = chunk.id
            if not item.id:
                item.id = f"obs_{chunk.id}_{index:03d}"
            item.start = min(max(item.start, chunk.start), chunk.end)
            item.end = min(max(item.end, item.start), chunk.end)
            if not item.evidence_refs:
                item.evidence_refs = [chunk.id]
        return result

    def update_knowledge(
        self,
        kb: LectureKnowledgeBase,
        batch: WindowObservations,
        added_observation_ids: list[str],
    ) -> KnowledgeUpdate:
        state = compact_knowledge_state(kb, self.config)
        new_observations = [
            item.model_dump(mode="json")
            for item in kb.observations
            if item.id in set(added_observation_ids)
        ]
        prompt = f"""Update the canonical event-sourced knowledge state of a lecture after one
overlapping evidence window. Return only a DELTA, not a rewritten knowledge base.

Current compact state:
{json.dumps(state, ensure_ascii=False, separators=(",", ":"))}

New observations from {batch.window_id}:
{json.dumps(new_observations, ensure_ascii=False, separators=(",", ":"))}

Rules:
1. Claims are canonical statements used later to write the lecture. Deduplicate repeated overlap
   evidence. Every claim must cite observation ids in evidence_ids.
2. If a later observation explicitly corrects/retracts an earlier active claim, create the corrected
   claim and put the old claim id in `supersedes`. Do NOT supersede merely because you believe the old
   claim is mathematically wrong.
3. `source_status` describes provenance. `math_status` is independent: normally keep it `unchecked`
   here. This pass is not the mathematical auditor.
4. Maintain scoped typed symbols. The same glyph may have different meanings in different scopes;
   create separate SymbolRecord entries instead of merging incompatible meanings.
5. Add semantic anchors for real topic/definition/theorem/proof/example/notation/correction
   transitions. Technical window boundaries are not anchors.
6. Preserve unresolved conflicts instead of inventing a resolution.
7. For an already-existing claim that only gains evidence, reuse its exact existing id and return an
   upsert with the union of relevant evidence ids. New ids may be left empty; the host assigns them.

Write strings in language code `{self.output_language}`.
"""
        return self._structured(
            prompt,
            KnowledgeUpdate,
            operation="knowledge_update",
            max_tokens=4096,
        )

    def plan_outline(
        self,
        kb: LectureKnowledgeBase,
        transcript: Transcript,
    ) -> LectureOutline:
        state = compact_knowledge_state(kb, self.config)
        contexts = anchor_contexts(kb, transcript, self.config)
        prompt = f"""Plan the final lecture structure AFTER the entire video has been processed.
Technical extraction windows overlap and must be ignored as document boundaries.

Canonical knowledge state:
{json.dumps(state, ensure_ascii=False, separators=(",", ":"))}

Raw transcript context around semantic anchors:
{json.dumps(contexts, ensure_ascii=False, separators=(",", ":"))}

Create at most {self.config.max_outline_sections} semantic sections in chronological order. Merge
anchors that are really one proof/definition/example split by a window boundary. A section may span
many extraction windows. Conversely, split only at genuine semantic transitions visible in the
evidence.

Each section must cite active claim ids and/or evidence observation ids. Do not include superseded
or retracted claims as final mathematical content. They may matter only to understand a correction.
Set start/end to cover the evidence supporting the section and ensure the intervals are monotone,
nonempty, and collectively cover the meaningful lecture content. Write titles in language code
`{self.output_language}`. Do not add textbook topics absent from the lecture.
"""
        outline = self._structured(
            prompt,
            LectureOutline,
            operation="outline_plan",
            max_tokens=4096,
        )
        outline.sections.sort(key=lambda item: (item.start, item.end))
        for index, section in enumerate(outline.sections):
            if not section.id:
                section.id = f"section_{index:03d}"
            if section.end < section.start:
                section.end = section.start
        return outline

    def write_section(
        self,
        section: OutlineSection,
        kb: LectureKnowledgeBase,
        transcript: Transcript,
    ) -> ChunkNotes:
        evidence = evidence_for_section(kb, section, transcript, self.config)
        prompt = f"""Write ONE final LaTeX-ready lecture-note section from the canonical lecture
knowledge base and local raw evidence. This is a synthesis pass, not a transcript chunk summary.

Section evidence:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Use active canonical claims as the default mathematical content.
- Preserve the lecturer's terminology, notation, proof order, and level of detail.
- A superseded/retracted lecturer mistake must not appear as a current theorem/formula. If the
  correction itself is pedagogically relevant, it may be mentioned briefly.
- Do not silently replace an unresolved lecturer statement with textbook knowledge.
- Do not introduce mathematical assertions not supported by a claim/observation/transcript excerpt.
- `source_claim_ids` and `source_evidence_ids` on every NoteBlock are mandatory provenance for
  substantive blocks.
- Use formal block types only when the lecturer presents the material as such. Do not emit raw
  LaTeX environment or section commands.
- Put uncertain material in `unresolved`, not in note blocks.
- Keep the section coherent even when its evidence came from several overlapping technical windows.
- Write prose in language code `{self.output_language}` and mathematics in LaTeX.
"""
        notes = self._structured(
            prompt,
            ChunkNotes,
            operation="section_write",
            max_tokens=8192,
        )
        notes.chunk_id = section.id
        notes.start = section.start
        notes.end = section.end
        notes.section_title = section.title.replace("$", "")
        return notes

    def validate_lecture(
        self,
        ir: LectureIR,
        kb: LectureKnowledgeBase,
    ) -> GlobalValidation:
        state = compact_knowledge_state(kb, self.config)
        state["claim_history"] = [
            item.model_dump(mode="json")
            for item in kb.claims[-2 * self.config.knowledge_max_active_claims :]
        ]
        draft = [
            {
                "section_index": section_index,
                "title": section.section_title,
                "blocks": [
                    {
                        "block_index": block_index,
                        **block.model_dump(mode="json"),
                    }
                    for block_index, block in enumerate(section.blocks)
                ],
            }
            for section_index, section in enumerate(ir.chunks)
        ]
        prompt = f"""Perform a final DOCUMENT-LEVEL validation of reconstructed lecture notes.

Canonical lecture state:
{json.dumps(state, ensure_ascii=False, separators=(",", ":"))}

Draft:
{json.dumps(draft, ensure_ascii=False, separators=(",", ":"))}

This pass exists to catch errors that local windows cannot see:
- a superseded lecturer typo accidentally surviving after a later correction;
- inconsistent symbol meaning/domain/codomain across sections;
- duplicated fragments caused by overlapping windows;
- a proof/definition split incorrectly at a former chunk boundary;
- algebraic/sign/type errors introduced by reconstruction;
- unsupported textbook extrapolation presented as if the lecturer said it.

Do not 'correct' an active lecturer statement solely because external mathematics says it is wrong.
If source-faithful content is mathematically suspicious but was not corrected in the lecture, add an
unresolved item. Apply a block correction only when the canonical evidence supports the replacement
or the draft itself contains a reconstruction inconsistency. Do not rewrite for style. Return the
complete corrected block latex for each correction and write reasons in language code
`{self.output_language}`.
"""
        return self._structured(
            prompt,
            GlobalValidation,
            operation="global_validation",
            max_tokens=8192,
        )
