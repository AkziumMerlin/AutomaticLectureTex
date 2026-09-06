from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import NotesConfig
from .episode_graph import reconcile_window_observations
from .schemas import (
    ChunkNotes,
    ClaimStatus,
    CorrectionRecord,
    EpisodeHierarchyPlan,
    EpisodeStatus,
    EpisodeTrackingUpdate,
    GlobalValidation,
    KnowledgeUpdate,
    LectureChunk,
    LectureIR,
    LectureKnowledgeBase,
    OutlineSection,
    Transcript,
    VisualEvidence,
    WindowObservations,
)

if TYPE_CHECKING:
    from .llm import LectureModelClient


def _merge_unique(left: list[str], right: Iterable[str]) -> list[str]:
    seen = set(left)
    result = list(left)
    for item in right:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def merge_window_observations(
    kb: LectureKnowledgeBase,
    batch: WindowObservations,
) -> list[str]:
    """Backward-compatible name for overlap reconciliation."""

    return reconcile_window_observations(kb, batch)


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
    """Legacy updater retained for compatibility tests/tools.

    The knowledge architecture no longer calls this function: canonical claims are derived from
    canonical observations inside semantic episodes. Keeping it here avoids breaking old artifacts
    and makes the migration explicit.
    """

    claim_ids = {item.id for item in kb.claims if item.id}
    symbol_ids = {item.id for item in kb.symbols if item.id}
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
            existing.episode_id = claim.episode_id
            existing.status = claim.status
            existing.math_status = claim.math_status
            existing.source_status = claim.source_status
            existing.evidence_ids = _merge_unique(existing.evidence_ids, claim.evidence_ids)
            existing.supersedes = _merge_unique(existing.supersedes, claim.supersedes)
            continue
        else:
            claim_ids.add(claim.id)
        kb.claims.append(claim)
        claim_by_id[claim.id] = claim

    for index, raw in enumerate(update.symbols):
        symbol = raw.model_copy(deep=True)
        key = (symbol.symbol, symbol.scope)
        existing = next(
            (item for item in kb.symbols if item.active and (item.symbol, item.scope) == key),
            None,
        )
        if existing is not None:
            if symbol.meaning:
                existing.meaning = symbol.meaning
            if symbol.type_hint:
                existing.type_hint = symbol.type_hint
            existing.evidence_ids = _merge_unique(existing.evidence_ids, symbol.evidence_ids)
            continue
        if not symbol.id or symbol.id in symbol_ids:
            symbol.id = _unique_id(f"sym_{window_id}", symbol_ids, index)
        kb.symbols.append(symbol)

    kb.unresolved = _merge_unique(kb.unresolved, update.unresolved)


def compact_knowledge_state(kb: LectureKnowledgeBase, config: NotesConfig) -> dict[str, Any]:
    active_claims = [item for item in kb.claims if item.status == ClaimStatus.ACTIVE]
    claims = active_claims[-config.knowledge_max_active_claims :]
    observations = kb.observations[-config.knowledge_recent_observations :]
    episodes = kb.episodes[-80:]
    return {
        "lecture_id": kb.lecture_id,
        "title": kb.title,
        "active_claims": [item.model_dump(mode="json") for item in claims],
        "symbols": [
            item.model_dump(mode="json")
            for item in kb.symbols
            if item.active
        ],
        "episodes": [item.model_dump(mode="json") for item in episodes],
        "recent_observations": [item.model_dump(mode="json") for item in observations],
        "observation_aliases": dict(kb.observation_aliases),
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


def episode_contexts(
    kb: LectureKnowledgeBase,
    transcript: Transcript,
    config: NotesConfig,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for episode in sorted(kb.episodes, key=lambda item: (item.start, item.end)):
        contexts.append(
            {
                "episode": episode.model_dump(mode="json"),
                "start_context": transcript_context(
                    transcript,
                    center=episode.start,
                    radius=config.boundary_context_seconds,
                ),
                "end_context": transcript_context(
                    transcript,
                    center=episode.end,
                    radius=config.boundary_context_seconds,
                ),
            }
        )
    return contexts


def evidence_for_section(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    transcript: Transcript,
    config: NotesConfig,
) -> dict[str, Any]:
    episode_ids = set(section.episode_ids)
    episodes = [
        item
        for item in sorted(kb.episodes, key=lambda item: (item.start, item.end))
        if item.id in episode_ids
    ]

    claim_ids = set(section.claim_ids)
    evidence_ids = set(section.evidence_ids)
    if episodes:
        for episode in episodes:
            claim_ids.update(episode.claim_ids)
            evidence_ids.update(episode.observation_ids)

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
    ]
    if not observations:
        observations = [
            item
            for item in kb.observations
            if item.end >= section.start and item.start <= section.end
        ]

    symbols = [
        item
        for item in kb.symbols
        if item.active and (item.episode_id in episode_ids or item.introduced_at <= section.end)
    ]
    transcript_text = "\n".join(
        f"[{segment.start:.3f}-{segment.end:.3f}] {segment.text}"
        for segment in transcript.segments
        if segment.end >= section.start - config.boundary_context_seconds
        and segment.start <= section.end + config.boundary_context_seconds
    )
    return {
        "section": section.model_dump(mode="json"),
        "episodes": [item.model_dump(mode="json") for item in episodes],
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
        recent_observations = [
            item.model_dump(mode="json")
            for item in kb.observations[-20:]
        ]
        open_episodes = [
            item.model_dump(mode="json")
            for item in kb.episodes
            if item.status == EpisodeStatus.OPEN
        ]
        prompt = f"""Extract evidence events from one OVERLAPPING technical window of a university
lecture. Do not write lecture notes and do not decide final document sections.

Window id: {chunk.id}
Window bounds: [{chunk.start:.3f}, {chunk.end:.3f}]
Timestamped transcript:
{chunk.timestamped_text or chunk.text}

Visual evidence:
{visual_json}

Known symbol registry:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Recent canonical observations from the previous overlap/context:
{json.dumps(recent_observations, ensure_ascii=False, separators=(",", ":"))}

Currently open semantic episodes:
{json.dumps(open_episodes, ensure_ascii=False, separators=(",", ":"))}

Return observations in temporal order. Use only these event meanings:
- definition/claim/equation/proof_step/example/notation/remark: something actually asserted or
  written in this window;
- correction: the lecturer explicitly corrects a previous statement, sign, symbol, derivation, or
  board entry. If it targets a recent canonical observation shown above, use that exact id in
  target_observation_id; if the target is earlier in THIS window, use the local observation id;
- retraction: the lecturer explicitly withdraws a statement, with target_observation_id when known;
- transition: a real topic/proof/example transition, not a technical window edge;
- unresolved: evidence is too ambiguous to reconstruct safely.

The lecturer may make mistakes and then fix them. Preserve both the mistaken event and the later
correction as evidence. Do not replace either with textbook knowledge. Technical window overlap is
not semantic structure: repeated material is expected and will be reconciled by the host.

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
            item.window_ids = _merge_unique(item.window_ids, [chunk.id])
            if not item.id:
                item.id = f"obs_{chunk.id}_{index:03d}"
            item.start = min(max(item.start, chunk.start), chunk.end)
            item.end = min(max(item.end, item.start), chunk.end)
            if not item.evidence_refs:
                item.evidence_refs = [chunk.id]
        return result

    def track_episodes(
        self,
        kb: LectureKnowledgeBase,
        batch: WindowObservations,
        added_observation_ids: list[str],
    ) -> EpisodeTrackingUpdate:
        new_ids = set(added_observation_ids)
        new_observations = [
            item.model_dump(mode="json")
            for item in kb.observations
            if item.id in new_ids
        ]
        recent_episodes = [
            item.model_dump(mode="json")
            for item in kb.episodes[-8:]
        ]
        active_symbols = [
            item.model_dump(mode="json")
            for item in kb.symbols
            if item.active
        ][-80:]
        prompt = f"""Track semantic episodes in a university lecture. The host owns all evidence
and will assign EVERY canonical observation to an episode. Your job is only to place semantic
boundaries and describe typed symbols; never create claims or document sections.

Recent/open episodes:
{json.dumps(recent_episodes, ensure_ascii=False, separators=(",", ":"))}

New canonical observations from {batch.window_id}:
{json.dumps(new_observations, ensure_ascii=False, separators=(",", ":"))}

Active symbols:
{json.dumps(active_symbols, ensure_ascii=False, separators=(",", ":"))}

For `boundaries`, emit a boundary BEFORE an observation only when a genuinely new semantic episode
starts: a new definition/theorem/proof/example/derivation/topic, not merely because a technical
window began. If the first observations continue the currently open proof/topic, emit no boundary.
A lecturer correction normally stays in the same episode. `close_after_observation_ids` is optional
and should be used only when an episode clearly ends without another episode starting immediately.

For symbols, give meaning/type_hint and cite canonical observation ids in evidence_ids. Do not choose
a global scope: the host derives symbol scope from the semantic episode containing the evidence.
Do not repeat unchanged symbols just because they reappear in an overlapping window.

This is state tracking, not summarization. Preserve ambiguity in `unresolved`. Write labels/descriptions
in language code `{self.output_language}`.
"""
        return self._structured(
            prompt,
            EpisodeTrackingUpdate,
            operation="episode_track",
            max_tokens=3072,
        )

    def plan_episode_hierarchy(
        self,
        kb: LectureKnowledgeBase,
        transcript: Transcript,
    ) -> EpisodeHierarchyPlan:
        episodes = [
            item.model_dump(mode="json")
            for item in sorted(kb.episodes, key=lambda item: (item.start, item.end))
            if item.observation_ids
        ]
        contexts = episode_contexts(kb, transcript, self.config)
        prompt = f"""Build a hierarchy over an ALREADY FIXED ordered sequence of semantic episodes.
You are not allowed to invent/remove/reorder episodes, claims, timestamps, or evidence. Return only
boundary decisions before existing episode ids.

Semantic episodes (immutable leaves, chronological):
{json.dumps(episodes, ensure_ascii=False, separators=(",", ":"))}

Transcript context around their boundaries:
{json.dumps(contexts, ensure_ascii=False, separators=(",", ":"))}

Use `level=topic` for a small number of major lecture sections and `level=subtopic` for useful
internal groupings. The first episode implicitly starts a topic; include a boundary before it only if
you want to provide a better title. Group adjacent episodes according to the actual lecture structure:
a theorem and its proof normally stay in one top-level topic, as do a definition and its immediate
properties. Technical window boundaries are irrelevant.

This operation only groups leaves. The host will derive every section's time range, claims and
evidence by unioning its episodes, so do not attempt to specify those fields. Do not add textbook
topics absent from the evidence. Write titles in language code `{self.output_language}`.
"""
        return self._structured(
            prompt,
            EpisodeHierarchyPlan,
            operation="episode_hierarchy",
            max_tokens=3072,
        )

    def write_section(
        self,
        section: OutlineSection,
        kb: LectureKnowledgeBase,
        transcript: Transcript,
    ) -> ChunkNotes:
        evidence = evidence_for_section(kb, section, transcript, self.config)
        prompt = f"""Write ONE final LaTeX-ready lecture-note section from a FIXED group of semantic
episodes. The episode sequence is the document structure; do not repartition it and do not create
material outside it.

Section evidence:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Follow semantic episodes in their given order. Their observations/claims are the source of truth.
- Preserve the lecturer's terminology, notation, proof order, corrections, and level of detail.
- A superseded/retracted lecturer mistake must not appear as current mathematical content.
- Do not silently replace an unresolved lecturer statement with textbook knowledge.
- Do not introduce mathematical assertions unsupported by the supplied episode evidence.
- Reconstructed/inferred observations are weaker evidence than directly observed ones; if the raw
  evidence does not support a safe statement, put it in `unresolved` instead of completing it from
  general knowledge.
- `source_claim_ids` and/or `source_evidence_ids` on every substantive NoteBlock must point into this
  section's episode evidence.
- Use formal block types only when the lecturer presents the material as such. Do not emit raw LaTeX
  environment or section commands.
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
        state["episodes"] = [item.model_dump(mode="json") for item in kb.episodes]
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

Canonical lecture state and semantic episodes:
{json.dumps(state, ensure_ascii=False, separators=(",", ":"))}

Draft:
{json.dumps(draft, ensure_ascii=False, separators=(",", ":"))}

The episode graph already determines structure. This pass must not create missing lecture content or
reorganize sections. It exists only to detect residual inconsistencies between the rendered draft and
the episode evidence: a superseded lecturer typo surviving after correction, duplicated rendered
content, a symbol meaning leaking across episode scopes, or algebra/type damage introduced during
section synthesis.

Do not complete an empty/ambiguous proof from unchecked or reconstructed claims. Do not 'correct' an
active lecturer statement solely because external mathematics says it is wrong. If evidence is
insufficient, add an unresolved item rather than a replacement. Apply a block correction only when
the replacement is directly supported by canonical episode evidence. Do not rewrite for style.
Return complete corrected block latex and write reasons in language code `{self.output_language}`.
"""
        return self._structured(
            prompt,
            GlobalValidation,
            operation="global_validation",
            max_tokens=8192,
        )