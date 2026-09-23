from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .generated_notes import GeneratedChunkNotes
from .llm import StructuredTaskTooLargeError
from .schemas import (
    ChunkNotes,
    ClaimStatus,
    CorrectionRecord,
    EpisodeHierarchyPlan,
    LectureKnowledgeBase,
    MathAudit,
    OutlineSection,
    SemanticEpisode,
)

if TYPE_CHECKING:
    from .config import NotesConfig
    from .knowledge import KnowledgeOrchestrator
    from .schemas import Transcript

HIERARCHY_CACHE_VERSION = 2
EPISODE_SYNTHESIS_CACHE_VERSION = 2


def _merge_unique(left: list[str], right: Iterable[str]) -> list[str]:
    seen = set(left)
    result = list(left)
    for item in right:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _compact_observation(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "start": item.start,
        "end": item.end,
        "kind": item.kind,
        "text": item.text,
        "latex": item.latex,
        "target_observation_id": item.target_observation_id,
        "confidence": item.confidence,
        "source_status": item.source_status,
    }


def _compact_claim(item, selected_observation_ids: set[str]) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "content": item.content,
        "latex": item.latex,
        "math_status": item.math_status,
        "source_status": item.source_status,
        "evidence_ids": [value for value in item.evidence_ids if value in selected_observation_ids],
        "supersedes": list(item.supersedes),
        "introduced_at": item.introduced_at,
    }


def _compact_symbol(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "symbol": item.symbol,
        "meaning": item.meaning,
        "type_hint": item.type_hint,
        "episode_id": item.episode_id,
        "introduced_at": item.introduced_at,
        "evidence_ids": list(item.evidence_ids),
    }


def _episode_summary(kb: LectureKnowledgeBase, episode: SemanticEpisode) -> dict[str, Any]:
    observations = [
        item for item in kb.observations if item.id in set(episode.observation_ids)
    ]
    observations.sort(key=lambda item: (item.start, item.end, item.id))

    def endpoint(item):
        if item is None:
            return None
        return {
            "id": item.id,
            "kind": item.kind,
            "text": item.text[:700],
            "latex": (item.latex or "")[:700] or None,
        }

    return {
        "id": episode.id,
        "title": episode.title,
        "kind": episode.kind,
        "start": episode.start,
        "end": episode.end,
        "first_observation": endpoint(observations[0] if observations else None),
        "last_observation": endpoint(observations[-1] if observations else None),
    }


def plan_episode_hierarchy_bounded(
    orchestrator: KnowledgeOrchestrator,
    kb: LectureKnowledgeBase,
) -> EpisodeHierarchyPlan:
    """Plan topic/subtopic boundaries in bounded episode batches.

    Only current-batch episode ids are accepted in each response. A small read-only prefix of the
    previous batch gives continuity without making the request grow with lecture length.
    """

    episodes = sorted(
        [item for item in kb.episodes if item.observation_ids],
        key=lambda item: (item.start, item.end, item.id),
    )
    if not episodes:
        return EpisodeHierarchyPlan()

    boundaries = []
    unresolved: list[str] = []
    batch_size = orchestrator.config.hierarchy_batch_episodes
    seen_boundaries: set[tuple[str, str]] = set()

    for start in range(0, len(episodes), batch_size):
        batch = episodes[start : start + batch_size]
        previous = episodes[max(0, start - 2) : start]
        allowed_ids = {item.id for item in batch}
        prompt = f"""Build hierarchy boundaries for ONE bounded batch of an already fixed sequence
of semantic lecture episodes. Episodes are immutable leaves: never invent, remove, reorder, resize,
or rewrite them. Return boundaries only before episode ids from CURRENT BATCH.

Previous context (read-only; do not return boundaries for these ids):
{json.dumps(
    [_episode_summary(kb, item) for item in previous],
    ensure_ascii=False,
    separators=(",", ":"),
)}

Current batch:
{json.dumps(
    [_episode_summary(kb, item) for item in batch],
    ensure_ascii=False,
    separators=(",", ":"),
)}

Use `level=topic` only for genuine major lecture-topic boundaries and `level=subtopic` for useful
internal groupings. A theorem and its proof normally remain in the same topic, as do a definition
and its immediate properties. The first current episode may continue the previous context; in that
case return no topic boundary before it. Do not add textbook topics absent from the evidence.
Write titles in language code `{orchestrator.output_language}`.
"""
        partial = orchestrator._structured(
            prompt,
            EpisodeHierarchyPlan,
            operation="episode_hierarchy",
            max_tokens=2048,
        )
        for boundary in partial.boundaries:
            if boundary.before_episode_id not in allowed_ids:
                unresolved.append(
                    "Ignored hierarchy boundary for out-of-batch episode "
                    f"{boundary.before_episode_id}."
                )
                continue
            key = (boundary.before_episode_id, boundary.level)
            if key in seen_boundaries:
                continue
            seen_boundaries.add(key)
            boundaries.append(boundary)
        unresolved = _merge_unique(unresolved, partial.unresolved)

    return EpisodeHierarchyPlan(boundaries=boundaries, unresolved=unresolved)


def _episode_payload_for_observation_ids(
    kb: LectureKnowledgeBase,
    episode: SemanticEpisode,
    observation_ids: list[str],
    config: NotesConfig,
    transcript: Transcript | None = None,
) -> dict[str, Any]:
    selected_ids = set(observation_ids)
    observations = [item for item in kb.observations if item.id in selected_ids]
    observations.sort(key=lambda item: (item.start, item.end, item.id))

    active_episode_claim_ids = set(episode.claim_ids)
    claims = [
        item
        for item in kb.claims
        if item.status == ClaimStatus.ACTIVE
        and item.id in active_episode_claim_ids
        and bool(selected_ids.intersection(item.evidence_ids))
    ]

    direct_symbols = [
        item
        for item in kb.symbols
        if item.active and bool(selected_ids.intersection(item.evidence_ids))
    ]
    direct_symbol_ids = {item.id for item in direct_symbols}
    batch_start = observations[0].start if observations else episode.start
    previous_symbol_candidates = sorted(
        [
            item
            for item in kb.symbols
            if item.active
            and item.id not in direct_symbol_ids
            and item.introduced_at <= batch_start
        ],
        key=lambda item: (item.introduced_at, item.id),
    )
    if config.episode_symbol_context_limit:
        previous_symbols = previous_symbol_candidates[-config.episode_symbol_context_limit :]
    else:
        previous_symbols = []

    transcript_segments = []
    if transcript is not None and observations:
        context = config.episode_transcript_context_seconds
        start = observations[0].start - context
        end = observations[-1].end + context
        transcript_segments = [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": segment.confidence,
            }
            for segment in transcript.segments
            if segment.end >= start and segment.start <= end
        ]

    return {
        "episode": {
            "id": episode.id,
            "title": episode.title,
            "kind": episode.kind,
            "start": episode.start,
            "end": episode.end,
        },
        "observations": [_compact_observation(item) for item in observations],
        "claims": [_compact_claim(item, selected_ids) for item in claims],
        "symbols": [_compact_symbol(item) for item in [*previous_symbols, *direct_symbols]],
        "transcript": transcript_segments,
    }


def _payload_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def episode_evidence_batches(
    kb: LectureKnowledgeBase,
    episode: SemanticEpisode,
    config: NotesConfig,
    transcript: Transcript | None = None,
) -> list[dict[str, Any]]:
    """Split one semantic episode into evidence payloads with a hard serialized-size budget."""

    available = {item.id for item in kb.observations}
    ordered_ids = [item for item in episode.observation_ids if item in available]
    if not ordered_ids:
        return []

    max_chars = config.episode_synthesis_max_evidence_chars
    # Reserve a small amount for batch metadata added after packing so the serialized final
    # payload still respects the configured hard budget.
    packing_limit = max_chars - 128
    batches: list[dict[str, Any]] = []
    current_ids: list[str] = []

    for observation_id in ordered_ids:
        candidate_ids = [*current_ids, observation_id]
        candidate = _episode_payload_for_observation_ids(
            kb, episode, candidate_ids, config, transcript
        )
        if current_ids and _payload_size(candidate) > packing_limit:
            payload = _episode_payload_for_observation_ids(
                kb, episode, current_ids, config, transcript
            )
            batches.append(payload)
            current_ids = [observation_id]
            single = _episode_payload_for_observation_ids(
                kb, episode, current_ids, config, transcript
            )
            if _payload_size(single) > packing_limit:
                raise ValueError(
                    f"single evidence atom in {episode.id} exceeds "
                    f"notes.episode_synthesis_max_evidence_chars={max_chars}"
                )
        else:
            current_ids = candidate_ids
            if _payload_size(candidate) > packing_limit:
                raise ValueError(
                    f"single evidence atom in {episode.id} exceeds "
                    f"notes.episode_synthesis_max_evidence_chars={max_chars}"
                )

    if current_ids:
        batches.append(
            _episode_payload_for_observation_ids(kb, episode, current_ids, config, transcript)
        )

    for index, payload in enumerate(batches):
        payload["batch"] = {"index": index, "count": len(batches)}
    return batches


def previous_block_context(notes: list[ChunkNotes], limit: int = 2) -> list[dict[str, Any]]:
    blocks = [block for item in notes for block in item.blocks]
    result = []
    for block in blocks[-limit:]:
        result.append(
            {
                "type": block.type,
                "title": block.title,
                "latex_tail": block.latex[-2000:],
                "source_claim_ids": block.source_claim_ids,
                "source_evidence_ids": block.source_evidence_ids,
            }
        )
    return result


def write_episode_batch(
    orchestrator: KnowledgeOrchestrator,
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    previous_context: list[dict[str, Any]],
) -> ChunkNotes:
    prompt = f"""Write LaTeX-ready lecture notes for ONE bounded evidence batch inside ONE fixed
semantic episode. Do not create document sections and do not pull material from outside this
payload.

Episode evidence:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Immediately preceding generated blocks from the same episode (continuity only; do not repeat them):
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Preserve the lecturer's terminology, notation, proof order, corrections, and level of detail.
- Active claims and canonical observations are the source of truth.
- Never resurrect superseded/retracted content or complete ambiguity from textbook knowledge.
- Reconstructed/inferred evidence is weaker than directly observed evidence; if support is
  insufficient, put the issue in `unresolved` instead of inventing content.
- Every substantive block must cite only `source_claim_ids` and/or `source_evidence_ids` present in
  this evidence batch.
- `latex` is the COMPLETE BODY of every returned block, including ordinary prose. It must never be
  empty. If a block body cannot be reconstructed safely, omit that block and record the issue in
  `unresolved` instead of returning an empty block.
- Use formal block types only when the lecturer presents the material as such.
- The deterministic renderer owns theorem/proof/definition/figure/section wrappers. Internal math
  environments such as aligned, cases, matrix, and split are allowed inside block LaTeX.
- Write prose in language code `{orchestrator.output_language}` and mathematics in LaTeX.
"""
    generated = orchestrator._structured(
        prompt,
        GeneratedChunkNotes,
        operation="episode_write",
        max_tokens=4096,
        split_oversized_task=True,
    )
    notes = generated.to_chunk_notes()
    batch = evidence["batch"]
    notes.chunk_id = f"{episode.id}_batch_{batch['index']:03d}"
    observations = evidence["observations"]
    notes.start = observations[0]["start"] if observations else episode.start
    notes.end = observations[-1]["end"] if observations else episode.end
    notes.section_title = episode.title.replace("$", "")

    allowed_claims = {item["id"] for item in evidence["claims"]}
    allowed_observations = {item["id"] for item in evidence["observations"]}
    kept = []
    for block in notes.blocks:
        original_claims = list(block.source_claim_ids)
        original_evidence = list(block.source_evidence_ids)
        block.source_claim_ids = [item for item in original_claims if item in allowed_claims]
        block.source_evidence_ids = [
            item for item in original_evidence if item in allowed_observations
        ]
        unknown_claims = sorted(set(original_claims) - allowed_claims)
        unknown_evidence = sorted(set(original_evidence) - allowed_observations)
        if unknown_claims or unknown_evidence:
            notes.unresolved.append(
                "Removed out-of-batch provenance from generated block: "
                f"claims={unknown_claims}, evidence={unknown_evidence}."
            )
        if not block.source_claim_ids and not block.source_evidence_ids:
            notes.unresolved.append(
                f"Dropped ungrounded generated {block.type} block: {block.latex[:160]}"
            )
            continue
        kept.append(block)
    notes.blocks = kept
    notes.unresolved = _merge_unique([], notes.unresolved)
    return notes


def validate_episode_batch(
    orchestrator: KnowledgeOrchestrator,
    evidence: dict[str, Any],
    notes: ChunkNotes,
) -> MathAudit:
    draft = [
        {"block_index": index, **block.model_dump(mode="json")}
        for index, block in enumerate(notes.blocks)
    ]
    prompt = f"""Validate ONE bounded generated lecture-note batch against exactly its source
evidence. This is a local faithfulness/math check, not a rewrite and not a document-level pass.

Evidence:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Generated blocks:
{json.dumps(draft, ensure_ascii=False, separators=(",", ":"))}

Report a correction only when a generated block contradicts or algebraically damages the supplied
evidence. Do not improve style, add textbook material, or infer missing proof steps. If evidence is
insufficient, add an `unresolved` item instead. `corrected_latex` must contain the complete
corrected content of that block. Internal mathematical environments are allowed; renderer-owned
outer section/theorem/proof/definition wrappers are not. Write reasons in language code
`{orchestrator.output_language}`.
"""
    return orchestrator._structured(
        prompt,
        MathAudit,
        operation="episode_validation",
        max_tokens=2048,
        split_oversized_task=True,
    )


def apply_episode_validation(
    notes: ChunkNotes,
    audit: MathAudit,
    *,
    threshold: float,
) -> None:
    for item in audit.corrections:
        if item.block_index >= len(notes.blocks):
            notes.unresolved.append(
                f"Episode validation returned invalid block index {item.block_index}: {item.reason}"
            )
            continue
        if item.confidence < threshold:
            notes.unresolved.append(
                "Неприменённая локальная правка "
                f"(confidence={item.confidence:.2f}): {item.reason}"
            )
            continue
        block = notes.blocks[item.block_index]
        if not item.corrected_latex.strip():
            notes.unresolved.append(
                "Ignored empty episode-validation correction for block "
                f"{item.block_index}: {item.reason}"
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
    notes.unresolved = _merge_unique(notes.unresolved, audit.unresolved)


def merge_episode_batches(
    episode: SemanticEpisode,
    batches: list[ChunkNotes],
) -> ChunkNotes:
    result = ChunkNotes(
        chunk_id=episode.id,
        start=episode.start,
        end=episode.end,
        section_title=episode.title.replace("$", ""),
        blocks=[],
    )
    for notes in batches:
        result.blocks.extend(notes.blocks)
        result.notation.extend(notes.notation)
        result.corrections.extend(notes.corrections)
        result.unresolved = _merge_unique(result.unresolved, notes.unresolved)
    return result


def assemble_outline_sections(
    sections: list[OutlineSection],
    episode_notes: dict[str, ChunkNotes],
    *,
    outline_unresolved: list[str] | None = None,
) -> list[ChunkNotes]:
    result: list[ChunkNotes] = []
    for section in sections:
        assembled = ChunkNotes(
            chunk_id=section.id,
            start=section.start,
            end=section.end,
            section_title=section.title.replace("$", ""),
            blocks=[],
        )
        for episode_id in section.episode_ids:
            notes = episode_notes.get(episode_id)
            if notes is None:
                assembled.unresolved.append(f"Missing synthesized episode {episode_id}.")
                continue
            assembled.blocks.extend(notes.blocks)
            assembled.notation.extend(notes.notation)
            assembled.corrections.extend(notes.corrections)
            assembled.unresolved = _merge_unique(assembled.unresolved, notes.unresolved)
        result.append(assembled)

    if result and outline_unresolved:
        result[-1].unresolved = _merge_unique(result[-1].unresolved, outline_unresolved)
    return result
