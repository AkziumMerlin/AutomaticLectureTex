from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .episode_synthesis import (
    apply_episode_validation,
    episode_evidence_batches as _base_evidence_batches,
    validate_episode_batch as _validate_once,
    write_episode_batch as _write_once,
)
from .schemas import ChunkNotes, MathAudit, SemanticEpisode

logger = logging.getLogger(__name__)

# Bump the downstream cache whenever split/merge semantics change. Upstream knowledge-window caches
# are intentionally unaffected by this version.
EPISODE_SYNTHESIS_CACHE_VERSION = 3

# Evidence character limits control input size, but do not bound output complexity. A second bound on
# semantic atoms keeps ordinary calls small; failures are split recursively below this threshold too.
MAX_OBSERVATIONS_PER_SYNTHESIS_CALL = 6

_STRUCTURED_ERRORS = (json.JSONDecodeError, ValidationError)


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen = set(left)
    result = list(left)
    for item in right:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _child_payload(
    evidence: dict[str, Any],
    observations: list[dict[str, Any]],
    side: int,
) -> dict[str, Any]:
    selected_ids = {item["id"] for item in observations}
    start = observations[0]["start"]
    end = observations[-1]["end"]

    claims: list[dict[str, Any]] = []
    for claim in evidence.get("claims", []):
        evidence_ids = [item for item in claim.get("evidence_ids", []) if item in selected_ids]
        if not evidence_ids:
            continue
        child_claim = deepcopy(claim)
        child_claim["evidence_ids"] = evidence_ids
        claims.append(child_claim)

    # Keep symbols introduced before this child as read-only context, plus symbols directly grounded
    # in this child. Do not leak symbols introduced only by the future/right child into the left one.
    symbols: list[dict[str, Any]] = []
    for symbol in evidence.get("symbols", []):
        symbol_evidence = set(symbol.get("evidence_ids", []))
        directly_used = bool(symbol_evidence.intersection(selected_ids))
        introduced_at = float(symbol.get("introduced_at", start) or 0.0)
        if directly_used or introduced_at <= start:
            child_symbol = deepcopy(symbol)
            child_symbol["evidence_ids"] = [
                item for item in symbol.get("evidence_ids", []) if item in selected_ids
            ]
            symbols.append(child_symbol)

    transcript = [
        deepcopy(segment)
        for segment in evidence.get("transcript", [])
        if float(segment.get("end", start)) >= start and float(segment.get("start", end)) <= end
    ]

    parent_batch = evidence.get("batch", {})
    split_path = [*parent_batch.get("split_path", []), side]
    return {
        "episode": deepcopy(evidence["episode"]),
        "observations": deepcopy(observations),
        "claims": claims,
        "symbols": symbols,
        "transcript": transcript,
        "batch": {
            "index": int(parent_batch.get("index", 0)),
            "count": int(parent_batch.get("count", 1)),
            "split_path": split_path,
        },
    }


def split_evidence_payload(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    observations = evidence.get("observations", [])
    if len(observations) < 2:
        return None
    midpoint = len(observations) // 2
    return (
        _child_payload(evidence, observations[:midpoint], 0),
        _child_payload(evidence, observations[midpoint:], 1),
    )


def _split_until_small(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    if len(evidence.get("observations", [])) <= MAX_OBSERVATIONS_PER_SYNTHESIS_CALL:
        return [evidence]
    split = split_evidence_payload(evidence)
    if split is None:
        return [evidence]
    left, right = split
    return [*_split_until_small(left), *_split_until_small(right)]


def episode_evidence_batches(kb, episode, config, transcript=None) -> list[dict[str, Any]]:
    """Create bounded synthesis leaves using both input-size and semantic-atom limits."""

    base_batches = _base_evidence_batches(kb, episode, config, transcript)
    leaves: list[dict[str, Any]] = []
    for payload in base_batches:
        leaves.extend(_split_until_small(payload))

    for index, payload in enumerate(leaves):
        payload["batch"]["index"] = index
        payload["batch"]["count"] = len(leaves)
    return leaves


def _context_after(previous_context: list[dict[str, Any]], notes: ChunkNotes) -> list[dict[str, Any]]:
    generated = []
    for block in notes.blocks[-2:]:
        generated.append(
            {
                "type": block.type,
                "title": block.title,
                "latex_tail": block.latex[-2000:],
                "source_claim_ids": block.source_claim_ids,
                "source_evidence_ids": block.source_evidence_ids,
            }
        )
    return [*previous_context, *generated][-2:]


def _merge_parts(
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    parts: list[ChunkNotes],
) -> ChunkNotes:
    observations = evidence.get("observations", [])
    batch = evidence.get("batch", {})
    result = ChunkNotes(
        chunk_id=f"{episode.id}_batch_{int(batch.get('index', 0)):03d}",
        start=observations[0]["start"] if observations else episode.start,
        end=observations[-1]["end"] if observations else episode.end,
        section_title=episode.title.replace("$", ""),
        blocks=[],
    )
    for notes in parts:
        result.blocks.extend(notes.blocks)
        result.notation.extend(notes.notation)
        result.corrections.extend(notes.corrections)
        result.unresolved = _merge_unique(result.unresolved, notes.unresolved)
    return result


def _indivisible_failure(
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    stage: str,
    exc: Exception,
) -> ChunkNotes:
    observations = evidence.get("observations", [])
    observation_ids = [item.get("id", "?") for item in observations]
    batch = evidence.get("batch", {})
    return ChunkNotes(
        chunk_id=f"{episode.id}_batch_{int(batch.get('index', 0)):03d}",
        start=observations[0]["start"] if observations else episode.start,
        end=observations[-1]["end"] if observations else episode.end,
        section_title=episode.title.replace("$", ""),
        blocks=[],
        unresolved=[
            f"{stage} failed for indivisible evidence {observation_ids}: "
            f"{type(exc).__name__}: {exc}"
        ],
    )


def _synthesize_tree(
    orchestrator,
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    previous_context: list[dict[str, Any]],
) -> ChunkNotes:
    try:
        notes = _write_once(orchestrator, episode, evidence, previous_context)
    except _STRUCTURED_ERRORS as exc:
        split = split_evidence_payload(evidence)
        if split is None:
            logger.error(
                "[%s] episode synthesis failed for one evidence atom; recording unresolved: %s",
                episode.id,
                exc,
            )
            return _indivisible_failure(episode, evidence, "Episode synthesis", exc)

        left_evidence, right_evidence = split
        logger.warning(
            "[%s] episode_write failed for %d observations; splitting into %d + %d",
            episode.id,
            len(evidence.get("observations", [])),
            len(left_evidence["observations"]),
            len(right_evidence["observations"]),
        )
        left_notes = _synthesize_tree(orchestrator, episode, left_evidence, previous_context)
        right_context = _context_after(previous_context, left_notes)
        right_notes = _synthesize_tree(orchestrator, episode, right_evidence, right_context)
        return _merge_parts(episode, evidence, [left_notes, right_notes])

    if not orchestrator.config.global_validation or not notes.blocks:
        return notes

    try:
        audit = _validate_once(orchestrator, evidence, notes)
    except _STRUCTURED_ERRORS as exc:
        split = split_evidence_payload(evidence)
        if split is None:
            notes.unresolved.append(
                "Episode validation failed for indivisible evidence; synthesized notes were kept "
                f"without automatic validation: {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "[%s] validation failed for one evidence atom; keeping synthesis: %s",
                episode.id,
                exc,
            )
            return notes

        left_evidence, right_evidence = split
        logger.warning(
            "[%s] episode_validation failed for %d observations; re-synthesizing as %d + %d",
            episode.id,
            len(evidence.get("observations", [])),
            len(left_evidence["observations"]),
            len(right_evidence["observations"]),
        )
        left_notes = _synthesize_tree(orchestrator, episode, left_evidence, previous_context)
        right_context = _context_after(previous_context, left_notes)
        right_notes = _synthesize_tree(orchestrator, episode, right_evidence, right_context)
        return _merge_parts(episode, evidence, [left_notes, right_notes])

    apply_episode_validation(
        notes,
        audit,
        threshold=orchestrator.config.global_validation_apply_threshold,
    )
    return notes


def write_episode_batch(
    orchestrator,
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    previous_context: list[dict[str, Any]],
) -> ChunkNotes:
    """Synthesize one batch as a recursive split-and-merge tree.

    The LLM is never asked to repair or continue partial JSON. A failed multi-observation node is
    replaced by two smaller synthesis nodes and the host deterministically concatenates the results.
    """

    return _synthesize_tree(orchestrator, episode, evidence, previous_context)


def validate_episode_batch(orchestrator, evidence, notes) -> MathAudit:
    """Outer pipeline validation is a no-op because each synthesis leaf is validated in-tree."""

    return MathAudit()
