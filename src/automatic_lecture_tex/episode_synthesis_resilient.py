from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .episode_synthesis import (
    apply_episode_validation,
    episode_evidence_batches as _base_evidence_batches,
    merge_episode_batches as _base_merge_episode_batches,
    validate_episode_batch as _validate_once,
    write_episode_batch as _write_once,
)
from .schemas import BlockType, ChunkNotes, MathAudit, SemanticEpisode

logger = logging.getLogger(__name__)

# Bump the downstream cache whenever split/merge, coverage, or stitching semantics change.
EPISODE_SYNTHESIS_CACHE_VERSION = 5

MAX_OBSERVATIONS_PER_SYNTHESIS_CALL = 6
_STRUCTURED_ERRORS = (json.JSONDecodeError, ValidationError)
_REQUIRED_COVERAGE_KINDS = {
    "definition",
    "claim",
    "equation",
    "proof_step",
    "example",
    "notation",
    "correction",
    "retraction",
}

_STATS: dict[str, float | int] = {}


def reset_synthesis_stats() -> None:
    _STATS.clear()
    _STATS.update(
        {
            "write_calls": 0,
            "validation_calls": 0,
            "boundary_validation_calls": 0,
            "synthesis_seconds": 0.0,
            "validation_seconds": 0.0,
            "proactive_splits": 0,
            "structured_splits": 0,
            "validation_splits": 0,
            "coverage_splits": 0,
            "indivisible_failures": 0,
            "coverage_unresolved": 0,
            "boundary_validation_failures": 0,
            "proof_merges": 0,
            "deduped_blocks": 0,
        }
    )


def synthesis_stats_snapshot() -> dict[str, float | int]:
    if not _STATS:
        reset_synthesis_stats()
    return dict(_STATS)


reset_synthesis_stats()


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


def episode_evidence_batches(kb, episode, config, transcript=None) -> list[dict[str, Any]]:
    """Create hard input-size bounded parent batches and attach only local transcript context."""

    return _base_evidence_batches(kb, episode, config, transcript)


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


def _normalize_body(value: str) -> str:
    return " ".join(value.split())


def _join_without_exact_overlap(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    max_overlap = min(len(left), len(right), 1200)
    for size in range(max_overlap, 39, -1):
        if _normalize_body(left[-size:]) == _normalize_body(right[:size]):
            return left + right[size:]
    return left + "\n\n" + right


def _merge_block_provenance(left, right) -> None:
    left.source_claim_ids = _merge_unique(left.source_claim_ids, right.source_claim_ids)
    left.source_evidence_ids = _merge_unique(left.source_evidence_ids, right.source_evidence_ids)


def stitch_chunk_notes(notes: ChunkNotes) -> ChunkNotes:
    """Deterministically clean split boundaries without introducing new mathematical content."""

    stitched = []
    for block in notes.blocks:
        if not stitched:
            stitched.append(block)
            continue
        previous = stitched[-1]
        same_body = (
            previous.type == block.type
            and _normalize_body(previous.latex) == _normalize_body(block.latex)
        )
        if same_body:
            _merge_block_provenance(previous, block)
            _STATS["deduped_blocks"] = int(_STATS["deduped_blocks"]) + 1
            continue

        right_title = (block.title or "").lower()
        proof_continuation = (
            previous.type == BlockType.PROOF
            and block.type == BlockType.PROOF
            and (
                not block.title
                or not previous.title
                or block.title == previous.title
                or "продолж" in right_title
                or right_title == "доказательство"
            )
        )
        if proof_continuation:
            previous.latex = _join_without_exact_overlap(previous.latex, block.latex)
            _merge_block_provenance(previous, block)
            if not previous.title and block.title:
                previous.title = block.title
            _STATS["proof_merges"] = int(_STATS["proof_merges"]) + 1
            continue
        stitched.append(block)

    notes.blocks = stitched
    return notes


def _post_merge_boundary_audit(orchestrator, evidence: dict[str, Any], notes: ChunkNotes) -> None:
    """Validate a reconstructed split parent against its original <=6-observation evidence."""

    if not orchestrator.config.global_validation or not notes.blocks:
        return
    started = time.perf_counter()
    _STATS["boundary_validation_calls"] = int(_STATS["boundary_validation_calls"]) + 1
    try:
        audit = _validate_once(orchestrator, evidence, notes)
    except _STRUCTURED_ERRORS as exc:
        _STATS["boundary_validation_failures"] = int(
            _STATS["boundary_validation_failures"]
        ) + 1
        notes.unresolved.append(
            "Bounded post-merge validation failed; child validations were kept: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        apply_episode_validation(
            notes,
            audit,
            threshold=orchestrator.config.global_validation_apply_threshold,
        )
    finally:
        _STATS["validation_seconds"] = float(_STATS["validation_seconds"]) + (
            time.perf_counter() - started
        )


def _merge_parts(
    orchestrator,
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

    _post_merge_boundary_audit(orchestrator, evidence, result)
    return stitch_chunk_notes(result)


def _indivisible_failure(
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    stage: str,
    exc: Exception,
) -> ChunkNotes:
    observations = evidence.get("observations", [])
    observation_ids = [item.get("id", "?") for item in observations]
    batch = evidence.get("batch", {})
    _STATS["indivisible_failures"] = int(_STATS["indivisible_failures"]) + 1
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


def _call_with_split_retry_policy(orchestrator, evidence: dict[str, Any], fn, *args):
    """Let multi-atom failures reach the split tree after the first invalid response."""

    if len(evidence.get("observations", [])) < 2:
        return fn(*args)

    llm = getattr(orchestrator, "llm", None)
    config = getattr(llm, "config", None)
    if config is None or not hasattr(config, "max_retries"):
        return fn(*args)

    retries = config.max_retries
    config.max_retries = 0
    try:
        return fn(*args)
    finally:
        config.max_retries = retries


def _required_observation_ids(evidence: dict[str, Any]) -> set[str]:
    return {
        str(item["id"])
        for item in evidence.get("observations", [])
        if str(item.get("kind", "")) in _REQUIRED_COVERAGE_KINDS
        and (str(item.get("text", "")).strip() or str(item.get("latex") or "").strip())
    }


def _covered_observation_ids(notes: ChunkNotes, evidence: dict[str, Any]) -> set[str]:
    claim_evidence = {
        str(claim["id"]): {str(item) for item in claim.get("evidence_ids", [])}
        for claim in evidence.get("claims", [])
    }
    covered: set[str] = set()
    for block in notes.blocks:
        covered.update(str(item) for item in block.source_evidence_ids)
        for claim_id in block.source_claim_ids:
            covered.update(claim_evidence.get(str(claim_id), set()))
    return covered


def missing_coverage(notes: ChunkNotes, evidence: dict[str, Any]) -> list[str]:
    required = _required_observation_ids(evidence)
    covered = _covered_observation_ids(notes, evidence)
    return sorted(required - covered)


def _split_for_failure(
    orchestrator,
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    previous_context: list[dict[str, Any]],
    *,
    reason: str,
) -> ChunkNotes | None:
    split = split_evidence_payload(evidence)
    if split is None:
        return None
    left_evidence, right_evidence = split
    logger.warning(
        "[%s] %s for %d observations; splitting into %d + %d",
        episode.id,
        reason,
        len(evidence.get("observations", [])),
        len(left_evidence["observations"]),
        len(right_evidence["observations"]),
    )
    left_notes = _synthesize_tree(orchestrator, episode, left_evidence, previous_context)
    right_context = _context_after(previous_context, left_notes)
    right_notes = _synthesize_tree(orchestrator, episode, right_evidence, right_context)
    return _merge_parts(orchestrator, episode, evidence, [left_notes, right_notes])


def _synthesize_tree(
    orchestrator,
    episode: SemanticEpisode,
    evidence: dict[str, Any],
    previous_context: list[dict[str, Any]],
) -> ChunkNotes:
    observations = evidence.get("observations", [])
    if len(observations) > MAX_OBSERVATIONS_PER_SYNTHESIS_CALL:
        _STATS["proactive_splits"] = int(_STATS["proactive_splits"]) + 1
        split_notes = _split_for_failure(
            orchestrator,
            episode,
            evidence,
            previous_context,
            reason="proactive semantic-atom cap",
        )
        if split_notes is not None:
            return split_notes

    started = time.perf_counter()
    _STATS["write_calls"] = int(_STATS["write_calls"]) + 1
    try:
        notes = _call_with_split_retry_policy(
            orchestrator,
            evidence,
            _write_once,
            orchestrator,
            episode,
            evidence,
            previous_context,
        )
    except _STRUCTURED_ERRORS as exc:
        _STATS["synthesis_seconds"] = float(_STATS["synthesis_seconds"]) + (
            time.perf_counter() - started
        )
        split_notes = _split_for_failure(
            orchestrator,
            episode,
            evidence,
            previous_context,
            reason="episode_write structured failure",
        )
        if split_notes is not None:
            _STATS["structured_splits"] = int(_STATS["structured_splits"]) + 1
            return split_notes
        logger.error(
            "[%s] episode synthesis failed for one evidence atom; recording unresolved: %s",
            episode.id,
            exc,
        )
        return _indivisible_failure(episode, evidence, "Episode synthesis", exc)
    else:
        _STATS["synthesis_seconds"] = float(_STATS["synthesis_seconds"]) + (
            time.perf_counter() - started
        )

    missing = missing_coverage(notes, evidence)
    if missing:
        split_notes = _split_for_failure(
            orchestrator,
            episode,
            evidence,
            previous_context,
            reason=f"coverage failure missing={missing}",
        )
        if split_notes is not None:
            _STATS["coverage_splits"] = int(_STATS["coverage_splits"]) + 1
            return split_notes
        _STATS["coverage_unresolved"] = int(_STATS["coverage_unresolved"]) + len(missing)
        notes.unresolved.append(
            "Host coverage invariant: substantive evidence was not represented by any generated "
            f"block: {missing}."
        )

    if not orchestrator.config.global_validation or not notes.blocks:
        return notes

    started = time.perf_counter()
    _STATS["validation_calls"] = int(_STATS["validation_calls"]) + 1
    try:
        audit = _call_with_split_retry_policy(
            orchestrator,
            evidence,
            _validate_once,
            orchestrator,
            evidence,
            notes,
        )
    except _STRUCTURED_ERRORS as exc:
        _STATS["validation_seconds"] = float(_STATS["validation_seconds"]) + (
            time.perf_counter() - started
        )
        split_notes = _split_for_failure(
            orchestrator,
            episode,
            evidence,
            previous_context,
            reason="episode_validation structured failure",
        )
        if split_notes is not None:
            _STATS["validation_splits"] = int(_STATS["validation_splits"]) + 1
            return split_notes
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
    else:
        _STATS["validation_seconds"] = float(_STATS["validation_seconds"]) + (
            time.perf_counter() - started
        )

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
    return _synthesize_tree(orchestrator, episode, evidence, previous_context)


def validate_episode_batch(orchestrator, evidence, notes) -> MathAudit:
    """Outer pipeline validation is a no-op because validation is performed inside the tree."""

    return MathAudit()


def merge_episode_batches(episode: SemanticEpisode, batches: list[ChunkNotes]) -> ChunkNotes:
    """Merge independent hard-size parent batches and clean only their immediate boundaries."""

    return stitch_chunk_notes(_base_merge_episode_batches(episode, batches))
