from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from .llm import StructuredTaskTooLargeError
from .schemas import (
    ClaimCompactionBatch,
    ClaimStatus,
    EpisodeClaimCompaction,
    KnowledgeClaim,
    LectureKnowledgeBase,
    MathStatus,
    ObservationKind,
    SourceStatus,
)
from .util import atomic_json_dump, stable_hash


STATE_CLAIM_COMPACTION_VERSION = 1
_CLAIM_COMPACTION_MAX_CHARS = 24_000


def _merge_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _source_status(claims: list[KnowledgeClaim]) -> SourceStatus:
    if any(item.source_status == SourceStatus.INFERRED for item in claims):
        return SourceStatus.INFERRED
    if any(item.source_status == SourceStatus.RECONSTRUCTED for item in claims):
        return SourceStatus.RECONSTRUCTED
    return SourceStatus.OBSERVED


def _math_status(claims: list[KnowledgeClaim]) -> MathStatus:
    order = {
        MathStatus.UNCHECKED: 0,
        MathStatus.CONSISTENT: 1,
        MathStatus.SUSPICIOUS: 2,
        MathStatus.INCORRECT: 3,
    }
    if not claims:
        return MathStatus.UNCHECKED
    return max((item.math_status for item in claims), key=order.__getitem__)


def _episode_payload(
    kb: LectureKnowledgeBase,
    episode_id: str,
) -> dict[str, Any] | None:
    episode = next((item for item in kb.episodes if item.id == episode_id), None)
    if episode is None:
        return None
    by_id = {item.id: item for item in kb.claims}
    claims = [
        by_id[claim_id]
        for claim_id in episode.claim_ids
        if claim_id in by_id and by_id[claim_id].status == ClaimStatus.ACTIVE
    ]
    if len(claims) <= 1:
        return None
    return {
        "episode": {
            "id": episode.id,
            "title": episode.title,
            "kind": str(episode.kind),
        },
        "claims": [
            {
                "id": item.id,
                "kind": str(item.kind),
                "content": item.content,
                "latex": item.latex,
                "source_status": str(item.source_status),
                "evidence_ids": list(item.evidence_ids),
                "introduced_at": item.introduced_at,
            }
            for item in claims
        ],
    }


def _payload_batches(kb: LectureKnowledgeBase) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 2

    for episode in sorted(kb.episodes, key=lambda item: (item.start, item.end, item.id)):
        payload = _episode_payload(kb, episode.id)
        if payload is None:
            continue
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        size = len(serialized)
        if current and current_chars + size > _CLAIM_COMPACTION_MAX_CHARS:
            batches.append(current)
            current = []
            current_chars = 2
        current.append(payload)
        current_chars += size + 1

    if current:
        batches.append(current)
    return batches


def _fingerprint_payload(kb: LectureKnowledgeBase) -> dict[str, Any]:
    active = {item.id: item for item in kb.claims if item.status == ClaimStatus.ACTIVE}
    return {
        "episodes": [
            {
                "id": episode.id,
                "title": episode.title,
                "kind": str(episode.kind),
                "claim_ids": [
                    claim_id for claim_id in episode.claim_ids if claim_id in active
                ],
            }
            for episode in sorted(kb.episodes, key=lambda item: (item.start, item.end, item.id))
        ],
        "claims": [
            {
                "id": claim.id,
                "kind": str(claim.kind),
                "content": claim.content,
                "latex": claim.latex,
                "episode_id": claim.episode_id,
                "source_status": str(claim.source_status),
                "math_status": str(claim.math_status),
                "evidence_ids": list(claim.evidence_ids),
                "introduced_at": claim.introduced_at,
            }
            for claim in sorted(active.values(), key=lambda item: (item.introduced_at, item.id))
        ],
    }


def _validate_episode_plan(
    plan: EpisodeClaimCompaction,
    *,
    episode_claim_ids: list[str],
    claim_by_id: dict[str, KnowledgeClaim],
) -> tuple[bool, str]:
    allowed = set(episode_claim_ids)
    for index, item in enumerate(plan.items):
        if item.type == "prose":
            unknown = [claim_id for claim_id in item.source_claim_ids if claim_id not in allowed]
            if unknown:
                return False, (
                    f"item {index} cites claims outside episode {plan.episode_id}: {unknown}"
                )
            if item.kind in {
                ObservationKind.CORRECTION,
                ObservationKind.RETRACTION,
                ObservationKind.TRANSITION,
                ObservationKind.UNRESOLVED,
            }:
                return False, f"item {index} uses non-canonical prose kind {item.kind}"
        else:
            source = claim_by_id.get(item.source_claim_id)
            if item.source_claim_id not in allowed or source is None:
                return False, (
                    f"item {index} selects formula outside episode {plan.episode_id}"
                )
            if not (source.latex or "").strip():
                return False, f"item {index} selects claim without LaTeX"
    return True, ""


def _compact_episode(
    kb: LectureKnowledgeBase,
    plan: EpisodeClaimCompaction,
) -> tuple[bool, str, int]:
    episode = next((item for item in kb.episodes if item.id == plan.episode_id), None)
    if episode is None:
        return False, f"unknown episode {plan.episode_id}", 0

    claim_by_id = {item.id: item for item in kb.claims}
    source_claims = [
        claim_by_id[claim_id]
        for claim_id in episode.claim_ids
        if claim_id in claim_by_id and claim_by_id[claim_id].status == ClaimStatus.ACTIVE
    ]
    source_ids = [item.id for item in source_claims]
    valid, reason = _validate_episode_plan(
        plan,
        episode_claim_ids=source_ids,
        claim_by_id=claim_by_id,
    )
    if not valid:
        return False, reason, 0
    if not plan.items:
        return False, f"empty compaction plan for {plan.episode_id}", 0

    derived: list[KnowledgeClaim] = []
    for index, item in enumerate(plan.items):
        claim_id = f"claim_compact_{episode.id}_{index:03d}"
        if item.type == "formula":
            source = claim_by_id[item.source_claim_id]
            derived.append(
                KnowledgeClaim(
                    id=claim_id,
                    kind=source.kind,
                    content="",
                    latex=source.latex,
                    scope=episode.id,
                    episode_id=episode.id,
                    status=ClaimStatus.ACTIVE,
                    math_status=source.math_status,
                    source_status=source.source_status,
                    evidence_ids=list(source.evidence_ids),
                    supersedes=[source.id],
                    introduced_at=source.introduced_at,
                )
            )
            continue

        selected = [claim_by_id[claim_id_] for claim_id_ in item.source_claim_ids]
        evidence_ids = _merge_unique(
            evidence_id
            for claim in selected
            for evidence_id in claim.evidence_ids
        )
        derived.append(
            KnowledgeClaim(
                id=claim_id,
                kind=item.kind,
                content=item.content.strip(),
                latex=None,
                scope=episode.id,
                episode_id=episode.id,
                status=ClaimStatus.ACTIVE,
                math_status=_math_status(selected),
                source_status=_source_status(selected),
                evidence_ids=evidence_ids,
                supersedes=list(item.source_claim_ids),
                introduced_at=min(item.introduced_at for item in selected),
            )
        )

    for source in source_claims:
        source.status = ClaimStatus.SUPERSEDED
    kb.claims.extend(derived)
    episode.claim_ids = [item.id for item in derived]
    if plan.unresolved:
        kb.unresolved = _merge_unique([*kb.unresolved, *plan.unresolved])
    return True, "", len(derived)


def compact_repaired_claims(
    orchestrator,
    *,
    kb: LectureKnowledgeBase,
    work,
    llm_config: dict[str, Any],
    force: bool,
) -> tuple[LectureKnowledgeBase, dict[str, Any], list[str]]:
    """Canonicalize repaired claims inside fixed semantic episodes before hierarchy realization.

    This is semantic-state compaction, not final-note writing. The model may merge claim prose and
    select essential existing formulas by id. Formula LaTeX remains host-owned and byte-for-byte
    copied from repaired claims.
    """

    fingerprint = stable_hash(
        {
            "version": STATE_CLAIM_COMPACTION_VERSION,
            "state": _fingerprint_payload(kb),
            "llm": llm_config,
        }
    )
    path = work / "repaired_claim_compaction.json"
    if path.exists() and not force:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint:
                cached = LectureKnowledgeBase.model_validate(payload["kb"])
                stats = dict(payload.get("stats") or {})
                stats["cache_hits"] = 1
                stats["model_calls"] = 0
                return cached, stats, list(payload.get("unresolved") or [])
        except (json.JSONDecodeError, KeyError, ValidationError):
            pass

    compacted = kb.model_copy(deep=True)
    batches = _payload_batches(compacted)
    plans: list[dict[str, Any]] = []
    unresolved: list[str] = []
    model_calls = 0
    fallback_episodes: list[str] = []
    claims_before = sum(item.status == ClaimStatus.ACTIVE for item in compacted.claims)
    episodes_compacted = 0

    for batch in batches:
        prompt = f"""Compact repaired mathematical claims inside FIXED semantic episodes.

Episodes with active repaired claims:
{json.dumps(batch, ensure_ascii=False, separators=(",", ":"))}

Return exactly one entry for every episode id above. This operation updates canonical semantic
state; it does NOT write final NoteBlocks or document sections.

For each episode, return an ordered sequence of semantic items:
- prose: one concise canonical mathematical statement with source_claim_ids from THIS episode;
- formula: select one existing source_claim_id whose LaTeX should remain explicitly visible.

Rules:
- merge repeated observations and intermediate restatements into the smallest set of distinct
  mathematical statements needed to study the episode;
- preserve enough proof steps to follow the actual argument, but drop bookkeeping such as
  "the previous inequality is repeated", "the next step begins", or restatements with no new role;
- for a definition/theorem episode, keep the actual definition/statement and only essential
  qualifications;
- for a proof/derivation, retain the logical spine and essential displayed formulas;
- for an example, retain setup, decisive steps and conclusion;
- formula items are ID selections only. Never generate, rewrite or repair LaTeX;
- prose must be plain semantic text, not LaTeX, and must cite every source claim it merges;
- do not mention lecturer/board/audio/OCR/reconstruction/chronology;
- do not import textbook facts, normalize theorem names, or silently fix a possible lecturer typo;
- preserve genuine ambiguity in unresolved instead of inventing missing mathematics;
- source claims may be omitted when they are duplicate, intermediate, or non-substantive;
- write prose in language code {orchestrator.output_language}.
"""
        try:
            generated = orchestrator._structured(
                prompt,
                ClaimCompactionBatch,
                operation="repaired_claim_compaction",
                split_oversized_task=True,
                thinking=False,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )
            model_calls += 1
        except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
            episode_ids = [str(item["episode"]["id"]) for item in batch]
            fallback_episodes.extend(episode_ids)
            unresolved.append(
                "Claim compaction failed for "
                + ", ".join(episode_ids)
                + f": {type(exc).__name__}: {exc}"
            )
            continue

        plans.append(generated.model_dump(mode="json"))
        requested = [str(item["episode"]["id"]) for item in batch]
        by_episode: dict[str, EpisodeClaimCompaction] = {}
        duplicate_ids: set[str] = set()
        for plan in generated.episodes:
            if plan.episode_id in by_episode:
                duplicate_ids.add(plan.episode_id)
            by_episode[plan.episode_id] = plan

        for episode_id in requested:
            plan = by_episode.get(episode_id)
            if plan is None or episode_id in duplicate_ids:
                fallback_episodes.append(episode_id)
                unresolved.append(
                    f"Claim compaction returned no unique plan for {episode_id}; original claims kept."
                )
                continue
            ok, reason, _ = _compact_episode(compacted, plan)
            if not ok:
                fallback_episodes.append(episode_id)
                unresolved.append(
                    f"Claim compaction rejected for {episode_id}: {reason}; original claims kept."
                )
                continue
            episodes_compacted += 1

    claims_after = sum(item.status == ClaimStatus.ACTIVE for item in compacted.claims)
    stats = {
        "cache_hits": 0,
        "model_calls": model_calls,
        "batches": len(batches),
        "episodes_compacted": episodes_compacted,
        "fallback_episodes": len(set(fallback_episodes)),
        "claims_before": claims_before,
        "claims_after": claims_after,
    }
    unresolved = _merge_unique(unresolved)
    atomic_json_dump(
        path,
        {
            "fingerprint": fingerprint,
            "version": STATE_CLAIM_COMPACTION_VERSION,
            "stats": stats,
            "plans": plans,
            "fallback_episode_ids": _merge_unique(fallback_episodes),
            "unresolved": unresolved,
            "kb": compacted.model_dump(mode="json"),
        },
    )
    return compacted, stats, unresolved
