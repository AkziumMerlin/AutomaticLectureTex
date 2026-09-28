from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .schemas import (
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
)
from .util import atomic_json_dump, stable_hash


STATE_CANONICALIZATION_VERSION = 1
_LOCAL_BATCH_SIZE = 64
_LOCAL_BATCH_OVERLAP = 8
_GLOBAL_AUDIT_MAX_CHARS = 60000

CanonicalRelationKind = Literal[
    "duplicate",
    "intermediate",
    "meta",
    "supersedes",
    "conflict",
]


class CanonicalObservationRelation(BaseModel):
    """ID-only relation proposed over already repaired observations.

    The schema deliberately contains no replacement text or LaTeX fields. Canonicalization is not
    allowed to write mathematics; it may only relate existing observations.
    """

    source_observation_id: str
    relation: CanonicalRelationKind
    target_observation_id: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_target(self) -> CanonicalObservationRelation:
        if self.relation == "meta":
            return self
        if not self.target_observation_id:
            raise ValueError(f"{self.relation} relation requires target_observation_id")
        if self.target_observation_id == self.source_observation_id:
            raise ValueError("canonicalization relation cannot target itself")
        return self


class CanonicalObservationPlan(BaseModel):
    relations: list[CanonicalObservationRelation] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _norm_latex(value: str | None) -> str:
    text = str(value or "").strip()
    text = text.replace(r"\,", "").replace(r"\;", "").replace(r"\!", "")
    return re.sub(r"\s+", "", text)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _compact_observation(item: LectureObservation, *, limit: int = 280) -> dict[str, Any]:
    return {
        "id": item.id,
        "episode_id": item.episode_id,
        "start": item.start,
        "end": item.end,
        "kind": item.kind,
        "text": item.text[:limit],
        "latex": (item.latex or "")[:limit] or None,
        "target_observation_id": item.target_observation_id,
    }


def _safe_duplicate(source: LectureObservation, target: LectureObservation) -> bool:
    if source.kind != target.kind:
        return False

    source_latex = _norm_latex(source.latex)
    target_latex = _norm_latex(target.latex)
    source_text = _norm_text(source.text)
    target_text = _norm_text(target.text)

    if source_latex or target_latex:
        if not source_latex or source_latex != target_latex:
            return False
        if not source_text or not target_text:
            return True
        return _similarity(source_text, target_text) >= 0.90

    return _similarity(source_text, target_text) >= 0.97


def _safe_intermediate(source: LectureObservation, target: LectureObservation) -> bool:
    if source.episode_id != target.episode_id:
        return False
    if source.kind != target.kind or source.kind == ObservationKind.PROOF_STEP:
        return False
    if target.start < source.start:
        return False

    source_latex = _norm_latex(source.latex)
    target_latex = _norm_latex(target.latex)
    if source_latex and target_latex:
        if source_latex == target_latex:
            return False
        if len(source_latex) < 8 or len(source_latex) / max(1, len(target_latex)) < 0.30:
            return False
        return source_latex in target_latex

    if source_latex or target_latex:
        return False

    source_text = _norm_text(source.text)
    target_text = _norm_text(target.text)
    if len(source_text) < 20 or len(source_text) / max(1, len(target_text)) < 0.50:
        return False
    return source_text != target_text and source_text in target_text


def _explicit_supersession(source: LectureObservation, target: LectureObservation) -> bool:
    if target.kind not in {ObservationKind.CORRECTION, ObservationKind.RETRACTION}:
        return False
    return target.target_observation_id == source.id


def _deterministic_relations(kb: LectureKnowledgeBase) -> list[CanonicalObservationRelation]:
    """Find only relations whose suppression can be checked without model judgement."""

    relations: list[CanonicalObservationRelation] = []
    by_episode: dict[str, list[LectureObservation]] = {}
    for item in sorted(kb.observations, key=lambda obs: (obs.start, obs.end, obs.id)):
        by_episode.setdefault(item.episode_id, []).append(item)

    for observations in by_episode.values():
        for index, source in enumerate(observations):
            if source.kind == ObservationKind.TRANSITION and not source.latex:
                relations.append(
                    CanonicalObservationRelation(
                        source_observation_id=source.id,
                        relation="meta",
                        reason="host: transition event has no mathematical payload",
                    )
                )

            for target in observations[index + 1 : index + 7]:
                if _safe_duplicate(source, target):
                    relations.append(
                        CanonicalObservationRelation(
                            source_observation_id=source.id,
                            target_observation_id=target.id,
                            relation="duplicate",
                            reason="host: exact/near-exact redundant payload",
                        )
                    )
                    break
                if _safe_intermediate(source, target):
                    relations.append(
                        CanonicalObservationRelation(
                            source_observation_id=source.id,
                            target_observation_id=target.id,
                            relation="intermediate",
                            reason="host: later observation contains the same partial payload",
                        )
                    )
                    break

    by_id = {item.id: item for item in kb.observations}
    for target in kb.observations:
        source_id = target.target_observation_id
        source = by_id.get(source_id or "")
        if source is not None and _explicit_supersession(source, target):
            relations.append(
                CanonicalObservationRelation(
                    source_observation_id=source.id,
                    target_observation_id=target.id,
                    relation="supersedes",
                    reason="host: explicit correction/retraction target",
                )
            )
    return relations


def _observation_batches(
    observations: list[LectureObservation],
    *,
    size: int = _LOCAL_BATCH_SIZE,
    overlap: int = _LOCAL_BATCH_OVERLAP,
) -> list[list[LectureObservation]]:
    if not observations:
        return []
    batches: list[list[LectureObservation]] = []
    start = 0
    while start < len(observations):
        batch = observations[start : start + size]
        batches.append(batch)
        if start + size >= len(observations):
            break
        start += max(1, size - overlap)
    return batches


def _load_plan(path: Path, fingerprint: str) -> CanonicalObservationPlan | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return CanonicalObservationPlan.model_validate(payload["plan"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _propose_local_relations(
    orchestrator,
    *,
    observations: list[LectureObservation],
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    force: bool,
) -> tuple[list[CanonicalObservationRelation], list[str], int]:
    relations: list[CanonicalObservationRelation] = []
    unresolved: list[str] = []
    cache_hits = 0

    for batch_index, batch in enumerate(_observation_batches(observations)):
        compact = [_compact_observation(item) for item in batch]
        allowed_ids = {item.id for item in batch}
        fingerprint = stable_hash(
            {
                "state_pipeline_version": pipeline_version,
                "canonicalization_version": STATE_CANONICALIZATION_VERSION,
                "mode": "local",
                "observations": compact,
                "llm": llm_config,
            }
        )
        path = work / "state_canonicalization_batches" / f"batch_{batch_index:03d}.json"
        plan = None if force else _load_plan(path, fingerprint)
        if plan is not None:
            cache_hits += 1
        else:
            prompt = f"""Conservatively relate already repaired mathematical lecture events.
You are NOT a writer and MUST NOT rewrite, repair, normalize, or invent any mathematics.
Return ONLY relations between the supplied observation IDs. The host may ignore your relation.

Observations:
{json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}

Allowed relation meanings:
- duplicate: same mathematical event repeated; choose one existing target ID carrying the same content.
- intermediate: source is a visibly partial/incomplete state later completed by target.
- meta: pure organizational transition with no mathematical content; target must be omitted.
- supersedes: a later existing event explicitly replaces/corrects the source.
- conflict: two existing events make materially incompatible mathematical statements.

Be conservative. Omit a relation when unsure. Do not use textbook knowledge to rewrite a formula.
Do not put mathematical content in reason. Never output text or LaTeX, only IDs, relation labels, and
short diagnostic reasons.
"""
            plan = orchestrator._structured(
                prompt,
                CanonicalObservationPlan,
                operation="state_canonicalize_local",
                split_oversized_task=True,
            )
            atomic_json_dump(
                path,
                {
                    "fingerprint": fingerprint,
                    "plan": plan.model_dump(mode="json"),
                },
            )

        for relation in plan.relations:
            if relation.source_observation_id not in allowed_ids:
                unresolved.append(
                    f"Canonicalization ignored unknown local source {relation.source_observation_id}."
                )
                continue
            if (
                relation.target_observation_id
                and relation.target_observation_id not in allowed_ids
            ):
                unresolved.append(
                    "Canonicalization ignored out-of-batch target "
                    f"{relation.target_observation_id}."
                )
                continue
            relations.append(relation)
        unresolved.extend(plan.unresolved)

    return relations, unresolved, cache_hits


def _propose_global_audit(
    orchestrator,
    *,
    observations: list[LectureObservation],
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    force: bool,
) -> tuple[list[CanonicalObservationRelation], list[str], int]:
    """Find long-range semantic relations. These are audit-only unless host-proven safe."""

    key_kinds = {
        ObservationKind.DEFINITION,
        ObservationKind.CLAIM,
        ObservationKind.EQUATION,
        ObservationKind.NOTATION,
        ObservationKind.CORRECTION,
        ObservationKind.RETRACTION,
    }
    compact = [
        _compact_observation(item, limit=180)
        for item in observations
        if item.kind in key_kinds
    ]
    serialized = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if not compact:
        return [], [], 0
    if len(serialized) > _GLOBAL_AUDIT_MAX_CHARS:
        return (
            [],
            [
                "Global canonicalization audit skipped because the compact mathematical catalog "
                f"exceeded {_GLOBAL_AUDIT_MAX_CHARS} characters."
            ],
            0,
        )

    fingerprint = stable_hash(
        {
            "state_pipeline_version": pipeline_version,
            "canonicalization_version": STATE_CANONICALIZATION_VERSION,
            "mode": "global",
            "observations": compact,
            "llm": llm_config,
        }
    )
    path = work / "state_canonicalization_global.json"
    plan = None if force else _load_plan(path, fingerprint)
    cache_hits = int(plan is not None)
    if plan is None:
        prompt = f"""Audit long-range relations in an already repaired mathematical lecture state.
You cannot write or modify mathematics. Return IDs only.

Compact mathematical catalog:
{serialized}

Return only:
- supersedes when a later EXISTING observation clearly replaces/refines an earlier EXISTING one;
- conflict when two EXISTING observations are mathematically incompatible and neither can be safely
  deleted from the evidence alone.

This pass is deliberately conservative and audit-oriented. Do not return duplicate/intermediate/meta.
Do not correct formulas, do not invent a canonical statement, and do not use textbook knowledge as
a substitute for lecture evidence. If uncertain, return nothing.
"""
        plan = orchestrator._structured(
            prompt,
            CanonicalObservationPlan,
            operation="state_canonicalize_global",
            split_oversized_task=True,
        )
        atomic_json_dump(
            path,
            {
                "fingerprint": fingerprint,
                "plan": plan.model_dump(mode="json"),
            },
        )

    allowed_ids = {item["id"] for item in compact}
    relations: list[CanonicalObservationRelation] = []
    unresolved = list(plan.unresolved)
    for relation in plan.relations:
        if relation.relation not in {"supersedes", "conflict"}:
            continue
        if relation.source_observation_id not in allowed_ids:
            continue
        if (
            relation.target_observation_id is None
            or relation.target_observation_id not in allowed_ids
        ):
            continue
        relations.append(relation)
    return relations, unresolved, cache_hits


def _dedupe_relations(
    relations: list[CanonicalObservationRelation],
) -> list[CanonicalObservationRelation]:
    seen: set[tuple[str, str, str | None]] = set()
    result: list[CanonicalObservationRelation] = []
    for relation in relations:
        key = (
            relation.source_observation_id,
            relation.relation,
            relation.target_observation_id,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result


def _apply_relations(
    kb: LectureKnowledgeBase,
    relations: list[CanonicalObservationRelation],
) -> tuple[LectureKnowledgeBase, dict[str, int], list[dict[str, Any]]]:
    """Apply only host-verifiable suppressions; semantic judgement remains audit-only."""

    result = kb.model_copy(deep=True)
    by_id = {item.id: item for item in result.observations}
    suppressed: set[str] = set()
    applied: list[dict[str, Any]] = []
    stats = {
        "suppressed_duplicate": 0,
        "suppressed_intermediate": 0,
        "suppressed_meta": 0,
        "suppressed_explicit_superseded": 0,
        "flagged_supersedes": 0,
        "flagged_conflict": 0,
        "rejected_unsafe_relation": 0,
    }

    for relation in _dedupe_relations(relations):
        source = by_id.get(relation.source_observation_id)
        target = by_id.get(relation.target_observation_id or "")
        accepted = False
        applied_as = "audit_only"

        if source is None:
            stats["rejected_unsafe_relation"] += 1
            continue

        if relation.relation == "duplicate" and target is not None:
            accepted = _safe_duplicate(source, target)
            applied_as = "suppressed" if accepted else "audit_only"
            key = "suppressed_duplicate"
        elif relation.relation == "intermediate" and target is not None:
            accepted = _safe_intermediate(source, target)
            applied_as = "suppressed" if accepted else "audit_only"
            key = "suppressed_intermediate"
        elif relation.relation == "meta":
            accepted = source.kind == ObservationKind.TRANSITION and not source.latex
            applied_as = "suppressed" if accepted else "audit_only"
            key = "suppressed_meta"
        elif relation.relation == "supersedes" and target is not None:
            accepted = _explicit_supersession(source, target)
            applied_as = "suppressed" if accepted else "audit_only"
            key = "suppressed_explicit_superseded" if accepted else "flagged_supersedes"
        elif relation.relation == "conflict" and target is not None:
            key = "flagged_conflict"
        else:
            key = "rejected_unsafe_relation"

        if relation.relation in {"duplicate", "intermediate", "meta"} and not accepted:
            stats["rejected_unsafe_relation"] += 1
        else:
            stats[key] += 1

        if accepted:
            suppressed.add(source.id)
            if target is not None:
                result.observation_aliases[source.id] = target.id
                target.evidence_refs = list(
                    dict.fromkeys([*target.evidence_refs, *source.evidence_refs])
                )
                target.window_ids = list(
                    dict.fromkeys([*target.window_ids, *source.window_ids])
                )

        applied.append(
            {
                **relation.model_dump(mode="json"),
                "applied_as": applied_as,
                "host_verified": accepted,
            }
        )

    result.observations = [
        item for item in result.observations if item.id not in suppressed
    ]
    surviving_ids = {item.id for item in result.observations}
    for episode in result.episodes:
        episode.observation_ids = [
            item for item in episode.observation_ids if item in surviving_ids
        ]

    for symbol in result.symbols:
        if not symbol.evidence_ids:
            continue
        remapped: list[str] = []
        for evidence_id in symbol.evidence_ids:
            current = evidence_id
            seen: set[str] = set()
            while current in result.observation_aliases and current not in seen:
                seen.add(current)
                current = result.observation_aliases[current]
            if current in surviving_ids and current not in remapped:
                remapped.append(current)
        symbol.evidence_ids = remapped
        if not remapped:
            symbol.active = False

    return result, stats, applied


def run_state_canonicalization(
    orchestrator,
    *,
    kb: LectureKnowledgeBase,
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    force: bool,
) -> tuple[LectureKnowledgeBase, dict[str, int], list[str]]:
    """Conservatively compact repaired state without allowing an LLM to write mathematics."""

    observations = sorted(kb.observations, key=lambda item: (item.start, item.end, item.id))
    deterministic = _deterministic_relations(kb)
    local, local_unresolved, local_cache_hits = _propose_local_relations(
        orchestrator,
        observations=observations,
        work=work,
        llm_config=llm_config,
        pipeline_version=pipeline_version,
        force=force,
    )
    global_relations, global_unresolved, global_cache_hits = _propose_global_audit(
        orchestrator,
        observations=observations,
        work=work,
        llm_config=llm_config,
        pipeline_version=pipeline_version,
        force=force,
    )

    compacted, stats, applied = _apply_relations(
        kb,
        [*deterministic, *local, *global_relations],
    )
    unresolved = list(dict.fromkeys([*local_unresolved, *global_unresolved]))
    compacted.unresolved = list(dict.fromkeys([*compacted.unresolved, *unresolved]))

    stats["input_observations"] = len(kb.observations)
    stats["output_observations"] = len(compacted.observations)
    stats["local_cache_hits"] = local_cache_hits
    stats["global_cache_hits"] = global_cache_hits
    stats["relations_total"] = len(_dedupe_relations([*deterministic, *local, *global_relations]))

    atomic_json_dump(
        work / "state_canonicalization.json",
        {
            "version": STATE_CANONICALIZATION_VERSION,
            "stats": stats,
            "relations": applied,
            "unresolved": unresolved,
        },
    )
    return compacted, stats, unresolved
