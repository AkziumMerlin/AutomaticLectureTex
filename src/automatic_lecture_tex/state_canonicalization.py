from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .schemas import LectureKnowledgeBase, LectureObservation, ObservationKind
from .util import atomic_json_dump, stable_hash


STATE_CANONICALIZATION_VERSION = 3
_GLOBAL_AUDIT_MAX_CHARS = 60000
_FORWARD_SCAN = 10

CanonicalRelationKind = Literal[
    "duplicate",
    "intermediate",
    "meta",
    "supersedes",
    "conflict",
]


class CanonicalObservationRelation(BaseModel):
    """ID-only relation over already repaired observations.

    There is deliberately no text/LaTeX replacement channel. The canonicalizer can only point at
    existing observations; host code decides whether one render channel is provably redundant.
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


class CanonicalRenderPolicy(BaseModel):
    """Render-time suppression mask; repaired state itself remains immutable."""

    suppress_text_ids: list[str] = Field(default_factory=list)
    suppress_latex_ids: list[str] = Field(default_factory=list)


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _norm_latex(value: str | None) -> str:
    text = str(value or "").strip()
    text = text.replace(r"\,", "").replace(r"\;", "").replace(r"\!", "")
    return re.sub(r"\s+", "", text)


def _ordered_after(source: LectureObservation, target: LectureObservation) -> bool:
    return (target.start, target.end, target.id) > (source.start, source.end, source.id)


def _incomplete_formula(value: str | None) -> bool:
    r"""Detect explicit board-state placeholders, not ordinary mathematical ellipses.

    In particular y_1,\ldots,y_m is a complete formula and must never be classified as an
    unfinished board state merely because it contains an ellipsis.
    """

    latex = str(value or "").strip()
    if r"\text{или}" in latex or r"\text{or}" in latex:
        return True
    if re.search(
        r"\\(?:bigl|Bigl)?\[\s*\\(?:cdots|ldots|dots)\s*\\(?:bigr|Bigr)?\]",
        latex,
    ):
        return True
    if re.search(r"^\s*\\(?:cdots|ldots|dots)\b", latex):
        return True
    return bool(
        re.search(
            r"\\(?:cdots|ldots|dots)(?:\\[,;!]|\s|\\[}\]])*$",
            latex,
        )
    )


def _math_tokens(value: str | None) -> set[str]:
    latex = str(value or "")
    for marker in (
        r"\cdots",
        r"\ldots",
        r"\dots",
        r"\text{или}",
        "cdots",
        "ldots",
        "dots",
    ):
        latex = latex.replace(marker, "")
    ignored = {
        "left",
        "right",
        "bigl",
        "bigr",
        "big",
        "middle",
        "mid",
        "text",
    }
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z]+|[А-Яа-яЁё]+|\d+", latex)
        if token and token.casefold() not in ignored
    }


def _formula_subsumed(source: LectureObservation, target: LectureObservation) -> bool:
    """Host-verifiable formula redundancy.

    This intentionally says nothing about prose. Suppressing a formula never suppresses the
    observation text, so a useful proof explanation cannot disappear because a later formula
    happens to contain the same symbols.
    """

    if source.episode_id != target.episode_id or not _ordered_after(source, target):
        return False
    source_latex = _norm_latex(source.latex)
    target_latex = _norm_latex(target.latex)
    if not source_latex or not target_latex:
        return False

    if source_latex == target_latex:
        return True

    # Literal completion/prefixing is safe at the formula channel: the complete target retains
    # every source symbol and adds context such as a name or the rest of a set-builder.
    if len(source_latex) >= 8 and source_latex in target_latex:
        return True

    # Board-state partials often contain \cdots or an explicit ambiguity placeholder, so literal
    # containment fails when the completed line also fixes a variable name. Only suppress the
    # partial formula when most of its mathematical vocabulary survives in the later line.
    if _incomplete_formula(source.latex) and not _incomplete_formula(target.latex):
        source_tokens = _math_tokens(source.latex)
        target_tokens = _math_tokens(target.latex)
        return bool(source_tokens) and source_tokens.issubset(target_tokens)

    return False


def _text_duplicate(source: LectureObservation, target: LectureObservation) -> bool:
    """Extremely strict text-only deduplication; semantic paraphrases remain visible."""

    if source.episode_id != target.episode_id or source.kind != target.kind:
        return False
    if not _ordered_after(source, target):
        return False
    source_text = _norm_text(source.text)
    target_text = _norm_text(target.text)
    if not source_text or not target_text:
        return False
    return source_text == target_text or (
        len(source_text) >= 24
        and len(source_text) / max(1, len(target_text)) >= 0.75
        and source_text in target_text
    )


def _explicit_supersession(source: LectureObservation, target: LectureObservation) -> bool:
    if target.kind not in {ObservationKind.CORRECTION, ObservationKind.RETRACTION}:
        return False
    return target.target_observation_id == source.id


def _deterministic_relations(kb: LectureKnowledgeBase) -> list[CanonicalObservationRelation]:
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
                        reason="host: transition text is structural evidence, not rendered mathematics",
                    )
                )

            for target in observations[index + 1 : index + 1 + _FORWARD_SCAN]:
                if _formula_subsumed(source, target):
                    relation = (
                        "intermediate"
                        if _norm_latex(source.latex) != _norm_latex(target.latex)
                        else "duplicate"
                    )
                    relations.append(
                        CanonicalObservationRelation(
                            source_observation_id=source.id,
                            target_observation_id=target.id,
                            relation=relation,
                            reason="host: later formula provably contains the same rendered math",
                        )
                    )
                    break
                if not source.latex and not target.latex and _text_duplicate(source, target):
                    relations.append(
                        CanonicalObservationRelation(
                            source_observation_id=source.id,
                            target_observation_id=target.id,
                            relation="duplicate",
                            reason="host: exact/literal text redundancy",
                        )
                    )
                    break

    by_id = {item.id: item for item in kb.observations}
    for target in kb.observations:
        source = by_id.get(target.target_observation_id or "")
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


def _compact_observation(item: LectureObservation, *, limit: int = 180) -> dict[str, Any]:
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


def _propose_semantic_audit(
    orchestrator,
    *,
    observations: list[LectureObservation],
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    force: bool,
) -> tuple[list[CanonicalObservationRelation], list[str], int]:
    """Optional one-call audit for long-range relations.

    Model-only semantic relations are diagnostic. They can never rewrite text/LaTeX, and unless a
    host-verifiable formula/text condition also holds they remain audit-only.
    """

    compact = [
        _compact_observation(item)
        for item in observations
        if item.kind
        in {
            ObservationKind.DEFINITION,
            ObservationKind.CLAIM,
            ObservationKind.EQUATION,
            ObservationKind.NOTATION,
            ObservationKind.CORRECTION,
            ObservationKind.RETRACTION,
        }
    ]
    serialized = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if not compact:
        return [], [], 0
    if len(serialized) > _GLOBAL_AUDIT_MAX_CHARS:
        return (
            [],
            [
                "Semantic canonicalization audit skipped because the compact catalog exceeded "
                f"{_GLOBAL_AUDIT_MAX_CHARS} characters."
            ],
            0,
        )

    fingerprint = stable_hash(
        {
            "state_pipeline_version": pipeline_version,
            "canonicalization_version": STATE_CANONICALIZATION_VERSION,
            "observations": compact,
            "llm": llm_config,
        }
    )
    path = work / "state_canonicalization_global.json"
    plan = None if force else _load_plan(path, fingerprint)
    cache_hits = int(plan is not None)
    if plan is None:
        prompt = f"""Audit long-range relations in an already repaired mathematical lecture state.
You are not a writer. You cannot change, normalize, or invent any mathematical text or formula.
Return observation IDs and relation labels only.

Catalog:
{serialized}

Use:
- supersedes when a later EXISTING observation clearly finalizes/replaces an earlier one;
- conflict when two EXISTING observations are mathematically incompatible.

This is audit-only semantic judgement. Do not propose prose rewrites or corrected formulas. If
uncertain, return nothing.
"""
        try:
            plan = orchestrator._structured(
                prompt,
                CanonicalObservationPlan,
                operation="state_canonicalize_global",
                split_oversized_task=True,
            )
        except Exception as exc:
            return (
                [],
                [
                    "Semantic canonicalization audit failed safely: "
                    f"{type(exc).__name__}: {exc}"
                ],
                0,
            )
        atomic_json_dump(
            path,
            {"fingerprint": fingerprint, "plan": plan.model_dump(mode="json")},
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


def _build_render_policy(
    kb: LectureKnowledgeBase,
    relations: list[CanonicalObservationRelation],
) -> tuple[CanonicalRenderPolicy, dict[str, int], list[dict[str, Any]]]:
    """Apply relations only to render channels; never mutate repaired observations."""

    by_id = {item.id: item for item in kb.observations}
    suppress_text: set[str] = set()
    suppress_latex: set[str] = set()
    applied: list[dict[str, Any]] = []
    stats = {
        "suppressed_text_blocks": 0,
        "suppressed_latex_blocks": 0,
        "flagged_supersedes": 0,
        "flagged_conflict": 0,
        "rejected_unsafe_relation": 0,
    }

    for relation in _dedupe_relations(relations):
        source = by_id.get(relation.source_observation_id)
        target = by_id.get(relation.target_observation_id or "")
        channels: list[str] = []
        host_verified = False

        if source is None:
            stats["rejected_unsafe_relation"] += 1
            continue

        if relation.relation == "meta":
            if source.kind == ObservationKind.TRANSITION and not source.latex:
                suppress_text.add(source.id)
                channels.append("text")
                host_verified = True
        elif relation.relation in {"duplicate", "intermediate"} and target is not None:
            if source.latex and target.latex and _formula_subsumed(source, target):
                suppress_latex.add(source.id)
                channels.append("latex")
                host_verified = True
            if not source.latex and not target.latex and _text_duplicate(source, target):
                suppress_text.add(source.id)
                channels.append("text")
                host_verified = True
        elif relation.relation == "supersedes" and target is not None:
            if _explicit_supersession(source, target):
                if source.text:
                    suppress_text.add(source.id)
                    channels.append("text")
                if source.latex:
                    suppress_latex.add(source.id)
                    channels.append("latex")
                host_verified = True
            else:
                stats["flagged_supersedes"] += 1
        elif relation.relation == "conflict" and target is not None:
            stats["flagged_conflict"] += 1

        if not host_verified and relation.relation in {"duplicate", "intermediate", "meta"}:
            stats["rejected_unsafe_relation"] += 1

        applied.append(
            {
                **relation.model_dump(mode="json"),
                "applied_as": "render_suppression" if host_verified else "audit_only",
                "host_verified": host_verified,
                "suppressed_channels": channels,
            }
        )

    stats["suppressed_text_blocks"] = len(suppress_text)
    stats["suppressed_latex_blocks"] = len(suppress_latex)
    return (
        CanonicalRenderPolicy(
            suppress_text_ids=sorted(suppress_text),
            suppress_latex_ids=sorted(suppress_latex),
        ),
        stats,
        applied,
    )


def run_state_canonicalization(
    orchestrator,
    *,
    kb: LectureKnowledgeBase,
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    semantic_audit: bool,
    force: bool,
) -> tuple[CanonicalRenderPolicy, dict[str, int], list[str]]:
    """Build a render-only compaction policy without changing repaired mathematical state."""

    observations = sorted(kb.observations, key=lambda item: (item.start, item.end, item.id))
    deterministic = _deterministic_relations(kb)

    semantic_relations: list[CanonicalObservationRelation] = []
    unresolved: list[str] = []
    audit_cache_hits = 0
    if semantic_audit:
        semantic_relations, unresolved, audit_cache_hits = _propose_semantic_audit(
            orchestrator,
            observations=observations,
            work=work,
            llm_config=llm_config,
            pipeline_version=pipeline_version,
            force=force,
        )

    relations = _dedupe_relations([*deterministic, *semantic_relations])
    policy, stats, applied = _build_render_policy(kb, relations)
    stats["observations_total"] = len(kb.observations)
    stats["relations_total"] = len(relations)
    stats["semantic_audit_enabled"] = int(semantic_audit)
    stats["semantic_audit_cache_hits"] = audit_cache_hits

    atomic_json_dump(
        work / "state_canonicalization.json",
        {
            "version": STATE_CANONICALIZATION_VERSION,
            "policy": policy.model_dump(mode="json"),
            "stats": stats,
            "relations": applied,
            "unresolved": unresolved,
        },
    )
    return policy, stats, unresolved
