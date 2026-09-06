from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from .schemas import (
    ClaimStatus,
    EpisodeBoundary,
    EpisodeHierarchyPlan,
    EpisodeKind,
    EpisodeStatus,
    EpisodeTrackingUpdate,
    HierarchyLevel,
    KnowledgeClaim,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    OutlineSection,
    OutlineSubsection,
    SemanticAnchor,
    SemanticEpisode,
    SourceStatus,
    SymbolRecord,
    WindowObservations,
)

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w\\]+", re.UNICODE)


def _merge_unique(left: list[str], right: Iterable[str]) -> list[str]:
    seen = set(left)
    result = list(left)
    for item in right:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return _WS.sub("", value).lower()


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {token.lower() for token in _TOKEN.findall(value)}


def _sequence_similarity(left: str | None, right: str | None) -> float:
    a = _norm(left)
    b = _norm(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _jaccard(left: str | None, right: str | None) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _temporal_overlap(left: LectureObservation, right: LectureObservation) -> float:
    overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    if overlap <= 0:
        return 0.0
    shortest = max(1e-6, min(left.end - left.start, right.end - right.start))
    return min(1.0, overlap / shortest)


def observation_similarity(left: LectureObservation, right: LectureObservation) -> float:
    """Conservative overlap-reconciliation score.

    Technical windows deliberately overlap. Two locally generated observations are considered two
    pieces of evidence for one canonical event only when their time intervals overlap and their
    mathematical/prose content is also strongly similar. This avoids treating the window boundary as
    a semantic boundary while remaining conservative around neighboring proof steps.
    """

    if left.kind != right.kind:
        return 0.0
    temporal = _temporal_overlap(left, right)
    if temporal <= 0.0:
        return 0.0
    latex = _sequence_similarity(left.latex, right.latex)
    prose = max(_sequence_similarity(left.text, right.text), _jaccard(left.text, right.text))
    semantic = max(latex, prose)
    return 0.45 * temporal + 0.55 * semantic


def canonical_observation_id(kb: LectureKnowledgeBase, observation_id: str | None) -> str | None:
    if not observation_id:
        return None
    current = observation_id
    seen: set[str] = set()
    while current in kb.observation_aliases and current not in seen:
        seen.add(current)
        current = kb.observation_aliases[current]
    return current


def reconcile_window_observations(
    kb: LectureKnowledgeBase,
    batch: WindowObservations,
) -> list[str]:
    """Merge an overlapping technical window into canonical observations.

    The result is a set of canonical evidence events. Repeated overlap evidence is represented by
    aliases and accumulated evidence_refs instead of parallel semantic objects.
    """

    added_ids: list[str] = []
    for index, raw in enumerate(batch.observations):
        obs = raw.model_copy(deep=True)
        obs.window_id = batch.window_id
        if not obs.id:
            obs.id = f"obs_{batch.window_id}_{index:03d}"
        if not obs.evidence_refs:
            obs.evidence_refs = [batch.window_id]
        obs.target_observation_id = canonical_observation_id(kb, obs.target_observation_id)

        candidates = [
            candidate
            for candidate in kb.observations
            if candidate.kind == obs.kind
            and min(candidate.end, obs.end) >= max(candidate.start, obs.start)
        ]
        duplicate = None
        best_score = 0.0
        for candidate in candidates:
            score = observation_similarity(candidate, obs)
            exact_formula = bool(candidate.latex and obs.latex and _norm(candidate.latex) == _norm(obs.latex))
            exact_text = bool(candidate.text and obs.text and _norm(candidate.text) == _norm(obs.text))
            if exact_formula or exact_text:
                score = max(score, 0.95)
            if score > best_score:
                best_score = score
                duplicate = candidate

        if duplicate is not None and best_score >= 0.78:
            kb.observation_aliases[obs.id] = duplicate.id
            duplicate.start = min(duplicate.start, obs.start)
            duplicate.end = max(duplicate.end, obs.end)
            duplicate.confidence = max(duplicate.confidence, obs.confidence)
            duplicate.evidence_refs = _merge_unique(duplicate.evidence_refs, obs.evidence_refs)
            duplicate.window_ids = _merge_unique(duplicate.window_ids, [batch.window_id])
            if obs.source_status == SourceStatus.OBSERVED:
                duplicate.source_status = SourceStatus.OBSERVED
            if obs.confidence >= duplicate.confidence:
                if len(obs.text.strip()) > len(duplicate.text.strip()):
                    duplicate.text = obs.text
                if obs.latex and (not duplicate.latex or len(obs.latex) >= len(duplicate.latex)):
                    duplicate.latex = obs.latex
            continue

        obs.window_ids = _merge_unique(obs.window_ids, [batch.window_id])
        kb.observations.append(obs)
        added_ids.append(obs.id)

    # Resolve targets that pointed at a raw observation which was reconciled earlier in this batch.
    for obs in kb.observations:
        obs.target_observation_id = canonical_observation_id(kb, obs.target_observation_id)
    kb.unresolved = _merge_unique(kb.unresolved, batch.unresolved)
    return added_ids


def _episode_kind_for_observation(obs: LectureObservation) -> EpisodeKind:
    mapping = {
        ObservationKind.DEFINITION: EpisodeKind.DEFINITION,
        ObservationKind.EXAMPLE: EpisodeKind.EXAMPLE,
        ObservationKind.NOTATION: EpisodeKind.NOTATION,
        ObservationKind.PROOF_STEP: EpisodeKind.PROOF,
        ObservationKind.EQUATION: EpisodeKind.DERIVATION,
        ObservationKind.REMARK: EpisodeKind.REMARK,
    }
    return mapping.get(obs.kind, EpisodeKind.TOPIC)


def _episode_title_for_observation(obs: LectureObservation) -> str:
    text = obs.text.strip()
    if text:
        return text[:120]
    if obs.latex:
        return obs.latex[:120]
    return "Фрагмент лекции"


def _next_episode_id(kb: LectureKnowledgeBase) -> str:
    existing = {item.id for item in kb.episodes}
    index = len(existing)
    while f"episode_{index:04d}" in existing:
        index += 1
    return f"episode_{index:04d}"


def _active_open_episode(kb: LectureKnowledgeBase) -> SemanticEpisode | None:
    open_episodes = [item for item in kb.episodes if item.status == EpisodeStatus.OPEN]
    if not open_episodes:
        return None
    return max(open_episodes, key=lambda item: (item.end, item.start))


def _close_episode(episode: SemanticEpisode, timestamp: float | None = None) -> None:
    if timestamp is not None:
        episode.end = max(episode.start, min(max(episode.end, timestamp), timestamp))
    episode.status = EpisodeStatus.CLOSED


def _make_episode(
    kb: LectureKnowledgeBase,
    *,
    title: str,
    kind: EpisodeKind,
    start: float,
) -> SemanticEpisode:
    episode = SemanticEpisode(
        id=_next_episode_id(kb),
        title=title.strip() or "Фрагмент лекции",
        kind=kind,
        start=start,
        end=start,
        status=EpisodeStatus.OPEN,
    )
    kb.episodes.append(episode)
    return episode


def _claim_for_observation(kb: LectureKnowledgeBase, obs: LectureObservation) -> KnowledgeClaim | None:
    for claim in kb.claims:
        if obs.id in claim.evidence_ids:
            return claim
    return None


def _claims_targeting_observation(
    kb: LectureKnowledgeBase,
    observation_id: str,
) -> list[KnowledgeClaim]:
    canonical = canonical_observation_id(kb, observation_id) or observation_id
    return [
        claim
        for claim in kb.claims
        if claim.status == ClaimStatus.ACTIVE and canonical in claim.evidence_ids
    ]


def _sync_claim_for_observation(
    kb: LectureKnowledgeBase,
    episode: SemanticEpisode,
    obs: LectureObservation,
) -> None:
    if obs.kind in {ObservationKind.TRANSITION, ObservationKind.UNRESOLVED}:
        return

    if obs.kind == ObservationKind.RETRACTION:
        target = canonical_observation_id(kb, obs.target_observation_id)
        if target:
            for claim in _claims_targeting_observation(kb, target):
                claim.status = ClaimStatus.RETRACTED
        else:
            kb.unresolved.append(
                f"Retraction {obs.id} has no resolvable target observation."
            )
        return

    if obs.kind == ObservationKind.CORRECTION:
        target = canonical_observation_id(kb, obs.target_observation_id)
        superseded: list[str] = []
        target_kind = ObservationKind.CLAIM
        if target:
            for claim in _claims_targeting_observation(kb, target):
                target_kind = claim.kind
                claim.status = ClaimStatus.SUPERSEDED
                superseded.append(claim.id)
        else:
            kb.unresolved.append(
                f"Correction {obs.id} has no resolvable target observation; kept as active evidence."
            )
        existing = _claim_for_observation(kb, obs)
        if existing is None:
            existing = KnowledgeClaim(
                id=f"claim_{obs.id}",
                kind=target_kind,
                content=obs.text or (obs.latex or ""),
                latex=obs.latex,
                scope=episode.id,
                episode_id=episode.id,
                source_status=obs.source_status,
                evidence_ids=[obs.id],
                supersedes=superseded,
                introduced_at=obs.start,
            )
            kb.claims.append(existing)
        else:
            existing.supersedes = _merge_unique(existing.supersedes, superseded)
        episode.claim_ids = _merge_unique(episode.claim_ids, [existing.id])
        return

    claim = _claim_for_observation(kb, obs)
    if claim is None:
        claim = KnowledgeClaim(
            id=f"claim_{obs.id}",
            kind=obs.kind,
            content=obs.text or (obs.latex or ""),
            latex=obs.latex,
            scope=episode.id,
            episode_id=episode.id,
            source_status=obs.source_status,
            evidence_ids=[obs.id],
            introduced_at=obs.start,
        )
        kb.claims.append(claim)
    else:
        claim.scope = episode.id
        claim.episode_id = episode.id
        claim.evidence_ids = _merge_unique(claim.evidence_ids, [obs.id])
    episode.claim_ids = _merge_unique(episode.claim_ids, [claim.id])


def _apply_symbols(
    kb: LectureKnowledgeBase,
    symbols: list[SymbolRecord],
) -> None:
    existing_ids = {item.id for item in kb.symbols if item.id}
    for index, raw in enumerate(symbols):
        symbol = raw.model_copy(deep=True)
        symbol.evidence_ids = [
            canonical_observation_id(kb, item) or item for item in symbol.evidence_ids
        ]
        episode_ids = [
            obs.episode_id
            for obs in kb.observations
            if obs.id in symbol.evidence_ids and obs.episode_id
        ]
        episode_id = episode_ids[0] if episode_ids else ""
        symbol.episode_id = episode_id
        # Scope is structural: the model may describe a meaning/type, but cannot choose a global
        # namespace independently of the semantic episode that introduced the symbol.
        symbol.scope = episode_id or "lecture"
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
            if symbol.introduced_at:
                existing.introduced_at = min(existing.introduced_at, symbol.introduced_at)
            if episode_id:
                episode = next((item for item in kb.episodes if item.id == episode_id), None)
                if episode is not None:
                    episode.symbol_ids = _merge_unique(episode.symbol_ids, [existing.id])
            continue

        if not symbol.id or symbol.id in existing_ids:
            suffix = index
            candidate = f"sym_{len(existing_ids) + suffix:04d}"
            while candidate in existing_ids:
                suffix += 1
                candidate = f"sym_{len(existing_ids) + suffix:04d}"
            symbol.id = candidate
        existing_ids.add(symbol.id)
        kb.symbols.append(symbol)
        if episode_id:
            episode = next((item for item in kb.episodes if item.id == episode_id), None)
            if episode is not None:
                episode.symbol_ids = _merge_unique(episode.symbol_ids, [symbol.id])


def apply_episode_tracking(
    kb: LectureKnowledgeBase,
    update: EpisodeTrackingUpdate,
    added_observation_ids: list[str],
    *,
    window_id: str,
) -> None:
    """Apply boundary decisions to canonical evidence.

    The LLM is allowed to decide *where* an episode starts/ends and how to name it. It is not allowed
    to decide which evidence survives: every canonical observation is assigned by the host to exactly
    one semantic episode. This is the core structural invariant of the knowledge architecture.
    """

    ids = []
    for raw_id in added_observation_ids:
        canonical = canonical_observation_id(kb, raw_id) or raw_id
        if canonical not in ids:
            ids.append(canonical)
    observations = sorted(
        [item for item in kb.observations if item.id in set(ids)],
        key=lambda item: (item.start, item.end, item.id),
    )
    if not observations:
        _apply_symbols(kb, update.symbols)
        kb.unresolved = _merge_unique(kb.unresolved, update.unresolved)
        return

    boundary_by_observation: dict[str, EpisodeBoundary] = {}
    for boundary in update.boundaries:
        canonical = canonical_observation_id(kb, boundary.before_observation_id)
        if canonical and canonical in {item.id for item in observations}:
            boundary_by_observation[canonical] = boundary
    close_after = {
        canonical_observation_id(kb, item) or item for item in update.close_after_observation_ids
    }

    current = _active_open_episode(kb)
    for obs in observations:
        boundary = boundary_by_observation.get(obs.id)
        if boundary is not None:
            if current is not None:
                current.end = max(current.end, obs.start)
                current.status = EpisodeStatus.CLOSED
            current = _make_episode(
                kb,
                title=boundary.title,
                kind=boundary.kind,
                start=obs.start,
            )
        elif current is None:
            current = _make_episode(
                kb,
                title=_episode_title_for_observation(obs),
                kind=_episode_kind_for_observation(obs),
                start=obs.start,
            )

        obs.episode_id = current.id
        current.start = min(current.start, obs.start)
        current.end = max(current.end, obs.end)
        current.observation_ids = _merge_unique(current.observation_ids, [obs.id])
        current.window_ids = _merge_unique(current.window_ids, obs.window_ids or [window_id])
        _sync_claim_for_observation(kb, current, obs)

        if obs.id in close_after:
            current.end = max(current.end, obs.end)
            current.status = EpisodeStatus.CLOSED
            current = None

    _apply_symbols(kb, update.symbols)
    kb.unresolved = _merge_unique(kb.unresolved, update.unresolved)
    refresh_derived_anchors(kb)


def close_open_episodes(kb: LectureKnowledgeBase) -> None:
    for episode in kb.episodes:
        if episode.status == EpisodeStatus.OPEN:
            episode.status = EpisodeStatus.CLOSED
    refresh_derived_anchors(kb)


def refresh_derived_anchors(kb: LectureKnowledgeBase) -> None:
    """Backward-compatible anchor view derived from episodes, never an independent structure."""

    anchors: list[SemanticAnchor] = []
    for episode in sorted(kb.episodes, key=lambda item: (item.start, item.end)):
        anchors.append(
            SemanticAnchor(
                id=f"anchor_{episode.id}",
                timestamp=episode.start,
                title=episode.title,
                kind=episode.anchor_kind,
                claim_ids=list(episode.claim_ids),
                evidence_ids=list(episode.observation_ids),
            )
        )
    kb.anchors = anchors


def _active_claim_ids_for_episodes(
    kb: LectureKnowledgeBase,
    episode_ids: list[str],
) -> list[str]:
    active = {item.id for item in kb.claims if item.status == ClaimStatus.ACTIVE}
    result: list[str] = []
    for episode in kb.episodes:
        if episode.id in episode_ids:
            result = _merge_unique(result, [item for item in episode.claim_ids if item in active])
    return result


def _evidence_ids_for_episodes(kb: LectureKnowledgeBase, episode_ids: list[str]) -> list[str]:
    result: list[str] = []
    for episode in kb.episodes:
        if episode.id in episode_ids:
            result = _merge_unique(result, episode.observation_ids)
    return result


def build_outline_from_episodes(
    kb: LectureKnowledgeBase,
    plan: EpisodeHierarchyPlan,
    *,
    lecture_title: str,
) -> list[OutlineSection]:
    """Build top-level sections by grouping the immutable ordered episode leaves.

    `plan` can only place boundaries before existing episodes. It cannot invent/remove/reorder a leaf,
    therefore every semantic episode is represented exactly once in the resulting outline by
    construction rather than by a post-hoc gate.
    """

    episodes = sorted(
        [item for item in kb.episodes if item.observation_ids],
        key=lambda item: (item.start, item.end, item.id),
    )
    if not episodes:
        return []

    episode_ids = {item.id for item in episodes}
    topic_boundaries = {
        item.before_episode_id: item
        for item in plan.boundaries
        if item.level == HierarchyLevel.TOPIC and item.before_episode_id in episode_ids
    }
    subtopic_boundaries = {
        item.before_episode_id: item
        for item in plan.boundaries
        if item.level == HierarchyLevel.SUBTOPIC and item.before_episode_id in episode_ids
    }

    sections: list[OutlineSection] = []
    current_episode_ids: list[str] = []
    current_title = topic_boundaries.get(episodes[0].id).title if episodes[0].id in topic_boundaries else episodes[0].title
    current_subsections: list[OutlineSubsection] = []
    current_subtopic_ids: list[str] = []
    current_subtopic_title: str | None = None

    def finish_subtopic() -> None:
        nonlocal current_subtopic_ids, current_subtopic_title
        if not current_subtopic_ids:
            return
        selected = [item for item in episodes if item.id in current_subtopic_ids]
        current_subsections.append(
            OutlineSubsection(
                title=current_subtopic_title or selected[0].title,
                start=selected[0].start,
                end=selected[-1].end,
                episode_ids=list(current_subtopic_ids),
            )
        )
        current_subtopic_ids = []
        current_subtopic_title = None

    def finish_topic() -> None:
        nonlocal current_episode_ids, current_subsections, current_title
        if not current_episode_ids:
            return
        finish_subtopic()
        selected = [item for item in episodes if item.id in current_episode_ids]
        section = OutlineSection(
            id=f"topic_{len(sections):03d}",
            title=current_title or lecture_title,
            start=selected[0].start,
            end=selected[-1].end,
            episode_ids=list(current_episode_ids),
            claim_ids=_active_claim_ids_for_episodes(kb, current_episode_ids),
            evidence_ids=_evidence_ids_for_episodes(kb, current_episode_ids),
            subsections=list(current_subsections),
        )
        sections.append(section)
        current_episode_ids = []
        current_subsections = []

    for index, episode in enumerate(episodes):
        if index > 0 and episode.id in topic_boundaries:
            finish_topic()
            current_title = topic_boundaries[episode.id].title
        if episode.id in subtopic_boundaries:
            finish_subtopic()
            current_subtopic_title = subtopic_boundaries[episode.id].title
        elif current_subtopic_title is None:
            current_subtopic_title = episode.title
        current_episode_ids.append(episode.id)
        current_subtopic_ids.append(episode.id)

    finish_topic()
    return sections
