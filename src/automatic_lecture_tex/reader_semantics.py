from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from .graph_revision import GraphNode, GraphState
from .graph_revision_render import _node_times


PROVENANCE_LANGUAGE = re.compile(
    r"\b(?:ASR|OCR|доск\w*|кадр\w*|окн\w*|лектор\w*|преподавател\w*|видео|"
    r"распознан\w*|реконструкц\w*|уверенност\w*|provenance|рукопис\w*|"
    r"пиксел\w*|панел\w*|гипотез\w*|чтени\w*|чита(?:ется|ем|ются)|"
    r"ведущ(?:ая|ий|ее)|подстроч\w*|неустойчив\w*|фонетич\w*|"
    r"по контексту|вариант чтения|артефакт\w*)\b",
    re.IGNORECASE,
)

TOPIC_PROPAGATION_RELATIONS = {
    "about",
    "refines",
    "uses",
    "supports",
    "supported_by",
    "proved_by",
    "has_proof_step",
    "leads_to",
    "follows_from",
    "derived_from",
    "based_on",
    "specializes",
    "specialization_of",
    "defined_on",
    "concerns",
    "yields",
    "introduces",
    "denotes_convergence_in",
}

_EXPLICIT_TOPIC_FORWARD = {"contains", "contains_node"}
_EXPLICIT_TOPIC_REVERSE = {"part_of", "in_section", "in_topic"}

_BLOCKING_LEGACY_KEYS = {
    "complex_convention",
    "inner_product_convention",
    "convention_ambiguous",
    "adjective_ambiguity",
    "orthogonal_complement_convention",
}
_ADVISORY_LEGACY_KEYS = {
    "topology_ambiguity",
    "relation_sign_unstable",
    "board_ambiguity",
    "symbol_ambiguity",
    "index_ambiguity",
    "reading_ambiguity",
    "notation_ambiguity",
}
_BLOCKING_VALUE = re.compile(
    r"(?:banach|банах|hilbert|гильберт|conjug|сопряж|inner.?product|scalar.?field|"
    r"ортогонал|convention|конвенц)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NodeSemanticPolicy:
    semantic_text: str
    disposition_override: Literal["omit", "unresolved"] | None = None
    blocking_reasons: tuple[str, ...] = ()
    advisory_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReaderOccurrenceSpec:
    occurrence_id: str
    node_id: str
    role: Literal["fact", "proof_step"]
    owner_node_id: str | None
    anchor_time: float


@dataclass(frozen=True)
class ReaderOccurrenceBlockSpec:
    block_id: str
    role: Literal["fact", "proof"]
    anchor_node_id: str
    node_ids: tuple[str, ...]
    occurrence_ids: tuple[str, ...]
    proof_status: Literal["not_applicable", "complete", "incomplete"] = "not_applicable"
    unresolved_support: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReaderOccurrencePlanSpec:
    occurrences: tuple[ReaderOccurrenceSpec, ...]
    blocks: tuple[ReaderOccurrenceBlockSpec, ...]
    incomplete_proofs: tuple[str, ...]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _sentences(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", value.strip())
        if item.strip()
    ]


def semantic_candidate_units(node: GraphNode) -> list[str]:
    """Reader-facing units only; audit/reconstruction prose never enters the surface writer."""

    units: list[str] = []
    title = (node.title or "").strip()
    if title and not PROVENANCE_LANGUAGE.search(title):
        units.append(title)

    body = node.semantic_text if node.semantic_text is not None else node.text
    for sentence in _sentences(body or ""):
        if not PROVENANCE_LANGUAGE.search(sentence):
            units.append(sentence)
    return _dedupe(units)


def semantic_title(node: GraphNode) -> str:
    title = (node.title or "").strip()
    if title and not PROVENANCE_LANGUAGE.search(title):
        return title
    return ""


def ambiguity_policy(node: GraphNode) -> NodeSemanticPolicy:
    blocking: list[str] = []
    advisory: list[str] = []

    for ambiguity in node.ambiguities:
        message = ambiguity.message.strip()
        if not message:
            continue
        if ambiguity.blocking:
            blocking.append(message)
        else:
            advisory.append(message)

    metadata = node.metadata or {}
    for key in _BLOCKING_LEGACY_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        resolved = isinstance(value, str) and value.strip().lower() in {
            "fixed",
            "resolved",
            "standard",
            "real",
            "complex",
        }
        if not resolved:
            blocking.append(f"{key}: {value}")

    ambiguity = metadata.get("ambiguity")
    if ambiguity not in (None, "", False, []):
        blob = f"ambiguity: {ambiguity}"
        (blocking if _BLOCKING_VALUE.search(blob) else advisory).append(blob)

    variants = metadata.get("reading_variants")
    if variants not in (None, "", False, []):
        blob = f"reading_variants: {variants}"
        (blocking if _BLOCKING_VALUE.search(blob) else advisory).append(blob)

    for key in _ADVISORY_LEGACY_KEYS:
        value = metadata.get(key)
        if value not in (None, "", False, []):
            advisory.append(f"{key}: {value}")

    for key in ("board_formula_note", "convention_note"):
        value = metadata.get(key)
        if value and re.search(r"conjug|сопряж|конвенц|convention", str(value), re.I):
            blocking.append(f"{key}: {value}")

    if metadata.get("y_definition_incomplete") is True and not metadata.get("y_line_completed"):
        blocking.append("y_definition_incomplete=true")
    if metadata.get("board_line_incomplete") is True:
        advisory.append("board_line_incomplete=true")

    units = semantic_candidate_units(node)
    semantic_text = " ".join(units).strip()

    if node.kind.strip().lower() in {"topic", "section", "subsection", "transition"}:
        return NodeSemanticPolicy(
            semantic_text=semantic_text,
            disposition_override="omit",
            advisory_notes=tuple(_dedupe(advisory)),
        )

    if (
        node.kind.strip().lower() == "remark"
        and not units
        and not (node.latex or "").strip()
    ):
        return NodeSemanticPolicy(
            semantic_text="",
            disposition_override="omit",
            advisory_notes=tuple(_dedupe(advisory)),
        )

    if metadata.get("board_line_incomplete") is True and node.kind.strip().lower() == "proof_step":
        return NodeSemanticPolicy(
            semantic_text=semantic_text,
            disposition_override="omit",
            advisory_notes=tuple(_dedupe(advisory)),
        )

    if blocking:
        return NodeSemanticPolicy(
            semantic_text=semantic_text,
            disposition_override="unresolved",
            blocking_reasons=tuple(_dedupe(blocking)),
            advisory_notes=tuple(_dedupe(advisory)),
        )

    return NodeSemanticPolicy(
        semantic_text=semantic_text,
        advisory_notes=tuple(_dedupe(advisory)),
    )


def dependency_graph(state: GraphState, node_ids: set[str] | None = None) -> dict[str, set[str]]:
    allowed = set(state.nodes) if node_ids is None else set(node_ids)
    deps: dict[str, set[str]] = defaultdict(set)

    for node_id in allowed:
        node = state.nodes.get(node_id)
        if node is None:
            continue
        deps[node_id].update(
            dependency for dependency in node.derived_from if dependency in allowed
        )

    for edge in state.edges:
        source, target = edge.source, edge.target
        if source not in allowed or target not in allowed:
            continue
        relation = edge.relation.strip().lower()
        if relation == "supported_by":
            deps[source].add(target)
        elif relation == "uses":
            if state.nodes[source].kind.strip().lower() in {"proof", "proof_step", "equation", "claim"}:
                deps[source].add(target)
        elif relation in {
            "follows_from",
            "derived_from",
            "based_on",
            "about",
            "refines",
            "specialization_of",
        }:
            deps[source].add(target)
        elif relation == "leads_to":
            deps[target].add(source)
    return deps


def propagate_unresolved(
    state: GraphState,
    dispositions: dict[str, str],
    reasons: dict[str, list[str]],
) -> None:
    deps = dependency_graph(state, set(dispositions))
    changed = True
    while changed:
        changed = False
        for node_id, disposition in list(dispositions.items()):
            if disposition != "render":
                continue
            bad = [
                dependency
                for dependency in deps.get(node_id, set())
                if dispositions.get(dependency) == "unresolved"
            ]
            if not bad:
                continue
            dispositions[node_id] = "unresolved"
            reasons.setdefault(node_id, []).append(
                "depends on unresolved: " + ", ".join(sorted(bad))
            )
            changed = True


def infer_topic_memberships(
    state: GraphState,
    *,
    renderable_node_ids: set[str],
    topic_ids: set[str],
    legacy_membership: dict[str, str],
) -> dict[str, set[str]]:
    """Prefer semantic relations; use chronological legacy assignment only as a last fallback."""

    memberships: dict[str, set[str]] = defaultdict(set)
    for edge in state.edges:
        source, target = edge.source, edge.target
        relation = edge.relation.strip().lower()
        if relation in _EXPLICIT_TOPIC_FORWARD and source in topic_ids and target in renderable_node_ids:
            memberships[target].add(source)
        elif relation in _EXPLICIT_TOPIC_REVERSE and target in topic_ids and source in renderable_node_ids:
            memberships[source].add(target)

    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in state.edges:
        source, target = edge.source, edge.target
        relation = edge.relation.strip().lower()
        if (
            source in renderable_node_ids
            and target in renderable_node_ids
            and relation in TOPIC_PROPAGATION_RELATIONS
        ):
            adjacency[source].append((target, relation))
            adjacency[target].append((source, relation))

    weights = {
        "refines": 3.0,
        "about": 3.0,
        "proved_by": 2.5,
        "has_proof_step": 2.5,
        "supports": 2.0,
        "supported_by": 2.0,
        "uses": 1.5,
        "leads_to": 1.5,
    }

    for _ in range(8):
        changed = False
        for node_id in renderable_node_ids:
            if node_id in topic_ids or memberships.get(node_id):
                continue
            scores: dict[str, float] = defaultdict(float)
            for neighbour, relation in adjacency.get(node_id, []):
                for topic_id in memberships.get(neighbour, set()):
                    scores[topic_id] += weights.get(relation, 1.0)
            if not scores:
                continue
            best = max(scores.values())
            winners = [topic for topic, score in scores.items() if score == best]
            if len(winners) == 1 and best >= 1.5:
                memberships[node_id].add(winners[0])
                changed = True
        if not changed:
            break

    for node_id in renderable_node_ids:
        if node_id in topic_ids or memberships.get(node_id):
            continue
        topic_id = legacy_membership.get(node_id)
        if topic_id:
            memberships[node_id].add(topic_id)
    return memberships


def _edge_time(state: GraphState, source: str, target: str, relations: set[str]) -> float:
    values: list[float] = []
    for edge in state.edges:
        if edge.source != source or edge.target != target:
            continue
        if edge.relation.strip().lower() not in relations:
            continue
        for evidence_id in edge.evidence_ids:
            evidence = state.evidence.get(evidence_id)
            if evidence is None:
                continue
            suffix = re.search(r"_(\d+)$", evidence_id)
            offset = (int(suffix.group(1)) if suffix else 0) * 1e-4
            values.append(float(evidence.start) + offset)
    return min(values) if values else math.inf


def _node_time(state: GraphState, node_id: str) -> float:
    node = state.nodes[node_id]
    start, _ = _node_times(state, node)
    return start


def proof_support_nodes(
    state: GraphState,
    *,
    claim_id: str,
    section_node_ids: set[str],
) -> set[str]:
    support: set[str] = set()
    for edge in state.edges:
        relation = edge.relation.strip().lower()
        if relation == "supports" and edge.target == claim_id and edge.source in section_node_ids:
            support.add(edge.source)
        elif (
            relation in {"has_proof_step", "proved_by"}
            and edge.source == claim_id
            and edge.target in section_node_ids
        ):
            support.add(edge.target)

    queue = deque(support)
    while queue:
        current = queue.popleft()
        for edge in state.edges:
            relation = edge.relation.strip().lower()
            candidate: str | None = None
            if edge.source == current and relation == "supported_by":
                candidate = edge.target
            elif edge.target == current and relation == "leads_to":
                candidate = edge.source
            if (
                candidate is not None
                and candidate in section_node_ids
                and candidate not in support
            ):
                support.add(candidate)
                queue.append(candidate)
    return support


def _topological_order(
    state: GraphState,
    node_ids: set[str],
    *,
    occurrence_times: dict[str, float] | None = None,
) -> list[str]:
    deps = dependency_graph(state, node_ids)
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, set[str]] = defaultdict(set)

    for node_id in node_ids:
        for dependency in deps.get(node_id, set()):
            if dependency in node_ids and node_id not in outgoing[dependency]:
                outgoing[dependency].add(node_id)
                indegree[node_id] += 1

    def key(node_id: str) -> tuple[float, str]:
        if occurrence_times is not None:
            value = occurrence_times.get(node_id, math.inf)
            if value != math.inf:
                return value, node_id
        return _node_time(state, node_id), node_id

    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=key,
    )
    result: list[str] = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for target in sorted(outgoing.get(current, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=key)

    if len(result) != len(node_ids):
        remaining = node_ids.difference(result)
        result.extend(sorted(remaining, key=key))
    return result


def _proof_occurrence_times(
    state: GraphState,
    *,
    claim_id: str,
    support: set[str],
) -> dict[str, float]:
    result = {node_id: _node_time(state, node_id) for node_id in support}
    for support_id in support:
        direct = _edge_time(state, support_id, claim_id, {"supports"})
        reverse = _edge_time(state, claim_id, support_id, {"has_proof_step", "proved_by"})
        candidates = [value for value in (direct, reverse) if value != math.inf]
        if candidates:
            result[support_id] = min(candidates)
    return result


def _theorem_like(node: GraphNode) -> bool:
    kind = node.kind.strip().lower()
    title = (node.title or "").lower()
    if kind in {"theorem", "lemma", "proposition", "corollary"}:
        return True
    return kind == "claim" and any(
        token in title
        for token in (
            "теорем",
            "характерист",
            "критер",
            "минималь",
            "равенство норм",
            "изоморф",
        )
    )


def build_occurrence_plan(
    state: GraphState,
    *,
    section_id: str,
    ordered_node_ids: list[str],
    dispositions: dict[str, str],
) -> ReaderOccurrencePlanSpec:
    """Build host-owned discourse occurrences.

    Semantic nodes remain unique in the graph, while the same node may occur in several proof
    contexts. Proof ordering uses relation-local evidence timestamps when available.
    """

    section_set = set(ordered_node_ids)
    proof_groups: dict[str, tuple[list[str], list[str]]] = {}
    support_consumed: set[str] = set()

    for node_id in ordered_node_ids:
        if dispositions.get(node_id) != "render":
            continue
        node = state.nodes[node_id]
        if node.kind.strip().lower() not in {
            "claim",
            "theorem",
            "lemma",
            "proposition",
            "corollary",
        }:
            continue
        support = proof_support_nodes(
            state,
            claim_id=node_id,
            section_node_ids=section_set,
        )
        if not support:
            continue
        times = _proof_occurrence_times(state, claim_id=node_id, support=support)
        ordered_support = _topological_order(
            state,
            support,
            occurrence_times=times,
        )
        unresolved = [
            support_id
            for support_id in ordered_support
            if dispositions.get(support_id) == "unresolved"
        ]
        proof_groups[node_id] = (ordered_support, unresolved)
        support_consumed.update(
            support_id for support_id in support if support_id in section_set
        )

    anchor_ids = [
        node_id
        for node_id in ordered_node_ids
        if dispositions.get(node_id) == "render" and node_id not in support_consumed
    ]
    ordered_anchors = _topological_order(state, set(anchor_ids))
    position = {node_id: index for index, node_id in enumerate(ordered_node_ids)}
    ordered_anchors.sort(
        key=lambda node_id: (
            0 if not state.nodes[node_id].derived_from else 1,
            position.get(node_id, 10**9),
        )
    )

    occurrences: list[ReaderOccurrenceSpec] = []
    blocks: list[ReaderOccurrenceBlockSpec] = []
    incomplete: list[str] = []

    for anchor_id in ordered_anchors:
        fact_occurrence_id = f"{section_id}::fact::{anchor_id}"
        occurrences.append(
            ReaderOccurrenceSpec(
                occurrence_id=fact_occurrence_id,
                node_id=anchor_id,
                role="fact",
                owner_node_id=None,
                anchor_time=_node_time(state, anchor_id),
            )
        )
        blocks.append(
            ReaderOccurrenceBlockSpec(
                block_id=f"{section_id}::block::fact::{anchor_id}",
                role="fact",
                anchor_node_id=anchor_id,
                node_ids=(anchor_id,),
                occurrence_ids=(fact_occurrence_id,),
            )
        )

        group = proof_groups.get(anchor_id)
        if group is None:
            continue
        support_order, unresolved_support = group
        if unresolved_support:
            incomplete.append(
                f"{anchor_id}: incomplete proof; unresolved support: "
                + ", ".join(unresolved_support)
            )
            continue

        renderable_support = [
            support_id
            for support_id in support_order
            if dispositions.get(support_id) == "render"
        ]
        if not renderable_support:
            continue

        times = _proof_occurrence_times(
            state,
            claim_id=anchor_id,
            support=set(renderable_support),
        )
        proof_occurrence_ids: list[str] = []
        for index, support_id in enumerate(renderable_support):
            occurrence_id = (
                f"{section_id}::proof::{anchor_id}::{index:03d}::{support_id}"
            )
            proof_occurrence_ids.append(occurrence_id)
            occurrences.append(
                ReaderOccurrenceSpec(
                    occurrence_id=occurrence_id,
                    node_id=support_id,
                    role="proof_step",
                    owner_node_id=anchor_id,
                    anchor_time=times.get(support_id, _node_time(state, support_id)),
                )
            )

        blocks.append(
            ReaderOccurrenceBlockSpec(
                block_id=f"{section_id}::block::proof::{anchor_id}",
                role="proof",
                anchor_node_id=anchor_id,
                node_ids=tuple(renderable_support),
                occurrence_ids=tuple(proof_occurrence_ids),
                proof_status="complete",
            )
        )

    return ReaderOccurrencePlanSpec(
        occurrences=tuple(occurrences),
        blocks=tuple(blocks),
        incomplete_proofs=tuple(incomplete),
    )
