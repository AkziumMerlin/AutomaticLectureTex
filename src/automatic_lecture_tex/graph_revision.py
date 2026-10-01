from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import LectureState
from .util import atomic_json_dump


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    start: float = 0.0
    end: float = 0.0
    kind: str = ""
    text: str = ""
    latex: str | None = None


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    title: str = ""
    text: str = ""
    latex: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    status: Literal["active", "alternative", "suppressed"] = "active"
    alternative_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation: str
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal["evidence", "math", "structure"]
    severity: int = Field(ge=1, le=3)
    message: str
    related_nodes: list[str] = Field(default_factory=list)


class GraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, EvidenceRecord]
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)
    evidence_disposition: dict[str, str] = Field(default_factory=dict)
    violations: dict[str, Violation] = Field(default_factory=dict)
    applied_patches: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def clone(self) -> GraphState:
        return self.model_copy(deep=True)

    def active_nodes(self) -> list[GraphNode]:
        return [node for node in self.nodes.values() if node.status == "active"]

    def explained_evidence(self) -> set[str]:
        explained = set(self.evidence_disposition)
        for node in self.nodes.values():
            if node.status != "suppressed":
                explained.update(node.evidence_ids)
        for edge in self.edges:
            explained.update(edge.evidence_ids)
        return explained


class AddNodeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_node"]
    node: GraphNode


class MergeNodesOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["merge_nodes"]
    node_ids: list[str] = Field(min_length=1)
    into_id: str
    kind: str | None = None
    title: str | None = None
    text: str | None = None
    latex: str | None = None


class SplitNodeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["split_node"]
    node_id: str
    replacements: list[GraphNode] = Field(min_length=2)


class RetypeNodeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["retype_node"]
    node_id: str
    kind: str


class ReplaceNodeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["replace_node"]
    node_id: str
    title: str | None = None
    text: str | None = None
    latex: str | None = None
    metadata_update: dict[str, Any] = Field(default_factory=dict)


class AddDerivedNodeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_derived"]
    node: GraphNode


class AddAliasOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_alias"]
    node_id: str
    alias: str


class AddRelationOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_relation"]
    edge: GraphEdge


class AttachEvidenceOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["attach_evidence"]
    node_id: str
    evidence_ids: list[str] = Field(min_length=1)


class MarkEvidenceOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["mark_evidence"]
    evidence_ids: list[str] = Field(min_length=1)
    disposition: str


class MarkAlternativeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["mark_alternative"]
    node_id: str
    group: str


class SuppressNodeOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["suppress_node"]
    node_id: str
    reason: str


class AddViolationOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add_violation"]
    violation: Violation


class ResolveViolationOp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["resolve_violation"]
    violation_id: str


PatchOp = Annotated[
    AddNodeOp
    | MergeNodesOp
    | SplitNodeOp
    | RetypeNodeOp
    | ReplaceNodeOp
    | AddDerivedNodeOp
    | AddAliasOp
    | AddRelationOp
    | AttachEvidenceOp
    | MarkEvidenceOp
    | MarkAlternativeOp
    | SuppressNodeOp
    | AddViolationOp
    | ResolveViolationOp,
    Field(discriminator="op"),
]


class GraphPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    operations: list[PatchOp] = Field(default_factory=list)
    decision_group: str | None = None
    incompatible_with: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class GraphRevisionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_patches: list[GraphPatch] = Field(default_factory=list)
    alternative_groups: list[list[GraphPatch]] = Field(default_factory=list)


class StateMetrics(BaseModel):
    evidence_violations: int
    math_violations: int
    structure_violations: int
    unexplained_evidence: int
    unsupported_nodes: int
    duplicate_nodes: int
    active_nodes: int
    edges: int

    @property
    def lexicographic(self) -> tuple[int, ...]:
        return (
            self.evidence_violations,
            self.math_violations,
            self.structure_violations,
            self.unexplained_evidence,
            self.unsupported_nodes,
            self.duplicate_nodes,
            self.active_nodes,
        )


@dataclass
class SearchResult:
    state: GraphState
    metrics: StateMetrics


def graph_state_from_lecture_state(state: LectureState) -> GraphState:
    evidence = {
        observation.id: EvidenceRecord(
            id=observation.id,
            start=observation.start,
            end=observation.end,
            kind=str(observation.kind),
            text=observation.text,
            latex=observation.latex,
            confidence=observation.confidence,
            source_status=str(observation.source_status),
            evidence_refs=list(observation.evidence_refs),
            window_id=observation.window_id,
            window_ids=list(observation.window_ids),
            episode_id=observation.episode_id,
        )
        for observation in state.observations
    }
    return GraphState(evidence=evidence)


def seed_observation_graph(state: LectureState) -> GraphState:
    """Create a deliberately weak graph used only as a search starting point.

    One provisional node per observation is an initialization convenience, never an invariant:
    later patches may merge, split, suppress, retype or replace these nodes freely.
    """

    graph = graph_state_from_lecture_state(state)
    for observation in state.observations:
        graph.nodes[f"obs::{observation.id}"] = GraphNode(
            id=f"obs::{observation.id}",
            kind=f"provisional_{observation.kind}",
            text=observation.text,
            latex=observation.latex,
            evidence_ids=[observation.id],
            metadata={
                "provisional": True,
                "start": observation.start,
                "end": observation.end,
            },
        )
    return graph


def graph_consensus(states: list[GraphState]) -> GraphState:
    """Keep the semantic intersection of equally viable frontier states.

    Divergent nodes are not arbitrarily selected for final notes. Their evidence is marked as
    frontier-ambiguous and a compact note records the competing readings.
    """

    if not states:
        raise ValueError("graph_consensus requires at least one state")
    if len(states) == 1:
        return states[0].clone()

    result = GraphState(evidence=states[0].evidence)
    node_ids = set.intersection(*(set(state.nodes) for state in states))

    def semantic_node(node: GraphNode) -> tuple:
        return (
            node.kind,
            node.title,
            node.text,
            node.latex,
            tuple(sorted(node.derived_from)),
            tuple(sorted(node.aliases)),
            node.status,
            node.alternative_group,
        )

    ambiguous_evidence: set[str] = set()
    for node_id in sorted(node_ids):
        nodes = [state.nodes[node_id] for state in states]
        if len({semantic_node(node) for node in nodes}) != 1:
            variants = [
                {
                    "kind": node.kind,
                    "title": node.title,
                    "text": node.text,
                    "latex": node.latex,
                }
                for node in nodes
            ]
            result.notes.append(
                f"Frontier ambiguity for {node_id}: {variants}"
            )
            for node in nodes:
                ambiguous_evidence.update(node.evidence_ids)
            continue

        node = nodes[0].model_copy(deep=True)
        node.evidence_ids = _dedupe(
            [evidence_id for item in nodes for evidence_id in item.evidence_ids]
        )

        metadata_keys = set.intersection(
            *(set(item.metadata) for item in nodes)
        ) if nodes else set()
        common_metadata = {
            key: nodes[0].metadata[key]
            for key in metadata_keys
            if all(item.metadata[key] == nodes[0].metadata[key] for item in nodes[1:])
        }
        metadata_variants = [
            item.metadata
            for item in nodes
            if item.metadata != common_metadata
        ]
        node.metadata = dict(common_metadata)
        if metadata_variants:
            node.metadata["frontier_metadata_alternatives"] = metadata_variants
            result.notes.append(
                f"Frontier metadata ambiguity for {node_id}: {metadata_variants}"
            )
        result.nodes[node_id] = node

    common_edges = None
    edge_payload: dict[tuple[str, str, str], GraphEdge] = {}
    for state in states:
        keys = {
            (edge.source, edge.target, edge.relation)
            for edge in state.edges
            if edge.source in result.nodes and edge.target in result.nodes
        }
        common_edges = keys if common_edges is None else common_edges.intersection(keys)
        for edge in state.edges:
            edge_payload[(edge.source, edge.target, edge.relation)] = edge
    for key in sorted(common_edges or set()):
        result.edges.append(edge_payload[key].model_copy(deep=True))

    for evidence_id in result.evidence:
        dispositions = {
            state.evidence_disposition.get(evidence_id)
            for state in states
        }
        if len(dispositions) == 1:
            value = next(iter(dispositions))
            if value is not None:
                result.evidence_disposition[evidence_id] = value
    for evidence_id in ambiguous_evidence:
        result.evidence_disposition[evidence_id] = "frontier_ambiguous"

    # A node can be semantically identical across branches while depending on a node that is
    # branch-specific and therefore absent from the consensus. Remove such dangling conclusions
    # instead of selecting one hidden premise.
    removed = True
    while removed:
        removed = False
        for node_id, node in list(result.nodes.items()):
            missing = [
                dependency
                for dependency in node.derived_from
                if dependency not in result.nodes
            ]
            if not missing:
                continue
            ambiguous_evidence.update(node.evidence_ids)
            result.notes.append(
                f"Consensus omitted {node_id} because dependencies are frontier-specific: "
                + ", ".join(missing)
            )
            del result.nodes[node_id]
            removed = True

    result.edges = [
        edge
        for edge in result.edges
        if edge.source in result.nodes and edge.target in result.nodes
    ]

    common_violations = set.intersection(
        *(set(state.violations) for state in states)
    )
    for violation_id in common_violations:
        values = [state.violations[violation_id] for state in states]
        if len({item.model_dump_json() for item in values}) == 1:
            result.violations[violation_id] = values[0].model_copy(deep=True)

    result.applied_patches = _dedupe(
        [
            patch_id
            for state in states
            for patch_id in state.applied_patches
            if all(patch_id in other.applied_patches for other in states)
        ]
    )
    _validate_state(result)
    return result


def _dedupe(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def _node_signature(node: GraphNode) -> tuple[str, str, str]:
    return (
        node.kind,
        " ".join(node.text.split()).casefold(),
        "".join((node.latex or "").split()),
    )


def metrics(state: GraphState) -> StateMetrics:
    counts = {"evidence": 0, "math": 0, "structure": 0}
    for violation in state.violations.values():
        counts[violation.category] += violation.severity

    explained = state.explained_evidence()
    unexplained = sum(evidence_id not in explained for evidence_id in state.evidence)
    unsupported = 0
    signatures: dict[tuple[str, str, str], int] = {}
    active = state.active_nodes()
    for node in active:
        if node.kind.startswith("provisional_"):
            unsupported += 1
        elif not node.evidence_ids and not node.derived_from:
            unsupported += 1
        signature = _node_signature(node)
        if signature[1] or signature[2]:
            signatures[signature] = signatures.get(signature, 0) + 1
    duplicates = sum(max(0, count - 1) for count in signatures.values())

    return StateMetrics(
        evidence_violations=counts["evidence"],
        math_violations=counts["math"],
        structure_violations=counts["structure"],
        unexplained_evidence=unexplained,
        unsupported_nodes=unsupported,
        duplicate_nodes=duplicates,
        active_nodes=len(active),
        edges=len(state.edges),
    )


def apply_patch(state: GraphState, patch: GraphPatch) -> GraphState:
    out = state.clone()

    for operation in patch.operations:
        if isinstance(operation, AddNodeOp | AddDerivedNodeOp):
            node = operation.node.model_copy(deep=True)
            if node.id in out.nodes:
                raise ValueError(f"node already exists: {node.id}")
            missing = [
                evidence_id
                for evidence_id in node.evidence_ids
                if evidence_id not in out.evidence
            ]
            if missing:
                raise ValueError(f"node {node.id} references missing evidence: {missing}")
            for dependency in node.derived_from:
                if dependency not in out.nodes:
                    raise ValueError(
                        f"node {node.id} derived_from missing node: {dependency}"
                    )
            out.nodes[node.id] = node

        elif isinstance(operation, MergeNodesOp):
            source_ids = _dedupe(operation.node_ids)
            missing = [
                node_id for node_id in source_ids if node_id not in out.nodes
            ]
            if missing:
                raise ValueError(f"merge missing nodes: {missing}")

            target_exists = operation.into_id in out.nodes
            if target_exists and operation.into_id not in source_ids:
                # Absorb one or more source nodes into an already-existing canonical target.
                member_ids = [operation.into_id, *source_ids]
            else:
                member_ids = source_ids

            member_ids = _dedupe(member_ids)
            if len(member_ids) < 2:
                raise ValueError(
                    "merge requires at least two distinct nodes unless into_id is an "
                    "existing distinct target"
                )

            members = [out.nodes[node_id] for node_id in member_ids]
            base = (
                out.nodes[operation.into_id]
                if target_exists
                else members[0]
            )

            evidence_ids = _dedupe(
                [item for node in members for item in node.evidence_ids]
            )
            absorbed_ids = {
                node_id for node_id in member_ids if node_id != operation.into_id
            }
            derived_from = _dedupe(
                [
                    dependency
                    for node in members
                    for dependency in node.derived_from
                    if dependency not in absorbed_ids
                    and dependency != operation.into_id
                ]
            )
            aliases = _dedupe([item for node in members for item in node.aliases])

            metadata = dict(base.metadata)
            previous_merged = metadata.get("merged_from", [])
            if not isinstance(previous_merged, list):
                previous_merged = [str(previous_merged)]
            metadata["merged_from"] = _dedupe(
                [
                    *[str(item) for item in previous_merged],
                    *[
                        node_id
                        for node_id in member_ids
                        if node_id != operation.into_id
                    ],
                ]
            )

            merged = GraphNode(
                id=operation.into_id,
                kind=operation.kind or base.kind,
                title=operation.title if operation.title is not None else base.title,
                text=operation.text if operation.text is not None else base.text,
                latex=operation.latex if operation.latex is not None else base.latex,
                evidence_ids=evidence_ids,
                derived_from=derived_from,
                aliases=aliases,
                status=base.status,
                alternative_group=base.alternative_group,
                metadata=metadata,
            )

            for node_id in member_ids:
                if node_id != operation.into_id:
                    del out.nodes[node_id]
            out.nodes[operation.into_id] = merged

            for edge in out.edges:
                if edge.source in absorbed_ids:
                    edge.source = operation.into_id
                if edge.target in absorbed_ids:
                    edge.target = operation.into_id
            out.edges = [edge for edge in out.edges if edge.source != edge.target]

        elif isinstance(operation, SplitNodeOp):
            if operation.node_id not in out.nodes:
                raise ValueError(f"split missing node: {operation.node_id}")
            old = out.nodes.pop(operation.node_id)
            covered: set[str] = set()
            for node in operation.replacements:
                if node.id in out.nodes:
                    raise ValueError(f"split replacement exists: {node.id}")
                covered.update(node.evidence_ids)
                out.nodes[node.id] = node.model_copy(deep=True)
            if set(old.evidence_ids) - covered:
                raise ValueError("split would drop evidence")
            out.edges = [
                edge
                for edge in out.edges
                if edge.source != operation.node_id and edge.target != operation.node_id
            ]

        elif isinstance(operation, RetypeNodeOp):
            out.nodes[operation.node_id].kind = operation.kind

        elif isinstance(operation, ReplaceNodeOp):
            node = out.nodes[operation.node_id]
            if operation.title is not None:
                node.title = operation.title
            if operation.text is not None:
                node.text = operation.text
            if operation.latex is not None:
                node.latex = operation.latex
            node.metadata.update(operation.metadata_update)

        elif isinstance(operation, AddAliasOp):
            node = out.nodes[operation.node_id]
            node.aliases = _dedupe([*node.aliases, operation.alias])

        elif isinstance(operation, AddRelationOp):
            edge = operation.edge
            if edge.source not in out.nodes or edge.target not in out.nodes:
                raise ValueError(
                    f"edge endpoint missing: {edge.source}->{edge.target}"
                )
            out.edges.append(edge.model_copy(deep=True))

        elif isinstance(operation, AttachEvidenceOp):
            node = out.nodes[operation.node_id]
            missing = [
                evidence_id
                for evidence_id in operation.evidence_ids
                if evidence_id not in out.evidence
            ]
            if missing:
                raise ValueError(f"missing evidence: {missing}")
            node.evidence_ids = _dedupe(
                [*node.evidence_ids, *operation.evidence_ids]
            )

        elif isinstance(operation, MarkEvidenceOp):
            for evidence_id in operation.evidence_ids:
                if evidence_id not in out.evidence:
                    raise ValueError(f"missing evidence: {evidence_id}")
                out.evidence_disposition[evidence_id] = operation.disposition

        elif isinstance(operation, MarkAlternativeOp):
            node = out.nodes[operation.node_id]
            node.status = "alternative"
            node.alternative_group = operation.group

        elif isinstance(operation, SuppressNodeOp):
            node = out.nodes[operation.node_id]
            node.status = "suppressed"
            node.metadata["suppressed_reason"] = operation.reason

        elif isinstance(operation, AddViolationOp):
            out.violations[operation.violation.id] = operation.violation

        elif isinstance(operation, ResolveViolationOp):
            out.violations.pop(operation.violation_id, None)

        else:
            raise TypeError(operation)

    _suppress_absorbed_provisionals(out)
    out.applied_patches.append(patch.id)
    _validate_state(out)
    return out


def _suppress_absorbed_provisionals(state: GraphState) -> None:
    """Drop weak one-observation placeholders once canonical mathematics owns their evidence.

    Provisional nodes are only an initialization device. Keeping them active after a canonical
    non-topic node cites the same evidence double-counts the observation and makes the search spend
    model calls on mechanical cleanup rather than mathematical reconstruction.
    """

    topic_kinds = {"topic", "section", "subsection"}
    canonical_evidence: set[str] = set()
    for node in state.nodes.values():
        if node.status != "active":
            continue
        if node.kind.startswith("provisional_"):
            continue
        if node.kind.strip().lower() in topic_kinds:
            continue
        canonical_evidence.update(node.evidence_ids)

    for node in state.nodes.values():
        if node.status != "active" or not node.kind.startswith("provisional_"):
            continue
        if not node.evidence_ids:
            continue
        covered = all(
            evidence_id in canonical_evidence
            or (
                evidence_id in state.evidence_disposition
                and state.evidence_disposition[evidence_id] != "frontier_ambiguous"
            )
            for evidence_id in node.evidence_ids
        )
        if covered:
            node.status = "suppressed"
            node.metadata["suppressed_reason"] = "absorbed_by_canonical_graph"


def _validate_state(state: GraphState) -> None:
    for node in state.nodes.values():
        for evidence_id in node.evidence_ids:
            if evidence_id not in state.evidence:
                raise ValueError(
                    f"node {node.id} references unknown evidence {evidence_id}"
                )
        for dependency in node.derived_from:
            if dependency not in state.nodes:
                raise ValueError(
                    f"node {node.id} references unknown dependency {dependency}"
                )

    seen_edges: set[tuple[str, str, str]] = set()
    deduped: list[GraphEdge] = []
    for edge in state.edges:
        if edge.source not in state.nodes or edge.target not in state.nodes:
            raise ValueError(f"dangling edge {edge.source}->{edge.target}")
        key = (edge.source, edge.target, edge.relation)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped.append(edge)
    state.edges = deduped


def search_alternatives(
    base: GraphState,
    fixed_patches: list[GraphPatch],
    alternative_groups: list[list[GraphPatch]],
    *,
    pareto: bool = True,
) -> list[SearchResult]:
    state = base
    for patch in fixed_patches:
        state = apply_patch(state, patch)

    results: list[SearchResult] = []
    combinations = product(*alternative_groups) if alternative_groups else [tuple()]
    for choices in combinations:
        selected_ids = {patch.id for patch in choices}
        if any(
            set(patch.incompatible_with) & selected_ids
            for patch in choices
        ):
            continue

        candidate = state
        try:
            for patch in choices:
                candidate = apply_patch(candidate, patch)
        except (KeyError, ValueError):
            continue
        results.append(SearchResult(candidate, metrics(candidate)))

    results.sort(key=lambda result: result.metrics.lexicographic)
    if not pareto:
        return results
    if not results:
        return []

    # Evidence/math/structure contradictions are hard priorities. A mathematically inconsistent
    # graph must not survive merely because it contains one fewer node.
    best_hard = min(
        (
            result.metrics.evidence_violations,
            result.metrics.math_violations,
            result.metrics.structure_violations,
        )
        for result in results
    )
    eligible = [
        result
        for result in results
        if (
            result.metrics.evidence_violations,
            result.metrics.math_violations,
            result.metrics.structure_violations,
        )
        == best_hard
    ]

    frontier: list[SearchResult] = []
    for result in eligible:
        value = (
            result.metrics.unexplained_evidence,
            result.metrics.unsupported_nodes,
            result.metrics.duplicate_nodes,
            result.metrics.active_nodes,
        )
        dominated = False
        for other in eligible:
            if other is result:
                continue
            other_value = (
                other.metrics.unexplained_evidence,
                other.metrics.unsupported_nodes,
                other.metrics.duplicate_nodes,
                other.metrics.active_nodes,
            )
            if all(a <= b for a, b in zip(other_value, value, strict=True)) and any(
                a < b for a, b in zip(other_value, value, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(result)

    frontier.sort(key=lambda result: result.metrics.lexicographic)
    return frontier


def run_revision_plan(
    state: LectureState,
    plan: GraphRevisionPlan,
    *,
    output_dir: Path,
) -> list[SearchResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = graph_state_from_lecture_state(state)
    frontier = search_alternatives(
        base,
        plan.fixed_patches,
        plan.alternative_groups,
        pareto=True,
    )
    all_results = search_alternatives(
        base,
        plan.fixed_patches,
        plan.alternative_groups,
        pareto=False,
    )

    for index, result in enumerate(frontier):
        atomic_json_dump(
            output_dir / f"frontier_{index:02d}.json",
            result.state.model_dump(mode="json"),
        )

    atomic_json_dump(
        output_dir / "summary.json",
        {
            "frontier": [
                {
                    "index": index,
                    "patches": result.state.applied_patches,
                    "metrics": result.metrics.model_dump(mode="json"),
                    "alternative_nodes": [
                        node.id
                        for node in result.state.nodes.values()
                        if node.status == "alternative"
                    ],
                }
                for index, result in enumerate(frontier)
            ],
            "all_states": [
                {
                    "patches": result.state.applied_patches,
                    "metrics": result.metrics.model_dump(mode="json"),
                }
                for result in all_results
            ],
        },
    )
    return frontier
