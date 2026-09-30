from __future__ import annotations

from collections import defaultdict

from .graph_revision import GraphNode, GraphState
from .latex import escape_tex
from .schemas import BlockType, ChunkNotes, LectureIR, NoteBlock


_TOPIC_KINDS = {"topic", "section", "subsection"}
_NONRENDER_KINDS = {
    "symbol",
    "notation_entity",
    "alias",
    "evidence",
}


def _kind(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _node_times(
    state: GraphState,
    node: GraphNode,
    _seen: set[str] | None = None,
) -> tuple[float, float]:
    starts = [
        state.evidence[evidence_id].start
        for evidence_id in node.evidence_ids
        if evidence_id in state.evidence
    ]
    ends = [
        state.evidence[evidence_id].end
        for evidence_id in node.evidence_ids
        if evidence_id in state.evidence
    ]
    if starts:
        return min(starts), max(ends)

    seen = set(_seen or ())
    if node.id in seen:
        return float("inf"), float("inf")
    seen.add(node.id)
    dependency_ranges = [
        _node_times(state, state.nodes[dependency], seen)
        for dependency in node.derived_from
        if dependency in state.nodes
    ]
    finite = [
        item for item in dependency_ranges
        if item[0] != float("inf")
    ]
    if finite:
        return min(item[0] for item in finite), max(item[1] for item in finite)
    return float("inf"), float("inf")


def _block_type(node: GraphNode) -> BlockType:
    kind = _kind(node.kind)
    mapping = {
        "definition": BlockType.DEFINITION,
        "theorem": BlockType.THEOREM,
        "lemma": BlockType.LEMMA,
        "proposition": BlockType.PROPOSITION,
        "corollary": BlockType.COROLLARY,
        "proof": BlockType.PROOF,
        "example": BlockType.EXAMPLE,
        "remark": BlockType.REMARK,
        "exercise": BlockType.EXERCISE,
        "equation": BlockType.EQUATION,
    }
    return mapping.get(kind, BlockType.PARAGRAPH)


def _render_node_body(node: GraphNode) -> str:
    pieces: list[str] = []
    text = node.text.strip()
    latex = (node.latex or "").strip()
    if text and text != latex:
        pieces.append(escape_tex(text))
    if latex:
        pieces.append("\\[\n" + latex + "\n\\]")
    return "\n\n".join(pieces)


def _is_renderable(node: GraphNode) -> bool:
    if node.status != "active":
        return False
    kind = _kind(node.kind)
    if kind.startswith("provisional_"):
        return False
    if kind in _TOPIC_KINDS or kind in _NONRENDER_KINDS:
        return False
    if node.metadata.get("render") is False:
        return False
    return bool(node.text.strip() or (node.latex or "").strip())


def _topic_membership(state: GraphState) -> dict[str, str]:
    topic_ids = {
        node.id
        for node in state.nodes.values()
        if node.status == "active" and _kind(node.kind) in _TOPIC_KINDS
    }
    membership: dict[str, str] = {}
    for edge in state.edges:
        relation = _kind(edge.relation)
        if relation in {"contains", "contains_node", "has_part"}:
            if edge.source in topic_ids:
                membership.setdefault(edge.target, edge.source)
        elif relation in {"part_of", "in_section", "in_topic"}:
            if edge.target in topic_ids:
                membership.setdefault(edge.source, edge.target)

    for node in state.nodes.values():
        section_id = node.metadata.get("section_id")
        if section_id in topic_ids:
            membership.setdefault(node.id, str(section_id))
    return membership


def _section_assignment(
    state: GraphState,
    renderable: list[GraphNode],
) -> tuple[list[GraphNode], dict[str, list[GraphNode]]]:
    topics = [
        node
        for node in state.nodes.values()
        if node.status == "active" and _kind(node.kind) in _TOPIC_KINDS
    ]
    topics.sort(key=lambda node: (_node_times(state, node)[0], node.id))
    membership = _topic_membership(state)
    grouped: dict[str, list[GraphNode]] = defaultdict(list)

    if not topics:
        grouped["__lecture__"] = list(renderable)
        return [], grouped

    topic_times = [
        (_node_times(state, topic)[0], topic.id)
        for topic in topics
    ]
    for node in renderable:
        explicit = membership.get(node.id)
        if explicit is not None:
            grouped[explicit].append(node)
            continue
        start, _ = _node_times(state, node)
        preceding = [
            (topic_start, topic_id)
            for topic_start, topic_id in topic_times
            if topic_start <= start
        ]
        target = preceding[-1][1] if preceding else topics[0].id
        grouped[target].append(node)
    return topics, grouped


def _order_nodes(state: GraphState, nodes: list[GraphNode]) -> list[GraphNode]:
    """Topologically order canonical mathematics; timestamps only break unrelated ties."""

    by_id = {node.id: node for node in nodes}
    precedence: set[tuple[str, str]] = set()

    for node in nodes:
        for dependency in node.derived_from:
            if dependency in by_id and dependency != node.id:
                precedence.add((dependency, node.id))

    nonordering_relations = {
        "contains",
        "contains_node",
        "part_of",
        "in_section",
        "in_topic",
        "has_part",
        "alias",
        "same_object",
        "equivalent",
    }
    for edge in state.edges:
        if edge.source not in by_id or edge.target not in by_id:
            continue
        if _kind(edge.relation) in nonordering_relations:
            continue
        if edge.source != edge.target:
            precedence.add((edge.source, edge.target))

    outgoing: dict[str, set[str]] = defaultdict(set)
    indegree = {node.id: 0 for node in nodes}
    for source, target in precedence:
        if target in outgoing[source]:
            continue
        outgoing[source].add(target)
        indegree[target] += 1

    def key(node_id: str) -> tuple[float, float, str]:
        node = by_id[node_id]
        raw_order = node.metadata.get("order", 0.0)
        try:
            order = float(raw_order)
        except (TypeError, ValueError):
            order = 0.0
        return (_node_times(state, node)[0], order, node_id)

    ready = sorted(
        [node_id for node_id, value in indegree.items() if value == 0],
        key=key,
    )
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for target in sorted(outgoing.get(node_id, set()), key=key):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=key)

    if len(ordered) != len(nodes):
        # A relation cycle should not make the document disappear. Preserve the acyclic prefix and
        # append the cyclic remainder in evidence order; the graph violation remains visible.
        remainder = sorted(
            [node_id for node_id in by_id if node_id not in ordered],
            key=key,
        )
        ordered.extend(remainder)
    return [by_id[node_id] for node_id in ordered]


def graph_state_to_ir(
    state: GraphState,
    *,
    lecture_id: str,
    title: str,
) -> LectureIR:
    renderable = [
        node for node in state.nodes.values()
        if _is_renderable(node)
    ]
    renderable = _order_nodes(state, renderable)
    topics, grouped = _section_assignment(state, renderable)
    chunks: list[ChunkNotes] = []

    unresolved_global = [
        violation.message
        for violation in state.violations.values()
    ]
    unresolved_global.extend(state.notes)
    ambiguous_groups = sorted(
        {
            node.alternative_group
            for node in state.nodes.values()
            if node.status == "alternative" and node.alternative_group
        }
    )
    unresolved_global.extend(
        f"Unresolved graph alternative group: {group}"
        for group in ambiguous_groups
    )

    unrevised_evidence = {
        evidence_id
        for node in state.nodes.values()
        if _kind(node.kind).startswith("provisional_") and node.status == "active"
        for evidence_id in node.evidence_ids
    }
    if unrevised_evidence:
        unresolved_global.append(
            f"{len(unrevised_evidence)} raw observations remained provisional and were not "
            "rendered as canonical mathematics."
        )

    if not topics:
        section_nodes = [(None, grouped.get("__lecture__", []))]
    else:
        section_nodes = [
            (topic, grouped.get(topic.id, []))
            for topic in topics
        ]

    for index, (topic, nodes) in enumerate(section_nodes):
        if not nodes and topic is None:
            continue
        if topic is not None and not nodes and not topic.text.strip() and not (topic.latex or ""):
            continue

        nodes = _order_nodes(state, nodes)
        ranges = [
            _node_times(state, node)
            for node in nodes
            if _node_times(state, node)[0] != float("inf")
        ]
        if topic is not None:
            topic_range = _node_times(state, topic)
            if topic_range[0] != float("inf"):
                ranges.append(topic_range)
        start = min((item[0] for item in ranges), default=0.0)
        end = max((item[1] for item in ranges), default=start)

        blocks: list[NoteBlock] = []
        if topic is not None:
            topic_body = _render_node_body(topic)
            if topic_body:
                blocks.append(
                    NoteBlock(
                        type=BlockType.PARAGRAPH,
                        title=None,
                        latex=topic_body,
                        source_evidence_ids=list(topic.evidence_ids),
                    )
                )

        for node in nodes:
            body = _render_node_body(node)
            if not body:
                continue
            blocks.append(
                NoteBlock(
                    type=_block_type(node),
                    title=node.title or None,
                    latex=body,
                    source_evidence_ids=list(node.evidence_ids),
                )
            )

        section_title = (
            (topic.title or topic.text).strip()
            if topic is not None
            else title
        )
        if not section_title:
            section_title = f"Раздел {index + 1}"
        chunks.append(
            ChunkNotes(
                chunk_id=topic.id if topic is not None else "graph_section_000",
                start=start,
                end=end,
                section_title=section_title.replace("$", ""),
                blocks=blocks,
                unresolved=list(
                    dict.fromkeys(
                        unresolved_global
                        if index == len(section_nodes) - 1
                        else []
                    )
                ),
            )
        )

    if not chunks:
        chunks.append(
            ChunkNotes(
                chunk_id="graph_section_000",
                start=0.0,
                end=0.0,
                section_title=title,
                blocks=[],
                unresolved=list(dict.fromkeys(unresolved_global)),
            )
        )

    return LectureIR(
        lecture_id=lecture_id,
        title=title,
        chunks=chunks,
    )
