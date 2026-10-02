from __future__ import annotations

from collections import defaultdict
import re

from .graph_revision import GraphNode, GraphState
from .latex import escape_tex
from .schemas import BlockType, ChunkNotes, LectureIR, NoteBlock


GRAPH_SURFACE_RENDER_VERSION = 2

_TOPIC_KINDS = {"topic", "section", "subsection"}
_NONRENDER_KINDS = {
    "symbol",
    "notation_entity",
    "alias",
    "evidence",
    "transition",
}

_PROOF_KINDS = {"proof", "proof_step"}
_PROOF_COMPONENT_KINDS = {"proof", "proof_step", "equation", "notation"}
_PROOF_CHAIN_RELATIONS = {
    "next_step",
    "precedes",
    "leads_to",
    "applies",
    "uses",
    "derived_from",
}
_FORWARD_ORDER_RELATIONS = {
    "precedes",
    "next_step",
    "leads_to",
    "generates",
    "defines",
    "formalizes",
    "subbasic_of",
    "used_in",
    "proved_by",
}
_REVERSE_ORDER_RELATIONS = {
    "applies",
    "uses",
    "derived_from",
    "proves",
    "supports",
    "verifies_property_of",
    "proves_reverse_of",
    "example_of",
    "specialization_of",
    "refines",
    "elaborates",
}
_NONORDERING_RELATIONS = {
    "contains",
    "contains_node",
    "part_of",
    "in_section",
    "in_topic",
    "has_part",
    "alias",
    "same_object",
    "equivalent",
    "equivalent_to",
    "concerns",
}

_PROVENANCE_LANGUAGE = re.compile(
    r"\b(?:лектор|доск(?:а|е|и|у|ой)|окн(?:о|е|а)|кадр(?:е|ы|ов)?|"
    r"ASR|OCR|рукопис|видео|визуальн|записан[оаы]?|видн[оы]|пометк|устно|"
    r"упоминает|подч[её]ркивает|объявляет|записыва(?:ет|ется)|читаются|"
    r"фиксиру(?:ет|ется)|произносит|начинает фразу|в нескольких окнах)\b",
    re.IGNORECASE,
)
_META_TITLE = re.compile(
    r"^(?:переход|начало|продолжение|обзор|введение обозначения.*доск)",
    re.IGNORECASE,
)
_CANONICAL_NAME_REPLACEMENTS = (
    ("Гейма–Банаха", "Хана–Банаха"),
    ("Гейне–Банаха", "Хана–Банаха"),
    ("Хан–Банаха", "Хана–Банаха"),
    ("Ризе", "Рисса"),
    ("Кас 1", "Случай 1"),
    ("Кас 2", "Случай 2"),
    ("Пример (Прим.)", "Пример"),
)


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
        "proof_step": BlockType.PROOF,
        "example": BlockType.EXAMPLE,
        "remark": BlockType.REMARK,
        "exercise": BlockType.EXERCISE,
        "equation": BlockType.EQUATION,
    }
    return mapping.get(kind, BlockType.PARAGRAPH)


def _canonical_surface_text(value: str) -> str:
    result = value.strip()
    for source, target in _CANONICAL_NAME_REPLACEMENTS:
        result = result.replace(source, target)
    return result


def _surface_title(node: GraphNode) -> str:
    return _canonical_surface_text(node.title or "")


def _split_tex_top_level(value: str, token: str) -> list[str]:
    """Split on a TeX token only outside {...} groups."""

    parts: list[str] = []
    start = 0
    depth = 0
    delimiter_depth = 0
    index = 0
    while index < len(value):
        if value.startswith(r"\left", index):
            delimiter_depth += 1
        elif value.startswith(r"\right", index):
            delimiter_depth = max(0, delimiter_depth - 1)
        char = value[index]
        escaped = index > 0 and value[index - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
            index += 1
            continue
        if char == "}" and not escaped:
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0 and delimiter_depth == 0 and not escaped and value.startswith(token, index):
            parts.append(value[start:index])
            index += len(token)
            start = index
            continue
        index += 1
    parts.append(value[start:])
    return parts


def _surface_display_latex(value: str) -> str:
    """Break long displays structurally; never shrink them to fit the page."""

    latex = value.strip()
    if len(latex) <= 140 or "\\begin{" in latex:
        return latex

    for separator in (r"\\qquad", r"\\quad"):
        parts = [part.strip() for part in _split_tex_top_level(latex, separator)]
        if len(parts) >= 3 and all(parts):
            return (
                "\\begin{aligned}\n"
                + " \\\\\n".join(f"&{part}" for part in parts)
                + "\n\\end{aligned}"
            )

    comma_parts = [part.strip() for part in _split_tex_top_level(latex, ",")]
    equality_parts = [part.strip() for part in _split_tex_top_level(latex, "=")]
    if len(equality_parts) >= 4 and all(equality_parts) and len(comma_parts) == 1:
        lines = [f"{equality_parts[0]} &={equality_parts[1]}"]
        lines.extend(f"&={part}" for part in equality_parts[2:])
        return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"

    for operator in (r"\\Longrightarrow", r"\\Longleftrightarrow"):
        parts = [part.strip() for part in _split_tex_top_level(latex, operator)]
        if len(parts) == 2 and all(parts):
            return (
                "\\begin{aligned}\n"
                + parts[0]
                + f" &{operator} \\\\\n"
                + "&\\quad "
                + parts[1]
                + "\n\\end{aligned}"
            )

    return latex


def _surface_text(node: GraphNode) -> str:
    """Return reader-facing prose only; provenance stays in graph/audit artifacts."""

    text = _canonical_surface_text(node.text)
    if not text:
        return ""
    if not _PROVENANCE_LANGUAGE.search(text):
        return text

    # Canonical nodes from older runs often contain a clean mathematical sentence followed by
    # board/ASR commentary. Strip only the observational sentences instead of discarding the whole
    # body. This is intentionally lexical: the surface layer may omit provenance but must not invent
    # or paraphrase mathematics.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", text)
    clean = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and not _PROVENANCE_LANGUAGE.search(sentence)
    ]
    if clean:
        return " ".join(clean)

    # If all prose was observational but a canonical formula exists, the formula is sufficient.
    if (node.latex or "").strip():
        return ""

    # Legacy prose-only nodes can still carry the mathematical statement in their title.
    title = _surface_title(node)
    if title and not _META_TITLE.search(title):
        return title.rstrip(".") + "."
    return ""


def _render_node_body(node: GraphNode) -> str:
    pieces: list[str] = []
    text = _surface_text(node)
    latex = (node.latex or "").strip()
    if text and text != latex:
        pieces.append(escape_tex(text))
    if latex:
        pieces.append("\\[\n" + _surface_display_latex(latex) + "\n\\]")
    return "\n\n".join(pieces)


def _render_proof_component(nodes: list[GraphNode]) -> str:
    pieces: list[str] = []
    for node in nodes:
        text = _surface_text(node)
        latex = (node.latex or "").strip()

        if text:
            pieces.append(escape_tex(text))
        if latex:
            pieces.append("\\[\n" + _surface_display_latex(latex) + "\n\\]")

    return "\n\n".join(pieces)


def _proof_components(
    state: GraphState,
    nodes: list[GraphNode],
) -> tuple[dict[str, str], dict[str, list[GraphNode]]]:
    """Group explicit proof chains without re-interpreting their mathematics."""

    by_id = {node.id: node for node in nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in state.edges:
        relation = _kind(edge.relation)
        if relation not in _PROOF_CHAIN_RELATIONS:
            continue
        if edge.source not in by_id or edge.target not in by_id:
            continue
        source_kind = _kind(by_id[edge.source].kind)
        target_kind = _kind(by_id[edge.target].kind)
        if (
            source_kind not in _PROOF_COMPONENT_KINDS
            or target_kind not in _PROOF_COMPONENT_KINDS
        ):
            continue
        if source_kind not in _PROOF_KINDS and target_kind not in _PROOF_KINDS:
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    component_of: dict[str, str] = {}
    components: dict[str, list[GraphNode]] = {}
    seen: set[str] = set()

    for node in nodes:
        if node.id in seen or _kind(node.kind) not in _PROOF_KINDS:
            continue
        stack = [node.id]
        member_ids: set[str] = set()
        while stack:
            current = stack.pop()
            if current in member_ids:
                continue
            member_ids.add(current)
            stack.extend(adjacency.get(current, set()) - member_ids)

        members = [by_id[node_id] for node_id in member_ids if node_id in by_id]
        members = _order_nodes(state, members)
        component_id = members[0].id
        components[component_id] = members
        for member in members:
            component_of[member.id] = component_id
        seen.update(member_ids)

    return component_of, components


def _surface_blocks(state: GraphState, nodes: list[GraphNode]) -> list[NoteBlock]:
    ordered = _order_nodes(state, nodes)
    component_of, components = _proof_components(state, ordered)
    blocks: list[NoteBlock] = []
    emitted_components: set[str] = set()

    for node in ordered:
        component_id = component_of.get(node.id)
        if component_id is not None:
            if component_id in emitted_components:
                continue
            emitted_components.add(component_id)
            members = components[component_id]
            body = _render_proof_component(members)
            if not body:
                continue
            blocks.append(
                NoteBlock(
                    type=BlockType.PROOF,
                    title=_surface_title(members[0]) or None if len(members) == 1 else None,
                    latex=body,
                    source_evidence_ids=list(
                        dict.fromkeys(
                            evidence_id
                            for member in members
                            for evidence_id in member.evidence_ids
                        )
                    ),
                )
            )
            continue

        title = _surface_title(node) or None
        text = _surface_text(node)
        latex = (node.latex or "").strip()
        kind = _kind(node.kind)

        if kind == "equation" and latex:
            blocks.append(
                NoteBlock(
                    type=BlockType.EQUATION,
                    title=title,
                    latex=_surface_display_latex(latex),
                    source_evidence_ids=list(node.evidence_ids),
                )
            )
            continue

        body = _render_node_body(node)
        if not body:
            continue

        block_title = title
        if text and title and text.rstrip(".") == title.rstrip(".") and not latex:
            block_title = None

        blocks.append(
            NoteBlock(
                type=_block_type(node),
                title=block_title,
                latex=body,
                source_evidence_ids=list(node.evidence_ids),
            )
        )

    return blocks


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

    latex = (node.latex or "").strip()
    # Do not print an explicitly unresolved OCR placeholder merely because it was preserved as a
    # canonical audit node. A later resolved replacement can still render normally.
    if latex and "?" in latex and _PROVENANCE_LANGUAGE.search(node.text):
        return False

    return bool(node.text.strip() or latex)


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


def _topic_time(state: GraphState, topic: GraphNode) -> float:
    start, _ = _node_times(state, topic)
    if start != float("inf"):
        return start
    child_ids = {
        edge.target
        for edge in state.edges
        if edge.source == topic.id
        and _kind(edge.relation) in {"contains", "contains_node", "has_part"}
    }
    starts = [
        _node_times(state, state.nodes[child_id])[0]
        for child_id in child_ids
        if child_id in state.nodes
    ]
    finite = [value for value in starts if value != float("inf")]
    return min(finite) if finite else float("inf")


def _topic_parents(state: GraphState, topic_ids: set[str]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for edge in state.edges:
        relation = _kind(edge.relation)
        if (
            relation in {"part_of", "in_section", "in_topic"}
            and edge.source in topic_ids
            and edge.target in topic_ids
        ):
            parents.setdefault(edge.source, edge.target)
        elif (
            relation in {"contains", "contains_node", "has_part"}
            and edge.source in topic_ids
            and edge.target in topic_ids
        ):
            parents.setdefault(edge.target, edge.source)
    return parents


def _root_topic(topic_id: str, parents: dict[str, str]) -> str:
    seen: set[str] = set()
    current = topic_id
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
    return current


def _section_assignment(
    state: GraphState,
    renderable: list[GraphNode],
) -> tuple[list[GraphNode], dict[str, list[GraphNode]]]:
    all_topics = [
        node
        for node in state.nodes.values()
        if node.status == "active" and _kind(node.kind) in _TOPIC_KINDS
    ]
    topic_by_id = {node.id: node for node in all_topics}
    topic_ids = set(topic_by_id)

    # LectureIR is flat: it has sections but no subsection hierarchy. Collapsing every child topic
    # into its root therefore destroys semantic boundaries (e.g. a late weak-topology topic can be
    # swallowed by a generic "Part 2" container). Render every substantive topic that owns content
    # as its own flat section; parent relations remain useful graph semantics but are not a license
    # to erase the child heading.
    topics = sorted(
        all_topics,
        key=lambda node: (_topic_time(state, node), node.id),
    )

    membership = _topic_membership(state)
    grouped: dict[str, list[GraphNode]] = defaultdict(list)

    if not topics:
        grouped["__lecture__"] = list(renderable)
        return [], grouped

    topic_times = [
        (_topic_time(state, topic), topic.id)
        for topic in topics
    ]
    first_topic_time = min((item[0] for item in topic_times), default=float("inf"))

    explicit_nodes = set(membership)
    assignment: dict[str, str] = {}
    assignment_origin: dict[str, str] = {}
    for node in renderable:
        explicit = membership.get(node.id)
        if explicit is not None and explicit in topic_ids:
            assignment[node.id] = explicit
            assignment_origin[node.id] = "explicit"
            continue

        start, _ = _node_times(state, node)
        if start < first_topic_time:
            assignment[node.id] = "__prelude__"
            assignment_origin[node.id] = "prelude"
            continue

        preceding = [
            (topic_start, topic_id)
            for topic_start, topic_id in topic_times
            if topic_start <= start
        ]
        assignment[node.id] = preceding[-1][1] if preceding else topics[0].id
        assignment_origin[node.id] = "chronological"

    # Chronology is only a fallback. Typed semantic relations such as "refines" provide stronger
    # ownership: a late clarification belongs with the mathematical object it refines. Once such a
    # refinement is anchored, its otherwise-unassigned premises follow it, which models a small
    # directed hypergraph rather than treating every dependency as an undirected proximity edge.
    strong_topic_relations = {
        "refines",
        "equivalent_to",
        "equivalent",
        "proves",
        "supports",
        "concerns",
    }
    by_id = {node.id: node for node in renderable}
    derived_dependencies = {
        node.id: set(node.derived_from)
        for node in renderable
    }

    for _ in range(max(1, len(renderable))):
        changed = False

        for edge in state.edges:
            if _kind(edge.relation) not in strong_topic_relations:
                continue
            if edge.source not in by_id or edge.target not in assignment:
                continue
            if assignment_origin.get(edge.source) in {"explicit", "prelude"}:
                continue
            target_topic = assignment.get(edge.target)
            if target_topic in {None, "__prelude__"}:
                continue
            if assignment.get(edge.source) != target_topic:
                assignment[edge.source] = target_topic
                assignment_origin[edge.source] = "semantic"
                changed = True

        # A semantically anchored refinement/proof may have premises introduced much later in time.
        # Pull only chronology-assigned premises into that same topic; never override explicit
        # membership or another already-semantic assignment.
        for node_id, dependencies in derived_dependencies.items():
            if assignment_origin.get(node_id) != "semantic":
                continue
            owner = assignment[node_id]
            for dependency in dependencies:
                if dependency not in by_id:
                    continue
                if assignment_origin.get(dependency) != "chronological":
                    continue
                if assignment.get(dependency) != owner:
                    assignment[dependency] = owner
                    assignment_origin[dependency] = "semantic"
                    changed = True

        if not changed:
            break

    # Ordinary derivation remains a weaker fallback: move a chronology-assigned node only when all
    # of its relevant premises already agree on one topic.
    edge_dependencies: dict[str, set[str]] = defaultdict(set)
    for edge in state.edges:
        if _kind(edge.relation) in {"derived_from", "depends_on"}:
            edge_dependencies[edge.source].add(edge.target)

    for _ in range(max(1, len(renderable))):
        changed = False
        for node in renderable:
            if assignment_origin.get(node.id) != "chronological":
                continue
            dependencies = set(node.derived_from)
            dependencies.update(edge_dependencies.get(node.id, set()))
            dependency_topics = {
                assignment[dependency]
                for dependency in dependencies
                if dependency in assignment
                and assignment[dependency] != "__prelude__"
            }
            if len(dependency_topics) != 1:
                continue
            inherited = next(iter(dependency_topics))
            if assignment.get(node.id) != inherited:
                assignment[node.id] = inherited
                changed = True
        if not changed:
            break

    for node in renderable:
        grouped[assignment[node.id]].append(node)

    return topics, grouped

def _order_nodes(state: GraphState, nodes: list[GraphNode]) -> list[GraphNode]:
    """Topologically order canonical mathematics; timestamps only break unrelated ties."""

    by_id = {node.id: node for node in nodes}
    precedence: set[tuple[str, str]] = set()

    for node in nodes:
        for dependency in node.derived_from:
            if dependency in by_id and dependency != node.id:
                precedence.add((dependency, node.id))

    for edge in state.edges:
        if edge.source not in by_id or edge.target not in by_id:
            continue
        relation = _kind(edge.relation)
        if relation in _NONORDERING_RELATIONS or edge.source == edge.target:
            continue
        if relation in _REVERSE_ORDER_RELATIONS:
            precedence.add((edge.target, edge.source))
        elif relation in _FORWARD_ORDER_RELATIONS:
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


def realize_graph_surface(
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
    # GraphState.notes are controller/audit diagnostics (frontier ambiguity descriptions,
    # rejected patch traces, provenance comments). They stay in graph artifacts but are not
    # reader-facing unresolved lecture content. Actual unresolved mathematics is represented by
    # graph violations or explicit surviving alternative groups below.
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
        section_nodes = []
        if grouped.get("__prelude__"):
            section_nodes.append((None, grouped["__prelude__"]))
        section_nodes.extend(
            (topic, grouped.get(topic.id, []))
            for topic in topics
        )

    for index, (topic, nodes) in enumerate(section_nodes):
        if not nodes and topic is None:
            continue
        if topic is not None and not nodes:
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

        # Topic nodes determine section headings only. Their descriptive text is latent/audit
        # context and would merely repeat the section contents in prose.
        blocks = _surface_blocks(state, nodes)

        section_title = (
            _canonical_surface_text(topic.title)
            if topic is not None
            else ("Начало лекции" if topics else title)
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


def graph_state_to_ir(
    state: GraphState,
    *,
    lecture_id: str,
    title: str,
) -> LectureIR:
    """Backward-compatible entry point for the deterministic graph surface realizer."""

    return realize_graph_surface(state, lecture_id=lecture_id, title=title)
