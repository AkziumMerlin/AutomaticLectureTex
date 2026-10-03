import logging
from pathlib import Path

from automatic_lecture_tex.graph_revision import (
    AddNodeOp,
    GraphEdge,
    GraphNode,
    GraphPatch,
    ReplaceNodeOp,
    RetypeNodeOp,
    AddRelationOp,
    Violation,
    apply_patch,
)
from automatic_lecture_tex.graph_revision_pipeline import (
    GraphRevisionProposal,
    _compact_catalog,
    _expand_frontier,
    _focus_raw_windows,
    _load_cached_split,
    _normalize_alternative_patch,
    _proposal_fingerprint,
    _apply_or_mark_failure,
    run_iterative_graph_revision,
)
from automatic_lecture_tex.graph_revision_render import graph_state_to_ir
from automatic_lecture_tex.llm import (
    StructuredOutputTruncatedError,
    StructuredTaskTooLargeError,
)
from automatic_lecture_tex.knowledge_pipeline import _load_state_raw_window_index
from automatic_lecture_tex.schemas import (
    BlockType,
    LectureObservation,
    LectureState,
    ObservationKind,
)


class FakeOrchestrator:
    output_language = "ru"

    def __init__(self) -> None:
        self.calls = 0
        self.kwargs = []

    def _structured(self, prompt, schema, **kwargs):
        del prompt, schema
        self.kwargs.append(dict(kwargs))
        self.calls += 1
        if self.calls == 1:
            return GraphRevisionProposal(
                focus_id="round_00_focus_000",
                common_patch=GraphPatch(
                    id="canonicalize_definition",
                    description="Create one canonical definition from repeated observations.",
                    operations=[
                        AddNodeOp(
                            op="add_node",
                            node=GraphNode(
                                id="weak_neighborhood",
                                kind="basis",
                                title="Weak neighborhood",
                                text="Single-functional neighborhoods.",
                                latex=r"V(x,\varphi,\varepsilon)",
                                evidence_ids=["o1", "o2"],
                            ),
                        ),
                    ],
                ),
            )
        return GraphRevisionProposal(
            focus_id="round_00_focus_001",
            common_patch=GraphPatch(
                id="late_revision",
                description="Late evidence revises the earlier type.",
                operations=[
                    RetypeNodeOp(
                        op="retype_node",
                        node_id="weak_neighborhood",
                        kind="subbasic_neighborhood",
                    ),
                    ReplaceNodeOp(
                        op="replace_node",
                        node_id="weak_neighborhood",
                        text="Single-functional neighborhoods are elementary/subbasic.",
                    ),
                ],
            ),
        )


def _lecture_state() -> LectureState:
    return LectureState(
        lecture_id="lecture",
        title="Lecture",
        observations=[
            LectureObservation(
                id="o1",
                window_id="w1",
                start=0.0,
                end=1.0,
                kind=ObservationKind.DEFINITION,
                text="first",
            ),
            LectureObservation(
                id="o2",
                window_id="w1",
                start=1.0,
                end=2.0,
                kind=ObservationKind.CLAIM,
                text="repeat",
            ),
            LectureObservation(
                id="o3",
                window_id="w2",
                start=100.0,
                end=101.0,
                kind=ObservationKind.CLAIM,
                text="later clarification",
            ),
        ],
    )


def test_late_focus_can_revise_an_earlier_canonical_node(tmp_path: Path) -> None:
    orchestrator = FakeOrchestrator()
    result = run_iterative_graph_revision(
        orchestrator,
        lecture_state=_lecture_state(),
        raw_windows=[],
        work=tmp_path,
        llm_config={"model": "fake"},
        rounds=1,
        batch_observations=2,
        overlap_observations=0,
        frontier_width=2,
        catalog_chars=10000,
        raw_context_chars=10000,
        max_images=0,
        max_tokens=4096,
        force=True,
    )

    assert orchestrator.calls == 2
    assert all(item["max_tokens"] == 4096 for item in orchestrator.kwargs)
    node = result.consensus.nodes["weak_neighborhood"]
    assert node.kind == "subbasic_neighborhood"
    assert "elementary/subbasic" in node.text
    assert (tmp_path / "graph_revision" / "consensus_graph.json").exists()


def test_graph_renderer_uses_topics_and_skips_provisional_nodes() -> None:
    state = _lecture_state()
    from automatic_lecture_tex.graph_revision import seed_observation_graph

    graph = seed_observation_graph(state)
    graph.nodes["topic"] = GraphNode(
        id="topic",
        kind="topic",
        title="Weak topology",
        evidence_ids=["o1"],
    )
    graph.nodes["definition"] = GraphNode(
        id="definition",
        kind="definition",
        title="Elementary neighborhood",
        text="An elementary weak neighborhood.",
        latex=r"V(x,\varphi,\varepsilon)",
        evidence_ids=["o1", "o2"],
    )
    graph.nodes["theorem"] = GraphNode(
        id="theorem",
        kind="theorem",
        title="Convergence criterion",
        text="Weak convergence is pointwise convergence on the family.",
        evidence_ids=["o3"],
    )
    graph.edges.extend(
        [
            GraphEdge(source="topic", target="definition", relation="contains"),
            GraphEdge(source="topic", target="theorem", relation="contains"),
        ]
    )

    ir = graph_state_to_ir(
        graph,
        lecture_id="lecture",
        title="Lecture",
    )

    assert len(ir.chunks) == 1
    assert ir.chunks[0].section_title == "Weak topology"
    assert [block.type.value for block in ir.chunks[0].blocks] == [
        "definition",
        "theorem",
    ]
    rendered = "\n".join(block.latex for block in ir.chunks[0].blocks)
    assert "first" not in rendered
    assert "repeat" not in rendered
    assert "later clarification" not in rendered
    assert ir.chunks[0].unresolved


def test_renderer_preserves_explicit_topic_membership() -> None:
    state = _lecture_state()
    from automatic_lecture_tex.graph_revision import graph_state_from_lecture_state

    graph = graph_state_from_lecture_state(state)
    graph.nodes["topic_a"] = GraphNode(
        id="topic_a",
        kind="topic",
        title="A",
        evidence_ids=["o1"],
    )
    graph.nodes["topic_b"] = GraphNode(
        id="topic_b",
        kind="topic",
        title="B",
        evidence_ids=["o3"],
    )
    graph.nodes["late"] = GraphNode(
        id="late",
        kind="statement",
        text="Late clarification of the first topic.",
        evidence_ids=["o3"],
    )
    graph.edges.append(
        GraphEdge(source="topic_a", target="late", relation="contains")
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert len(ir.chunks) == 1
    assert ir.chunks[0].section_title == "A"
    assert "Late clarification" in ir.chunks[0].blocks[0].latex


def test_renderer_dependencies_override_observation_timestamps() -> None:
    state = _lecture_state()
    from automatic_lecture_tex.graph_revision import graph_state_from_lecture_state

    graph = graph_state_from_lecture_state(state)
    graph.nodes["definition"] = GraphNode(
        id="definition",
        kind="definition",
        title="g",
        text="Define g.",
        evidence_ids=["o3"],
    )
    graph.nodes["proof"] = GraphNode(
        id="proof",
        kind="proof",
        title="Complex linearity",
        text="Then prove g is complex-linear.",
        evidence_ids=["o1"],
        derived_from=["definition"],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert [block.title for block in ir.chunks[0].blocks] == [
        "g",
        "Complex linearity",
    ]


class SplittingOrchestrator:
    output_language = "ru"

    def __init__(self, *, allow_calls: bool = True) -> None:
        self.calls = 0
        self.allow_calls = allow_calls

    def _structured(self, prompt, schema, **kwargs):
        del prompt, schema, kwargs
        if not self.allow_calls:
            raise AssertionError("cached split run should not call the model")
        self.calls += 1
        if self.calls == 1:
            raise StructuredTaskTooLargeError("parent focus input is too large")
        if self.calls == 2:
            return GraphRevisionProposal(
                focus_id="child-left",
                common_patch=GraphPatch(
                    id="left_adds_shared_node",
                    description="Left child creates a canonical node.",
                    operations=[
                        AddNodeOp(
                            op="add_node",
                            node=GraphNode(
                                id="shared",
                                kind="claim",
                                text="Initial interpretation.",
                                evidence_ids=["o1"],
                            ),
                        ),
                    ],
                ),
            )
        return GraphRevisionProposal(
            focus_id="child-right",
            common_patch=GraphPatch(
                id="right_revises_shared_node",
                description="Right child sees and revises the node from the left child.",
                operations=[
                    RetypeNodeOp(
                        op="retype_node",
                        node_id="shared",
                        kind="definition",
                    ),
                    ReplaceNodeOp(
                        op="replace_node",
                        node_id="shared",
                        text="Revised using later evidence.",
                    ),
                ],
            ),
        )


def test_oversized_graph_focus_splits_sequentially_and_caches_split(
    tmp_path: Path,
) -> None:
    first = SplittingOrchestrator()
    result = run_iterative_graph_revision(
        first,
        lecture_state=_lecture_state(),
        raw_windows=[],
        work=tmp_path,
        llm_config={"model": "fake"},
        rounds=1,
        batch_observations=3,
        overlap_observations=0,
        frontier_width=2,
        catalog_chars=10000,
        raw_context_chars=10000,
        max_images=0,
        max_tokens=4096,
        force=False,
    )

    assert first.calls == 3
    assert result.stats["split_focuses"] == 1
    assert result.stats["max_split_depth"] == 1
    assert result.consensus.nodes["shared"].kind == "definition"
    assert result.consensus.nodes["shared"].text == "Revised using later evidence."

    cached = SplittingOrchestrator(allow_calls=False)
    cached_result = run_iterative_graph_revision(
        cached,
        lecture_state=_lecture_state(),
        raw_windows=[],
        work=tmp_path,
        llm_config={"model": "fake"},
        rounds=1,
        batch_observations=3,
        overlap_observations=0,
        frontier_width=2,
        catalog_chars=10000,
        raw_context_chars=10000,
        max_images=0,
        max_tokens=4096,
        force=False,
    )

    assert cached.calls == 0
    assert cached_result.stats["split_cache_hits"] == 1
    assert cached_result.consensus.nodes["shared"].kind == "definition"


def test_global_catalog_index_keeps_late_canonical_node_under_tight_budget() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    evidence = {
        "early": EvidenceRecord(id="early", start=0.0, end=1.0, text="early"),
        "late": EvidenceRecord(id="late", start=4000.0, end=4001.0, text="late"),
    }
    graph = GraphState(evidence=evidence)
    for index in range(30):
        graph.nodes[f"early_{index:02d}"] = GraphNode(
            id=f"early_{index:02d}",
            kind="claim",
            title=f"Early canonical node {index}",
            text="x" * 300,
            evidence_ids=["early"],
        )
    graph.nodes["riesz_late"] = GraphNode(
        id="riesz_late",
        kind="theorem",
        title="Late Riesz theorem",
        text="late canonical mathematics",
        evidence_ids=["late"],
    )

    catalog = _compact_catalog(
        graph,
        5000,
        focus_evidence_ids=["late"],
    )

    index_ids = {row[0] for row in catalog["index"]}
    detail_ids = {row["id"] for row in catalog["detail"]}
    assert "riesz_late" in index_ids
    assert "riesz_late" in detail_ids


def test_renderer_flattens_child_topic_and_preserves_pretopic_material() -> None:
    from automatic_lecture_tex.graph_revision import (
        EvidenceRecord,
        GraphState,
    )

    graph = GraphState(
        evidence={
            "pre": EvidenceRecord(id="pre", start=0.0, end=1.0, text="pre"),
            "root": EvidenceRecord(id="root", start=10.0, end=11.0, text="root"),
            "child": EvidenceRecord(id="child", start=20.0, end=21.0, text="child"),
        }
    )
    graph.nodes["pre_math"] = GraphNode(
        id="pre_math",
        kind="definition",
        title="Initial definition",
        text="Material before the first topic.",
        evidence_ids=["pre"],
    )
    graph.nodes["root_topic"] = GraphNode(
        id="root_topic",
        kind="topic",
        title="Root topic",
        evidence_ids=["root"],
    )
    graph.nodes["child_topic"] = GraphNode(
        id="child_topic",
        kind="topic",
        title="Child topic",
        evidence_ids=["child"],
    )
    graph.nodes["child_math"] = GraphNode(
        id="child_math",
        kind="theorem",
        title="Child theorem",
        text="Nested mathematics.",
        evidence_ids=["child"],
    )
    graph.edges.extend(
        [
            GraphEdge(
                source="child_topic",
                target="root_topic",
                relation="part_of",
            ),
            GraphEdge(
                source="child_topic",
                target="child_math",
                relation="contains",
            ),
        ]
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert [chunk.section_title for chunk in ir.chunks] == [
        "Начало лекции",
        "Child topic",
    ]
    assert [block.title for block in ir.chunks[0].blocks] == ["Initial definition"]
    assert [block.title for block in ir.chunks[1].blocks] == ["Child theorem"]


def test_catalog_includes_distant_derived_node_in_focus_detail() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "early": EvidenceRecord(id="early", start=10.0, end=11.0, text="early"),
            "late": EvidenceRecord(id="late", start=4000.0, end=4001.0, text="late"),
        },
        nodes={
            "elementary": GraphNode(
                id="elementary",
                kind="equation",
                title="Elementary neighbourhood",
                text="V(x, phi, eps)",
                evidence_ids=["early"],
            ),
            "finite_basis": GraphNode(
                id="finite_basis",
                kind="definition",
                title="Finite-intersection basis",
                text="Finite intersections of elementary neighbourhoods.",
                evidence_ids=["late"],
                derived_from=["elementary"],
            ),
        },
    )

    catalog = _compact_catalog(
        graph,
        10000,
        focus_evidence_ids=["early"],
    )

    assert "finite_basis" in {row["id"] for row in catalog["detail"]}


def test_single_alternative_keeps_common_state_as_competing_frontier() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    base = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="x")},
        nodes={
            "n": GraphNode(
                id="n",
                kind="equation",
                text="literal reading",
                evidence_ids=["e1"],
            )
        },
    )
    proposal = GraphRevisionProposal(
        focus_id="focus",
        alternatives=[
            GraphPatch(
                id="alternative",
                description="Competing reading.",
                operations=[
                    ReplaceNodeOp(
                        op="replace_node",
                        node_id="n",
                        text="alternative reading",
                    )
                ],
            )
        ],
    )

    frontier = _expand_frontier([base], proposal, width=3)

    assert len(frontier) == 2
    assert {state.nodes["n"].text for state in frontier} == {
        "literal reading",
        "alternative reading",
    }


def test_renderer_inherits_topic_from_long_range_dependency() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "weak": EvidenceRecord(id="weak", start=100.0, end=101.0, text="weak"),
            "l2": EvidenceRecord(id="l2", start=3000.0, end=3001.0, text="l2"),
            "late": EvidenceRecord(id="late", start=4000.0, end=4001.0, text="late"),
        },
        nodes={
            "weak_topic": GraphNode(
                id="weak_topic",
                kind="topic",
                title="Weak topology",
                evidence_ids=["weak"],
            ),
            "l2_topic": GraphNode(
                id="l2_topic",
                kind="topic",
                title="l2 example",
                evidence_ids=["l2"],
            ),
            "elementary": GraphNode(
                id="elementary",
                kind="equation",
                title="Elementary neighbourhood",
                text="V",
                evidence_ids=["weak"],
            ),
            "late_basis": GraphNode(
                id="late_basis",
                kind="definition",
                title="Finite-intersection basis",
                text="beta",
                evidence_ids=["late"],
                derived_from=["elementary"],
            ),
        },
        edges=[
            GraphEdge(
                source="weak_topic",
                target="elementary",
                relation="contains",
            )
        ],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")
    chunks = {chunk.section_title: chunk for chunk in ir.chunks}

    assert any(
        block.title == "Finite-intersection basis"
        for block in chunks["Weak topology"].blocks
    )
    if "l2 example" in chunks:
        assert not any(
            block.title == "Finite-intersection basis"
            for block in chunks["l2 example"].blocks
        )


def test_graph_revision_emits_high_level_progress_logs(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="automatic_lecture_tex.graph_revision_pipeline",
    )
    run_iterative_graph_revision(
        FakeOrchestrator(),
        lecture_state=_lecture_state(),
        raw_windows=[],
        work=tmp_path,
        llm_config={"model": "fake"},
        rounds=1,
        batch_observations=3,
        overlap_observations=0,
        frontier_width=2,
        catalog_chars=10000,
        raw_context_chars=10000,
        max_images=0,
        max_tokens=4096,
        force=True,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("[graph_revision] start:" in message for message in messages)
    assert any("round 1/1 start" in message for message in messages)
    assert any("focus round_00_focus_000 start" in message for message in messages)
    assert any("proposal ready: common_ops=" in message for message in messages)
    assert any("focus round_00_focus_000 done" in message for message in messages)
    assert any("[graph_revision] complete:" in message for message in messages)


def test_same_turn_diagnosis_does_not_survive_successful_common_patch() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    base = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="old")},
        nodes={
            "n": GraphNode(
                id="n",
                kind="claim",
                text="old",
                evidence_ids=["e1"],
            )
        },
    )
    proposal = GraphRevisionProposal(
        focus_id="focus",
        diagnosed_violations=[
            Violation(
                id="pre_edit_problem",
                category="structure",
                severity=2,
                message="The current node text is incomplete.",
                related_nodes=["n"],
            )
        ],
        common_patch=GraphPatch(
            id="fix",
            description="Complete the node.",
            operations=[
                ReplaceNodeOp(
                    op="replace_node",
                    node_id="n",
                    text="complete",
                )
            ],
        ),
    )

    frontier = _expand_frontier([base], proposal, width=2)

    assert len(frontier) == 1
    assert frontier[0].nodes["n"].text == "complete"
    assert "pre_edit_problem" not in frontier[0].violations


def test_renderer_refinement_anchor_pulls_late_premises_into_refined_topic() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "weak": EvidenceRecord(id="weak", start=100.0, end=101.0, text="weak"),
            "l2": EvidenceRecord(id="l2", start=3000.0, end=3001.0, text="l2"),
            "beta": EvidenceRecord(id="beta", start=4300.0, end=4301.0, text="beta"),
            "claim": EvidenceRecord(id="claim", start=4310.0, end=4311.0, text="claim"),
        },
        nodes={
            "weak_topic": GraphNode(
                id="weak_topic",
                kind="topic",
                title="Weak topology",
                evidence_ids=["weak"],
            ),
            "l2_topic": GraphNode(
                id="l2_topic",
                kind="topic",
                title="l2 example",
                evidence_ids=["l2"],
            ),
            "sigma": GraphNode(
                id="sigma",
                kind="definition",
                title="Early generating family",
                text="sigma",
                evidence_ids=["weak"],
            ),
            "beta": GraphNode(
                id="beta",
                kind="definition",
                title="Finite-intersection family",
                text="beta",
                evidence_ids=["beta"],
            ),
            "beta_tilde": GraphNode(
                id="beta_tilde",
                kind="definition",
                title="Common-center finite-intersection family",
                text="beta tilde",
                evidence_ids=["beta"],
            ),
            "equivalence": GraphNode(
                id="equivalence",
                kind="claim",
                title="Equivalent basis refinement",
                text="refines sigma",
                evidence_ids=["claim"],
                derived_from=["beta", "beta_tilde"],
            ),
        },
        edges=[
            GraphEdge(source="weak_topic", target="sigma", relation="contains"),
            GraphEdge(source="equivalence", target="sigma", relation="refines"),
        ],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")
    chunks = {chunk.section_title: chunk for chunk in ir.chunks}

    weak_titles = {block.title for block in chunks["Weak topology"].blocks}
    assert "Equivalent basis refinement" in weak_titles
    assert "Finite-intersection family" in weak_titles
    assert "Common-center finite-intersection family" in weak_titles
    if "l2 example" in chunks:
        l2_titles = {block.title for block in chunks["l2 example"].blocks}
        assert "Equivalent basis refinement" not in l2_titles
        assert "Finite-intersection family" not in l2_titles


def test_sibling_alternative_node_is_normalized_to_competing_same_id_variant() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    base = GraphState(
        evidence={
            "plain": EvidenceRecord(id="plain", text="plain"),
            "conj": EvidenceRecord(id="conj", text="conjugated"),
        },
        nodes={
            "riesz_vector": GraphNode(
                id="riesz_vector",
                kind="equation",
                title="Plain coefficient",
                text="plain coefficient",
                evidence_ids=["plain", "conj"],
                derived_from=[],
                alternative_group="riesz_convention",
            )
        },
    )
    alternative = GraphPatch(
        id="conjugated",
        description="Competing conjugated reading.",
        operations=[
            AddNodeOp(
                op="add_node",
                node=GraphNode(
                    id="riesz_vector_conj",
                    kind="equation",
                    title="Conjugated coefficient",
                    text="conjugated coefficient",
                    evidence_ids=["conj"],
                    derived_from=[],
                    status="alternative",
                    alternative_group="riesz_convention",
                ),
            )
        ],
    )

    normalized = _normalize_alternative_patch(base, alternative)
    revised = apply_patch(base, normalized)

    assert "riesz_vector_conj" not in revised.nodes
    assert revised.nodes["riesz_vector"].text == "conjugated coefficient"
    assert revised.nodes["riesz_vector"].metadata["frontier_variant_source_id"] == (
        "riesz_vector_conj"
    )


def test_nested_substantive_topic_renders_as_own_flat_section() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "intro": EvidenceRecord(id="intro", start=0.0, end=1.0, text="intro"),
            "weak": EvidenceRecord(id="weak", start=100.0, end=101.0, text="weak"),
        },
        nodes={
            "part": GraphNode(
                id="part",
                kind="topic",
                title="Part 2",
                evidence_ids=["intro"],
            ),
            "weak_topic": GraphNode(
                id="weak_topic",
                kind="topic",
                title="Weak topology",
                evidence_ids=["weak"],
            ),
            "intro_def": GraphNode(
                id="intro_def",
                kind="definition",
                title="Intro definition",
                text="intro",
                evidence_ids=["intro"],
            ),
            "weak_def": GraphNode(
                id="weak_def",
                kind="definition",
                title="Weak definition",
                text="weak",
                evidence_ids=["weak"],
            ),
        },
        edges=[
            GraphEdge(source="part", target="intro_def", relation="contains"),
            GraphEdge(source="weak_topic", target="part", relation="part_of"),
            GraphEdge(source="weak_topic", target="weak_def", relation="contains"),
        ],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")
    chunks = {chunk.section_title: chunk for chunk in ir.chunks}

    assert "Part 2" in chunks
    assert "Weak topology" in chunks
    assert {block.title for block in chunks["Part 2"].blocks} == {"Intro definition"}
    assert {block.title for block in chunks["Weak topology"].blocks} == {"Weak definition"}


def test_proposal_fingerprint_changes_when_prompt_changes() -> None:
    from automatic_lecture_tex.graph_revision import GraphState

    state = GraphState(evidence={})
    kwargs = dict(
        state=state,
        focus_id="focus",
        focus_evidence=[],
        raw_windows=[],
        catalog={"index": [], "detail": []},
        frontier_summary=[],
        llm_config={"model": "fake"},
    )

    first = _proposal_fingerprint(
        **kwargs,
        proposal_prompt="instruction version A",
    )
    second = _proposal_fingerprint(
        **kwargs,
        proposal_prompt="instruction version B",
    )

    assert first != second


def test_rejected_patch_is_audit_note_not_graph_violation() -> None:
    from automatic_lecture_tex.graph_revision import GraphState

    state = GraphState(evidence={})
    failed = _apply_or_mark_failure(
        state,
        GraphPatch(
            id="bad",
            description="References a missing node.",
            operations=[
                ReplaceNodeOp(
                    op="replace_node",
                    node_id="missing",
                    text="x",
                )
            ],
        ),
        failure_id="focus::common_failed::0",
    )

    assert failed.violations == {}
    assert len(failed.notes) == 1
    assert "Unapplied patch bad" in failed.notes[0]


def test_renderer_keeps_audit_notes_out_of_unresolved() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", start=0.0, end=1.0, text="x")},
        nodes={
            "topic": GraphNode(
                id="topic",
                kind="topic",
                title="Section",
                evidence_ids=["e1"],
            ),
            "claim": GraphNode(
                id="claim",
                kind="claim",
                title="Claim",
                text="x",
                evidence_ids=["e1"],
            ),
        },
        edges=[GraphEdge(source="topic", target="claim", relation="contains")],
        notes=["Frontier ambiguity for internal_node: audit details"],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert sum(len(chunk.unresolved) for chunk in ir.chunks) == 0


def test_raw_window_index_preserves_extraction_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "knowledge_windows"
    root.mkdir()
    (root / "window_0001.json").write_text(
        """{
          "chunk": {
            "id": "window_0001",
            "start": 10.0,
            "end": 20.0,
            "timestamped_text": "[00:10-00:20] ambiguous"
          },
          "observations": {
            "window_id": "window_0001",
            "start": 10.0,
            "end": 20.0,
            "observations": [],
            "unresolved": [
              "Two formula readings remain compatible with the board."
            ]
          },
          "visual_evidence": []
        }""",
        encoding="utf-8",
    )

    raw = _load_state_raw_window_index(tmp_path)

    assert raw[0]["extraction_unresolved"] == [
        "Two formula readings remain compatible with the board."
    ]


def test_focus_raw_windows_exposes_only_nonempty_extraction_uncertainty() -> None:
    evidence = [
        {
            "id": "o1",
            "start": 10.0,
            "end": 20.0,
            "window_id": "window_0001",
            "window_ids": ["window_0001"],
        }
    ]
    raw = [
        {
            "window_id": "window_0001",
            "start": 10.0,
            "end": 20.0,
            "asr": "ambiguous",
            "visual_latex": [],
            "math_ocr_candidates": [],
            "extraction_unresolved": [
                "Two formula readings remain compatible with the board."
            ],
        },
        {
            "window_id": "window_0002",
            "start": 21.0,
            "end": 25.0,
            "asr": "clear",
            "visual_latex": [],
            "math_ocr_candidates": [],
            "extraction_unresolved": [],
        },
    ]

    selected = _focus_raw_windows(evidence, raw, max_chars=10000)

    by_id = {item["window_id"]: item for item in selected}
    assert by_id["window_0001"]["extraction_unresolved"] == [
        "Two formula readings remain compatible with the board."
    ]
    assert "extraction_unresolved" not in by_id["window_0002"]


def test_renderer_concerns_relation_inherits_topic_from_all_targets() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "weak": EvidenceRecord(id="weak", start=100.0, end=101.0, text="weak"),
            "l2": EvidenceRecord(id="l2", start=3000.0, end=3001.0, text="l2"),
            "late": EvidenceRecord(id="late", start=4000.0, end=4001.0, text="late"),
        },
        nodes={
            "weak_topic": GraphNode(
                id="weak_topic",
                kind="topic",
                title="Weak topology",
                evidence_ids=["weak"],
            ),
            "l2_topic": GraphNode(
                id="l2_topic",
                kind="topic",
                title="l2",
                evidence_ids=["l2"],
            ),
            "beta": GraphNode(
                id="beta",
                kind="definition",
                title="beta",
                text="beta",
                evidence_ids=["late"],
            ),
            "beta_tilde": GraphNode(
                id="beta_tilde",
                kind="definition",
                title="beta tilde",
                text="beta tilde",
                evidence_ids=["late"],
            ),
            "equivalence": GraphNode(
                id="equivalence",
                kind="claim",
                title="equivalence",
                text="same topology",
                evidence_ids=["late"],
            ),
        },
        edges=[
            GraphEdge(source="weak_topic", target="beta", relation="contains"),
            GraphEdge(source="weak_topic", target="beta_tilde", relation="contains"),
            GraphEdge(source="equivalence", target="beta", relation="concerns"),
            GraphEdge(source="equivalence", target="beta_tilde", relation="concerns"),
        ],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")
    chunks = {chunk.section_title: chunk for chunk in ir.chunks}

    assert "equivalence" in {
        block.title for block in chunks["Weak topology"].blocks
    }
    if "l2" in chunks:
        assert "equivalence" not in {
            block.title for block in chunks["l2"].blocks
        }


def test_surface_realizer_drops_topic_body_and_transition_nodes() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", start=0.0, end=1.0, text="topic"),
            "e2": EvidenceRecord(id="e2", start=1.0, end=2.0, text="transition"),
            "e3": EvidenceRecord(id="e3", start=2.0, end=3.0, text="definition"),
        },
        nodes={
            "topic": GraphNode(
                id="topic",
                kind="topic",
                title="Weak topology",
                text="Lecturer-facing summary that must not be repeated.",
                evidence_ids=["e1"],
            ),
            "transition": GraphNode(
                id="transition",
                kind="transition",
                text="Лектор переходит к следующему разделу.",
                evidence_ids=["e2"],
            ),
            "definition": GraphNode(
                id="definition",
                kind="definition",
                title="Definition",
                text="A reader-facing mathematical definition.",
                evidence_ids=["e3"],
            ),
        },
        edges=[
            GraphEdge(source="topic", target="transition", relation="contains"),
            GraphEdge(source="topic", target="definition", relation="contains"),
        ],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert [chunk.section_title for chunk in ir.chunks] == ["Weak topology"]
    assert len(ir.chunks[0].blocks) == 1
    assert ir.chunks[0].blocks[0].title == "Definition"
    assert "Lecturer-facing summary" not in ir.chunks[0].blocks[0].latex
    assert "переходит" not in ir.chunks[0].blocks[0].latex


def test_surface_realizer_drops_observational_prose_when_formula_is_canonical() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="formula")},
        nodes={
            "eq": GraphNode(
                id="eq",
                kind="equation",
                title="Norm equality",
                text="На доске лектор записывает равенство норм.",
                latex=r"\lVert f\rVert=\lVert u\rVert",
                evidence_ids=["e1"],
            )
        },
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")
    block = ir.chunks[0].blocks[0]

    assert block.type == BlockType.EQUATION
    assert block.title == "Norm equality"
    assert block.latex == r"\lVert f\rVert=\lVert u\rVert"
    assert "лектор" not in block.latex


def test_surface_realizer_uses_mathematical_title_when_legacy_body_is_only_provenance() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="remark")},
        nodes={
            "remark": GraphNode(
                id="remark",
                kind="remark",
                title="Вещественный случай является частным случаем комплексного",
                text="Устно лектор несколько раз повторяет это замечание у доски.",
                evidence_ids=["e1"],
            )
        },
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")
    block = ir.chunks[0].blocks[0]

    assert block.title is None
    assert "Вещественный случай является частным случаем комплексного" in block.latex
    assert "лектор" not in block.latex


def test_surface_realizer_coalesces_explicit_proof_chain() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", start=1.0, end=2.0, text="step1"),
            "e2": EvidenceRecord(id="e2", start=2.0, end=3.0, text="eq"),
            "e3": EvidenceRecord(id="e3", start=3.0, end=4.0, text="step2"),
        },
        nodes={
            "step1": GraphNode(
                id="step1",
                kind="proof_step",
                title="Choose a vector",
                text="Choose a nonzero vector.",
                latex=r"z\ne0",
                evidence_ids=["e1"],
            ),
            "middle": GraphNode(
                id="middle",
                kind="equation",
                title="Decomposition",
                text="Write the decomposition.",
                latex=r"x=\alpha z+y",
                evidence_ids=["e2"],
            ),
            "step2": GraphNode(
                id="step2",
                kind="proof_step",
                title="Conclude",
                text="The conclusion follows.",
                latex=r"f(x)=(x,y_f)",
                evidence_ids=["e3"],
            ),
        },
        edges=[
            GraphEdge(source="step1", target="middle", relation="precedes"),
            GraphEdge(source="middle", target="step2", relation="precedes"),
        ],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert len(ir.chunks[0].blocks) == 1
    block = ir.chunks[0].blocks[0]
    assert block.type == BlockType.PROOF
    assert r"z\ne0" in block.latex
    assert r"x=\alpha z+y" in block.latex
    assert r"f(x)=(x,y_f)" in block.latex


def test_surface_order_places_claim_before_proof_that_proves_it() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    graph = GraphState(
        evidence={
            "claim": EvidenceRecord(id="claim", start=20.0, end=21.0, text="claim"),
            "proof": EvidenceRecord(id="proof", start=10.0, end=11.0, text="proof"),
        },
        nodes={
            "claim": GraphNode(
                id="claim",
                kind="claim",
                title="Statement",
                text="The statement.",
                evidence_ids=["claim"],
            ),
            "proof": GraphNode(
                id="proof",
                kind="proof_step",
                title="Proof",
                text="The proof.",
                evidence_ids=["proof"],
            ),
        },
        edges=[GraphEdge(source="proof", target="claim", relation="proves")],
    )

    ir = graph_state_to_ir(graph, lecture_id="lecture", title="Lecture")

    assert [block.type for block in ir.chunks[0].blocks] == [
        BlockType.PARAGRAPH,
        BlockType.PROOF,
    ]
    assert ir.chunks[0].blocks[0].title == "Statement"



class OutputRetryOrchestrator:
    output_language = "ru"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def _structured(self, prompt, schema, **kwargs):
        del prompt, schema
        budget = int(kwargs["max_tokens"])
        self.calls.append(budget)
        if len(self.calls) == 1:
            raise StructuredOutputTruncatedError(
                "graph_revision_proposal structured output was truncated",
                max_tokens=budget,
                raw_chars=0,
            )
        return GraphRevisionProposal(focus_id="round_00_focus_000", stable=True)


def test_graph_revision_retries_same_focus_at_global_output_budget(tmp_path: Path) -> None:
    orchestrator = OutputRetryOrchestrator()
    result = run_iterative_graph_revision(
        orchestrator,
        lecture_state=_lecture_state(),
        raw_windows=[],
        work=tmp_path,
        llm_config={"model": "fake", "max_tokens": 32768},
        rounds=1,
        batch_observations=3,
        overlap_observations=0,
        frontier_width=2,
        catalog_chars=10000,
        raw_context_chars=10000,
        max_images=0,
        max_tokens=16384,
        force=True,
    )

    assert orchestrator.calls == [16384, 32768]
    assert result.stats["output_budget_retries"] == 1
    assert result.stats["split_focuses"] == 0


def test_cached_output_truncation_split_is_not_replayed(tmp_path: Path) -> None:
    path = tmp_path / "proposal.json"
    path.write_text(
        '{"fingerprint":"fp","split":{"reason":'
        '"graph_revision_proposal structured output was truncated at max_tokens=16384",'
        '"children":["a","b"]}}',
        encoding="utf-8",
    )

    assert _load_cached_split(path, "fp") is False


def test_catalog_does_not_expand_singleton_focus_with_nearby_provisionals() -> None:
    from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphState

    evidence = {
        "focus": EvidenceRecord(id="focus", start=70.0, end=71.0, text="focus"),
        "near": EvidenceRecord(id="near", start=200.0, end=201.0, text="near"),
    }
    graph = GraphState(
        evidence=evidence,
        nodes={
            "obs::focus": GraphNode(
                id="obs::focus",
                kind="provisional_claim",
                text="focus",
                evidence_ids=["focus"],
            ),
            "obs::near": GraphNode(
                id="obs::near",
                kind="provisional_claim",
                text="near",
                evidence_ids=["near"],
            ),
            "canonical_near": GraphNode(
                id="canonical_near",
                kind="definition",
                text="canonical",
                evidence_ids=["near"],
            ),
        },
    )

    catalog = _compact_catalog(graph, 10000, focus_evidence_ids=["focus"])
    detail_ids = {row["id"] for row in catalog["detail"]}

    assert "obs::focus" in detail_ids
    assert "obs::near" not in detail_ids
    assert "canonical_near" in detail_ids
