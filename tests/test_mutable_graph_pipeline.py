from pathlib import Path

from automatic_lecture_tex.graph_revision import (
    AddNodeOp,
    GraphEdge,
    GraphNode,
    GraphPatch,
    ReplaceNodeOp,
    RetypeNodeOp,
    AddRelationOp,
)
from automatic_lecture_tex.graph_revision_pipeline import (
    GraphRevisionProposal,
    run_iterative_graph_revision,
)
from automatic_lecture_tex.graph_revision_render import graph_state_to_ir
from automatic_lecture_tex.llm import StructuredTaskTooLargeError
from automatic_lecture_tex.schemas import (
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
