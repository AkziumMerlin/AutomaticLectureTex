import pytest
from pydantic import ValidationError

from automatic_lecture_tex.graph_revision import (
    AddNodeOp,
    AddViolationOp,
    EvidenceRecord,
    GraphNode,
    GraphPatch,
    GraphState,
    MergeNodesOp,
    ResolveViolationOp,
    RetypeNodeOp,
    ReplaceNodeOp,
    SplitNodeOp,
    SuppressNodeOp,
    Violation,
    apply_patch,
    graph_consensus,
    metrics,
    search_alternatives,
)


def _state() -> GraphState:
    return GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", text="first"),
            "e2": EvidenceRecord(id="e2", text="second"),
        }
    )


def test_merge_preserves_all_provenance() -> None:
    state = _state()
    state = apply_patch(
        state,
        GraphPatch(
            id="seed",
            description="seed nodes",
            operations=[
                AddNodeOp(
                    op="add_node",
                    node=GraphNode(
                        id="n1",
                        kind="statement",
                        text="same event",
                        evidence_ids=["e1"],
                    ),
                ),
                AddNodeOp(
                    op="add_node",
                    node=GraphNode(
                        id="n2",
                        kind="statement",
                        text="same event",
                        evidence_ids=["e2"],
                    ),
                ),
            ],
        ),
    )
    merged = apply_patch(
        state,
        GraphPatch(
            id="merge",
            description="merge repeated measurements",
            operations=[
                MergeNodesOp(
                    op="merge_nodes",
                    node_ids=["n1", "n2"],
                    into_id="event",
                )
            ],
        ),
    )

    assert set(merged.nodes["event"].evidence_ids) == {"e1", "e2"}
    assert metrics(merged).unexplained_evidence == 0


def test_split_cannot_drop_evidence() -> None:
    state = _state()
    state = apply_patch(
        state,
        GraphPatch(
            id="seed",
            description="seed",
            operations=[
                AddNodeOp(
                    op="add_node",
                    node=GraphNode(
                        id="n",
                        kind="statement",
                        evidence_ids=["e1", "e2"],
                    ),
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match="drop evidence"):
        apply_patch(
            state,
            GraphPatch(
                id="bad-split",
                description="bad",
                operations=[
                    SplitNodeOp(
                        op="split_node",
                        node_id="n",
                        replacements=[
                            GraphNode(
                                id="left",
                                kind="statement",
                                evidence_ids=["e1"],
                            ),
                            GraphNode(
                                id="right",
                                kind="statement",
                                evidence_ids=[],
                            ),
                        ],
                    )
                ],
            ),
        )


def test_math_consistency_is_hard_priority_over_node_count() -> None:
    base = _state()
    fixed = GraphPatch(
        id="seed",
        description="literal early graph",
        operations=[
            AddNodeOp(
                op="add_node",
                node=GraphNode(
                    id="v",
                    kind="basis",
                    text="single functional neighborhoods form a basis",
                    evidence_ids=["e1", "e2"],
                ),
            ),
            AddViolationOp(
                op="add_violation",
                violation=Violation(
                    id="basis-error",
                    category="math",
                    severity=3,
                    message="finite intersections require a separate basis node",
                ),
            ),
        ],
    )
    corrected = GraphPatch(
        id="correct",
        description="late evidence revises early type",
        decision_group="basis",
        operations=[
            RetypeNodeOp(
                op="retype_node",
                node_id="v",
                kind="subbasic_neighborhood",
            ),
            AddNodeOp(
                op="add_node",
                node=GraphNode(
                    id="basis",
                    kind="basis",
                    text="finite intersections",
                    evidence_ids=["e2"],
                ),
            ),
            ResolveViolationOp(
                op="resolve_violation",
                violation_id="basis-error",
            ),
        ],
    )
    literal = GraphPatch(
        id="literal",
        description="keep smaller but wrong graph",
        decision_group="basis",
    )

    frontier = search_alternatives(
        base,
        [fixed],
        [[corrected, literal]],
        pareto=True,
    )

    assert len(frontier) == 1
    assert frontier[0].state.applied_patches[-1] == "correct"
    assert frontier[0].metrics.math_violations == 0
    assert frontier[0].metrics.active_nodes == 2


def test_unresolved_global_alternatives_can_survive_frontier() -> None:
    base = _state()
    fixed = GraphPatch(
        id="seed",
        description="shared core",
        operations=[
            AddNodeOp(
                op="add_node",
                node=GraphNode(
                    id="riesz",
                    kind="theorem",
                    text="f(x)=(x,y_f)",
                    evidence_ids=["e1", "e2"],
                ),
            )
        ],
    )
    real = GraphPatch(
        id="real",
        description="real convention",
        decision_group="field",
    )
    complex_ = GraphPatch(
        id="complex",
        description="complex convention with conjugation",
        decision_group="field",
    )

    frontier = search_alternatives(
        base,
        [fixed],
        [[real, complex_]],
        pareto=True,
    )

    assert {item.state.applied_patches[-1] for item in frontier} == {
        "real",
        "complex",
    }


def test_consensus_preserves_invariant_node_and_marks_metadata_ambiguity() -> None:
    base = _state()
    base.nodes["riesz"] = GraphNode(
        id="riesz",
        kind="theorem",
        text="f(x)=(x,y_f)",
        evidence_ids=["e1", "e2"],
        metadata={"field": "real"},
    )
    other = base.clone()
    other.nodes["riesz"].metadata = {"field": "complex", "conjugation": True}

    consensus = graph_consensus([base, other])

    assert "riesz" in consensus.nodes
    assert consensus.nodes["riesz"].text == "f(x)=(x,y_f)"
    assert "frontier_metadata_alternatives" in consensus.nodes["riesz"].metadata
    assert any("riesz" in note for note in consensus.notes)


def test_canonical_node_automatically_suppresses_absorbed_provisional() -> None:
    state = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="raw observation")},
        nodes={
            "obs::e1": GraphNode(
                id="obs::e1",
                kind="provisional_equation",
                text="raw observation",
                evidence_ids=["e1"],
            )
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="canonicalize",
            description="Canonicalize one observation.",
            operations=[
                AddNodeOp(
                    op="add_node",
                    node=GraphNode(
                        id="canonical",
                        kind="equation",
                        text="canonical mathematics",
                        evidence_ids=["e1"],
                    ),
                )
            ],
        ),
    )

    assert revised.nodes["obs::e1"].status == "suppressed"
    assert revised.nodes["canonical"].status == "active"
    assert metrics(revised).unsupported_nodes == 0


def test_singleton_merge_absorbs_into_existing_canonical_target() -> None:
    state = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", text="canonical"),
            "e2": EvidenceRecord(id="e2", text="repeat measurement"),
        },
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="theorem",
                title="Canonical theorem",
                text="Keep this canonical content.",
                evidence_ids=["e1"],
                aliases=["old alias"],
                metadata={"scope": "global"},
            ),
            "obs::e2": GraphNode(
                id="obs::e2",
                kind="provisional_claim",
                text="repeat measurement",
                evidence_ids=["e2"],
                aliases=["surface alias"],
            ),
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="absorb",
            description="Absorb one provisional measurement into an existing theorem.",
            operations=[
                MergeNodesOp(
                    op="merge_nodes",
                    node_ids=["obs::e2"],
                    into_id="canonical",
                )
            ],
        ),
    )

    assert "obs::e2" not in revised.nodes
    target = revised.nodes["canonical"]
    assert target.kind == "theorem"
    assert target.title == "Canonical theorem"
    assert target.text == "Keep this canonical content."
    assert set(target.evidence_ids) == {"e1", "e2"}
    assert set(target.aliases) == {"old alias", "surface alias"}
    assert target.metadata["scope"] == "global"
    assert target.metadata["merged_from"] == ["obs::e2"]


def test_singleton_self_merge_is_rejected() -> None:
    state = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="x")},
        nodes={
            "n": GraphNode(
                id="n",
                kind="claim",
                text="x",
                evidence_ids=["e1"],
            )
        },
    )

    with pytest.raises(ValueError, match="at least two distinct nodes"):
        apply_patch(
            state,
            GraphPatch(
                id="bad-self-merge",
                description="Meaningless self merge.",
                operations=[
                    MergeNodesOp(
                        op="merge_nodes",
                        node_ids=["n"],
                        into_id="n",
                    )
                ],
            ),
        )


def test_stale_provisional_cleanup_does_not_rollback_useful_patch() -> None:
    state = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="x")},
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="claim",
                text="old text",
                evidence_ids=["e1"],
            )
        },
        violations={
            "stale": Violation(
                id="stale",
                category="structure",
                severity=1,
                message="stale cleanup",
            )
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="cleanup",
            description="Useful edit plus stale provisional cleanup.",
            operations=[
                ReplaceNodeOp(
                    op="replace_node",
                    node_id="canonical",
                    text="new text",
                ),
                MergeNodesOp(
                    op="merge_nodes",
                    node_ids=["obs::already_absorbed"],
                    into_id="canonical",
                ),
                SuppressNodeOp(
                    op="suppress_node",
                    node_id="obs::already_absorbed",
                    reason="already absorbed",
                ),
                ResolveViolationOp(
                    op="resolve_violation",
                    violation_id="stale",
                ),
            ],
        ),
    )

    assert revised.nodes["canonical"].text == "new text"
    assert "stale" not in revised.violations


def test_graph_patch_repairs_bare_graph_node_operation() -> None:
    patch = GraphPatch.model_validate(
        {
            "id": "repair-bare-node",
            "description": "Model forgot the add_node wrapper.",
            "operations": [
                {
                    "id": "section_g_complex",
                    "kind": "topic",
                    "title": "Complex functionals",
                    "text": "",
                    "latex": None,
                    "evidence_ids": ["e1"],
                    "derived_from": [],
                    "aliases": [],
                    "status": "active",
                    "alternative_group": None,
                    "metadata": {},
                }
            ],
        }
    )

    assert len(patch.operations) == 1
    operation = patch.operations[0]
    assert isinstance(operation, AddNodeOp)
    assert operation.op == "add_node"
    assert operation.node.id == "section_g_complex"
    assert operation.node.evidence_ids == ["e1"]


def test_graph_patch_does_not_guess_ambiguous_missing_operation_tag() -> None:
    with pytest.raises(ValidationError, match="Unable to extract tag"):
        GraphPatch.model_validate(
            {
                "id": "ambiguous",
                "description": "Missing discriminator but not a valid GraphNode.",
                "operations": [
                    {
                        "node_id": "x",
                        "text": "new text",
                    }
                ],
            }
        )


def test_merge_nodes_repairs_metadata_alias_and_updates_target_metadata() -> None:
    state = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", text="canonical"),
            "e2": EvidenceRecord(id="e2", text="support"),
        },
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="theorem",
                text="core",
                evidence_ids=["e1"],
                metadata={"field": "complex"},
            ),
            "obs::e2": GraphNode(
                id="obs::e2",
                kind="provisional_claim",
                text="support",
                evidence_ids=["e2"],
            ),
        },
    )

    patch = GraphPatch.model_validate(
        {
            "id": "merge-with-metadata",
            "description": "Absorb support and record ambiguity metadata.",
            "operations": [
                {
                    "op": "merge_nodes",
                    "node_ids": ["obs::e2"],
                    "into_id": "canonical",
                    "metadata": {
                        "conjugation_ambiguity": "lecture evidence does not fix convention"
                    },
                }
            ],
        }
    )

    operation = patch.operations[0]
    assert isinstance(operation, MergeNodesOp)
    assert operation.metadata_update == {
        "conjugation_ambiguity": "lecture evidence does not fix convention"
    }

    revised = apply_patch(state, patch)
    target = revised.nodes["canonical"]
    assert target.metadata["field"] == "complex"
    assert target.metadata["conjugation_ambiguity"] == (
        "lecture evidence does not fix convention"
    )
    assert target.metadata["merged_from"] == ["obs::e2"]
