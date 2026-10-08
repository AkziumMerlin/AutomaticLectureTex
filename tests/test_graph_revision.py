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


def test_suppress_node_marks_evidence_as_handled() -> None:
    state = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="degenerate ASR")},
        nodes={
            "obs::e1": GraphNode(
                id="obs::e1",
                kind="provisional_remark",
                text="degenerate ASR",
                evidence_ids=["e1"],
            )
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="suppress-noise",
            description="Discard non-formal repeated/noisy evidence.",
            operations=[
                SuppressNodeOp(
                    op="suppress_node",
                    node_id="obs::e1",
                    reason="degenerate ASR without independent mathematical content",
                )
            ],
        ),
    )

    assert revised.nodes["obs::e1"].status == "suppressed"
    assert revised.evidence_disposition["e1"].startswith("suppressed:")
    assert metrics(revised).unexplained_evidence == 0


def test_resolving_violation_clears_tagged_failure_note() -> None:
    state = GraphState(
        evidence={},
        violations={
            "focus::common_failed::0": Violation(
                id="focus::common_failed::0",
                category="structure",
                severity=1,
                message="failed patch",
            )
        },
        notes=[
            "[focus::common_failed::0] Unapplied patch patch_x: ValueError: stale source"
        ],
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="resolve",
            description="Resolve a previously failed patch marker.",
            operations=[
                ResolveViolationOp(
                    op="resolve_violation",
                    violation_id="focus::common_failed::0",
                )
            ],
        ),
    )

    assert "focus::common_failed::0" not in revised.violations
    assert revised.notes == []


def test_consensus_keeps_invariant_conclusion_over_ambiguous_same_id_premise() -> None:
    left = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", text="premise"),
            "e2": EvidenceRecord(id="e2", text="conclusion"),
        },
        nodes={
            "variant": GraphNode(
                id="variant",
                kind="equation",
                text="plain coefficient",
                evidence_ids=["e1"],
                alternative_group="coefficient_convention",
            ),
            "invariant": GraphNode(
                id="invariant",
                kind="equation",
                text="same norm identity",
                evidence_ids=["e2"],
                derived_from=["variant"],
            ),
        },
    )
    right = left.clone()
    right.nodes["variant"].text = "conjugated coefficient"

    consensus = graph_consensus([left, right])

    assert "variant" not in consensus.nodes
    assert "invariant" in consensus.nodes
    assert consensus.nodes["invariant"].derived_from == []
    assert consensus.nodes["invariant"].metadata["frontier_ambiguous_dependencies"] == [
        "variant"
    ]
    assert any("branch-invariant invariant" in note for note in consensus.notes)


def test_stale_current_observation_ids_do_not_rollback_useful_patch() -> None:
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
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="cleanup-current-observation-id",
            description="Useful edit plus stale obs_window cleanup.",
            operations=[
                ReplaceNodeOp(
                    op="replace_node",
                    node_id="canonical",
                    text="new text",
                ),
                MergeNodesOp(
                    op="merge_nodes",
                    node_ids=[
                        "obs_window_0039_000",
                        "obs_window_0040_000",
                    ],
                    into_id="canonical",
                ),
                SuppressNodeOp(
                    op="suppress_node",
                    node_id="obs_window_0040_000",
                    reason="already absorbed",
                ),
            ],
        ),
    )

    assert revised.nodes["canonical"].text == "new text"
    assert revised.applied_patches[-1] == "cleanup-current-observation-id"


def test_merge_ignores_only_missing_provisionals_when_real_source_survives() -> None:
    state = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", text="canonical"),
            "e2": EvidenceRecord(id="e2", text="remaining observation"),
        },
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="claim",
                text="canonical",
                evidence_ids=["e1"],
            ),
            "obs_window_0044_000": GraphNode(
                id="obs_window_0044_000",
                kind="provisional_claim",
                text="remaining",
                evidence_ids=["e2"],
            ),
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="partial-stale-merge",
            description="Some provisional sources were already absorbed.",
            operations=[
                MergeNodesOp(
                    op="merge_nodes",
                    node_ids=[
                        "obs_window_0039_000",
                        "obs_window_0040_000",
                        "obs_window_0044_000",
                    ],
                    into_id="canonical",
                )
            ],
        ),
    )

    assert "obs_window_0044_000" not in revised.nodes
    assert set(revised.nodes["canonical"].evidence_ids) == {"e1", "e2"}


def test_missing_canonical_merge_source_remains_error() -> None:
    state = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="canonical")},
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="claim",
                text="canonical",
                evidence_ids=["e1"],
            )
        },
    )

    with pytest.raises(ValueError, match="merge missing nodes"):
        apply_patch(
            state,
            GraphPatch(
                id="hard-missing",
                description="Canonical source typo must still fail.",
                operations=[
                    MergeNodesOp(
                        op="merge_nodes",
                        node_ids=["missing_canonical_node"],
                        into_id="canonical",
                    )
                ],
            ),
        )


def test_replace_node_hoists_canonical_fields_from_metadata_update() -> None:
    state = GraphState(
        evidence={"e1": EvidenceRecord(id="e1", text="weak convergence")},
        nodes={
            "claim": GraphNode(
                id="claim",
                kind="claim",
                title="Overstated criterion",
                text="Coordinate convergence is equivalent to weak convergence.",
                latex=r"x_m \\rightharpoonup 0 \\iff (x_m,e_n)\\to0",
                evidence_ids=["e1"],
                metadata={"source": "revision"},
            )
        },
    )

    patch = GraphPatch.model_validate(
        {
            "id": "repair-l2-criterion",
            "description": "Model emitted a full replacement inside metadata_update.",
            "operations": [
                {
                    "op": "replace_node",
                    "node_id": "claim",
                    "title": None,
                    "text": None,
                    "latex": None,
                    "metadata_update": {
                        "title": "Weak convergence criterion with boundedness",
                        "text": "The converse requires sup_m ||x_m|| < infinity.",
                        "latex": (
                            r"x_m \\rightharpoonup 0 \\Rightarrow (x_m,e_n)\\to0; "
                            r"\\sup_m\\|x_m\\|<\\infty"
                        ),
                        "aliases": ["bounded coordinate criterion"],
                        "review_note": "mathematical correction",
                    },
                }
            ],
        }
    )

    operation = patch.operations[0]
    assert isinstance(operation, ReplaceNodeOp)
    assert operation.text == "The converse requires sup_m ||x_m|| < infinity."
    assert operation.aliases == ["bounded coordinate criterion"]
    assert operation.metadata_update == {"review_note": "mathematical correction"}

    revised = apply_patch(state, patch)
    node = revised.nodes["claim"]
    assert node.title == "Weak convergence criterion with boundedness"
    assert node.text == "The converse requires sup_m ||x_m|| < infinity."
    assert r"\\sup_m\\|x_m\\|<\\infty" in (node.latex or "")
    assert node.aliases == ["bounded coordinate criterion"]
    assert node.metadata == {
        "source": "revision",
        "review_note": "mathematical correction",
    }
    assert "text" not in node.metadata
    assert "latex" not in node.metadata


def test_merge_node_hoists_canonical_fields_from_metadata_update() -> None:
    state = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", text="old canonical"),
            "e2": EvidenceRecord(id="e2", text="new observation"),
        },
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="claim",
                title="Old title",
                text="Old text",
                evidence_ids=["e1"],
            ),
            "obs::e2": GraphNode(
                id="obs::e2",
                kind="provisional_claim",
                text="new observation",
                evidence_ids=["e2"],
            ),
        },
    )

    patch = GraphPatch.model_validate(
        {
            "id": "merge-and-rewrite",
            "description": "Canonicalize while absorbing one observation.",
            "operations": [
                {
                    "op": "merge_nodes",
                    "node_ids": ["obs::e2"],
                    "into_id": "canonical",
                    "metadata_update": {
                        "kind": "theorem",
                        "title": "Correct title",
                        "text": "Correct canonical text.",
                        "latex": r"f(x)=0",
                        "aliases": ["canonical alias"],
                        "editor_note": "keep as metadata",
                    },
                }
            ],
        }
    )

    operation = patch.operations[0]
    assert isinstance(operation, MergeNodesOp)
    assert operation.kind == "theorem"
    assert operation.text == "Correct canonical text."
    assert operation.metadata_update == {"editor_note": "keep as metadata"}

    revised = apply_patch(state, patch)
    node = revised.nodes["canonical"]
    assert node.kind == "theorem"
    assert node.title == "Correct title"
    assert node.text == "Correct canonical text."
    assert node.latex == r"f(x)=0"
    assert set(node.evidence_ids) == {"e1", "e2"}
    assert "canonical alias" in node.aliases
    assert node.metadata["editor_note"] == "keep as metadata"
    assert "text" not in node.metadata
