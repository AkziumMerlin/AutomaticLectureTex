from automatic_lecture_tex.graph_revision import (
    EvidenceRecord,
    GraphAmbiguity,
    GraphEdge,
    GraphNode,
    GraphPatch,
    GraphState,
    MergeNodesOp,
    ReplaceNodeOp,
    apply_patch,
)
from automatic_lecture_tex.reader_semantics import (
    ambiguity_policy,
    build_occurrence_plan,
    infer_topic_memberships,
    propagate_unresolved,
    semantic_candidate_units,
)


def test_semantic_units_prefer_explicit_semantic_text_and_hide_audit_prose() -> None:
    node = GraphNode(
        id="n",
        kind="claim",
        title="Непрерывность функционала",
        text="OCR читается неустойчиво. На доске видна другая буква.",
        semantic_text="Функционал непрерывен.",
        evidence_ids=["e1"],
        provenance_notes=["OCR ambiguous"],
    )

    units = semantic_candidate_units(node)

    assert "Функционал непрерывен." in units
    assert all("OCR" not in item and "доск" not in item.lower() for item in units)


def test_typed_blocking_ambiguity_overrides_reader_assertion() -> None:
    node = GraphNode(
        id="riesz",
        kind="claim",
        semantic_text="Представляющий вектор задаёт функционал.",
        latex=r"f(x)=(x,y_f)",
        ambiguities=[
            GraphAmbiguity(
                kind="convention",
                blocking=True,
                message="Не зафиксирована конвенция линейности скалярного произведения.",
                affects_fields=["latex"],
            )
        ],
    )

    policy = ambiguity_policy(node)

    assert policy.disposition_override == "unresolved"
    assert any("конвенц" in item.lower() for item in policy.blocking_reasons)


def test_legacy_riesz_convention_metadata_is_blocking() -> None:
    node = GraphNode(
        id="riesz",
        kind="claim",
        latex=r"y_f=\frac{f(z_f)}{\|z_f\|^2}z_f",
        metadata={
            "complex_convention": "not_fixed",
            "board_formula_note": "possible conjugation depends on convention",
        },
    )

    policy = ambiguity_policy(node)

    assert policy.disposition_override == "unresolved"
    assert len(policy.blocking_reasons) >= 1


def test_unresolved_dependency_propagates_to_dependent_fact() -> None:
    state = GraphState(
        evidence={},
        nodes={
            "premise": GraphNode(id="premise", kind="claim"),
            "conclusion": GraphNode(
                id="conclusion",
                kind="claim",
                derived_from=["premise"],
            ),
        },
    )
    dispositions = {"premise": "unresolved", "conclusion": "render"}
    reasons = {"premise": ["ambiguous"], "conclusion": []}

    propagate_unresolved(state, dispositions, reasons)

    assert dispositions["conclusion"] == "unresolved"
    assert "premise" in reasons["conclusion"][0]


def test_semantic_topic_relation_beats_chronological_fallback() -> None:
    state = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", start=10.0, end=11.0),
            "e2": EvidenceRecord(id="e2", start=100.0, end=101.0),
            "e3": EvidenceRecord(id="e3", start=200.0, end=201.0),
        },
        nodes={
            "topic_weak": GraphNode(id="topic_weak", kind="topic", evidence_ids=["e1"]),
            "topic_l2": GraphNode(id="topic_l2", kind="topic", evidence_ids=["e2"]),
            "weak_basis": GraphNode(id="weak_basis", kind="claim", evidence_ids=["e1"]),
            "l2_fact": GraphNode(id="l2_fact", kind="claim", evidence_ids=["e2"]),
            "late_weak_proof": GraphNode(
                id="late_weak_proof",
                kind="proof_step",
                evidence_ids=["e3"],
            ),
        },
        edges=[
            GraphEdge(source="topic_weak", target="weak_basis", relation="contains"),
            GraphEdge(source="topic_l2", target="l2_fact", relation="contains"),
            GraphEdge(source="late_weak_proof", target="weak_basis", relation="refines"),
        ],
    )

    memberships = infer_topic_memberships(
        state,
        renderable_node_ids={"weak_basis", "l2_fact", "late_weak_proof"},
        topic_ids={"topic_weak", "topic_l2"},
        legacy_membership={
            "weak_basis": "topic_weak",
            "l2_fact": "topic_l2",
            "late_weak_proof": "topic_l2",
        },
    )

    assert memberships["late_weak_proof"] == {"topic_weak"}


def test_occurrence_plan_uses_relation_local_time_for_shared_proof_fact() -> None:
    state = GraphState(
        evidence={
            "early": EvidenceRecord(id="early", start=10.0, end=11.0),
            "u": EvidenceRecord(id="u", start=100.0, end=101.0),
            "hb": EvidenceRecord(id="hb", start=101.0, end=102.0),
            "reconstruct": EvidenceRecord(id="reconstruct", start=102.0, end=103.0),
        },
        nodes={
            "theorem": GraphNode(
                id="theorem",
                kind="theorem",
                title="Комплексный Хан--Банах",
                evidence_ids=["hb"],
            ),
            "define_u": GraphNode(
                id="define_u",
                kind="proof_step",
                latex=r"u=\operatorname{Re}h",
                evidence_ids=["u"],
            ),
            "real_hb": GraphNode(
                id="real_hb",
                kind="proof_step",
                evidence_ids=["hb"],
            ),
            "complex_reconstruction": GraphNode(
                id="complex_reconstruction",
                kind="equation",
                latex=r"f(x)=w(x)-iw(ix)",
                evidence_ids=["early"],
            ),
        },
        edges=[
            GraphEdge(
                source="define_u",
                target="theorem",
                relation="supports",
                evidence_ids=["u"],
            ),
            GraphEdge(
                source="real_hb",
                target="theorem",
                relation="supports",
                evidence_ids=["hb"],
            ),
            GraphEdge(
                source="complex_reconstruction",
                target="theorem",
                relation="supports",
                evidence_ids=["reconstruct"],
            ),
            GraphEdge(source="real_hb", target="define_u", relation="uses"),
            GraphEdge(
                source="complex_reconstruction",
                target="real_hb",
                relation="uses",
            ),
        ],
    )

    plan = build_occurrence_plan(
        state,
        section_id="hb",
        ordered_node_ids=[
            "complex_reconstruction",
            "define_u",
            "real_hb",
            "theorem",
        ],
        dispositions={
            "theorem": "render",
            "define_u": "render",
            "real_hb": "render",
            "complex_reconstruction": "render",
        },
    )

    proof = next(block for block in plan.blocks if block.role == "proof")
    assert proof.node_ids == (
        "define_u",
        "real_hb",
        "complex_reconstruction",
    )
    assert len(
        [
            item
            for item in plan.occurrences
            if item.node_id == "complex_reconstruction"
        ]
    ) == 1
    reconstruction_occurrence = next(
        item
        for item in plan.occurrences
        if item.node_id == "complex_reconstruction"
    )
    assert reconstruction_occurrence.anchor_time > 100.0


def test_incomplete_proof_is_not_rendered_as_complete_proof_block() -> None:
    state = GraphState(
        evidence={},
        nodes={
            "theorem": GraphNode(id="theorem", kind="theorem"),
            "step_ok": GraphNode(id="step_ok", kind="proof_step"),
            "step_bad": GraphNode(id="step_bad", kind="proof_step"),
        },
        edges=[
            GraphEdge(source="step_ok", target="theorem", relation="supports"),
            GraphEdge(source="step_bad", target="theorem", relation="supports"),
        ],
    )

    plan = build_occurrence_plan(
        state,
        section_id="s",
        ordered_node_ids=["theorem", "step_ok", "step_bad"],
        dispositions={
            "theorem": "render",
            "step_ok": "render",
            "step_bad": "unresolved",
        },
    )

    assert not any(block.role == "proof" for block in plan.blocks)
    assert any("step_bad" in item for item in plan.incomplete_proofs)


def test_graph_merge_preserves_audit_notes_and_typed_ambiguities() -> None:
    state = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1"),
            "e2": EvidenceRecord(id="e2"),
        },
        nodes={
            "canonical": GraphNode(
                id="canonical",
                kind="claim",
                evidence_ids=["e1"],
                provenance_notes=["first provenance"],
            ),
            "obs::e2": GraphNode(
                id="obs::e2",
                kind="claim",
                evidence_ids=["e2"],
                reconstruction_notes=["second reconstruction"],
                ambiguities=[
                    GraphAmbiguity(
                        kind="reading",
                        blocking=False,
                        message="symbol reading uncertain",
                    )
                ],
            ),
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="merge",
            description="merge",
            operations=[
                MergeNodesOp(
                    op="merge_nodes",
                    node_ids=["obs::e2"],
                    into_id="canonical",
                )
            ],
        ),
    )

    node = revised.nodes["canonical"]
    assert node.provenance_notes == ["first provenance"]
    assert node.reconstruction_notes == ["second reconstruction"]
    assert len(node.ambiguities) == 1


def test_replace_node_can_set_semantic_text_and_clear_ambiguities() -> None:
    state = GraphState(
        evidence={},
        nodes={
            "n": GraphNode(
                id="n",
                kind="claim",
                ambiguities=[
                    GraphAmbiguity(
                        kind="formula",
                        blocking=True,
                        message="old ambiguity",
                    )
                ],
            )
        },
    )

    revised = apply_patch(
        state,
        GraphPatch(
            id="resolve",
            description="resolve",
            operations=[
                ReplaceNodeOp(
                    op="replace_node",
                    node_id="n",
                    semantic_text="Однозначно восстановленное утверждение.",
                    ambiguities=[],
                )
            ],
        ),
    )

    assert revised.nodes["n"].semantic_text == "Однозначно восстановленное утверждение."
    assert revised.nodes["n"].ambiguities == []
