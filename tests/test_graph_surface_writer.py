from __future__ import annotations

from types import SimpleNamespace

import json
import pytest

from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphEdge, GraphNode, GraphState
from automatic_lecture_tex.graph_revision_render import graph_state_to_ir
from automatic_lecture_tex.graph_surface_writer import graph_section_specs, write_graph_surface
from automatic_lecture_tex.reader_surface import (
    DraftGeneratedReaderBlock,
    DraftReaderSurfaceSegment,
    GeneratedReaderBlock,
    PlannedReaderBlock,
    ReaderDiscoursePlan,
    ReaderGroundingIssue,
    ReaderGroundingReview,
    ReaderProjectionChoice,
    ReaderProjectionChoices,
    ReaderSurfaceSegment,
    _candidate_units,
    _canonicalize_plan,
    _deterministic_block_fallback,
    _grounding_prompt,
    _materialize_projection,
    _normalize_draft_block,
    _projection_input,
    _reader_candidate_units,
    _verify_block,
    _verify_plan,
    _verify_projection_choices,
)
from automatic_lecture_tex.schemas import BlockType


class StubOrchestrator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.operations: list[str] = []
        self.output_language = "ru"
        self.config = SimpleNamespace(
            state_section_writer_thinking=False,
            state_section_writer_temperature=0.2,
            state_section_writer_top_p=0.9,
            state_section_writer_top_k=20,
            state_section_writer_min_p=0.0,
            state_section_writer_presence_penalty=0.0,
            state_section_writer_repetition_penalty=1.0,
        )

    def _structured(self, prompt, schema, **kwargs):
        del schema
        self.prompts.append(prompt)
        self.operations.append(kwargs["operation"])
        return self.responses.pop(0)


def _definition_graph() -> GraphState:
    return GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", start=0.0, end=1.0, text="definition"),
            "e2": EvidenceRecord(id="e2", start=1.0, end=2.0, text="formula"),
        },
        nodes={
            "topic": GraphNode(
                id="topic",
                kind="topic",
                title="Комплексно-линейные функционалы",
                evidence_ids=["e1"],
            ),
            "definition": GraphNode(
                id="definition",
                kind="definition",
                title="Комплексно-линейный функционал",
                text=(
                    "Функционал называется комплексно-линейным. "
                    "На доске ниже записано определяющее тождество."
                ),
                evidence_ids=["e1"],
            ),
            "linearity": GraphNode(
                id="linearity",
                kind="equation",
                title="Условие комплексной линейности",
                text="Условие комплексной линейности задаётся равенством.",
                latex=r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)",
                evidence_ids=["e2"],
                derived_from=["definition"],
            ),
        },
        edges=[
            GraphEdge(source="topic", target="definition", relation="contains"),
            GraphEdge(source="topic", target="linearity", relation="contains"),
        ],
    )


def test_reader_projection_prefilters_provenance_before_model_selection():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    raw_units = _candidate_units(definition)
    safe_units = _reader_candidate_units(definition)
    payload = _projection_input(graph, spec)
    definition_payload = next(item for item in payload["nodes"] if item["id"] == "definition")

    assert raw_units[0] == "Комплексно-линейный функционал"
    assert "Функционал называется комплексно-линейным." in safe_units
    assert any("доске" in unit for unit in raw_units)
    assert all("доске" not in unit for unit in safe_units)
    assert definition_payload["candidate_units"] == safe_units


def test_projection_materializes_only_selected_source_units():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    units = _reader_candidate_units(definition)
    semantic_index = units.index("Функционал называется комплексно-линейным.")

    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[semantic_index],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )

    projection = _materialize_projection(graph, spec, choices)
    fact = next(item for item in projection.facts if item.node_id == "definition")

    assert fact.statement == "Функционал называется комплексно-линейным."
    assert "доске" not in fact.statement
    expression = projection.expressions[0]
    assert expression.id == "expr::linearity"
    assert expression.latex == r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)"


def test_discourse_plan_requires_exact_ordered_coverage():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id=node.id,
                disposition="render",
                selected_unit_indices=[0],
            )
            for node in spec.nodes
        ]
    )
    projection = _materialize_projection(graph, spec, choices)

    reversed_plan = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="b0",
                type=BlockType.DEFINITION,
                purpose="Определение",
                node_ids=["linearity", "definition"],
            )
        ]
    )

    assert _verify_plan(projection, reversed_plan)



def test_discourse_plan_canonicalization_restores_reordered_groups():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id=node.id,
                disposition="render",
                selected_unit_indices=[0],
            )
            for node in spec.nodes
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    expected = [
        fact.node_id for fact in projection.facts if fact.disposition == "render"
    ]

    broken = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="later",
                type=BlockType.EQUATION,
                purpose="Формула.",
                node_ids=["linearity"],
            ),
            PlannedReaderBlock(
                block_id="earlier",
                type=BlockType.DEFINITION,
                purpose="Определение.",
                node_ids=["definition"],
            ),
        ]
    )

    normalized = _canonicalize_plan(projection, broken)

    assert [node_id for block in normalized.blocks for node_id in block.node_ids] == expected
    assert _verify_plan(projection, normalized) == []


def test_discourse_plan_canonicalization_fills_missing_nodes():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id=node.id,
                disposition="render",
                selected_unit_indices=[0],
            )
            for node in spec.nodes
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    expected = [
        fact.node_id for fact in projection.facts if fact.disposition == "render"
    ]

    broken = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="partial",
                type=BlockType.DEFINITION,
                purpose="Определение.",
                node_ids=[expected[0]],
            )
        ]
    )

    normalized = _canonicalize_plan(projection, broken)

    assert [node_id for block in normalized.blocks for node_id in block.node_ids] == expected
    assert _verify_plan(projection, normalized) == []


def test_discourse_plan_canonicalization_breaks_duplicate_memberships_safely():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id=node.id,
                disposition="render",
                selected_unit_indices=[0],
            )
            for node in spec.nodes
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    expected = [
        fact.node_id for fact in projection.facts if fact.disposition == "render"
    ]

    broken = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="a",
                type=BlockType.PARAGRAPH,
                purpose="Первая группа.",
                node_ids=expected,
            ),
            PlannedReaderBlock(
                block_id="b",
                type=BlockType.REMARK,
                purpose="Дубликат.",
                node_ids=[expected[0]],
            ),
        ]
    )

    normalized = _canonicalize_plan(projection, broken)

    assert [node_id for block in normalized.blocks for node_id in block.node_ids] == expected
    assert len({block.block_id for block in normalized.blocks}) == len(normalized.blocks)
    assert _verify_plan(projection, normalized) == []

def test_typed_inline_math_cannot_smuggle_a_new_relation():
    with pytest.raises(ValueError, match="cannot contain mathematical relations"):
        ReaderSurfaceSegment(
            kind="inline_math",
            source_node_ids=["definition"],
            latex=r"H=L_2",
        )


def test_block_requires_all_host_owned_expressions():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id=node.id,
                disposition="render",
                selected_unit_indices=[0],
            )
            for node in spec.nodes
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    block = PlannedReaderBlock(
        block_id="definition",
        type=BlockType.DEFINITION,
        purpose="Определить функционал.",
        node_ids=["definition", "linearity"],
    )
    generated = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["definition", "linearity"],
                text="Определим комплексно-линейный функционал.",
            )
        ]
    )

    errors = _verify_block(projection=projection, block=block, generated=generated)

    assert any("omits canonical expressions" in error for error in errors)


def test_surface_pipeline_projection_plan_writer_end_to_end(tmp_path):
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    definition_units = _reader_candidate_units(definition)
    semantic_index = definition_units.index("Функционал называется комплексно-линейным.")

    projection = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[semantic_index],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    plan = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="definition",
                type=BlockType.DEFINITION,
                purpose="Дать определение и точное условие линейности.",
                node_ids=["definition", "linearity"],
            )
        ]
    )
    generated = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["definition"],
                text="Будем называть функционал комплексно-линейным, если выполняется условие",
            ),
            ReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["linearity"],
                expression_id="expr::linearity",
                display=True,
            ),
        ]
    )
    orchestrator = StubOrchestrator(
        [projection, generated, ReaderGroundingReview(issues=[])]
    )
    metadata_ir = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=metadata_ir,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    assert orchestrator.operations == [
        "graph_reader_projection",
        "graph_block_write",
        "graph_block_grounding_review",
    ]
    assert len(ir.chunks) == 1
    assert len(ir.chunks[0].blocks) == 1
    output = ir.chunks[0].blocks[0].latex
    assert "доске" not in output
    assert r"\[" in output
    assert r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)" in output
    assert ir.chunks[0].blocks[0].source_claim_ids == ["definition", "linearity"]
    section_root = tmp_path / "graph_surface_writer" / "section_000"
    assert (section_root / "reader_projection.json").is_file()
    assert (section_root / "plan.json").is_file()
    assert (section_root / "block_000.json").is_file()


def test_projection_can_omit_pure_audit_node_without_surface_fallback(tmp_path):
    graph = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", start=0.0, end=1.0, text="audit"),
            "e2": EvidenceRecord(id="e2", start=1.0, end=2.0, text="claim"),
        },
        nodes={
            "audit": GraphNode(
                id="audit",
                kind="remark",
                title="Неоднозначность записи",
                text="В окне OCR не позволяет надёжно прочитать строку.",
                evidence_ids=["e1"],
            ),
            "claim": GraphNode(
                id="claim",
                kind="claim",
                title="Непрерывность функционала",
                text="Функционал непрерывен.",
                evidence_ids=["e2"],
            ),
        },
    )
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    audit_units = _candidate_units(graph.nodes["audit"])
    claim_units = _candidate_units(graph.nodes["claim"])
    projection = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="audit",
                disposition="omit",
                selected_unit_indices=[],
            ),
            ReaderProjectionChoice(
                node_id="claim",
                disposition="render",
                selected_unit_indices=[claim_units.index("Функционал непрерывен.")],
            ),
        ]
    )
    plan = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="claim",
                type=BlockType.PARAGRAPH,
                purpose="Сформулировать утверждение.",
                node_ids=["claim"],
            )
        ]
    )
    generated = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["claim"],
                text="Функционал непрерывен.",
            )
        ]
    )
    orchestrator = StubOrchestrator(
        [projection, generated, ReaderGroundingReview(issues=[])]
    )
    metadata_ir = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=metadata_ir,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    rendered = "\n".join(block.latex for block in ir.chunks[0].blocks)
    assert "OCR" not in rendered
    assert "окне" not in rendered
    summary = (
        tmp_path / "graph_surface_writer" / "section_000" / "summary.json"
    ).read_text(encoding="utf-8")
    assert '"audit"' in summary



def test_grounding_issue_forces_block_repair(tmp_path):
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    semantic_index = _reader_candidate_units(definition).index(
        "Функционал называется комплексно-линейным."
    )
    projection = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[semantic_index],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    plan = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="definition",
                type=BlockType.DEFINITION,
                purpose="Дать определение.",
                node_ids=["definition", "linearity"],
            )
        ]
    )
    bad = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["definition"],
                text="Пространство совпадает с эл два.",
            ),
            ReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["linearity"],
                expression_id="expr::linearity",
                display=True,
            ),
        ]
    )
    good = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["definition"],
                text="Рассматривается комплексно-линейный функционал.",
            ),
            ReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["linearity"],
                expression_id="expr::linearity",
                display=True,
            ),
        ]
    )
    orchestrator = StubOrchestrator(
        [
            projection,
            bad,
            ReaderGroundingReview(
                issues=[
                    ReaderGroundingIssue(
                        segment_index=0,
                        reason="Совпадение пространств не следует из cited fact.",
                    )
                ]
            ),
            good,
            ReaderGroundingReview(issues=[]),
        ]
    )
    metadata_ir = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=metadata_ir,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=True,
    )

    assert "совпадает" not in ir.chunks[0].blocks[0].latex
    assert orchestrator.operations.count("graph_block_write_repair") == 1
    assert orchestrator.operations.count("graph_block_grounding_review") == 2



def test_provenance_only_node_without_expression_must_be_omitted():
    graph = GraphState(
        evidence={
            "e1": EvidenceRecord(id="e1", start=0.0, end=1.0, text="audit"),
            "e2": EvidenceRecord(id="e2", start=1.0, end=2.0, text="claim"),
        },
        nodes={
            "audit": GraphNode(
                id="audit",
                kind="remark",
                title="Комментарий на доске",
                text="Лектор указывает на неоднозначное чтение кадра OCR.",
                evidence_ids=["e1"],
            ),
            "claim": GraphNode(
                id="claim",
                kind="claim",
                title="Непрерывность функционала",
                text="Функционал непрерывен.",
                evidence_ids=["e2"],
            ),
        },
    )
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]

    assert _reader_candidate_units(graph.nodes["audit"]) == []

    bad = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="audit",
                disposition="render",
                selected_unit_indices=[],
            ),
            ReaderProjectionChoice(
                node_id="claim",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    errors = _verify_projection_choices(spec=spec, choices=bad)

    assert any("must be omitted" in error for error in errors)


def test_materialization_indices_are_against_prefiltered_units():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    safe_units = _reader_candidate_units(definition)
    semantic_index = safe_units.index("Функционал называется комплексно-линейным.")

    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[semantic_index],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )

    projection = _materialize_projection(graph, spec, choices)
    fact = next(item for item in projection.facts if item.node_id == "definition")

    assert fact.statement == "Функционал называется комплексно-линейным."
    assert "доске" not in fact.statement



def test_grounding_prompt_includes_canonical_expression_latex():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    semantic_index = _reader_candidate_units(definition).index(
        "Функционал называется комплексно-линейным."
    )
    projection = _materialize_projection(
        graph,
        spec,
        ReaderProjectionChoices(
            choices=[
                ReaderProjectionChoice(
                    node_id="definition",
                    disposition="render",
                    selected_unit_indices=[semantic_index],
                ),
                ReaderProjectionChoice(
                    node_id="linearity",
                    disposition="render",
                    selected_unit_indices=[0],
                ),
            ]
        ),
    )
    block = PlannedReaderBlock(
        block_id="definition",
        type=BlockType.DEFINITION,
        purpose="Дать определение.",
        node_ids=["definition", "linearity"],
    )
    generated = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["linearity"],
                text="Выполняется условие комплексной линейности.",
            ),
            ReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["linearity"],
                expression_id="expr::linearity",
                display=True,
            ),
        ]
    )

    prompt = _grounding_prompt(
        projection=projection,
        block=block,
        generated=generated,
        output_language="ru",
    )

    assert r'"latex":"f(\\alpha x+\\beta y)=\\alpha f(x)+\\beta f(y)"' in prompt
    assert '"expressions":[{"id":"expr::linearity","latex":' in prompt
    assert "cited_facts.expressions" in prompt


def test_deterministic_block_fallback_uses_exact_facts_and_expressions():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    semantic_index = _reader_candidate_units(definition).index(
        "Функционал называется комплексно-линейным."
    )
    projection = _materialize_projection(
        graph,
        spec,
        ReaderProjectionChoices(
            choices=[
                ReaderProjectionChoice(
                    node_id="definition",
                    disposition="render",
                    selected_unit_indices=[semantic_index],
                ),
                ReaderProjectionChoice(
                    node_id="linearity",
                    disposition="render",
                    selected_unit_indices=[0],
                ),
            ]
        ),
    )
    block = PlannedReaderBlock(
        block_id="definition",
        type=BlockType.DEFINITION,
        purpose="Дать определение.",
        node_ids=["definition", "linearity"],
    )

    generated = _deterministic_block_fallback(projection, block)

    assert _verify_block(projection=projection, block=block, generated=generated) == []
    assert any(
        segment.kind == "text"
        and segment.text == "Функционал называется комплексно-линейным."
        for segment in generated.segments
    )
    assert any(
        segment.kind == "expression"
        and segment.expression_id == "expr::linearity"
        for segment in generated.segments
    )


def test_second_grounding_failure_falls_back_deterministically(tmp_path):
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    definition = next(node for node in spec.nodes if node.id == "definition")
    semantic_index = _reader_candidate_units(definition).index(
        "Функционал называется комплексно-линейным."
    )
    projection_choice = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[semantic_index],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    plan = ReaderDiscoursePlan(
        blocks=[
            PlannedReaderBlock(
                block_id="definition",
                type=BlockType.DEFINITION,
                purpose="Дать определение.",
                node_ids=["definition", "linearity"],
            )
        ]
    )
    bad = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["definition"],
                text="Добавляется неподдержанное утверждение.",
            ),
            ReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["linearity"],
                expression_id="expr::linearity",
                display=True,
            ),
        ]
    )
    repaired_but_bad = GeneratedReaderBlock(
        segments=[
            ReaderSurfaceSegment(
                kind="text",
                source_node_ids=["definition"],
                text="Снова добавляется неподдержанное утверждение.",
            ),
            ReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["linearity"],
                expression_id="expr::linearity",
                display=True,
            ),
        ]
    )
    orchestrator = StubOrchestrator(
        [
            projection_choice,
            bad,
            ReaderGroundingReview(
                issues=[ReaderGroundingIssue(segment_index=0, reason="unsupported")]
            ),
            repaired_but_bad,
            ReaderGroundingReview(
                issues=[ReaderGroundingIssue(segment_index=0, reason="still unsupported")]
            ),
        ]
    )
    metadata_ir = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=metadata_ir,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=True,
    )

    rendered = ir.chunks[0].blocks[0].latex
    assert "неподдержанное" not in rendered
    assert "Функционал называется комплексно-линейным." in rendered
    assert r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)" in rendered
    payload = json.loads(
        (
            tmp_path
            / "graph_surface_writer"
            / "section_000"
            / "block_000.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["deterministic_fallback"] is True



def test_reader_block_draft_ignores_extra_payload_on_canonical_expression():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[0],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    block = PlannedReaderBlock(
        block_id="b",
        type=BlockType.PARAGRAPH,
        purpose="Render exact formula.",
        node_ids=["definition", "linearity"],
    )
    draft = DraftGeneratedReaderBlock(
        segments=[
            DraftReaderSurfaceSegment(
                kind="expression",
                source_node_ids=["definition", "linearity"],
                expression_id="expr::linearity",
                latex=r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)",
                text="redundant",
                display=True,
            )
        ]
    )

    normalized = _normalize_draft_block(projection, block, draft)

    assert normalized is not None
    assert len(normalized.segments) == 1
    segment = normalized.segments[0]
    assert segment.kind == "expression"
    assert segment.expression_id == "expr::linearity"
    assert segment.source_node_ids == ["linearity"]
    assert segment.latex is None
    assert segment.text is None


def test_reader_block_draft_promotes_exact_relation_to_canonical_expression():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[0],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    block = PlannedReaderBlock(
        block_id="b",
        type=BlockType.PARAGRAPH,
        purpose="Render exact formula.",
        node_ids=["definition", "linearity"],
    )
    draft = DraftGeneratedReaderBlock(
        segments=[
            DraftReaderSurfaceSegment(
                kind="inline_math",
                source_node_ids=["linearity"],
                latex=r" f(\alpha x + \beta y) = \alpha f(x) + \beta f(y) ",
            )
        ]
    )

    normalized = _normalize_draft_block(projection, block, draft)

    assert normalized is not None
    assert normalized.segments[0].kind == "expression"
    assert normalized.segments[0].expression_id == "expr::linearity"


def test_reader_block_draft_drops_uncanonical_relation_instead_of_inventing_math():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    choices = ReaderProjectionChoices(
        choices=[
            ReaderProjectionChoice(
                node_id="definition",
                disposition="render",
                selected_unit_indices=[0],
            ),
            ReaderProjectionChoice(
                node_id="linearity",
                disposition="render",
                selected_unit_indices=[0],
            ),
        ]
    )
    projection = _materialize_projection(graph, spec, choices)
    block = PlannedReaderBlock(
        block_id="b",
        type=BlockType.PARAGRAPH,
        purpose="Do not invent relations.",
        node_ids=["definition", "linearity"],
    )
    draft = DraftGeneratedReaderBlock(
        segments=[
            DraftReaderSurfaceSegment(
                kind="inline_math",
                source_node_ids=["definition"],
                latex=r"\varphi \in \Phi",
            )
        ]
    )

    normalized = _normalize_draft_block(projection, block, draft)

    assert normalized is None
