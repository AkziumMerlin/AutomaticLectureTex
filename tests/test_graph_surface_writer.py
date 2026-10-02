from __future__ import annotations

from types import SimpleNamespace

from automatic_lecture_tex.graph_revision import EvidenceRecord, GraphEdge, GraphNode, GraphState
from automatic_lecture_tex.graph_revision_render import graph_state_to_ir
from automatic_lecture_tex.graph_surface_writer import (
    GeneratedGraphSectionBlock,
    GeneratedGraphSectionNotes,
    _surface_writer_prompt,
    _verify_generated,
    graph_section_specs,
    write_graph_surface,
)
from automatic_lecture_tex.schemas import BlockType


class StubOrchestrator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.output_language = "ru"
        self.config = SimpleNamespace(
            state_section_writer_thinking=False,
            state_section_writer_temperature=0.6,
            state_section_writer_top_p=0.8,
            state_section_writer_top_k=20,
            state_section_writer_min_p=0.0,
            state_section_writer_presence_penalty=0.5,
            state_section_writer_repetition_penalty=1.0,
        )

    def _structured(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
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
                text="Определяется комплексная линейность функционала.",
                evidence_ids=["e1"],
            ),
            "linearity": GraphNode(
                id="linearity",
                kind="equation",
                title="Линейность",
                text="Условие комплексной линейности.",
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


def test_surface_writer_prompt_encodes_reference_handout_style():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    payload = {
        "section_id": spec.section_id,
        "title": spec.title,
        "nodes": [],
        "relations": [],
    }

    prompt = _surface_writer_prompt(payload=payload, output_language="ru")

    assert "Пусть ..." in prompt
    assert "Будем называть ... , если ..." in prompt
    assert "Заметим, что" in prompt
    assert "Do NOT create a bold/paragraph heading" in prompt
    assert "every intermediate equation" in prompt
    assert "Every mathematical symbol occurring inside prose must be in math mode" in prompt
    assert "[[MATH:<node_id>]]" in prompt
    assert "Never wrap it in $...$, $$...$$" in prompt
    assert "Never use mathematical placeholders" in prompt


def test_graph_surface_writer_merges_atoms_into_polished_definition(tmp_path):
    graph = _definition_graph()
    fallback = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")
    generated = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                title=None,
                latex=(
                    r"Пусть $X$ --- комплексное линейное пространство. "
                    r"Будем называть функционал $f\colon X\to\mathbb C$ комплексно-линейным, если "
                    r"для любых $\alpha,\beta\in\mathbb C$ и $x,y\in X$ выполняется"
                    "\n\\[\n"
                    r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)"
                    "\n\\]"
                ),
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    orchestrator = StubOrchestrator([generated])

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=fallback,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    assert len(ir.chunks) == 1
    assert len(ir.chunks[0].blocks) == 1
    block = ir.chunks[0].blocks[0]
    assert block.type == BlockType.DEFINITION
    assert "Будем называть функционал" in block.latex
    assert r"$f\colon X\to\mathbb C$" in block.latex
    assert block.source_evidence_ids == ["e1", "e2"]
    assert len(orchestrator.prompts) == 1


def test_graph_surface_writer_repairs_provenance_language(tmp_path):
    graph = _definition_graph()
    fallback = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")
    bad = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                latex="По данным OCR это определение.",
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    good = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                latex=(
                    r"Пусть $X$ --- комплексное линейное пространство. "
                    r"Будем называть $f$ комплексно-линейным, если"
                    "\n\\[\n"
                    r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)"
                    "\n\\]"
                ),
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    orchestrator = StubOrchestrator([bad, good])

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=fallback,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    assert len(orchestrator.prompts) == 2
    assert "Verification errors" in orchestrator.prompts[1]
    assert "Будем называть $f$" in ir.chunks[0].blocks[0].latex


def test_graph_surface_verifier_requires_full_node_and_formula_coverage():
    graph = _definition_graph()
    spec = graph_section_specs(graph, lecture_title="Lecture")[0]
    generated = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.PARAGRAPH,
                latex="Только текст.",
                source_node_ids=["definition"],
            )
        ]
    )

    errors = _verify_generated(spec=spec, generated=generated)

    assert any("uncovered canonical nodes: linearity" in item for item in errors)
    assert any("missing canonical formulas from: linearity" in item for item in errors)


def test_graph_surface_writer_injects_renderer_owned_formula_without_repair(tmp_path):
    graph = _definition_graph()
    fallback = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")
    generated = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                latex=(
                    r"Пусть $X$ --- комплексное линейное пространство. "
                    r"Будем называть $f$ комплексно-линейным."
                ),
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    orchestrator = StubOrchestrator([generated])

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=fallback,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    assert len(orchestrator.prompts) == 1
    assert r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)" in ir.chunks[0].blocks[0].latex
    assert "[[MATH:" not in ir.chunks[0].blocks[0].latex



def test_graph_surface_writer_normalizes_loose_wrapped_formula_markers(tmp_path):
    graph = _definition_graph()
    fallback = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")
    generated = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                latex=(
                    r"Пусть $X$ --- комплексное линейное пространство. "
                    r"Условие линейности имеет вид $$[MATH:linearity]$$."
                ),
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    orchestrator = StubOrchestrator([generated])

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=fallback,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    block = ir.chunks[0].blocks[0]
    assert len(orchestrator.prompts) == 1
    assert "[MATH:" not in block.latex
    assert "$$" not in block.latex
    assert r"f(\alpha x+\beta y)=\alpha f(x)+\beta f(y)" in block.latex


def test_graph_surface_writer_repairs_placeholder_unicode_and_bad_tex(tmp_path):
    graph = _definition_graph()
    fallback = graph_state_to_ir(graph, lecture_id="l1", title="Lecture")
    bad = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                latex=(
                    r"Пусть $f:X\to\text{...}$ и $x \tin X$, "
                    "а φ ∈ Φ. [[MATH:linearity]]"
                ),
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    good = GeneratedGraphSectionNotes(
        blocks=[
            GeneratedGraphSectionBlock(
                type=BlockType.DEFINITION,
                latex=(
                    r"Пусть $X$ --- комплексное линейное пространство и $x\in X$. "
                    r"Условие комплексной линейности имеет вид [[MATH:linearity]]."
                ),
                source_node_ids=["definition", "linearity"],
            )
        ]
    )
    orchestrator = StubOrchestrator([bad, good])

    ir = write_graph_surface(
        orchestrator,
        state=graph,
        lecture_id="l1",
        lecture_title="Lecture",
        fallback_ir=fallback,
        work=tmp_path,
        llm_config={"model": "fake"},
        force=False,
    )

    assert len(orchestrator.prompts) == 2
    assert "Verification errors" in orchestrator.prompts[1]
    block = ir.chunks[0].blocks[0]
    assert r"$x\in X$" in block.latex
    assert r"\text{...}" not in block.latex
    assert "φ" not in block.latex
    assert r"\tin" not in block.latex
