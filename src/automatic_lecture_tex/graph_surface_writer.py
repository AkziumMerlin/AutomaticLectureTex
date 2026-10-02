from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from .generated_notes import GeneratedStateSectionBlock, GeneratedStateSectionNotes
from .graph_revision import GraphNode, GraphState
from .graph_revision_render import (
    _canonical_surface_text,
    _is_renderable,
    _node_times,
    _order_nodes,
    _section_assignment,
)
from .schemas import ChunkNotes, LectureIR, NoteBlock
from .util import atomic_json_dump, stable_hash

logger = logging.getLogger(__name__)

GRAPH_SURFACE_WRITER_VERSION = 1

_AUDIT_LANGUAGE = re.compile(
    r"\\b(?:ASR|OCR|доск(?:а|е|и|у|ой)|кадр(?:е|ы|ов)?|окн(?:о|е|а)|"
    r"лектор|видео|распознан|реконструкц|уверенност|provenance)\\b",
    re.IGNORECASE,
)
_UNICODE_MATH = re.compile(
    r"[∀∃∈∉∋∑∏∫√∞≤≥≠≈≡→←↔⇒⇔⊂⊃⊆⊇∩∪ℂℝℕℤℚ"
    r"αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)


class GeneratedGraphSectionBlock(GeneratedStateSectionBlock):
    source_node_ids: list[str] = Field(min_length=1)


class GeneratedGraphSectionNotes(GeneratedStateSectionNotes):
    blocks: list[GeneratedGraphSectionBlock] = Field(default_factory=list)


@dataclass
class GraphSectionSpec:
    section_id: str
    title: str
    start: float
    end: float
    nodes: list[GraphNode]


def graph_section_specs(
    state: GraphState,
    *,
    lecture_title: str,
) -> list[GraphSectionSpec]:
    renderable = _order_nodes(
        state,
        [node for node in state.nodes.values() if _is_renderable(node)],
    )
    topics, grouped = _section_assignment(state, renderable)

    pairs: list[tuple[GraphNode | None, list[GraphNode], str]] = []
    if not topics:
        pairs.append((None, grouped.get("__lecture__", []), "__lecture__"))
    else:
        if grouped.get("__prelude__"):
            pairs.append((None, grouped["__prelude__"], "__prelude__"))
        for topic in topics:
            nodes = grouped.get(topic.id, [])
            if nodes:
                pairs.append((topic, nodes, topic.id))

    specs: list[GraphSectionSpec] = []
    for topic, nodes, section_id in pairs:
        ordered = _order_nodes(state, nodes)
        if not ordered:
            continue
        starts: list[float] = []
        ends: list[float] = []
        for node in ordered:
            start, end = _node_times(state, node)
            if start != float("inf"):
                starts.append(start)
                ends.append(end)
        title = (
            _canonical_surface_text(topic.title)
            if topic is not None
            else ("Начало лекции" if topics else lecture_title)
        )
        specs.append(
            GraphSectionSpec(
                section_id=section_id,
                title=title,
                start=min(starts) if starts else 0.0,
                end=max(ends) if ends else 0.0,
                nodes=ordered,
            )
        )
    return specs


def _section_payload(state: GraphState, spec: GraphSectionSpec) -> dict[str, Any]:
    node_ids = {node.id for node in spec.nodes}
    return {
        "section_id": spec.section_id,
        "title": spec.title,
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "title": node.title,
                "text": node.text,
                "latex": node.latex,
                "derived_from": [
                    dependency
                    for dependency in node.derived_from
                    if dependency in node_ids
                ],
            }
            for node in spec.nodes
        ],
        "relations": [
            {
                "source": edge.source,
                "relation": edge.relation,
                "target": edge.target,
            }
            for edge in state.edges
            if edge.source in node_ids and edge.target in node_ids
        ],
    }


def _surface_writer_prompt(
    *,
    payload: dict[str, Any],
    output_language: str,
) -> str:
    return f"""Write one section of polished university mathematics lecture notes from an
already-canonical mathematical graph.

CANONICAL SECTION GRAPH:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

The graph is the only source of mathematical facts. Do not use outside knowledge to correct,
complete, strengthen, or reinterpret it. Do not mention the graph, reconstruction, ASR, OCR,
frames, a board, a lecturer, confidence, or provenance.

STYLE CONTRACT:
- Write dense but readable Russian university notes, in the style of a carefully typeset functional
  analysis handout rather than a graph dump.
- Definitions must be self-contained mathematical sentences. Introduce the ambient objects first
  ("Пусть ..."), then use normal mathematical phrasing such as "Будем называть ... , если ...".
- Theorem/lemma/proposition statements must be complete before their proofs.
- Proofs must read as connected arguments: use short logical transitions such as "Заметим, что",
  "Покажем", "Поскольку", "Следовательно", "А значит", "Таким образом" where they are supported by
  the canonical steps.
- Merge adjacent atomic graph nodes into coherent blocks. Do NOT create a bold/paragraph heading for
  every intermediate equation or algebraic manipulation. Titles like "Снятие фазы", "Оценка ...",
  "Полярное представление ..." are normally prose inside a proof/remark, not standalone headings.
- Keep useful displayed equations, but surround nontrivial formulas with enough prose to explain
  what is being defined/proved and why the next step follows.
- Prefer a few substantial definition/theorem/proof/remark blocks over many tiny blocks.
- Preserve the graph's order/dependencies; never move a proof before the statement it proves.

LATEX CONTRACT:
- Output actual LaTeX source, not Unicode mathematical glyphs.
- Every mathematical symbol occurring inside prose must be in math mode using $...$ or \\(...\\).
  For example write $f$, $u$, $v$, $X^*$, $\\alpha$, not text-mode f/u/v or Unicode symbols.
- Do not use renderer-owned environments such as \\begin{{theorem}} or \\begin{{proof}}; block type
  carries that structure.
- Every non-empty canonical node field latex is immutable. If you use that formula, copy the
  canonical LaTeX string EXACTLY somewhere in a block body; do not rename symbols or algebraically
  rewrite it.
- Do not introduce new mathematical identities.

COVERAGE CONTRACT:
- Every canonical node id must appear in source_node_ids of at least one returned block.
- source_node_ids may contain several ids when you merge graph atoms into one exposition block.
- Never invent node ids.

Write prose in language code {output_language}. Return strict structured JSON only.
"""


def _formula_key(value: str) -> str:
    return re.sub(r"\\s+", "", value.strip())


def _verify_generated(
    *,
    spec: GraphSectionSpec,
    generated: GeneratedGraphSectionNotes,
) -> list[str]:
    errors: list[str] = []
    allowed_ids = {node.id for node in spec.nodes}
    covered_ids = {
        node_id
        for block in generated.blocks
        for node_id in block.source_node_ids
    }
    unknown = sorted(covered_ids - allowed_ids)
    missing_nodes = sorted(allowed_ids - covered_ids)
    if unknown:
        errors.append("unknown source_node_ids: " + ", ".join(unknown))
    if missing_nodes:
        errors.append("uncovered canonical nodes: " + ", ".join(missing_nodes))

    combined = "\\n".join(block.latex for block in generated.blocks)
    combined_key = _formula_key(combined)
    missing_formulas = [
        node.id
        for node in spec.nodes
        if (node.latex or "").strip()
        and _formula_key(node.latex or "") not in combined_key
    ]
    if missing_formulas:
        errors.append("missing immutable formulas from: " + ", ".join(missing_formulas))

    if _AUDIT_LANGUAGE.search(combined):
        errors.append("reader-facing output contains provenance/audit language")
    if _UNICODE_MATH.search(combined):
        errors.append("reader-facing output contains Unicode math instead of LaTeX")

    return errors


def _generated_to_chunk(
    *,
    state: GraphState,
    spec: GraphSectionSpec,
    generated: GeneratedGraphSectionNotes,
    fallback: ChunkNotes,
) -> ChunkNotes:
    node_by_id = {node.id: node for node in spec.nodes}
    blocks: list[NoteBlock] = []
    for block in generated.blocks:
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for node_id in block.source_node_ids
                for evidence_id in node_by_id[node_id].evidence_ids
            )
        )
        blocks.append(block.to_note_block(source_evidence_ids=evidence_ids))

    return ChunkNotes(
        chunk_id=spec.section_id,
        start=spec.start,
        end=spec.end,
        section_title=spec.title,
        blocks=blocks,
        unresolved=list(fallback.unresolved),
    )


def _load_cached(
    path: Path,
    fingerprint: str,
) -> GeneratedGraphSectionNotes | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return GeneratedGraphSectionNotes.model_validate(payload["generated"])
    except (OSError, ValueError, KeyError):
        return None


def write_graph_surface(
    orchestrator: Any,
    *,
    state: GraphState,
    lecture_id: str,
    lecture_title: str,
    fallback_ir: LectureIR,
    work: Path,
    llm_config: dict[str, Any],
    force: bool,
) -> LectureIR:
    """Realize canonical graph sections into mathematical prose without reopening evidence."""

    specs = graph_section_specs(state, lecture_title=lecture_title)
    if len(specs) != len(fallback_ir.chunks):
        logger.warning(
            "[graph_surface_writer] section mismatch (%d specs vs %d fallback); using deterministic surface",
            len(specs),
            len(fallback_ir.chunks),
        )
        return fallback_ir

    root = work / "graph_surface_writer"
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[ChunkNotes] = []

    for index, (spec, fallback) in enumerate(zip(specs, fallback_ir.chunks, strict=True)):
        payload = _section_payload(state, spec)
        prompt = _surface_writer_prompt(
            payload=payload,
            output_language=orchestrator.output_language,
        )
        fingerprint = stable_hash(
            {
                "writer_version": GRAPH_SURFACE_WRITER_VERSION,
                "section": payload,
                "prompt": prompt,
                "llm": llm_config,
                "writer": {
                    "thinking": orchestrator.config.state_section_writer_thinking,
                    "temperature": orchestrator.config.state_section_writer_temperature,
                    "top_p": orchestrator.config.state_section_writer_top_p,
                    "top_k": orchestrator.config.state_section_writer_top_k,
                    "min_p": orchestrator.config.state_section_writer_min_p,
                    "presence_penalty": (
                        orchestrator.config.state_section_writer_presence_penalty
                    ),
                    "repetition_penalty": (
                        orchestrator.config.state_section_writer_repetition_penalty
                    ),
                },
            }
        )
        path = root / f"section_{index:03d}.json"
        generated = None if force else _load_cached(path, fingerprint)
        cache_hit = generated is not None

        if generated is None:
            generated = orchestrator._structured(
                prompt,
                GeneratedGraphSectionNotes,
                operation="graph_surface_write",
                split_oversized_task=True,
                temperature=float(orchestrator.config.state_section_writer_temperature),
                thinking=bool(orchestrator.config.state_section_writer_thinking),
                top_p=float(orchestrator.config.state_section_writer_top_p),
                top_k=int(orchestrator.config.state_section_writer_top_k),
                min_p=float(orchestrator.config.state_section_writer_min_p),
                presence_penalty=float(
                    orchestrator.config.state_section_writer_presence_penalty
                ),
                repetition_penalty=float(
                    orchestrator.config.state_section_writer_repetition_penalty
                ),
            )

        errors = _verify_generated(spec=spec, generated=generated)
        if errors and not cache_hit:
            repair_prompt = (
                prompt
                + "\\n\\nThe previous structured realization failed host verification. "
                + "Repair it without changing mathematics. Verification errors:\\n- "
                + "\\n- ".join(errors)
                + "\\nPrevious JSON:\\n"
                + generated.model_dump_json()
            )
            generated = orchestrator._structured(
                repair_prompt,
                GeneratedGraphSectionNotes,
                operation="graph_surface_write_repair",
                split_oversized_task=True,
                temperature=float(orchestrator.config.state_section_writer_temperature),
                thinking=bool(orchestrator.config.state_section_writer_thinking),
                top_p=float(orchestrator.config.state_section_writer_top_p),
                top_k=int(orchestrator.config.state_section_writer_top_k),
                min_p=float(orchestrator.config.state_section_writer_min_p),
                presence_penalty=float(
                    orchestrator.config.state_section_writer_presence_penalty
                ),
                repetition_penalty=float(
                    orchestrator.config.state_section_writer_repetition_penalty
                ),
            )
            errors = _verify_generated(spec=spec, generated=generated)

        if errors:
            logger.warning(
                "[graph_surface_writer] section %s rejected (%s); using deterministic fallback",
                spec.section_id,
                "; ".join(errors),
            )
            chunks.append(fallback)
            continue

        atomic_json_dump(
            path,
            {
                "fingerprint": fingerprint,
                "generated": generated.model_dump(mode="json"),
            },
        )
        chunks.append(
            _generated_to_chunk(
                state=state,
                spec=spec,
                generated=generated,
                fallback=fallback,
            )
        )
        logger.info(
            "[graph_surface_writer] section %d/%d ready: id=%s nodes=%d blocks=%d cache_hit=%s",
            index + 1,
            len(specs),
            spec.section_id,
            len(spec.nodes),
            len(generated.blocks),
            cache_hit,
        )

    return LectureIR(
        lecture_id=lecture_id,
        title=lecture_title,
        chunks=chunks,
    )
