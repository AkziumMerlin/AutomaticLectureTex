from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from .generated_notes import GeneratedStateSectionBlock, GeneratedStateSectionNotes
from .graph_revision import GraphNode, GraphState
from .graph_revision_render import (
    _canonical_surface_text,
    _is_renderable,
    _node_times,
    _order_nodes,
    _section_assignment,
)
from .llm import StructuredTaskTooLargeError
from .schemas import BlockType, ChunkNotes, LectureIR, NoteBlock
from .util import atomic_json_dump, stable_hash

logger = logging.getLogger(__name__)

GRAPH_SURFACE_WRITER_VERSION = 4

_AUDIT_LANGUAGE = re.compile(
    r"\b(?:ASR|OCR|доск(?:а|е|и|у|ой)|кадр(?:е|ы|ов)?|окн(?:о|е|а)|"
    r"лектор|видео|распознан|реконструкц|уверенност|provenance|рукопис\w*|"
    r"нечитаем\w*|восстановлен\w*|пиксел\w*|панел\w*|гипотез\w*|"
    r"чтени\w*)\b",
    re.IGNORECASE,
)
_FORMULA_MARKER = re.compile(r"\[\[MATH:([^\]\n]+)\]\]")
_WRAPPED_FORMULA_MARKER = re.compile(
    r"(?P<open>\$\$|\$|\\\[|\\\()\s*\[\[?MATH:([^\]\n]+)\]\]?\s*"
    r"(?P<close>\$\$|\$|\\\]|\\\))"
)
_LOOSE_FORMULA_MARKER = re.compile(r"(?<!\[)\[MATH:([^\]\n]+)\](?!\])")
_UNICODE_MATH = re.compile(
    r"[∀∃∈∉∋∑∏∫√∞≤≥≠≈≡→←↔⇒⇔⊂⊃⊆⊇∩∪⋂⋃∅ℂℝℕℤℚℓ"
    r"αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)
_SURFACE_PLACEHOLDER = re.compile(r"\\text\{\s*(?:\.{3}|…)\s*\}")
_BAD_SURFACE_TEX = re.compile(r"\\t(?:le|in|eq)(?![A-Za-z])|\\t\{")
_INLINE_MATH = re.compile(r"\$([^$\n]*)\$")
_BAD_INLINE_ASCII_MATH = re.compile(
    r"(?:->|\|\||(?<!\\)\b(?:le|ge|neq|in|to|Re|Im|Arg)\b|"
    r"\\textbf\{[A-Za-z]+\}|\\text\{(?:R|C|N|Z|Q)\})"
)
_RAW_MODEL_DISPLAY = re.compile(r"\\\[|\\\]|\\begin\{(?:aligned|alignedat|gathered|multlined|cases|array|matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|split)\}")


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
- Canonical display formulas are renderer-owned. Do NOT retype or rewrite them. For every node
  whose latex field is non-empty, place the literal marker [[MATH:<node_id>]] in the block that
  explains that node. The host expands the marker to the exact canonical LaTeX after validation.
- Write that marker exactly as plain text. Never wrap it in $...$, $$...$$, \\(...\\), or
  \\[...\\], and never shorten it to [MATH:<node_id>].
- Never use mathematical placeholders such as \\text{{...}}, "...", "см. ниже", or "см. выше" in
  place of a mathematical object. If a display formula already carries the needed mathematics,
  write the surrounding prose without restating an uncertain inline formula.
- Do not use Markdown markup such as **bold**. The renderer owns document typography.
- Inline symbol mentions in explanatory prose are allowed, but they must be valid LaTeX. Use
  standard commands such as \\in, \\neq, \\le, \\varphi and \\mathbb{{C}}; never emit
  malformed commands such as \\tin, \\teq, or \\tle.
- Do not introduce new mathematical identities.
- Do not author display mathematics yourself. Use [[MATH:<node_id>]] for every displayed canonical
  formula; the host is the only component allowed to emit display formulas.
- If a canonical node explicitly says that the role or interpretation of a visible formula is
  unresolved, do not resolve that ambiguity yourself. State only the unambiguous mathematical
  content, or introduce the canonical formula neutrally without assigning it a stronger role.

COVERAGE CONTRACT:
- Every canonical node id must appear in source_node_ids of at least one returned block.
- source_node_ids may contain several ids when you merge graph atoms into one exposition block.
- Never invent node ids.

Write prose in language code {output_language}. Return strict structured JSON only.
"""


def _formula_key(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def _formula_marker(node_id: str) -> str:
    return f"[[MATH:{node_id}]]"


def _normalize_generated_surface(generated: GeneratedGraphSectionNotes) -> None:
    """Normalize harmless model formatting drift before semantic/TeX verification."""

    for block in generated.blocks:
        latex = _WRAPPED_FORMULA_MARKER.sub(
            lambda match: _formula_marker(match.group(2)),
            block.latex,
        )
        latex = _LOOSE_FORMULA_MARKER.sub(
            lambda match: _formula_marker(match.group(1)),
            latex,
        )
        # Markdown emphasis is a presentation choice, not mathematics. The LaTeX renderer owns it.
        block.latex = latex.replace("**", "")


def _inject_formula_markers(
    *,
    spec: GraphSectionSpec,
    generated: GeneratedGraphSectionNotes,
) -> None:
    """Attach renderer-owned canonical formulas to the block claiming each source node."""

    combined = "\n".join(block.latex for block in generated.blocks)
    combined_key = _formula_key(combined)
    for node in spec.nodes:
        latex = (node.latex or "").strip()
        if not latex:
            continue
        marker = _formula_marker(node.id)
        if marker in combined or _formula_key(latex) in combined_key:
            continue
        target = next(
            (block for block in generated.blocks if node.id in block.source_node_ids),
            None,
        )
        if target is None:
            continue
        target.latex = target.latex.rstrip() + "\n\n" + marker
        combined += "\n" + marker
        combined_key = _formula_key(combined)


def _expand_formula_markers(value: str, node_by_id: dict[str, GraphNode]) -> str:
    def replace(match: re.Match[str]) -> str:
        node = node_by_id[match.group(1)]
        return "\\[\n" + (node.latex or "").strip() + "\n\\]"

    return _FORMULA_MARKER.sub(replace, value)


def _verify_generated(
    *,
    spec: GraphSectionSpec,
    generated: GeneratedGraphSectionNotes,
) -> list[str]:
    errors: list[str] = []
    allowed_ids = {node.id for node in spec.nodes}
    node_by_id = {node.id: node for node in spec.nodes}
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

    body_combined = "\n".join(block.latex for block in generated.blocks)
    combined = "\n".join(
        part
        for block in generated.blocks
        for part in (block.title or "", block.latex)
        if part
    )
    combined_key = _formula_key(body_combined)
    marker_ids = set(_FORMULA_MARKER.findall(body_combined))
    unknown_markers = sorted(marker_ids - allowed_ids)
    if unknown_markers:
        errors.append("unknown formula markers: " + ", ".join(unknown_markers))
    nonformula_markers = sorted(
        node_id
        for node_id in marker_ids
        if node_id in node_by_id and not (node_by_id[node_id].latex or "").strip()
    )
    if nonformula_markers:
        errors.append("formula markers for nodes without latex: " + ", ".join(nonformula_markers))

    missing_formulas = [
        node.id
        for node in spec.nodes
        if (node.latex or "").strip()
        and _formula_marker(node.id) not in body_combined
        and _formula_key(node.latex or "") not in combined_key
    ]
    if missing_formulas:
        errors.append("missing canonical formulas from: " + ", ".join(missing_formulas))

    if _AUDIT_LANGUAGE.search(combined):
        errors.append("reader-facing output contains provenance/audit language")
    if _UNICODE_MATH.search(combined):
        errors.append("reader-facing output contains Unicode math instead of LaTeX")
    if _SURFACE_PLACEHOLDER.search(combined):
        errors.append("reader-facing output contains mathematical placeholder \\text{...}")
    if _BAD_SURFACE_TEX.search(combined):
        errors.append("reader-facing output contains malformed TeX command such as \\tin/\\teq/\\tle")
    bad_inline = [
        math
        for math in _INLINE_MATH.findall(combined)
        if _BAD_INLINE_ASCII_MATH.search(math)
    ]
    if bad_inline:
        errors.append(
            "reader-facing output contains malformed/non-LaTeX inline math: "
            + "; ".join(bad_inline[:3])
        )
    if _RAW_MODEL_DISPLAY.search(body_combined):
        errors.append(
            "reader-facing output contains model-authored display math; use canonical MATH markers"
        )
    if combined.find("$" + "$") >= 0:
        errors.append("reader-facing output contains raw double-dollar display delimiters")
    residual_marker_text = _FORMULA_MARKER.sub("", combined)
    if "[MATH:" in residual_marker_text:
        errors.append("reader-facing output contains malformed formula marker")

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
        marker_ids = _FORMULA_MARKER.findall(block.latex)
        block_type = block.type
        latex = block.latex
        if marker_ids:
            single_marker = (
                len(marker_ids) == 1
                and latex.strip() == _formula_marker(marker_ids[0])
            )
            if single_marker and block_type == BlockType.EQUATION:
                latex = (node_by_id[marker_ids[0]].latex or "").strip()
            else:
                latex = _expand_formula_markers(latex, node_by_id)
                if block_type == BlockType.EQUATION:
                    block_type = BlockType.PARAGRAPH
        blocks.append(
            NoteBlock(
                type=block_type,
                title=block.title,
                latex=latex,
                source_claim_ids=[],
                source_evidence_ids=evidence_ids,
            )
        )

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



def _surface_writer_fingerprint(
    *,
    payload: dict[str, Any],
    prompt: str,
    llm_config: dict[str, Any],
    orchestrator: Any,
) -> str:
    return stable_hash(
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
                "presence_penalty": orchestrator.config.state_section_writer_presence_penalty,
                "repetition_penalty": orchestrator.config.state_section_writer_repetition_penalty,
            },
        }
    )


def _subspec(
    state: GraphState,
    parent: GraphSectionSpec,
    nodes: list[GraphNode],
    suffix: str,
) -> GraphSectionSpec:
    starts: list[float] = []
    ends: list[float] = []
    for node in nodes:
        start, end = _node_times(state, node)
        if start != float("inf"):
            starts.append(start)
            ends.append(end)
    return GraphSectionSpec(
        section_id=f"{parent.section_id}{suffix}",
        title=parent.title,
        start=min(starts) if starts else parent.start,
        end=max(ends) if ends else parent.end,
        nodes=nodes,
    )


def _surface_call_kwargs(orchestrator: Any) -> dict[str, Any]:
    return {
        "split_oversized_task": True,
        "temperature": float(orchestrator.config.state_section_writer_temperature),
        "thinking": bool(orchestrator.config.state_section_writer_thinking),
        "top_p": float(orchestrator.config.state_section_writer_top_p),
        "top_k": int(orchestrator.config.state_section_writer_top_k),
        "min_p": float(orchestrator.config.state_section_writer_min_p),
        "presence_penalty": float(orchestrator.config.state_section_writer_presence_penalty),
        "repetition_penalty": float(orchestrator.config.state_section_writer_repetition_penalty),
    }


def _split_surface_realization(
    orchestrator: Any,
    *,
    state: GraphState,
    spec: GraphSectionSpec,
    path: Path,
    fingerprint: str,
    work_force: bool,
    llm_config: dict[str, Any],
    reason: str,
    depth: int,
) -> tuple[GeneratedGraphSectionNotes, bool]:
    if len(spec.nodes) <= 1:
        node_id = spec.nodes[0].id if spec.nodes else "<empty>"
        raise RuntimeError(
            "graph surface realization failed for indivisible canonical node "
            f"{node_id} in section {spec.section_id}: {reason}"
        )

    midpoint = len(spec.nodes) // 2
    left = _subspec(state, spec, spec.nodes[:midpoint], "::__a")
    right = _subspec(state, spec, spec.nodes[midpoint:], "::__b")
    logger.warning(
        "[graph_surface_writer] section %s rejected at depth=%d (%s); "
        "split %d nodes -> %d + %d",
        spec.section_id,
        depth,
        reason,
        len(spec.nodes),
        len(left.nodes),
        len(right.nodes),
    )

    left_path = path.with_name(path.stem + "__a" + path.suffix)
    right_path = path.with_name(path.stem + "__b" + path.suffix)
    left_generated, _ = _realize_surface_spec(
        orchestrator,
        state=state,
        spec=left,
        path=left_path,
        llm_config=llm_config,
        force=work_force,
        depth=depth + 1,
    )
    right_generated, _ = _realize_surface_spec(
        orchestrator,
        state=state,
        spec=right,
        path=right_path,
        llm_config=llm_config,
        force=work_force,
        depth=depth + 1,
    )

    generated = GeneratedGraphSectionNotes(
        blocks=[*left_generated.blocks, *right_generated.blocks]
    )
    _normalize_generated_surface(generated)
    _inject_formula_markers(spec=spec, generated=generated)
    errors = _verify_generated(spec=spec, generated=generated)
    if errors:
        raise RuntimeError(
            f"split realization of section {spec.section_id} failed combined verification: "
            + "; ".join(errors)
        )

    atomic_json_dump(
        path,
        {
            "fingerprint": fingerprint,
            "generated": generated.model_dump(mode="json"),
            "split": {
                "reason": reason,
                "children": [left.section_id, right.section_id],
            },
        },
    )
    return generated, False


def _realize_surface_spec(
    orchestrator: Any,
    *,
    state: GraphState,
    spec: GraphSectionSpec,
    path: Path,
    llm_config: dict[str, Any],
    force: bool,
    depth: int = 0,
) -> tuple[GeneratedGraphSectionNotes, bool]:
    payload = _section_payload(state, spec)
    prompt = _surface_writer_prompt(
        payload=payload,
        output_language=orchestrator.output_language,
    )
    fingerprint = _surface_writer_fingerprint(
        payload=payload,
        prompt=prompt,
        llm_config=llm_config,
        orchestrator=orchestrator,
    )

    generated = None if force else _load_cached(path, fingerprint)
    cache_hit = generated is not None
    if generated is not None:
        _normalize_generated_surface(generated)
        _inject_formula_markers(spec=spec, generated=generated)
        cached_errors = _verify_generated(spec=spec, generated=generated)
        if not cached_errors:
            return generated, True
        logger.warning(
            "[graph_surface_writer] invalid cached section %s (%s); regenerating",
            spec.section_id,
            "; ".join(cached_errors),
        )
        generated = None
        cache_hit = False

    try:
        generated = orchestrator._structured(
            prompt,
            GeneratedGraphSectionNotes,
            operation="graph_surface_write",
            **_surface_call_kwargs(orchestrator),
        )
    except (ValidationError, StructuredTaskTooLargeError, ValueError) as exc:
        return _split_surface_realization(
            orchestrator,
            state=state,
            spec=spec,
            path=path,
            fingerprint=fingerprint,
            work_force=force,
            llm_config=llm_config,
            reason=f"writer call failed: {type(exc).__name__}: {exc}",
            depth=depth,
        )

    _normalize_generated_surface(generated)
    _inject_formula_markers(spec=spec, generated=generated)
    errors = _verify_generated(spec=spec, generated=generated)
    if errors:
        repair_prompt = (
            prompt
            + "\n\nThe previous structured realization failed host verification. "
            + "Repair it without changing mathematics. Verification errors:\n- "
            + "\n- ".join(errors)
            + "\nPrevious JSON:\n"
            + generated.model_dump_json()
        )
        try:
            generated = orchestrator._structured(
                repair_prompt,
                GeneratedGraphSectionNotes,
                operation="graph_surface_write_repair",
                **_surface_call_kwargs(orchestrator),
            )
            _normalize_generated_surface(generated)
            _inject_formula_markers(spec=spec, generated=generated)
            errors = _verify_generated(spec=spec, generated=generated)
        except (ValidationError, StructuredTaskTooLargeError, ValueError) as exc:
            errors = [f"repair call failed: {type(exc).__name__}: {exc}"]

    if errors:
        return _split_surface_realization(
            orchestrator,
            state=state,
            spec=spec,
            path=path,
            fingerprint=fingerprint,
            work_force=force,
            llm_config=llm_config,
            reason="; ".join(errors),
            depth=depth,
        )

    atomic_json_dump(
        path,
        {
            "fingerprint": fingerprint,
            "generated": generated.model_dump(mode="json"),
        },
    )
    return generated, cache_hit


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
    """Realize canonical graph sections into mathematical prose without reopening evidence.

    Deterministic graph rendering is metadata/debug support only. Reader-facing sections are
    accepted exclusively after LLM realization and host verification. Large or repeatedly invalid
    sections are recursively realized in smaller canonical-node batches; an indivisible failure
    aborts the run instead of silently leaking a graph dump into the final notes.
    """

    specs = graph_section_specs(state, lecture_title=lecture_title)
    if len(specs) != len(fallback_ir.chunks):
        raise RuntimeError(
            "graph surface section mismatch: "
            f"{len(specs)} canonical specs vs {len(fallback_ir.chunks)} deterministic metadata chunks"
        )

    root = work / "graph_surface_writer"
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[ChunkNotes] = []

    for index, (spec, fallback) in enumerate(zip(specs, fallback_ir.chunks, strict=True)):
        path = root / f"section_{index:03d}.json"
        generated, cache_hit = _realize_surface_spec(
            orchestrator,
            state=state,
            spec=spec,
            path=path,
            llm_config=llm_config,
            force=force,
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

