from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .generated_notes import GeneratedBlockType
from .graph_revision import GraphNode, GraphState
from .graph_revision_render import (
    _canonical_surface_text,
    _is_renderable,
    _node_times,
    _order_nodes,
    _section_assignment,
)
from .schemas import BlockType, ChunkNotes, LectureIR, NoteBlock
from .util import atomic_json_dump, stable_hash

logger = logging.getLogger(__name__)

READER_SURFACE_PIPELINE_VERSION = 3

_PROVENANCE_LANGUAGE = re.compile(
    r"\b(?:ASR|OCR|доск\w*|кадр\w*|окн\w*|лектор\w*|видео|распознан\w*|"
    r"реконструкц\w*|уверенност\w*|provenance|рукопис\w*|пиксел\w*|"
    r"панел\w*|гипотез\w*|чтени\w*)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])")
_TEXT_FORBIDDEN = re.compile(r"[$\\]")
_UNICODE_MATH = re.compile(
    r"[∀∃∈∉∋∑∏∫√∞≤≥≠≈≡→←↔⇒⇔⊂⊃⊆⊇∩∪⋂⋃∅ℂℝℕℤℚℓ"
    r"αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)
_INLINE_RELATION = re.compile(
    r"(?:=|<|>|→|←|↦|⇒|⇔|≤|≥|≠|≈|≡|∈|∉|⊂|⊆|⊃|⊇|"
    r"\\(?:to|mapsto|Rightarrow|Leftarrow|Leftrightarrow|leq?|geq?|neq|"
    r"approx|equiv|in|notin|subset(?:eq)?|supset(?:eq)?)\b)"
)


@dataclass(frozen=True)
class GraphSectionSpec:
    section_id: str
    title: str
    start: float
    end: float
    nodes: list[GraphNode]


@dataclass(frozen=True)
class ReaderExpression:
    id: str
    source_node_id: str
    latex: str


@dataclass(frozen=True)
class ReaderFact:
    node_id: str
    kind: str
    title: str
    statement: str
    disposition: Literal["render", "omit", "unresolved"]
    expression_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReaderSectionProjection:
    section_id: str
    title: str
    facts: tuple[ReaderFact, ...]
    expressions: tuple[ReaderExpression, ...]
    relations: tuple[dict[str, str], ...]


class ReaderProjectionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    disposition: Literal["render", "omit", "unresolved"]
    selected_unit_indices: list[int] = Field(default_factory=list)


class ReaderProjectionChoices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choices: list[ReaderProjectionChoice] = Field(default_factory=list)


class PlannedReaderBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    type: GeneratedBlockType
    title: str | None = None
    purpose: str = Field(min_length=1, max_length=800)
    node_ids: list[str] = Field(min_length=1)


class ReaderDiscoursePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[PlannedReaderBlock] = Field(default_factory=list)


class ReaderSurfaceSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["text", "inline_math", "expression"]
    source_node_ids: list[str] = Field(min_length=1)
    text: str | None = None
    latex: str | None = None
    expression_id: str | None = None
    display: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> "ReaderSurfaceSegment":
        if self.kind == "text":
            if not self.text or self.latex is not None or self.expression_id is not None:
                raise ValueError("text segment requires only text")
            if _TEXT_FORBIDDEN.search(self.text):
                raise ValueError("text segment must be plain prose without TeX delimiters/commands")
            if _UNICODE_MATH.search(self.text):
                raise ValueError("text segment contains mathematical glyphs; use inline_math")
            return self

        if self.kind == "inline_math":
            if not self.latex or self.text is not None or self.expression_id is not None:
                raise ValueError("inline_math segment requires only latex")
            if len(self.latex) > 160:
                raise ValueError("inline_math segment must be a short mathematical atom")
            if _INLINE_RELATION.search(self.latex):
                raise ValueError(
                    "inline_math cannot contain mathematical relations; use a canonical expression"
                )
            if "$" in self.latex or r"\[" in self.latex or r"\]" in self.latex:
                raise ValueError("inline_math must not contain math delimiters")
            return self

        if not self.expression_id or self.text is not None or self.latex is not None:
            raise ValueError("expression segment requires only expression_id")
        return self


class GeneratedReaderBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[ReaderSurfaceSegment] = Field(min_length=1)


class ReaderGroundingIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_index: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=800)


class ReaderGroundingReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ReaderGroundingIssue] = Field(default_factory=list)


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
        ranges = [
            _node_times(state, node)
            for node in ordered
            if _node_times(state, node)[0] != float("inf")
        ]
        title = (
            _canonical_surface_text(topic.title)
            if topic is not None
            else ("Начало лекции" if topics else lecture_title)
        )
        specs.append(
            GraphSectionSpec(
                section_id=section_id,
                title=title,
                start=min((item[0] for item in ranges), default=0.0),
                end=max((item[1] for item in ranges), default=0.0),
                nodes=ordered,
            )
        )
    return specs


def _candidate_units(node: GraphNode) -> list[str]:
    """Raw extractive units retained for diagnostics/tests; may contain provenance language."""

    units: list[str] = []
    title = _canonical_surface_text(node.title or "").strip()
    if title:
        units.append(title)
    text = _canonical_surface_text(node.text or "").strip()
    if text:
        units.extend(item.strip() for item in _SENTENCE_SPLIT.split(text) if item.strip())
    return list(dict.fromkeys(units))


def _reader_candidate_units(node: GraphNode) -> list[str]:
    """Only units that are legal to expose to reader-facing stages."""

    return [
        unit
        for unit in _candidate_units(node)
        if not _PROVENANCE_LANGUAGE.search(unit)
    ]


def _reader_safe_title(node: GraphNode) -> str:
    title = _canonical_surface_text(node.title or "").strip()
    if title and not _PROVENANCE_LANGUAGE.search(title):
        return title
    return ""


def _expression_for_node(node: GraphNode) -> ReaderExpression | None:
    latex = (node.latex or "").strip()
    if not latex:
        return None
    return ReaderExpression(id=f"expr::{node.id}", source_node_id=node.id, latex=latex)


def _projection_input(state: GraphState, spec: GraphSectionSpec) -> dict[str, Any]:
    node_ids = {node.id for node in spec.nodes}
    return {
        "section_id": spec.section_id,
        "title": spec.title,
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "candidate_units": _reader_candidate_units(node),
                "expression_ids": (
                    [f"expr::{node.id}"] if (node.latex or "").strip() else []
                ),
            }
            for node in spec.nodes
        ],
        "relations": [
            {"source": edge.source, "relation": edge.relation, "target": edge.target}
            for edge in state.edges
            if edge.source in node_ids and edge.target in node_ids
        ],
    }


def _projection_prompt(payload: dict[str, Any], output_language: str) -> str:
    return f"""Project an internal canonical mathematics graph into reader-facing facts.

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

This is an EXTRACTIVE projection. You may not write or paraphrase mathematics.
For every input node return exactly one choice with the same node_id and choose:
- render: the node contains independent mathematical content suitable for notes;
- omit: the node is only transition/provenance/redundant commentary and adds no independent
  mathematical content;
- unresolved: the node contains a genuine mathematical ambiguity that the canonical graph did not
  resolve and should be surfaced as an unresolved item rather than asserted.

selected_unit_indices refer only to candidate_units of that node. Candidate units have already
been deterministically filtered to exclude provenance language. Select the smallest set that
contains the reader-facing mathematical content. Prefer the canonical title when it is present.
Nodes with an exact expression may use a short prose unit, or no prose unit when the expression
alone carries the mathematical content. If candidate_units is empty and expression_ids is also
empty, disposition MUST be omit.

Do not invent text, formulas, node IDs, or indices. Language code is {output_language}.
Return strict structured JSON only.
"""


def _verify_projection_choices(
    *,
    spec: GraphSectionSpec,
    choices: ReaderProjectionChoices,
) -> list[str]:
    errors: list[str] = []
    expected = [node.id for node in spec.nodes]
    got = [choice.node_id for choice in choices.choices]
    counts = Counter(got)
    missing = [node_id for node_id in expected if counts[node_id] == 0]
    duplicates = [node_id for node_id, count in counts.items() if count > 1]
    expected_set = set(expected)
    unknown = [node_id for node_id in got if node_id not in expected_set]
    if missing:
        errors.append("missing projection choices: " + ", ".join(missing))
    if duplicates:
        errors.append("duplicate projection choices: " + ", ".join(duplicates))
    if unknown:
        errors.append("unknown projection node ids: " + ", ".join(unknown))

    by_id = {node.id: node for node in spec.nodes}
    for choice in choices.choices:
        node = by_id.get(choice.node_id)
        if node is None:
            continue
        units = _reader_candidate_units(node)
        invalid = [
            index
            for index in choice.selected_unit_indices
            if index < 0 or index >= len(units)
        ]
        if invalid:
            errors.append(f"{node.id}: invalid selected unit indices {invalid}")
            continue
        if choice.disposition in {"render", "unresolved"} and not choice.selected_unit_indices:
            if not (node.latex or "").strip():
                errors.append(
                    f"{node.id}: {choice.disposition} requires a selected reader unit"
                )
        if not units and not (node.latex or "").strip() and choice.disposition != "omit":
            errors.append(
                f"{node.id}: node has no reader-safe units or expression and must be omitted"
            )
        selected = " ".join(units[index] for index in choice.selected_unit_indices)
        if selected and _PROVENANCE_LANGUAGE.search(selected):
            errors.append(f"{node.id}: selected unit contains provenance language")
    return errors


def _materialize_projection(
    state: GraphState,
    spec: GraphSectionSpec,
    choices: ReaderProjectionChoices,
) -> ReaderSectionProjection:
    choice_by_id = {choice.node_id: choice for choice in choices.choices}
    expressions: list[ReaderExpression] = []
    facts: list[ReaderFact] = []
    for node in spec.nodes:
        expression = _expression_for_node(node)
        if expression is not None:
            expressions.append(expression)
        choice = choice_by_id[node.id]
        units = _reader_candidate_units(node)
        statement = " ".join(units[index] for index in choice.selected_unit_indices).strip()
        facts.append(
            ReaderFact(
                node_id=node.id,
                kind=node.kind,
                title=_reader_safe_title(node),
                statement=statement,
                disposition=choice.disposition,
                expression_ids=((expression.id,) if expression is not None else ()),
            )
        )

    node_ids = {node.id for node in spec.nodes}
    relations = tuple(
        {"source": edge.source, "relation": edge.relation, "target": edge.target}
        for edge in state.edges
        if edge.source in node_ids and edge.target in node_ids
    )
    return ReaderSectionProjection(
        section_id=spec.section_id,
        title=spec.title,
        facts=tuple(facts),
        expressions=tuple(expressions),
        relations=relations,
    )


def _projection_json(projection: ReaderSectionProjection) -> dict[str, Any]:
    return {
        "section_id": projection.section_id,
        "title": projection.title,
        "facts": [
            {
                "node_id": fact.node_id,
                "kind": fact.kind,
                "title": fact.title,
                "statement": fact.statement,
                "disposition": fact.disposition,
                "expression_ids": list(fact.expression_ids),
            }
            for fact in projection.facts
        ],
        "expressions": [
            {
                "id": expression.id,
                "source_node_id": expression.source_node_id,
                "latex": expression.latex,
            }
            for expression in projection.expressions
        ],
        "relations": list(projection.relations),
    }


def _planner_prompt(projection: ReaderSectionProjection, output_language: str) -> str:
    visible = [
        {
            "node_id": fact.node_id,
            "kind": fact.kind,
            "title": fact.title,
            "statement": fact.statement,
            "expression_ids": list(fact.expression_ids),
        }
        for fact in projection.facts
        if fact.disposition == "render"
    ]
    visible_ids = {item["node_id"] for item in visible}
    relations = [
        relation
        for relation in projection.relations
        if relation["source"] in visible_ids and relation["target"] in visible_ids
    ]
    return f"""Plan the discourse structure of one university mathematics section.

READER-SAFE FACTS IN CANONICAL ORDER:
{json.dumps(visible, ensure_ascii=False, separators=(",", ":"))}

RELATIONS:
{json.dumps(relations, ensure_ascii=False, separators=(",", ":"))}

Return a small sequence of coherent blocks. This stage plans structure only; it must not write
mathematical prose or formulas.

Requirements:
- Every reader-safe node_id must occur in exactly one block.
- Keep node order. A block must cover a contiguous range of the listed nodes.
- Group proof steps/equations that belong to one argument instead of creating tiny headings.
- Keep a theorem/lemma/proposition before the proof that belongs to it.
- Use block type to express document structure.
- purpose is a short editorial instruction describing what the block must communicate.
- title is optional reader-facing prose. Do not put LaTeX commands in it.
- Never invent node IDs or mathematical facts.

Language code is {output_language}. Return strict structured JSON only.
"""


def _verify_plan(
    projection: ReaderSectionProjection,
    plan: ReaderDiscoursePlan,
) -> list[str]:
    errors: list[str] = []
    expected = [
        fact.node_id for fact in projection.facts if fact.disposition == "render"
    ]
    flattened = [node_id for block in plan.blocks for node_id in block.node_ids]
    if flattened != expected:
        errors.append("plan must cover each rendered node exactly once in canonical order")
    block_ids = [block.block_id for block in plan.blocks]
    if len(block_ids) != len(set(block_ids)):
        errors.append("plan contains duplicate block_id values")
    if any(block.title and _TEXT_FORBIDDEN.search(block.title) for block in plan.blocks):
        errors.append("plan titles must be plain prose without TeX")
    return errors



def _fallback_block_type(
    projection: ReaderSectionProjection,
    node_ids: list[str],
) -> GeneratedBlockType:
    kinds = {
        fact.kind.strip().lower()
        for fact in projection.facts
        if fact.node_id in set(node_ids)
    }
    if len(kinds) == 1:
        kind = next(iter(kinds))
        mapping: dict[str, GeneratedBlockType] = {
            "definition": BlockType.DEFINITION,
            "theorem": BlockType.THEOREM,
            "lemma": BlockType.LEMMA,
            "proposition": BlockType.PROPOSITION,
            "corollary": BlockType.COROLLARY,
            "proof": BlockType.PROOF,
            "proof_step": BlockType.PROOF,
            "example": BlockType.EXAMPLE,
            "remark": BlockType.REMARK,
            "equation": BlockType.EQUATION,
            "exercise": BlockType.EXERCISE,
        }
        block_type = mapping.get(kind)
        if block_type is not None:
            if block_type != BlockType.EQUATION or len(node_ids) == 1:
                return block_type
    return BlockType.PARAGRAPH


def _canonicalize_plan(
    projection: ReaderSectionProjection,
    plan: ReaderDiscoursePlan,
) -> ReaderDiscoursePlan:
    """Project an imperfect planner suggestion onto host-owned canonical node order.

    The model may suggest grouping/type/title/purpose, but it never owns coverage or order.
    Only pairwise grouping hints that are internally ordered, unique and contiguous survive.
    Missing, duplicated, unknown or reordered ids simply create deterministic boundaries.
    """

    expected = [
        fact.node_id for fact in projection.facts if fact.disposition == "render"
    ]
    if not expected:
        return ReaderDiscoursePlan(blocks=[])

    expected_set = set(expected)
    occurrences: dict[str, list[tuple[int, int]]] = {node_id: [] for node_id in expected}
    filtered_blocks: list[list[str]] = []
    for block_index, block in enumerate(plan.blocks):
        filtered: list[str] = []
        for local_index, node_id in enumerate(block.node_ids):
            if node_id not in expected_set:
                continue
            occurrences[node_id].append((block_index, local_index))
            filtered.append(node_id)
        filtered_blocks.append(filtered)

    join_pairs: set[tuple[str, str]] = set()
    position = {node_id: index for index, node_id in enumerate(expected)}
    for block_index, filtered in enumerate(filtered_blocks):
        if not filtered:
            continue
        # A block contributes grouping hints only for ids that occur exactly once globally and
        # appear in canonical order inside this proposal block.
        unique = [
            node_id
            for node_id in filtered
            if len(occurrences.get(node_id, [])) == 1
        ]
        if unique != sorted(unique, key=position.__getitem__):
            continue
        for left, right in zip(unique, unique[1:], strict=False):
            if position[right] == position[left] + 1:
                join_pairs.add((left, right))

    ranges: list[list[str]] = []
    current = [expected[0]]
    for left, right in zip(expected, expected[1:], strict=False):
        if (left, right) in join_pairs:
            current.append(right)
        else:
            ranges.append(current)
            current = [right]
    ranges.append(current)

    blocks: list[PlannedReaderBlock] = []
    for block_index, node_ids in enumerate(ranges):
        matching: list[PlannedReaderBlock] = []
        node_set = set(node_ids)
        for proposal in plan.blocks:
            filtered = [node_id for node_id in proposal.node_ids if node_id in expected_set]
            if node_set.issubset(filtered):
                ordered_subset = [node_id for node_id in filtered if node_id in node_set]
                if ordered_subset == node_ids:
                    matching.append(proposal)

        source = matching[0] if len(matching) == 1 else None
        title = source.title if source is not None else None
        if title and _TEXT_FORBIDDEN.search(title):
            title = None
        purpose = (
            source.purpose
            if source is not None
            else "Present the supplied canonical facts in order without adding new content."
        )
        block_type = (
            source.type
            if source is not None
            else _fallback_block_type(projection, node_ids)
        )
        blocks.append(
            PlannedReaderBlock(
                block_id=f"reader_block_{block_index:03d}",
                type=block_type,
                title=title,
                purpose=purpose,
                node_ids=node_ids,
            )
        )

    return ReaderDiscoursePlan(blocks=blocks)


def _block_prompt(
    *,
    projection: ReaderSectionProjection,
    block: PlannedReaderBlock,
    output_language: str,
) -> str:
    fact_by_id = {fact.node_id: fact for fact in projection.facts}
    expression_by_id = {item.id: item for item in projection.expressions}
    facts = [
        {
            "node_id": node_id,
            "title": fact_by_id[node_id].title,
            "statement": fact_by_id[node_id].statement,
            "expressions": [
                {"id": expression_id, "latex": expression_by_id[expression_id].latex}
                for expression_id in fact_by_id[node_id].expression_ids
            ],
        }
        for node_id in block.node_ids
    ]
    return f"""Realize one already-planned mathematics block as typed reader-facing segments.

BLOCK:
{json.dumps(block.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

READER-SAFE FACTS:
{json.dumps(facts, ensure_ascii=False, separators=(",", ":"))}

You are not allowed to introduce new mathematical claims.

Segment contract:
- text: ordinary prose only. No LaTeX commands, dollar signs, or Unicode math glyphs. Each text
  segment must cite the source_node_ids whose reader-safe facts it directly paraphrases.
- inline_math: a short mathematical atom such as X, f, X^*, T_\\Phi, x_n, or \\|f\\|. It may not
  contain an equality, inequality, membership, map, implication, or any other relation. Cite the
  source node(s) that establish the notation.
- expression: reference one exact host-owned expression id from the supplied facts. Do not retype
  the formula. Set display=true for substantial formulas and false only when the exact canonical
  expression naturally belongs inline.

Use the planned block type/title/purpose as structure, but return only the ordered segments.
Do not mention reconstruction, evidence, a board, OCR/ASR, confidence, or provenance.
Write prose in language code {output_language}. Return strict structured JSON only.
"""


def _verify_block(
    *,
    projection: ReaderSectionProjection,
    block: PlannedReaderBlock,
    generated: GeneratedReaderBlock,
) -> list[str]:
    errors: list[str] = []
    allowed_nodes = set(block.node_ids)
    allowed_expressions = {
        expression_id
        for fact in projection.facts
        if fact.node_id in allowed_nodes
        for expression_id in fact.expression_ids
    }
    cited_nodes = {
        node_id
        for segment in generated.segments
        for node_id in segment.source_node_ids
    }
    if not cited_nodes.issubset(allowed_nodes):
        errors.append("block segments cite node IDs outside the planned block")
    missing_nodes = allowed_nodes - cited_nodes
    if missing_nodes:
        errors.append(
            "block segments do not cite planned nodes: " + ", ".join(sorted(missing_nodes))
        )

    referenced_expressions = {
        segment.expression_id
        for segment in generated.segments
        if segment.kind == "expression" and segment.expression_id
    }
    if not referenced_expressions.issubset(allowed_expressions):
        errors.append("block references an expression outside its planned nodes")
    missing_expressions = allowed_expressions - referenced_expressions
    if missing_expressions:
        errors.append(
            "block omits canonical expressions: " + ", ".join(sorted(missing_expressions))
        )
    return errors


def _grounding_prompt(
    *,
    projection: ReaderSectionProjection,
    block: PlannedReaderBlock,
    generated: GeneratedReaderBlock,
    output_language: str,
) -> str:
    fact_by_id = {fact.node_id: fact for fact in projection.facts}
    payload = []
    for index, segment in enumerate(generated.segments):
        payload.append(
            {
                "segment_index": index,
                "segment": segment.model_dump(mode="json"),
                "cited_facts": [
                    {
                        "node_id": node_id,
                        "title": fact_by_id[node_id].title,
                        "statement": fact_by_id[node_id].statement,
                        "expression_ids": list(fact_by_id[node_id].expression_ids),
                    }
                    for node_id in segment.source_node_ids
                ],
            }
        )
    return f"""Audit one typed lecture-note block for semantic grounding.

PLANNED BLOCK:
{json.dumps(block.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

SEGMENTS WITH THEIR CITED READER FACTS:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

For each text or inline_math segment, decide whether it states anything mathematically stronger,
more specific, or different from its cited reader facts. Flag a segment if it introduces an
uncited theorem name, identifies an unresolved object, changes a space/domain/codomain, asserts a
new equality/implication/property in prose, or otherwise adds mathematical content not entailed by
the cited facts. Exact expression segments are host-owned and need no mathematical reinterpretation.

Do not rewrite the block. Return only issue indices and concise reasons. If every segment is
grounded, return an empty issues list. Reasons use language code {output_language}.
Return strict structured JSON only.
"""


def _verify_grounding_review(
    generated: GeneratedReaderBlock,
    review: ReaderGroundingReview,
) -> list[str]:
    errors: list[str] = []
    for issue in review.issues:
        if issue.segment_index >= len(generated.segments):
            errors.append(f"grounding review references invalid segment {issue.segment_index}")
    return errors


def _review_grounding(
    orchestrator: Any,
    *,
    projection: ReaderSectionProjection,
    block: PlannedReaderBlock,
    generated: GeneratedReaderBlock,
) -> ReaderGroundingReview:
    prompt = _grounding_prompt(
        projection=projection,
        block=block,
        generated=generated,
        output_language=orchestrator.output_language,
    )
    review = orchestrator._structured(
        prompt,
        ReaderGroundingReview,
        operation="graph_block_grounding_review",
        **_call_kwargs(orchestrator),
    )
    errors = _verify_grounding_review(generated, review)
    if errors:
        raise RuntimeError(
            f"grounding review malformed for {block.block_id}: " + "; ".join(errors)
        )
    return review


def _render_generated_block(
    *,
    state: GraphState,
    projection: ReaderSectionProjection,
    block: PlannedReaderBlock,
    generated: GeneratedReaderBlock,
) -> NoteBlock:
    expression_by_id = {item.id: item for item in projection.expressions}

    pieces: list[str] = []
    for segment in generated.segments:
        if segment.kind == "text":
            pieces.append((segment.text or "").strip())
        elif segment.kind == "inline_math":
            pieces.append(r"\(" + (segment.latex or "").strip() + r"\)")
        else:
            expression = expression_by_id[segment.expression_id or ""]
            if segment.display:
                pieces.append("\\[\n" + expression.latex + "\n\\]")
            else:
                pieces.append(r"\(" + expression.latex + r"\)")

    latex = " ".join(piece for piece in pieces if piece).strip()
    latex = re.sub(r"\s+([,.;:!?])", r"\1", latex)

    block_type = block.type
    if block_type == BlockType.EQUATION:
        expression_segments = [
            segment for segment in generated.segments if segment.kind == "expression"
        ]
        if len(generated.segments) == 1 and len(expression_segments) == 1:
            latex = expression_by_id[expression_segments[0].expression_id or ""].latex
        else:
            block_type = BlockType.PARAGRAPH

    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for node_id in block.node_ids
            for evidence_id in state.nodes[node_id].evidence_ids
        )
    )
    return NoteBlock(
        type=block_type,
        title=block.title,
        latex=latex,
        source_claim_ids=list(block.node_ids),
        source_evidence_ids=evidence_ids,
    )


def _call_kwargs(orchestrator: Any) -> dict[str, Any]:
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


def _load_cached_model(
    path: Path,
    fingerprint: str,
    schema: type[BaseModel],
    key: str,
) -> BaseModel | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return schema.model_validate(payload[key])
    except (OSError, ValueError, KeyError):
        return None


def _project_section(
    orchestrator: Any,
    *,
    state: GraphState,
    spec: GraphSectionSpec,
    root: Path,
    llm_config: dict[str, Any],
    force: bool,
) -> ReaderSectionProjection:
    payload = _projection_input(state, spec)
    prompt = _projection_prompt(payload, orchestrator.output_language)
    fingerprint = stable_hash(
        {
            "version": READER_SURFACE_PIPELINE_VERSION,
            "stage": "projection",
            "payload": payload,
            "prompt": prompt,
            "llm": llm_config,
        }
    )
    path = root / "projection.json"
    choices = None if force else _load_cached_model(
        path, fingerprint, ReaderProjectionChoices, "choices"
    )
    if choices is None:
        choices = orchestrator._structured(
            prompt,
            ReaderProjectionChoices,
            operation="graph_reader_projection",
            **_call_kwargs(orchestrator),
        )
        errors = _verify_projection_choices(spec=spec, choices=choices)
        if errors:
            repair = (
                prompt
                + "\n\nYour previous extraction violated the host contract:\n- "
                + "\n- ".join(errors)
                + "\nPrevious JSON:\n"
                + choices.model_dump_json()
            )
            choices = orchestrator._structured(
                repair,
                ReaderProjectionChoices,
                operation="graph_reader_projection_repair",
                **_call_kwargs(orchestrator),
            )
    errors = _verify_projection_choices(spec=spec, choices=choices)
    if errors:
        raise RuntimeError(
            f"reader projection failed for section {spec.section_id}: " + "; ".join(errors)
        )
    atomic_json_dump(
        path,
        {"fingerprint": fingerprint, "choices": choices.model_dump(mode="json")},
    )
    projection = _materialize_projection(state, spec, choices)
    atomic_json_dump(root / "reader_projection.json", _projection_json(projection))
    return projection


def _plan_section(
    orchestrator: Any,
    *,
    projection: ReaderSectionProjection,
    root: Path,
    llm_config: dict[str, Any],
    force: bool,
) -> ReaderDiscoursePlan:
    prompt = _planner_prompt(projection, orchestrator.output_language)
    fingerprint = stable_hash(
        {
            "version": READER_SURFACE_PIPELINE_VERSION,
            "stage": "plan",
            "projection": _projection_json(projection),
            "prompt": prompt,
            "llm": llm_config,
        }
    )
    path = root / "plan.json"
    plan = None if force else _load_cached_model(path, fingerprint, ReaderDiscoursePlan, "plan")
    if plan is None:
        plan = orchestrator._structured(
            prompt,
            ReaderDiscoursePlan,
            operation="graph_discourse_plan",
            **_call_kwargs(orchestrator),
        )
        errors = _verify_plan(projection, plan)
        if errors:
            repair = (
                prompt
                + "\n\nThe previous plan violated the structural contract:\n- "
                + "\n- ".join(errors)
                + "\nPrevious JSON:\n"
                + plan.model_dump_json()
            )
            plan = orchestrator._structured(
                repair,
                ReaderDiscoursePlan,
                operation="graph_discourse_plan_repair",
                **_call_kwargs(orchestrator),
            )
    errors = _verify_plan(projection, plan)
    if errors:
        logger.warning(
            "discourse plan for section %s still violated the host contract after repair; "
            "canonicalizing coverage/order deterministically: %s",
            projection.section_id,
            "; ".join(errors),
        )
        plan = _canonicalize_plan(projection, plan)
        errors = _verify_plan(projection, plan)
    if errors:
        raise RuntimeError(
            f"discourse plan failed for section {projection.section_id}: " + "; ".join(errors)
        )
    atomic_json_dump(path, {"fingerprint": fingerprint, "plan": plan.model_dump(mode="json")})
    return plan


def _write_block(
    orchestrator: Any,
    *,
    projection: ReaderSectionProjection,
    block: PlannedReaderBlock,
    root: Path,
    index: int,
    llm_config: dict[str, Any],
    force: bool,
) -> GeneratedReaderBlock:
    prompt = _block_prompt(
        projection=projection,
        block=block,
        output_language=orchestrator.output_language,
    )
    fingerprint = stable_hash(
        {
            "version": READER_SURFACE_PIPELINE_VERSION,
            "stage": "block",
            "block": block.model_dump(mode="json"),
            "projection": _projection_json(projection),
            "prompt": prompt,
            "llm": llm_config,
        }
    )
    path = root / f"block_{index:03d}.json"
    generated = None if force else _load_cached_model(
        path, fingerprint, GeneratedReaderBlock, "generated"
    )
    if generated is None:
        generated = orchestrator._structured(
            prompt,
            GeneratedReaderBlock,
            operation="graph_block_write",
            **_call_kwargs(orchestrator),
        )

    errors = _verify_block(projection=projection, block=block, generated=generated)
    review = None
    if not errors:
        review = _review_grounding(
            orchestrator,
            projection=projection,
            block=block,
            generated=generated,
        )
        if review.issues:
            errors.extend(
                f"unsupported segment {issue.segment_index}: {issue.reason}"
                for issue in review.issues
            )

    if errors:
        repair = (
            prompt
            + "\n\nThe previous block violated the structural/grounding contract:\n- "
            + "\n- ".join(errors)
            + "\nPrevious JSON:\n"
            + generated.model_dump_json()
        )
        generated = orchestrator._structured(
            repair,
            GeneratedReaderBlock,
            operation="graph_block_write_repair",
            **_call_kwargs(orchestrator),
        )
        errors = _verify_block(projection=projection, block=block, generated=generated)
        if not errors:
            review = _review_grounding(
                orchestrator,
                projection=projection,
                block=block,
                generated=generated,
            )
            if review.issues:
                errors.extend(
                    f"unsupported segment {issue.segment_index}: {issue.reason}"
                    for issue in review.issues
                )

    if errors:
        raise RuntimeError(f"block writer failed for {block.block_id}: " + "; ".join(errors))
    atomic_json_dump(
        path,
        {
            "fingerprint": fingerprint,
            "generated": generated.model_dump(mode="json"),
            "grounding_review": (
                review.model_dump(mode="json") if review is not None else {"issues": []}
            ),
        },
    )
    return generated


def write_reader_surface(
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
    """Render graph through extractive projection, discourse planning and typed block writing.

    The deterministic graph renderer is metadata-only here: its blocks are never used as final
    reader-facing content.
    """

    specs = graph_section_specs(state, lecture_title=lecture_title)
    if len(specs) != len(fallback_ir.chunks):
        raise RuntimeError(
            "reader surface section mismatch: "
            f"{len(specs)} canonical specs vs {len(fallback_ir.chunks)} metadata chunks"
        )

    root = work / "graph_surface_writer"
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[ChunkNotes] = []

    for index, (spec, metadata_chunk) in enumerate(zip(specs, fallback_ir.chunks, strict=True)):
        section_root = root / f"section_{index:03d}"
        section_root.mkdir(parents=True, exist_ok=True)

        projection = _project_section(
            orchestrator,
            state=state,
            spec=spec,
            root=section_root,
            llm_config=llm_config,
            force=force,
        )
        plan = _plan_section(
            orchestrator,
            projection=projection,
            root=section_root,
            llm_config=llm_config,
            force=force,
        )

        blocks: list[NoteBlock] = []
        for block_index, planned in enumerate(plan.blocks):
            generated = _write_block(
                orchestrator,
                projection=projection,
                block=planned,
                root=section_root,
                index=block_index,
                llm_config=llm_config,
                force=force,
            )
            blocks.append(
                _render_generated_block(
                    state=state,
                    projection=projection,
                    block=planned,
                    generated=generated,
                )
            )

        projection_unresolved = [
            fact.statement or fact.title
            for fact in projection.facts
            if fact.disposition == "unresolved" and (fact.statement or fact.title)
        ]
        chunks.append(
            ChunkNotes(
                chunk_id=spec.section_id,
                start=spec.start,
                end=spec.end,
                section_title=spec.title,
                blocks=blocks,
                unresolved=list(
                    dict.fromkeys([*metadata_chunk.unresolved, *projection_unresolved])
                ),
            )
        )
        atomic_json_dump(
            section_root / "summary.json",
            {
                "section_id": spec.section_id,
                "rendered_nodes": [
                    fact.node_id for fact in projection.facts if fact.disposition == "render"
                ],
                "omitted_nodes": [
                    fact.node_id for fact in projection.facts if fact.disposition == "omit"
                ],
                "unresolved_nodes": [
                    fact.node_id for fact in projection.facts if fact.disposition == "unresolved"
                ],
                "blocks": [block.model_dump(mode="json") for block in plan.blocks],
            },
        )
        logger.info(
            "[reader_surface] section %d/%d ready: id=%s facts=%d blocks=%d",
            index + 1,
            len(specs),
            spec.section_id,
            len(projection.facts),
            len(blocks),
        )

    return LectureIR(lecture_id=lecture_id, title=lecture_title, chunks=chunks)
