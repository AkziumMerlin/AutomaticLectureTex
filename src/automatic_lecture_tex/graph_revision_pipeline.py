from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .graph_revision import (
    AddAliasOp,
    AddNodeOp,
    AddViolationOp,
    AttachEvidenceOp,
    GraphPatch,
    GraphState,
    ReplaceNodeOp,
    SearchResult,
    StateMetrics,
    Violation,
    apply_patch,
    graph_consensus,
    metrics,
    seed_observation_graph,
)
from .knowledge import KnowledgeOrchestrator
from .llm import (
    StructuredBackendAmbiguousRejectionError,
    StructuredInputTooLargeError,
    StructuredOutputTruncatedError,
    StructuredTaskTooLargeError,
)
from .schemas import LectureState
from .util import atomic_json_dump, stable_hash

GRAPH_REVISION_PROPOSAL_VERSION = 8
GRAPH_REVISION_RUNTIME_VERSION = 11

logger = logging.getLogger(__name__)


class GraphRevisionProposal(BaseModel):
    """One model review of a focus region against the current whole-lecture graph."""

    model_config = ConfigDict(extra="forbid")

    focus_id: str
    diagnosed_violations: list[Violation] = Field(default_factory=list)
    common_patch: GraphPatch | None = None
    alternatives: list[GraphPatch] = Field(default_factory=list)
    stable: bool = False
    summary: str = ""


@dataclass
class GraphRevisionRun:
    frontier: list[GraphState]
    consensus: GraphState
    stats: dict[str, Any]


def _metrics_log_line(state: GraphState) -> str:
    value = metrics(state)
    return (
        f"active={value.active_nodes} "
        f"unexplained={value.unexplained_evidence} "
        f"unsupported={value.unsupported_nodes} "
        f"dup={value.duplicate_nodes} "
        f"violations=e{value.evidence_violations}/m{value.math_violations}/"
        f"s{value.structure_violations}"
    )


def _frontier_log_line(frontier: list[GraphState]) -> str:
    if not frontier:
        return "frontier=0"
    representative = graph_consensus(frontier)
    return f"frontier={len(frontier)} {_metrics_log_line(representative)}"


def _node_time(state: GraphState, node_id: str) -> float:
    node = state.nodes[node_id]
    times = [
        state.evidence[evidence_id].start
        for evidence_id in node.evidence_ids
        if evidence_id in state.evidence
    ]
    if times:
        return min(times)
    dependency_times = [
        _node_time(state, dependency)
        for dependency in node.derived_from
        if dependency in state.nodes and dependency != node_id
    ]
    return min(dependency_times) if dependency_times else float("inf")


def _compact_catalog(
    state: GraphState,
    max_chars: int,
    *,
    focus_evidence_ids: list[str],
) -> dict[str, list[Any]]:
    """Represent the whole graph without truncating it to an early chronological prefix.

    Every canonical node id appears in a compact global index. Rich text/LaTeX is reserved for the
    current focus and its temporal/relational neighbourhood, so a late focus can always discover
    that a Riesz/weak-topology node already exists without paying to serialize the whole lecture in
    full detail.
    """

    focus_ids = set(focus_evidence_ids)
    focus_times = [
        state.evidence[evidence_id].start
        for evidence_id in focus_ids
        if evidence_id in state.evidence
    ]
    focus_start = min(focus_times, default=float("-inf"))
    focus_end = max(
        (
            state.evidence[evidence_id].end
            for evidence_id in focus_ids
            if evidence_id in state.evidence
        ),
        default=float("inf"),
    )

    active = [
        node
        for node in state.nodes.values()
        if node.status != "suppressed"
    ]
    active.sort(key=lambda item: (_node_time(state, item.id), item.id))

    # Provisional nodes outside the focus are raw initialization noise. Canonical nodes, topics,
    # and current-focus provisionals form the global identity index.
    indexed = [
        node
        for node in active
        if not node.kind.startswith("provisional_")
        or bool(focus_ids.intersection(node.evidence_ids))
    ]
    index: list[list[Any]] = []
    for node in indexed:
        start = _node_time(state, node.id)
        index.append(
            [
                node.id,
                node.kind,
                node.status,
                node.title[:96] or None,
                None if start == float("inf") else round(start, 1),
            ]
        )

    focus_node_ids = {
        node.id
        for node in active
        if focus_ids.intersection(node.evidence_ids)
    }
    related_ids: set[str] = set()
    for edge in state.edges:
        if edge.source in focus_node_ids:
            related_ids.add(edge.target)
        if edge.target in focus_node_ids:
            related_ids.add(edge.source)

    # Node-level derivation dependencies are first-class graph relations too. They are not always
    # duplicated into state.edges, and omitting them hid late mathematical refinements from reverse
    # passes (for example a late finite-intersection basis derived from an early elementary weak
    # neighbourhood).
    for node in active:
        if node.id in focus_node_ids:
            related_ids.update(
                dependency
                for dependency in node.derived_from
                if dependency in state.nodes
            )
        if any(dependency in focus_node_ids for dependency in node.derived_from):
            related_ids.add(node.id)

    def detail_priority(node: Any) -> tuple[int, float, str]:
        direct = bool(focus_ids.intersection(node.evidence_ids))
        start = _node_time(state, node.id)
        near = (
            start != float("inf")
            and focus_start != float("-inf")
            and focus_start - 300.0 <= start <= focus_end + 300.0
        )
        provisional = node.kind.startswith("provisional_")
        topic = node.kind.strip().lower() in {"topic", "section", "subsection"}
        if direct:
            bucket = 0
        elif node.id in related_ids:
            bucket = 1
        elif near and not provisional:
            # Nearby provisional observations are raw initialization noise. Including hundreds of
            # seconds of them defeats recursive focus splitting: a singleton focus still sees a
            # broad pseudo-transcript through graph detail. Canonical neighbours remain available,
            # while every canonical id is still present in the compact global index.
            bucket = 2
        elif topic:
            bucket = 3
        else:
            bucket = 4
        distance = (
            abs(start - 0.5 * (focus_start + focus_end))
            if start != float("inf")
            and focus_start != float("-inf")
            and focus_end != float("inf")
            else float("inf")
        )
        return bucket, distance, node.id

    detail_candidates = sorted(active, key=detail_priority)
    payload: dict[str, list[Any]] = {
        "index": index,
        "detail": [],
    }
    used = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    for node in detail_candidates:
        if detail_priority(node)[0] >= 4:
            break
        row = {
            "id": node.id,
            "kind": node.kind,
            "title": node.title,
            "text": node.text[:320],
            "latex": (node.latex or "")[:600] or None,
            "aliases": node.aliases[:8],
            "status": node.status,
            "alternative_group": node.alternative_group,
            "evidence_ids": node.evidence_ids[:20],
            "derived_from": node.derived_from[:12],
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if payload["detail"] and used + len(encoded) + 1 > max_chars:
            break
        payload["detail"].append(row)
        used += len(encoded) + 1

    return payload

def _focus_batches(
    state: GraphState,
    *,
    batch_observations: int,
    overlap_observations: int,
) -> list[list[str]]:
    evidence = sorted(
        state.evidence.values(),
        key=lambda item: (item.start, item.end, item.id),
    )
    if not evidence:
        return []
    stride = max(1, batch_observations - overlap_observations)
    result: list[list[str]] = []
    for start in range(0, len(evidence), stride):
        batch = evidence[start : start + batch_observations]
        if not batch:
            continue
        result.append([item.id for item in batch])
        if start + batch_observations >= len(evidence):
            break
    return result


def _focus_evidence(state: GraphState, evidence_ids: list[str]) -> list[dict[str, Any]]:
    return [
        state.evidence[evidence_id].model_dump(mode="json")
        for evidence_id in evidence_ids
        if evidence_id in state.evidence
    ]


def _focus_raw_windows(
    evidence: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not evidence:
        return []
    start = min(float(item.get("start", 0.0)) for item in evidence)
    end = max(float(item.get("end", start)) for item in evidence)
    window_ids = {
        str(value)
        for item in evidence
        for value in [
            item.get("window_id"),
            *(item.get("window_ids") or []),
        ]
        if value
    }

    selected = []
    for raw in raw_windows:
        raw_id = str(raw.get("window_id") or "")
        overlaps = float(raw.get("end", 0.0)) >= start - 30.0 and float(
            raw.get("start", 0.0)
        ) <= end + 30.0
        if raw_id not in window_ids and not overlaps:
            continue
        item = {
            "window_id": raw_id,
            "start": raw.get("start"),
            "end": raw.get("end"),
            "asr": str(raw.get("asr") or "")[:1200],
            "visual_latex": [
                str(value)[:500]
                for value in raw.get("visual_latex", [])[:3]
            ],
            "math_ocr_candidates": [
                {
                    "timestamp": candidate.get("timestamp"),
                    "text": str(candidate.get("text") or "")[:500],
                    "source_id": candidate.get("source_id"),
                }
                for candidate in raw.get("math_ocr_candidates", [])[:8]
            ],
        }
        extraction_unresolved = [
            str(value)[:500]
            for value in raw.get("extraction_unresolved", [])[:6]
            if str(value).strip()
        ]
        if extraction_unresolved:
            item["extraction_unresolved"] = extraction_unresolved
        selected.append(item)

    result: list[dict[str, Any]] = []
    used = 2
    for item in selected:
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if result and used + len(encoded) + 1 > max_chars:
            break
        result.append(item)
        used += len(encoded) + 1
    return result


def _focus_images(
    evidence: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    *,
    max_images: int,
) -> list[Path]:
    if max_images <= 0 or not evidence:
        return []
    start = min(float(item.get("start", 0.0)) for item in evidence)
    end = max(float(item.get("end", start)) for item in evidence)
    center = 0.5 * (start + end)
    candidates: list[tuple[float, Path]] = []
    seen: set[Path] = set()

    for raw in raw_windows:
        if float(raw.get("end", 0.0)) < start - 20.0:
            continue
        if float(raw.get("start", 0.0)) > end + 20.0:
            continue
        for crop in raw.get("formula_crops", []):
            path = Path(str(crop.get("image_path") or ""))
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            timestamp = float(crop.get("timestamp") or center)
            candidates.append((abs(timestamp - center), path))
        for frame in raw.get("board_frames", []):
            path = Path(str(frame.get("image_path") or ""))
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            timestamp = float(frame.get("timestamp") or center)
            candidates.append((abs(timestamp - center) + 5.0, path))

    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates[:max_images]]


def _frontier_summary(
    frontier: list[GraphState],
    consensus: GraphState,
    *,
    max_chars: int = 8000,
) -> list[dict[str, Any]]:
    consensus_ids = set(consensus.nodes)
    result: list[dict[str, Any]] = []
    used = 2
    for index, state in enumerate(frontier):
        branch_only = [
            {
                "id": node.id,
                "kind": node.kind,
                "title": node.title,
                "text": node.text[:220],
                "latex": (node.latex or "")[:350] or None,
                "status": node.status,
            }
            for node in state.nodes.values()
            if node.id not in consensus_ids and node.status != "suppressed"
        ]
        differing_metadata = [
            {
                "id": node_id,
                "metadata": state.nodes[node_id].metadata,
            }
            for node_id in consensus_ids.intersection(state.nodes)
            if state.nodes[node_id].metadata
            != consensus.nodes[node_id].metadata
        ]
        row = {
            "branch": index,
            "applied_patches": state.applied_patches[-12:],
            "metrics": metrics(state).model_dump(mode="json"),
            "branch_only_nodes": branch_only[:12],
            "differing_metadata": differing_metadata[:12],
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if result and used + len(encoded) + 1 > max_chars:
            break
        result.append(row)
        used += len(encoded) + 1
    return result


def _proposal_prompt(
    *,
    focus_id: str,
    focus_evidence: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    catalog: dict[str, list[Any]],
    violations: list[dict[str, Any]],
    graph_notes: list[str],
    frontier_summary: list[dict[str, Any]],
    output_language: str,
) -> str:
    return f"""Revise a mutable latent mathematical graph reconstructed from one complete lecture.

FOCUS ID:
{focus_id}

FOCUS OBSERVATIONS:
{json.dumps(focus_evidence, ensure_ascii=False, separators=(",", ":"))}

LITERAL ASR/OCR CONTEXT:
{json.dumps(raw_windows, ensure_ascii=False, separators=(",", ":"))}

CURRENT WHOLE-LECTURE GRAPH INDEX + FOCUS-RELEVANT DETAIL:
{json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))}

CURRENT UNRESOLVED GRAPH VIOLATIONS:
{json.dumps(violations, ensure_ascii=False, separators=(",", ":"))}

FRONTIER NOTES:
{json.dumps(graph_notes[-12:], ensure_ascii=False, separators=(",", ":"))}

CURRENT FRONTIER VARIANTS OUTSIDE THE CONSENSUS:
{json.dumps(frontier_summary, ensure_ascii=False, separators=(",", ":"))}

Goal: recover the globally mathematically correct state conveyed by the lecture, not a literal
transcript. Raw observations are noisy measurements. Later evidence may revise an earlier node.
A lecturer may change notation; OCR/ASR may miss it. Normalize notation in the canonical graph when
global mathematics identifies one object, while preserving evidence_ids and aliases/provenance.

This is NOT a staged pipeline. Propose generic graph edits only when they improve the current graph.
A single observation need not become a node; several observations may support one node; one
observation may support several nodes. You may revise any node in the whole-lecture catalog, not
only nodes local to the focus interval.

Useful operations include add_node, merge_nodes, split_node, retype_node, replace_node,
add_derived, add_alias, add_relation, attach_evidence, mark_evidence, mark_alternative,
suppress_node, add_violation, and resolve_violation.

Every entry in operations must be an operation object with an explicit "op" discriminator. Never
place a GraphNode directly in operations. For a new node, emit exactly the wrapper
{{"op":"add_node","node":{{...}}}} (or {{"op":"add_derived","node":{{...}}}} when appropriate).
For operations that target an existing node (replace_node, retype_node, split_node, add_alias,
attach_evidence, mark_alternative, suppress_node), the target field is node_id, NOT id.
Every suppress_node operation must include a non-empty reason.
Patch rationale and incompatible_with are JSON arrays; emit [] rather than null.
Each diagnosed_violations entry MUST use exactly:
{{"id":"...", "category":"evidence|math|structure", "severity":1|2|3,
  "message":"...", "related_nodes":["..."]}}.
Do not use kind/description/node_ids aliases for diagnosed_violations.

merge_nodes semantics:
- node_ids are source nodes to absorb;
- into_id may be a new id when merging 2+ source nodes;
- if into_id already exists and is not listed in node_ids, even a single source node is valid:
  its provenance/dependencies/aliases are absorbed into that existing canonical target;
- canonical GraphNode replacements (kind/title/text/latex/evidence_ids/derived_from/aliases/status/
  alternative_group) belong in the corresponding TOP-LEVEL operation fields;
- metadata_update is ONLY for auxiliary metadata not represented by canonical GraphNode fields;
- never emit merge_nodes with node_ids=[into_id] only; that is a meaningless self-merge.

replace_node semantics:
- put every canonical replacement directly in the corresponding TOP-LEVEL operation field;
- metadata_update is ONLY for auxiliary metadata. Never put kind, title, text, semantic_text,
  latex, evidence_ids, derived_from, aliases, status, alternative_group, provenance_notes,
  reconstruction_notes, or ambiguities inside metadata_update.

Important invariants:
- Never delete or rewrite raw evidence.
- Every non-derived canonical claim must retain direct evidence_ids.
- Derived nodes must cite derived_from nodes.
- Prefer mathematical correctness and coherent global definitions/proofs over local OCR wording.
- GraphNode.semantic_text is reader-facing mathematical prose only. Put board/OCR/ASR/provenance
  commentary in provenance_notes and reconstruction/debug commentary in reconstruction_notes.
  Keep legacy text for compatibility when needed, but do not mix audit prose into semantic_text.
- Use GraphNode.ambiguities for unresolved semantic uncertainty. Set blocking=true whenever the
  ambiguity can change mathematical truth, object identity, space type, scalar/inner-product
  convention, or the displayed formula. Advisory reading/notation uncertainty may use
  blocking=false. Do not hide a truth-changing ambiguity only inside metadata.
- A blocking ambiguity must remain explicit until evidence resolves it; do not render one branch
  as canonical merely because it is typographically convenient.
- Do not import unrelated textbook material merely because it would be true.
- Distinguish a real mathematical contradiction from harmless notation/wording variation.
- If two globally coherent interpretations remain possible, return them as alternatives rather
  than forcing one. In particular, do not silently choose a scalar-field, inner-product convention,
  symbol identity, or conjugation convention when the lecture evidence does not determine it.
- Topic/section nodes and contains/part_of relations may be added when they clarify final lecture
  organization, but do not invent a rigid outline just to satisfy formatting.
- Do not create duplicate nodes for repeated/overlapping measurements of the same mathematical
  event.
- Nodes whose kind starts with provisional_ are only weak initialization from raw observations.
  Once their evidence has been absorbed into canonical nodes, merge/suppress them or explicitly
  mark that evidence as context/repetition; do not leave a second canonical copy.
- Maintain a small coherent set of topic/section nodes with contains/part_of relations when the
  lecture has clear thematic structure. These nodes are for organization only and must follow the
  mathematics rather than imposing arbitrary fixed-duration sections. If an existing topic already
  covers the same mathematical block, attach/revise/merge it instead of creating another overlapping
  topic under a new id.
- A patch may resolve violations diagnosed in this same response by their exact ids.
- The global index lists every existing canonical node id. Never add a node under an id already
  present there; revise/retype/attach/merge the existing node instead.
- Later evidence may refine the structural role of an earlier object. If a later construction
  introduces finite intersections, completion, closure, normalization, or another refinement of an
  earlier generating family, revise the earlier canonical role rather than keeping two globally
  inconsistent labels merely because both appeared literally during the lecture.
- For an initial topology generated by a family Phi of scalar-valued functionals, the single
  inverse-image neighborhoods V(x, phi, epsilon) with one phi are, in general, elementary/subbasic
  neighborhoods. The neighborhood basis is formed by FINITE INTERSECTIONS of such sets. Do not
  claim that the single-phi family is closed under finite intersections for a general Phi. If later
  evidence introduces beta_Phi as finite intersections, use it to correct any earlier "single V is
  the full basis" wording rather than preserving both claims.
- Use standard mathematical names when the mathematical identity is globally unambiguous; noisy
  ASR spellings are aliases, not canonical theorem names. In Russian canonical prose use
  "теорема Хана–Банаха" for Hahn–Banach and "теорема/представление Рисса" for Riesz; forms such as
  "Гейма–Банаха", "Гейне–Банаха" or "Ризе" are ASR/name corruptions, not canonical names. Infer canonical terminology from the
  actual defining objects/relations, not from a lecturer's shorthand surface label. In particular,
  for constructions defined by a dual pairing, distinguish the topology/convergence generated by
  the pairing that is actually present rather than collapsing distinct standard notions under the
  word "weak". In particular, on X* the topology generated only by evaluations f -> f(x), x in X,
  is the weak-* topology sigma(X*, X); the weak topology on X* would be sigma(X*, X**).
- common_patch may contain only edits valid under every surviving interpretation. Put
  convention-dependent formulas or identities in alternatives. For a local convention ambiguity,
  keep the convention-independent theorem/core in common_patch and put the smallest possible
  convention-sensitive replace/retype operations in alternatives; do not collapse a genuine
  ambiguity into a metadata note while rendering one branch as canonical. Competing alternatives
  for one mathematical object must mutate the SAME canonical node id (normally replace_node);
  never represent a competing reading by adding a second sibling node while leaving the first
  branch-specific formula active at the same time.
- Patch ids and new node ids must be globally descriptive and stable; reuse existing ids when
  revising existing mathematics.
- Write graph prose in language code {output_language}.

Return:
1. diagnosed_violations that are true of the CURRENT graph before your edits. These are pre-edit
   diagnostics; when common_patch fixes them successfully the controller will not carry those
   same-turn diagnoses into the revised state. Use resolve_violation only for violations that were
   already present in CURRENT UNRESOLVED GRAPH VIOLATIONS;
2. one common_patch for revisions that should happen under every reasonable interpretation;
3. alternatives only when there are genuinely competing globally coherent graph revisions;
4. stable=true only if this focus needs no revision.
"""


def _proposal_fingerprint(
    *,
    state: GraphState,
    focus_id: str,
    focus_evidence: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    catalog: dict[str, list[Any]],
    frontier_summary: list[dict[str, Any]],
    llm_config: dict[str, Any],
    proposal_prompt: str,
) -> str:
    return stable_hash(
        {
            "version": GRAPH_REVISION_PROPOSAL_VERSION,
            "state_signature": _state_signature(state),
            "focus_id": focus_id,
            "focus_evidence": focus_evidence,
            "raw_windows": raw_windows,
            "catalog": catalog,
            "frontier_summary": frontier_summary,
            "violations": [
                item.model_dump(mode="json")
                for item in state.violations.values()
            ],
            "llm": llm_config,
            # Cache validity follows the actual model instruction, not only a manually bumped
            # version constant. Any semantic prompt edit therefore invalidates stale proposals.
            "proposal_prompt_hash": stable_hash(proposal_prompt),
        }
    )


def _load_cached(path: Path, fingerprint: str) -> GraphRevisionProposal | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return GraphRevisionProposal.model_validate(payload["proposal"])
    except (OSError, json.JSONDecodeError, KeyError, ValidationError):
        return None


def _load_cached_split(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return False
        split = payload.get("split")
        if not isinstance(split, dict):
            return False
        kind = str(split.get("kind") or "")
        if kind:
            # Runtime-v8 typed splits are semantic controller decisions. An output split is only
            # persisted after retrying the same focus with the full usable completion budget, so
            # replaying it is as valid as replaying an input-context split. The fingerprint still
            # binds the decision to the exact state, prompt and LLM configuration.
            return kind in {"input_overflow", "output_overflow"}

        # Migration for pre-v8 artifacts. Output truncation used to be cached as if it proved the
        # input focus was too large; replaying that split would preserve the original bug forever.
        reason = str(split.get("reason") or "").lower()
        if "structured output was truncated" in reason or "output ceiling" in reason:
            return False
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _diagnosis_patch(proposal: GraphRevisionProposal) -> GraphPatch | None:
    if not proposal.diagnosed_violations:
        return None
    return GraphPatch(
        id=f"{proposal.focus_id}::diagnosis",
        description="Model-diagnosed violations shared by all revisions of this focus.",
        operations=[
            AddViolationOp(op="add_violation", violation=item)
            for item in proposal.diagnosed_violations
        ],
    )


def _apply_or_mark_failure(
    state: GraphState,
    patch: GraphPatch | None,
    *,
    failure_id: str,
) -> GraphState:
    if patch is None:
        return state
    try:
        return apply_patch(state, patch)
    except (KeyError, ValueError) as exc:
        # A rejected proposal is a controller/audit event, not a defect in the graph that remains
        # after the edit was rejected. Keeping it as a hard structure violation biases frontier
        # pruning by model serialization mistakes rather than mathematical state quality.
        failed = state.clone()
        message = f"[{failure_id}] Unapplied patch {patch.id}: {type(exc).__name__}: {exc}"
        failed.notes.append(message)
        logger.warning("[graph_revision] %s", message)
        return failed


def _state_signature(state: GraphState) -> str:
    return stable_hash(
        {
            "nodes": {
                key: value.model_dump(mode="json")
                for key, value in sorted(state.nodes.items())
            },
            "edges": [
                edge.model_dump(mode="json")
                for edge in sorted(
                    state.edges,
                    key=lambda item: (item.source, item.target, item.relation),
                )
            ],
            "violations": {
                key: value.model_dump(mode="json")
                for key, value in sorted(state.violations.items())
            },
            "evidence_disposition": dict(sorted(state.evidence_disposition.items())),
        }
    )


def _prune_frontier(states: list[GraphState], width: int) -> list[GraphState]:
    unique: dict[str, GraphState] = {}
    for state in states:
        unique.setdefault(_state_signature(state), state)
    values = list(unique.values())
    if not values:
        return []

    scored = [(state, metrics(state)) for state in values]
    best_hard = min(
        (
            value.evidence_violations,
            value.math_violations,
            value.structure_violations,
        )
        for _, value in scored
    )
    eligible = [
        (state, value)
        for state, value in scored
        if (
            value.evidence_violations,
            value.math_violations,
            value.structure_violations,
        )
        == best_hard
    ]
    eligible.sort(
        key=lambda item: (
            item[1].unexplained_evidence,
            item[1].unsupported_nodes,
            item[1].duplicate_nodes,
            item[1].active_nodes,
            len(item[0].violations),
            _state_signature(item[0]),
        )
    )
    return [state for state, _ in eligible[:width]]


def _normalize_alternative_patch(
    state: GraphState,
    patch: GraphPatch,
) -> GraphPatch:
    """Turn a model-emitted sibling alternative into a true competing realization when safe.

    The proposer occasionally encodes a convention variant by adding a second node in the same
    alternative_group while leaving the common branch-specific node active. That makes the
    "alternative" branch a superset rather than a competing hypothesis, so consensus incorrectly
    keeps the common formula. When one new node clearly corresponds to exactly one existing node in
    the same group, same kind, and same dependency set, reinterpret it as an in-place variant of
    that canonical object. This is a controller-level representation repair, not a mathematical
    choice between the variants.
    """

    group_add_counts: dict[str, int] = {}
    for operation in patch.operations:
        if isinstance(operation, AddNodeOp) and operation.node.alternative_group:
            group = operation.node.alternative_group
            group_add_counts[group] = group_add_counts.get(group, 0) + 1

    operations: list[Any] = []
    changed = False
    for operation in patch.operations:
        if not isinstance(operation, AddNodeOp):
            operations.append(operation)
            continue

        variant = operation.node
        group = variant.alternative_group
        if not group or group_add_counts.get(group) != 1:
            operations.append(operation)
            continue

        candidates = [
            node
            for node in state.nodes.values()
            if node.status != "suppressed" and node.alternative_group == group
        ]
        if len(candidates) != 1:
            operations.append(operation)
            continue

        target = candidates[0]
        if (
            target.kind != variant.kind
            or set(target.derived_from) != set(variant.derived_from)
        ):
            operations.append(operation)
            continue

        metadata_update = dict(variant.metadata)
        metadata_update["frontier_variant_source_id"] = variant.id
        operations.append(
            ReplaceNodeOp(
                op="replace_node",
                node_id=target.id,
                title=variant.title,
                text=variant.text,
                latex=variant.latex,
                metadata_update=metadata_update,
            )
        )

        missing_evidence = [
            evidence_id
            for evidence_id in variant.evidence_ids
            if evidence_id not in target.evidence_ids
        ]
        if missing_evidence:
            operations.append(
                AttachEvidenceOp(
                    op="attach_evidence",
                    node_id=target.id,
                    evidence_ids=missing_evidence,
                )
            )
        for alias in variant.aliases:
            if alias not in target.aliases:
                operations.append(
                    AddAliasOp(
                        op="add_alias",
                        node_id=target.id,
                        alias=alias,
                    )
                )
        changed = True

    if not changed:
        return patch
    return patch.model_copy(update={"operations": operations}, deep=True)


def _expand_frontier(
    frontier: list[GraphState],
    proposal: GraphRevisionProposal,
    *,
    width: int,
) -> list[GraphState]:
    diagnosis = _diagnosis_patch(proposal)
    expanded: list[GraphState] = []

    for branch_index, state in enumerate(frontier):
        common_succeeded = proposal.common_patch is None
        if proposal.common_patch is None:
            current = state
        else:
            try:
                current = apply_patch(state, proposal.common_patch)
                common_succeeded = True
            except (KeyError, ValueError):
                current = _apply_or_mark_failure(
                    state,
                    proposal.common_patch,
                    failure_id=f"{proposal.focus_id}::common_failed::{branch_index}",
                )

        # diagnosed_violations describe the CURRENT PRE-EDIT graph. If the common revision applies
        # successfully, carrying those diagnoses into the revised state creates stale hard
        # violations. Persist them only when no revision was supplied or the revision failed; a
        # genuinely remaining issue can be diagnosed again by a later focus.
        if proposal.common_patch is None or not common_succeeded:
            current = _apply_or_mark_failure(
                current,
                diagnosis,
                failure_id=f"{proposal.focus_id}::diagnosis_failed::{branch_index}",
            )

        if not proposal.alternatives:
            expanded.append(current)
            continue

        # The common state remains a viable unresolved hypothesis. Alternatives are refinements of
        # that state, not a command to silently discard the baseline. This also makes a single
        # model-supplied alternative meaningful: baseline + alternative form a two-branch frontier.
        expanded.append(current)
        for alt_index, alternative in enumerate(proposal.alternatives):
            try:
                normalized = _normalize_alternative_patch(current, alternative)
                expanded.append(apply_patch(current, normalized))
            except (KeyError, ValueError):
                continue

    return _prune_frontier(expanded, width)


def _process_focus_resilient(
    orchestrator: KnowledgeOrchestrator,
    *,
    frontier: list[GraphState],
    evidence_ids: list[str],
    focus_id: str,
    raw_windows: list[dict[str, Any]],
    proposal_root: Path,
    llm_config: dict[str, Any],
    frontier_width: int,
    catalog_chars: int,
    raw_context_chars: int,
    max_images: int,
    max_tokens: int,
    force: bool,
    stats: dict[str, Any],
    split_depth: int = 0,
) -> tuple[list[GraphState], bool]:
    """Process one focus, recursively splitting only when the structured input cannot fit.

    Child focuses are applied sequentially. The second child is therefore proposed against the
    frontier already revised by the first child rather than against a stale pre-split graph.
    """

    focus_started = time.perf_counter()
    logger.info(
        "[graph_revision] focus %s start: observations=%d depth=%d %s",
        focus_id,
        len(evidence_ids),
        split_depth,
        _frontier_log_line(frontier),
    )

    representative = graph_consensus(frontier)
    focus = _focus_evidence(representative, evidence_ids)
    raw_context = _focus_raw_windows(
        focus,
        raw_windows,
        max_chars=raw_context_chars,
    )
    catalog = _compact_catalog(
        representative,
        catalog_chars,
        focus_evidence_ids=evidence_ids,
    )
    frontier_summary = _frontier_summary(
        frontier,
        representative,
    )
    proposal_prompt = _proposal_prompt(
        focus_id=focus_id,
        focus_evidence=focus,
        raw_windows=raw_context,
        catalog=catalog,
        violations=[
            item.model_dump(mode="json")
            for item in representative.violations.values()
        ],
        graph_notes=representative.notes,
        frontier_summary=frontier_summary,
        output_language=orchestrator.output_language,
    )
    fingerprint = _proposal_fingerprint(
        state=representative,
        focus_id=focus_id,
        focus_evidence=focus,
        raw_windows=raw_context,
        catalog=catalog,
        frontier_summary=frontier_summary,
        llm_config=llm_config,
        proposal_prompt=proposal_prompt,
    )
    path = proposal_root / f"{focus_id}.json"

    if not force and _load_cached_split(path, fingerprint):
        stats["split_cache_hits"] += 1
        logger.info(
            "[graph_revision] focus %s cached split: observations=%d",
            focus_id,
            len(evidence_ids),
        )
        midpoint = len(evidence_ids) // 2
        if midpoint <= 0:
            raise StructuredTaskTooLargeError(
                f"{focus_id} cached split cannot divide a single-observation focus"
            )
        left_ids = evidence_ids[:midpoint]
        right_ids = evidence_ids[midpoint:]
        frontier, left_changed = _process_focus_resilient(
            orchestrator,
            frontier=frontier,
            evidence_ids=left_ids,
            focus_id=f"{focus_id}__a",
            raw_windows=raw_windows,
            proposal_root=proposal_root,
            llm_config=llm_config,
            frontier_width=frontier_width,
            catalog_chars=catalog_chars,
            raw_context_chars=raw_context_chars,
            max_images=max_images,
            max_tokens=max_tokens,
            force=force,
            stats=stats,
            split_depth=split_depth + 1,
        )
        frontier, right_changed = _process_focus_resilient(
            orchestrator,
            frontier=frontier,
            evidence_ids=right_ids,
            focus_id=f"{focus_id}__b",
            raw_windows=raw_windows,
            proposal_root=proposal_root,
            llm_config=llm_config,
            frontier_width=frontier_width,
            catalog_chars=catalog_chars,
            raw_context_chars=raw_context_chars,
            max_images=max_images,
            max_tokens=max_tokens,
            force=force,
            stats=stats,
            split_depth=split_depth + 1,
        )
        return frontier, left_changed or right_changed

    proposal = None if force else _load_cached(path, fingerprint)
    if proposal is not None:
        proposal.focus_id = focus_id
        stats["cache_hits"] += 1
        logger.info("[graph_revision] focus %s proposal cache hit", focus_id)
    else:
        images = _focus_images(
            focus,
            raw_windows,
            max_images=max_images,
        )
        logger.info(
            "[graph_revision] focus %s requesting proposal: observations=%d "
            "catalog_index=%d catalog_detail=%d images=%d max_tokens=%d",
            focus_id,
            len(evidence_ids),
            len(catalog.get("index", [])),
            len(catalog.get("detail", [])),
            len(images),
            max_tokens,
        )

        def request_proposal(request_max_tokens: int) -> GraphRevisionProposal:
            stats["focus_calls"] += 1
            return orchestrator._structured(
                proposal_prompt,
                GraphRevisionProposal,
                operation="graph_revision_proposal",
                images=images or None,
                guided_json=not bool(images),
                split_oversized_task=True,
                max_tokens=request_max_tokens,
                thinking=True,
                temperature=0.6,
                top_p=0.9,
            )

        split_exc: StructuredTaskTooLargeError | None = None
        split_kind = "input_overflow"
        try:
            proposal = request_proposal(max_tokens)
        except StructuredOutputTruncatedError as exc:
            global_max_tokens = int(llm_config.get("max_tokens") or max_tokens)
            actual_budget = int(exc.max_tokens or max_tokens)
            if global_max_tokens > actual_budget:
                stats["output_budget_retries"] += 1
                logger.warning(
                    "[graph_revision] focus %s output truncated at max_tokens=%d; "
                    "retrying same focus with global max_tokens=%d",
                    focus_id,
                    actual_budget,
                    global_max_tokens,
                )
                try:
                    proposal = request_proposal(global_max_tokens)
                except StructuredInputTooLargeError as retry_exc:
                    split_exc = retry_exc
                    split_kind = "input_overflow"
                except StructuredOutputTruncatedError as retry_exc:
                    split_exc = retry_exc
                    split_kind = "output_overflow"
                except StructuredBackendAmbiguousRejectionError as retry_exc:
                    split_exc = retry_exc
                    split_kind = "ambiguous_backend_rejection"
                except StructuredTaskTooLargeError as retry_exc:
                    split_exc = retry_exc
            else:
                split_exc = exc
                split_kind = "output_overflow"
        except StructuredInputTooLargeError as exc:
            split_exc = exc
            split_kind = "input_overflow"
        except StructuredBackendAmbiguousRejectionError as exc:
            split_exc = exc
            split_kind = "ambiguous_backend_rejection"
        except StructuredTaskTooLargeError as exc:
            # Compatibility with lightweight/older adapters that only know the base exception.
            split_exc = exc

        if split_exc is not None:
            if len(evidence_ids) <= 1:
                if split_kind == "output_overflow":
                    raise StructuredOutputTruncatedError(
                        f"{focus_id} still exceeds the full structured-output budget",
                        max_tokens=getattr(split_exc, "max_tokens", max_tokens),
                        raw_chars=getattr(split_exc, "raw_chars", None),
                    ) from split_exc
                if isinstance(split_exc, StructuredBackendAmbiguousRejectionError):
                    raise split_exc.backend_error
                raise StructuredInputTooLargeError(
                    f"{focus_id} still cannot fit after recursive focus splitting"
                ) from split_exc

            midpoint = len(evidence_ids) // 2
            left_ids = evidence_ids[:midpoint]
            right_ids = evidence_ids[midpoint:]
            stats["split_focuses"] += 1
            if split_kind == "output_overflow":
                stats["output_split_focuses"] += 1
            elif split_kind == "ambiguous_backend_rejection":
                stats["ambiguous_backend_split_focuses"] = (
                    int(stats.get("ambiguous_backend_split_focuses", 0)) + 1
                )
            else:
                stats["input_split_focuses"] += 1
            logger.warning(
                "[graph_revision] focus %s %s; split %d observations -> %d + %d",
                focus_id,
                split_kind,
                len(evidence_ids),
                len(left_ids),
                len(right_ids),
            )
            stats["max_split_depth"] = max(
                int(stats["max_split_depth"]),
                split_depth + 1,
            )
            atomic_json_dump(
                path,
                {
                    "fingerprint": fingerprint,
                    "focus_evidence_ids": evidence_ids,
                    "split": {
                        "kind": split_kind,
                        "reason": str(split_exc),
                        "children": [
                            f"{focus_id}__a",
                            f"{focus_id}__b",
                        ],
                    },
                },
            )

            frontier, left_changed = _process_focus_resilient(
                orchestrator,
                frontier=frontier,
                evidence_ids=left_ids,
                focus_id=f"{focus_id}__a",
                raw_windows=raw_windows,
                proposal_root=proposal_root,
                llm_config=llm_config,
                frontier_width=frontier_width,
                catalog_chars=catalog_chars,
                raw_context_chars=raw_context_chars,
                max_images=max_images,
                max_tokens=max_tokens,
                force=force,
                stats=stats,
                split_depth=split_depth + 1,
            )
            frontier, right_changed = _process_focus_resilient(
                orchestrator,
                frontier=frontier,
                evidence_ids=right_ids,
                focus_id=f"{focus_id}__b",
                raw_windows=raw_windows,
                proposal_root=proposal_root,
                llm_config=llm_config,
                frontier_width=frontier_width,
                catalog_chars=catalog_chars,
                raw_context_chars=raw_context_chars,
                max_images=max_images,
                max_tokens=max_tokens,
                force=force,
                stats=stats,
                split_depth=split_depth + 1,
            )
            return frontier, left_changed or right_changed

        proposal.focus_id = focus_id
        logger.info(
            "[graph_revision] focus %s proposal ready: common_ops=%d alternatives=%d "
            "diagnosed_violations=%d stable=%s",
            focus_id,
            len(proposal.common_patch.operations) if proposal.common_patch is not None else 0,
            len(proposal.alternatives),
            len(proposal.diagnosed_violations),
            proposal.stable,
        )
        atomic_json_dump(
            path,
            {
                "fingerprint": fingerprint,
                "focus_evidence_ids": evidence_ids,
                "proposal": proposal.model_dump(mode="json"),
            },
        )

    before = {_state_signature(state) for state in frontier}
    if (
        proposal.stable
        and not proposal.diagnosed_violations
        and proposal.common_patch is None
        and not proposal.alternatives
    ):
        stats["stable_focuses"] += 1
        logger.info(
            "[graph_revision] focus %s stable: %.1fs %s",
            focus_id,
            time.perf_counter() - focus_started,
            _frontier_log_line(frontier),
        )
        return frontier, False

    frontier = _expand_frontier(
        frontier,
        proposal,
        width=frontier_width,
    )
    after = {_state_signature(state) for state in frontier}
    stats["frontier_sizes"].append(len(frontier))
    changed = before != after
    logger.info(
        "[graph_revision] focus %s done: changed=%s alternatives=%d %.1fs %s",
        focus_id,
        changed,
        len(proposal.alternatives),
        time.perf_counter() - focus_started,
        _frontier_log_line(frontier),
    )
    return frontier, changed


def run_iterative_graph_revision(
    orchestrator: KnowledgeOrchestrator,
    *,
    lecture_state: LectureState,
    raw_windows: list[dict[str, Any]],
    work: Path,
    llm_config: dict[str, Any],
    rounds: int,
    batch_observations: int,
    overlap_observations: int,
    frontier_width: int,
    catalog_chars: int,
    raw_context_chars: int,
    max_images: int,
    max_tokens: int,
    force: bool,
) -> GraphRevisionRun:
    root = work / "graph_revision"
    proposal_root = root / "proposals"
    proposal_root.mkdir(parents=True, exist_ok=True)

    frontier = [seed_observation_graph(lecture_state)]
    batches = _focus_batches(
        frontier[0],
        batch_observations=batch_observations,
        overlap_observations=overlap_observations,
    )
    stats: dict[str, Any] = {
        "version": GRAPH_REVISION_PROPOSAL_VERSION,
        "rounds_requested": rounds,
        "rounds_completed": 0,
        "focus_calls": 0,
        "cache_hits": 0,
        "stable_focuses": 0,
        "split_focuses": 0,
        "split_cache_hits": 0,
        "input_split_focuses": 0,
        "output_split_focuses": 0,
        "output_budget_retries": 0,
        "max_split_depth": 0,
        "frontier_sizes": [],
    }

    logger.info(
        "[graph_revision] start: observations=%d batches=%d rounds=%d frontier_width=%d",
        len(lecture_state.observations),
        len(batches),
        rounds,
        frontier_width,
    )

    for round_index in range(rounds):
        round_started = time.perf_counter()
        changed = False
        direction = "forward" if round_index % 2 == 0 else "reverse"
        ordered_batches = batches if round_index % 2 == 0 else list(reversed(batches))
        logger.info(
            "[graph_revision] round %d/%d start: direction=%s focuses=%d %s",
            round_index + 1,
            rounds,
            direction,
            len(ordered_batches),
            _frontier_log_line(frontier),
        )

        for batch_index, evidence_ids in enumerate(ordered_batches):
            focus_id = f"round_{round_index:02d}_focus_{batch_index:03d}"
            frontier, focus_changed = _process_focus_resilient(
                orchestrator,
                frontier=frontier,
                evidence_ids=evidence_ids,
                focus_id=focus_id,
                raw_windows=raw_windows,
                proposal_root=proposal_root,
                llm_config=llm_config,
                frontier_width=frontier_width,
                catalog_chars=catalog_chars,
                raw_context_chars=raw_context_chars,
                max_images=max_images,
                max_tokens=max_tokens,
                force=force,
                stats=stats,
            )
            changed = changed or focus_changed

        stats["rounds_completed"] = round_index + 1
        logger.info(
            "[graph_revision] round %d/%d done: changed=%s %.1fs %s",
            round_index + 1,
            rounds,
            changed,
            time.perf_counter() - round_started,
            _frontier_log_line(frontier),
        )
        if not changed:
            logger.info(
                "[graph_revision] converged after round %d; stopping early",
                round_index + 1,
            )
            break

    consensus = graph_consensus(frontier)
    logger.info(
        "[graph_revision] consensus ready: %s",
        _frontier_log_line(frontier),
    )
    root.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(frontier):
        atomic_json_dump(
            root / f"frontier_{index:02d}.json",
            state.model_dump(mode="json"),
        )
    atomic_json_dump(
        root / "consensus_graph.json",
        consensus.model_dump(mode="json"),
    )
    atomic_json_dump(
        root / "summary.json",
        {
            **stats,
            "frontier": [
                {
                    "index": index,
                    "metrics": metrics(state).model_dump(mode="json"),
                    "applied_patches": state.applied_patches,
                }
                for index, state in enumerate(frontier)
            ],
            "consensus_metrics": metrics(consensus).model_dump(mode="json"),
        },
    )
    logger.info(
        "[graph_revision] complete: rounds=%d llm_calls=%d cache_hits=%d "
        "splits=%d split_cache_hits=%d consensus=%s",
        stats["rounds_completed"],
        stats["focus_calls"],
        stats["cache_hits"],
        stats["split_focuses"],
        stats["split_cache_hits"],
        _metrics_log_line(consensus),
    )
    return GraphRevisionRun(
        frontier=frontier,
        consensus=consensus,
        stats=stats,
    )
