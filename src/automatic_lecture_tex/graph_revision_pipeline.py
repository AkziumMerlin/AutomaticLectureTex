from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .graph_revision import (
    AddViolationOp,
    GraphPatch,
    GraphState,
    SearchResult,
    StateMetrics,
    Violation,
    apply_patch,
    graph_consensus,
    metrics,
    seed_observation_graph,
)
from .knowledge import KnowledgeOrchestrator
from .llm import StructuredTaskTooLargeError
from .schemas import LectureState
from .util import atomic_json_dump, stable_hash

GRAPH_REVISION_PROPOSAL_VERSION = 1


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


def _compact_catalog(state: GraphState, max_chars: int) -> list[dict[str, Any]]:
    rows = []
    for node in sorted(
        state.nodes.values(),
        key=lambda item: (_node_time(state, item.id), item.id),
    ):
        if node.status == "suppressed":
            continue
        rows.append(
            {
                "id": node.id,
                "kind": node.kind,
                "title": node.title,
                "text": node.text[:280],
                "latex": (node.latex or "")[:500] or None,
                "aliases": node.aliases[:8],
                "status": node.status,
                "alternative_group": node.alternative_group,
                "evidence_ids": node.evidence_ids[:16],
                "derived_from": node.derived_from[:12],
            }
        )

    selected: list[dict[str, Any]] = []
    used = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if selected and used + len(encoded) + 1 > max_chars:
            break
        selected.append(row)
        used += len(encoded) + 1
    return selected


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
        selected.append(
            {
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
                        "timestamp": item.get("timestamp"),
                        "text": str(item.get("text") or "")[:500],
                        "source_id": item.get("source_id"),
                    }
                    for item in raw.get("math_ocr_candidates", [])[:8]
                ],
            }
        )

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
    catalog: list[dict[str, Any]],
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

CURRENT WHOLE-LECTURE GRAPH CATALOG:
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

Important invariants:
- Never delete or rewrite raw evidence.
- Every non-derived canonical claim must retain direct evidence_ids.
- Derived nodes must cite derived_from nodes.
- Prefer mathematical correctness and coherent global definitions/proofs over local OCR wording.
- Do not import unrelated textbook material merely because it would be true.
- Distinguish a real mathematical contradiction from harmless notation/wording variation.
- If two globally coherent interpretations remain possible, return them as alternatives rather
  than forcing one.
- Topic/section nodes and contains/part_of relations may be added when they clarify final lecture
  organization, but do not invent a rigid outline just to satisfy formatting.
- Do not create duplicate nodes for repeated/overlapping measurements of the same mathematical
  event.
- Nodes whose kind starts with provisional_ are only weak initialization from raw observations.
  Once their evidence has been absorbed into canonical nodes, merge/suppress them or explicitly
  mark that evidence as context/repetition; do not leave a second canonical copy.
- Maintain a small coherent set of topic/section nodes with contains/part_of relations when the
  lecture has clear thematic structure. These nodes are for organization only and must follow the
  mathematics rather than imposing arbitrary fixed-duration sections.
- A patch may resolve violations diagnosed in this same response by their exact ids.
- Patch ids and new node ids must be globally descriptive and stable; reuse existing ids when
  revising existing mathematics.
- Write graph prose in language code {output_language}.

Return:
1. diagnosed_violations that are true of the CURRENT graph before your edits;
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
    catalog: list[dict[str, Any]],
    frontier_summary: list[dict[str, Any]],
    llm_config: dict[str, Any],
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
        return payload.get("fingerprint") == fingerprint and bool(payload.get("split"))
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
        failed = state.clone()
        failed.violations[failure_id] = Violation(
            id=failure_id,
            category="structure",
            severity=1,
            message=f"Graph patch could not be applied: {type(exc).__name__}: {exc}",
        )
        failed.notes.append(
            f"Unapplied patch {patch.id}: {type(exc).__name__}: {exc}"
        )
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


def _expand_frontier(
    frontier: list[GraphState],
    proposal: GraphRevisionProposal,
    *,
    width: int,
) -> list[GraphState]:
    diagnosis = _diagnosis_patch(proposal)
    expanded: list[GraphState] = []

    for branch_index, state in enumerate(frontier):
        current = _apply_or_mark_failure(
            state,
            diagnosis,
            failure_id=f"{proposal.focus_id}::diagnosis_failed::{branch_index}",
        )
        current = _apply_or_mark_failure(
            current,
            proposal.common_patch,
            failure_id=f"{proposal.focus_id}::common_failed::{branch_index}",
        )
        if not proposal.alternatives:
            expanded.append(current)
            continue

        successful = False
        for alt_index, alternative in enumerate(proposal.alternatives):
            try:
                expanded.append(apply_patch(current, alternative))
                successful = True
            except (KeyError, ValueError):
                continue
        if not successful:
            expanded.append(current)

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

    representative = graph_consensus(frontier)
    focus = _focus_evidence(representative, evidence_ids)
    raw_context = _focus_raw_windows(
        focus,
        raw_windows,
        max_chars=raw_context_chars,
    )
    catalog = _compact_catalog(representative, catalog_chars)
    frontier_summary = _frontier_summary(
        frontier,
        representative,
    )
    fingerprint = _proposal_fingerprint(
        state=representative,
        focus_id=focus_id,
        focus_evidence=focus,
        raw_windows=raw_context,
        catalog=catalog,
        frontier_summary=frontier_summary,
        llm_config=llm_config,
    )
    path = proposal_root / f"{focus_id}.json"

    if not force and _load_cached_split(path, fingerprint):
        stats["split_cache_hits"] += 1
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
    else:
        images = _focus_images(
            focus,
            raw_windows,
            max_images=max_images,
        )
        stats["focus_calls"] += 1
        try:
            proposal = orchestrator._structured(
                _proposal_prompt(
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
                ),
                GraphRevisionProposal,
                operation="graph_revision_proposal",
                images=images or None,
                guided_json=not bool(images),
                split_oversized_task=True,
                max_tokens=max_tokens,
                thinking=True,
                temperature=0.6,
                top_p=0.9,
            )
        except StructuredTaskTooLargeError as exc:
            if len(evidence_ids) <= 1:
                raise StructuredTaskTooLargeError(
                    f"{focus_id} still cannot fit after recursive focus splitting"
                ) from exc

            midpoint = len(evidence_ids) // 2
            left_ids = evidence_ids[:midpoint]
            right_ids = evidence_ids[midpoint:]
            stats["split_focuses"] += 1
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
                        "reason": str(exc),
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
        return frontier, False

    frontier = _expand_frontier(
        frontier,
        proposal,
        width=frontier_width,
    )
    after = {_state_signature(state) for state in frontier}
    stats["frontier_sizes"].append(len(frontier))
    return frontier, before != after


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
        "max_split_depth": 0,
        "frontier_sizes": [],
    }

    for round_index in range(rounds):
        changed = False
        ordered_batches = batches if round_index % 2 == 0 else list(reversed(batches))

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
        if not changed:
            break

    consensus = graph_consensus(frontier)
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
    return GraphRevisionRun(
        frontier=frontier,
        consensus=consensus,
        stats=stats,
    )
