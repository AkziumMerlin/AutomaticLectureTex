from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import AppConfig, LectureConfig
from .knowledge import KnowledgeOrchestrator
from .schemas import LectureObservation, LectureState, ObservationKind
from .util import atomic_json_dump, stable_hash

GRAPH_RECONSTRUCTION_VERSION = 1
_CANDIDATE_BATCH_MAX_CHARS = 24_000
_EDGE_BATCH_MAX_CHARS = 28_000


RelationKind = Literal[
    "continuation",
    "same_object",
    "supports",
    "contradicts",
    "independent",
    "uncertain",
]


class CandidateDraft(BaseModel):
    """One locally plausible interpretation proposed from noisy evidence only."""

    model_config = ConfigDict(extra="forbid")

    kind: ObservationKind
    text: str = ""
    latex: str | None = None
    evidence_score: float = Field(ge=-3.0, le=3.0)
    evidence_refs: list[str] = Field(default_factory=list)
    note: str = ""


class ObservationCandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    source_evidence_score: float = Field(ge=-3.0, le=3.0)
    null_evidence_score: float = Field(ge=-3.0, le=3.0)
    candidates: list[CandidateDraft] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class CandidateProposalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[ObservationCandidateProposal] = Field(default_factory=list)


class CandidateHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    observation_id: str
    kind: ObservationKind | Literal["null"]
    text: str = ""
    latex: str | None = None
    unary_score: float
    source: Literal["source", "model", "null"]
    evidence_refs: list[str] = Field(default_factory=list)
    note: str = ""


class CandidateSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    start: float
    end: float
    episode_id: str = ""
    candidates: list[CandidateHypothesis]
    unresolved: list[str] = Field(default_factory=list)


class PairwiseScoreDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_candidate_id: str
    right_candidate_id: str
    score: int = Field(ge=-3, le=3)
    relation: RelationKind
    reason: str = ""


class PairwiseScoreBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[PairwiseScoreDraft] = Field(default_factory=list)


class PairwisePotential(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_candidate_id: str
    right_candidate_id: str
    score: float
    relation: RelationKind | Literal["null"]
    reason: str = ""


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_observation_id: str
    right_observation_id: str
    reasons: list[str] = Field(default_factory=list)
    potentials: list[PairwisePotential] = Field(default_factory=list)


class GlobalHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    assignments: dict[str, str]


@dataclass(frozen=True)
class EdgeSpec:
    left_observation_id: str
    right_observation_id: str
    reasons: tuple[str, ...]


def _compact_observation(observation: LectureObservation) -> dict[str, Any]:
    return {
        "id": observation.id,
        "start": observation.start,
        "end": observation.end,
        "kind": str(observation.kind),
        "text": observation.text,
        "latex": observation.latex,
        "confidence": observation.confidence,
        "source_status": str(observation.source_status),
        "evidence_refs": list(observation.evidence_refs),
    }


def _load_source_state(work: Path) -> tuple[LectureState, Path]:
    candidates = [
        work / "lecture_state_pre_semantic_graph.json",
        work / "lecture_state_pre_claim_compaction.json",
        work / "lecture_state.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        return LectureState.model_validate_json(path.read_text(encoding="utf-8")), path
    raise FileNotFoundError(
        "No state artifact found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_raw_context_index(work: Path) -> dict[str, dict[str, Any]]:
    """Merge sensor-level context retained by local repair artifacts.

    The reconstruction prototype ignores every repair decision. Duplicate resolution artifacts are
    common because the same observation can appear in topic-level and repaired-episode passes, so
    their raw evidence is merged rather than whichever path happens to be visited last winning.
    """

    result: dict[str, dict[str, Any]] = {}
    root = work / "state_observation_resolutions"
    if not root.exists():
        return result

    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        current = payload.get("current") or {}
        observation_id = str(current.get("id") or path.stem)
        if not observation_id:
            continue
        entry = result.setdefault(
            observation_id,
            {
                "current_sensor_hypothesis": current,
                "raw_windows": [],
                "evidence_catalog": [],
                "images": [],
            },
        )
        if current and not entry.get("current_sensor_hypothesis"):
            entry["current_sensor_hypothesis"] = current

        for key in ("raw_windows", "evidence_catalog", "images"):
            seen = {
                json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                for item in entry[key]
            }
            for item in payload.get(key) or []:
                signature = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                if signature in seen:
                    continue
                entry[key].append(item)
                seen.add(signature)
    return result


def _bounded_json(value: Any, limit: int) -> Any:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(raw) <= limit:
        return value
    return {
        "truncated": True,
        "prefix": raw[:limit],
    }


def _proposal_payload(
    observations: list[LectureObservation],
    index: int,
    raw_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    observation = observations[index]
    lo = max(0, index - 2)
    hi = min(len(observations), index + 3)
    neighbors = [
        _compact_observation(item)
        for j, item in enumerate(observations[lo:hi], start=lo)
        if j != index
    ]
    return {
        "observation": _compact_observation(observation),
        "sensor_context": _bounded_json(raw_context.get(observation.id, {}), 8_000),
        "neighboring_uncommitted_observations": neighbors,
    }


def _chunk_payloads(
    payloads: list[dict[str, Any]],
    *,
    max_items: int,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 2
    for payload in payloads:
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if current and (len(current) >= max_items or chars + size > max_chars):
            batches.append(current)
            current = []
            chars = 2
        current.append(payload)
        chars += size + 1
    if current:
        batches.append(current)
    return batches


def _candidate_fingerprint(
    payload: dict[str, Any],
    *,
    candidate_count: int,
    llm_config: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "version": GRAPH_RECONSTRUCTION_VERSION,
            "payload": payload,
            "candidate_count": candidate_count,
            "llm": llm_config,
        }
    )


def _proposal_prompt(
    payloads: list[dict[str, Any]],
    *,
    candidate_count: int,
    output_language: str,
) -> str:
    return f"""Generate competing LOCAL hypotheses for noisy mathematical lecture observations.

Inputs:
{json.dumps(payloads, ensure_ascii=False, separators=(",", ":"))}

This is candidate generation, not reconstruction and not note writing. Do NOT select a final
interpretation. For every observation_id score the supplied source hypothesis and return up to
{candidate_count - 2} materially distinct non-null alternatives. The host keeps the source as an
explicit candidate; do not repeat it among candidates.

Rules:
- source_evidence_score, null_evidence_score and candidate evidence_score measure LOCAL support
  from the supplied sensor evidence only, on [-3, 3], and must be directly comparable within that
  observation;
- null_evidence_score scores the hypothesis that the local evidence is insufficient to commit to
  any specific mathematical reading; it is not a generic penalty;
- do not use later textbook knowledge or global mathematical consistency to repair the source;
- preserve lecturer mistakes if they are a locally plausible reading;
- if several glyphs, formulas, referents or statements remain plausible, keep alternatives separate;
- do not create cosmetic paraphrases as separate candidates;
- evidence_refs must be copied from refs visible in the input; never invent refs;
- a candidate may have plain semantic text, exact LaTeX, or both;
- use an empty candidate list when the source reading is already the only locally supported reading;
- report ambiguity that cannot be represented compactly in unresolved;
- write prose in language code {output_language}.

The host adds both the original source hypothesis and an explicit NULL/unknown hypothesis. Your job
is only to propose alternatives that should remain alive for later global inference.
"""


def _source_candidate(
    observation: LectureObservation,
    *,
    evidence_score: float,
) -> CandidateHypothesis:
    score = float(evidence_score)
    return CandidateHypothesis(
        id=f"{observation.id}:source",
        observation_id=observation.id,
        kind=observation.kind,
        text=observation.text,
        latex=observation.latex,
        unary_score=score,
        source="source",
        evidence_refs=list(observation.evidence_refs),
        note="Original extracted observation; not canonical truth.",
    )


def _null_candidate(
    observation: LectureObservation,
    *,
    evidence_score: float,
) -> CandidateHypothesis:
    return CandidateHypothesis(
        id=f"{observation.id}:null",
        observation_id=observation.id,
        kind="null",
        unary_score=float(evidence_score),
        source="null",
        note="Observation left unresolved by global reconstruction.",
    )


def _candidate_key(
    kind: ObservationKind | Literal["null"],
    text: str,
    latex: str | None,
) -> tuple[str, str, str]:
    return (
        str(kind),
        re.sub(r"\s+", " ", text).strip().casefold(),
        re.sub(r"\s+", "", latex or ""),
    )


def _materialize_candidate_set(
    observation: LectureObservation,
    proposal: ObservationCandidateProposal | None,
    *,
    raw_context: dict[str, Any],
    candidate_count: int,
) -> CandidateSet:
    candidates: list[CandidateHypothesis] = [
        _source_candidate(
            observation,
            evidence_score=(
                proposal.source_evidence_score
                if proposal is not None
                else 2.0 * float(observation.confidence) - 1.0
            ),
        )
    ]
    seen = {
        _candidate_key(
            candidates[0].kind,
            candidates[0].text,
            candidates[0].latex,
        )
    }

    if proposal is not None:
        allowed_refs = set(observation.evidence_refs)
        allowed_refs.update(
            str(item["ref"])
            for item in raw_context.get("evidence_catalog", [])
            if isinstance(item, dict) and item.get("ref")
        )
        for index, draft in enumerate(proposal.candidates):
            if len(candidates) >= candidate_count - 1:
                break
            key = _candidate_key(draft.kind, draft.text, draft.latex)
            if key in seen:
                # If the model simply rediscovered the source reading, let its local evidence score
                # calibrate the source unary instead of duplicating a state.
                candidates[0].unary_score = max(candidates[0].unary_score, draft.evidence_score)
                continue
            seen.add(key)
            candidates.append(
                CandidateHypothesis(
                    id=f"{observation.id}:model{index}",
                    observation_id=observation.id,
                    kind=draft.kind,
                    text=draft.text,
                    latex=draft.latex,
                    unary_score=float(draft.evidence_score),
                    source="model",
                    evidence_refs=[
                        ref for ref in draft.evidence_refs if ref in allowed_refs
                    ],
                    note=draft.note,
                )
            )

    candidates.append(
        _null_candidate(
            observation,
            evidence_score=(
                proposal.null_evidence_score
                if proposal is not None
                else -0.15 - 0.7 * float(observation.confidence)
            ),
        )
    )
    return CandidateSet(
        observation_id=observation.id,
        start=observation.start,
        end=observation.end,
        episode_id=observation.episode_id,
        candidates=candidates,
        unresolved=list(proposal.unresolved) if proposal is not None else [],
    )


def generate_candidate_sets(
    orchestrator: KnowledgeOrchestrator,
    *,
    observations: list[LectureObservation],
    raw_context: dict[str, dict[str, Any]],
    cache_dir: Path,
    llm_config: dict[str, Any],
    candidate_count: int,
    batch_size: int,
    force: bool,
) -> list[CandidateSet]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        _proposal_payload(observations, index, raw_context)
        for index in range(len(observations))
    ]
    proposals: dict[str, ObservationCandidateProposal] = {}
    missing_payloads: list[dict[str, Any]] = []

    for payload in payloads:
        observation_id = str(payload["observation"]["id"])
        fingerprint = _candidate_fingerprint(
            payload,
            candidate_count=candidate_count,
            llm_config=llm_config,
        )
        path = cache_dir / f"{observation_id}.json"
        if path.exists() and not force:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    proposals[observation_id] = ObservationCandidateProposal.model_validate(
                        cached["proposal"]
                    )
                    continue
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                pass
        payload["_fingerprint"] = fingerprint
        missing_payloads.append(payload)

    batches = _chunk_payloads(
        missing_payloads,
        max_items=batch_size,
        max_chars=_CANDIDATE_BATCH_MAX_CHARS,
    )
    for batch in batches:
        clean_batch = [
            {key: value for key, value in payload.items() if key != "_fingerprint"}
            for payload in batch
        ]
        generated = orchestrator._structured(
            _proposal_prompt(
                clean_batch,
                candidate_count=candidate_count,
                output_language=orchestrator.output_language,
            ),
            CandidateProposalBatch,
            operation="graph_candidate_generation",
            split_oversized_task=True,
            thinking=False,
            temperature=0.5,
            top_p=0.9,
        )
        by_id = {item.observation_id: item for item in generated.observations}

        for payload in batch:
            observation_id = str(payload["observation"]["id"])
            proposal = by_id.get(
                observation_id,
                ObservationCandidateProposal(
                    observation_id=observation_id,
                    source_evidence_score=(
                        2.0 * float(payload["observation"]["confidence"]) - 1.0
                    ),
                    null_evidence_score=(
                        -0.15 - 0.7 * float(payload["observation"]["confidence"])
                    ),
                    unresolved=["Candidate generator omitted this observation; source/null kept."],
                ),
            )
            proposals[observation_id] = proposal
            atomic_json_dump(
                cache_dir / f"{observation_id}.json",
                {
                    "fingerprint": payload["_fingerprint"],
                    "proposal": proposal.model_dump(mode="json"),
                },
            )

    return [
        _materialize_candidate_set(
            observation,
            proposals.get(observation.id),
            raw_context=raw_context.get(observation.id, {}),
            candidate_count=candidate_count,
        )
        for observation in observations
    ]


_LATEX_SYMBOL_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z](?:_[A-Za-z0-9{}]+)?")
_TEXT_SYMBOL_RE = re.compile(r"(?<![\w])(?:[A-Z][A-Za-z0-9_*']*|[a-z]_[A-Za-z0-9{}]+)(?![\w])")


def _candidate_symbols(candidate: CandidateHypothesis) -> set[str]:
    symbols = set(_LATEX_SYMBOL_RE.findall(candidate.latex or ""))
    symbols.update(_TEXT_SYMBOL_RE.findall(candidate.text))
    return {
        symbol
        for symbol in symbols
        if symbol not in {"text", "frac", "left", "right", "begin", "end"}
    }


def _set_symbols(candidate_set: CandidateSet) -> set[str]:
    result: set[str] = set()
    for candidate in candidate_set.candidates:
        if candidate.kind == "null":
            continue
        result.update(_candidate_symbols(candidate))
    return result


def build_sparse_edges(
    candidate_sets: list[CandidateSet],
    *,
    neighbor_span: int,
    max_gap_seconds: float,
    symbol_gap_seconds: float,
) -> list[EdgeSpec]:
    """Build a deliberately sparse graph.

    Temporal edges carry local continuity. Non-local edges are reserved for a small number of
    distinctive shared symbols; ubiquitous glyphs such as x/y/f and structural LaTeX commands do
    not create graph-wide cliques.
    """

    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    symbols = [_set_symbols(item) for item in candidate_sets]
    structural = {
        "\\in",
        "\\quad",
        "\\qquad",
        "\\text",
        "\\forall",
        "\\exists",
        "\\bigl",
        "\\bigr",
        "\\lVert",
        "\\rVert",
        "\\lvert",
        "\\rvert",
        "\\to",
        "\\neq",
        "\\le",
        "\\ge",
        "\\Longrightarrow",
        "\\mathbb",
        "\\left",
        "\\right",
        "\\frac",
        "\\begin",
        "\\end",
        "\\cdot",
    }
    for index, item_symbols in enumerate(symbols):
        symbols[index] = {
            symbol
            for symbol in item_symbols
            if symbol not in structural
            and not (len(symbol) == 1 and symbol.isalpha())
        }

    frequency: dict[str, int] = defaultdict(int)
    for item_symbols in symbols:
        for symbol in item_symbols:
            frequency[symbol] += 1
    max_document_frequency = max(4, len(candidate_sets) // 3)
    symbols = [
        {
            symbol
            for symbol in item_symbols
            if frequency[symbol] <= max_document_frequency
        }
        for item_symbols in symbols
    ]

    for i, left in enumerate(candidate_sets):
        # Always retain a narrow temporal backbone.
        for j in range(i + 1, min(len(candidate_sets), i + 1 + neighbor_span)):
            right = candidate_sets[j]
            gap = max(0.0, right.start - left.end)
            if gap > max_gap_seconds:
                continue
            pair = (left.observation_id, right.observation_id)
            edges[pair].add("temporal")

        # Add at most two non-local symbol links per node. Prefer rare symbols, then short gaps.
        nonlocal_candidates: list[tuple[float, float, int]] = []
        for j in range(i + 1, len(candidate_sets)):
            right = candidate_sets[j]
            gap = max(0.0, right.start - left.end)
            if gap > symbol_gap_seconds:
                break
            shared = symbols[i].intersection(symbols[j])
            if not shared:
                continue
            rarity = sum(1.0 / max(1, frequency[symbol]) for symbol in shared)
            nonlocal_candidates.append((-rarity, gap, j))

        for _rarity, _gap, j in sorted(nonlocal_candidates)[:2]:
            right = candidate_sets[j]
            edges[(left.observation_id, right.observation_id)].add("shared_symbol")

    return [
        EdgeSpec(left, right, tuple(sorted(reasons)))
        for (left, right), reasons in sorted(edges.items())
    ]


def _edge_payload(
    edge: EdgeSpec,
    by_id: dict[str, CandidateSet],
) -> dict[str, Any]:
    left = by_id[edge.left_observation_id]
    right = by_id[edge.right_observation_id]

    def compact(candidate: CandidateHypothesis) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "kind": str(candidate.kind),
            "text": candidate.text,
            "latex": candidate.latex,
        }

    return {
        "left_observation_id": edge.left_observation_id,
        "right_observation_id": edge.right_observation_id,
        "edge_reasons": list(edge.reasons),
        "left": [compact(item) for item in left.candidates if item.kind != "null"],
        "right": [compact(item) for item in right.candidates if item.kind != "null"],
    }


def _edge_fingerprint(payload: dict[str, Any], llm_config: dict[str, Any]) -> str:
    return stable_hash(
        {
            "version": GRAPH_RECONSTRUCTION_VERSION,
            "payload": payload,
            "llm": llm_config,
        }
    )


def _edge_prompt(payloads: list[dict[str, Any]], *, output_language: str) -> str:
    return f"""Score pairwise compatibility between competing lecture-graph hypotheses.

Edges:
{json.dumps(payloads, ensure_ascii=False, separators=(",", ":"))}

For EVERY non-null candidate pair on every edge, emit one PairwiseScoreDraft.

The score is an INTEGER RELATIONAL potential in {-3,-2,-1,0,1,2,3}, not another estimate of local
sensor confidence:
+3  strongly mutually consistent / one clearly continues or supports the other
+1  weakly useful compatibility
 0  independent / no information
-1  weak tension
-3  mutually incompatible interpretations

Use only relationships that can be justified from the two candidate statements plus their temporal
or shared-symbol relation. Standard mathematics is a consistency prior, not permission to silently
rewrite the lecture. A lecturer mistake can therefore be globally selected when it is what the
evidence trajectory supports. Do not favour polished textbook statements merely for being true.

relation must be one of continuation, same_object, supports, contradicts, independent, uncertain.
Keep reason short and write it in language code {output_language}. Do not score NULL candidates;
the host assigns zero pairwise potential to them.
"""


def _score_edge_batch(
    orchestrator: KnowledgeOrchestrator,
    payloads: list[dict[str, Any]],
) -> PairwiseScoreBatch:
    return orchestrator._structured(
        _edge_prompt(payloads, output_language=orchestrator.output_language),
        PairwiseScoreBatch,
        operation="graph_pairwise_scoring",
        split_oversized_task=True,
        thinking=False,
        temperature=0.2,
        top_p=0.8,
    )


def score_sparse_edges(
    orchestrator: KnowledgeOrchestrator,
    *,
    candidate_sets: list[CandidateSet],
    edge_specs: list[EdgeSpec],
    cache_dir: Path,
    llm_config: dict[str, Any],
    batch_size: int,
    force: bool,
) -> list[GraphEdge]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_id = {item.observation_id: item for item in candidate_sets}
    payload_by_key = {
        (edge.left_observation_id, edge.right_observation_id): _edge_payload(edge, by_id)
        for edge in edge_specs
    }
    generated_by_key: dict[tuple[str, str], list[PairwiseScoreDraft]] = {}
    missing: list[dict[str, Any]] = []

    for edge in edge_specs:
        key = (edge.left_observation_id, edge.right_observation_id)
        payload = payload_by_key[key]
        fingerprint = _edge_fingerprint(payload, llm_config)
        safe_name = stable_hash({"left": key[0], "right": key[1]})[:20]
        path = cache_dir / f"{safe_name}.json"
        if path.exists() and not force:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    generated_by_key[key] = [
                        PairwiseScoreDraft.model_validate(item)
                        for item in cached.get("scores", [])
                    ]
                    continue
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        missing.append(
            {
                "key": key,
                "fingerprint": fingerprint,
                "path": path,
                "payload": payload,
            }
        )

    payload_batches = _chunk_payloads(
        [item["payload"] for item in missing],
        max_items=batch_size,
        max_chars=_EDGE_BATCH_MAX_CHARS,
    )
    cursor = 0
    for batch_payloads in payload_batches:
        batch_items = missing[cursor : cursor + len(batch_payloads)]
        cursor += len(batch_payloads)
        generated = _score_edge_batch(orchestrator, batch_payloads)
        scores_by_edge: dict[tuple[str, str], list[PairwiseScoreDraft]] = defaultdict(list)

        candidate_to_observation = {
            candidate.id: item.observation_id
            for item in candidate_sets
            for candidate in item.candidates
        }
        for score in generated.scores:
            left_obs = candidate_to_observation.get(score.left_candidate_id)
            right_obs = candidate_to_observation.get(score.right_candidate_id)
            if left_obs is None or right_obs is None:
                continue
            scores_by_edge[(left_obs, right_obs)].append(score)

        for item in batch_items:
            key = item["key"]
            scores = scores_by_edge.get(key, [])
            generated_by_key[key] = scores
            atomic_json_dump(
                item["path"],
                {
                    "fingerprint": item["fingerprint"],
                    "payload": item["payload"],
                    "scores": [score.model_dump(mode="json") for score in scores],
                },
            )

    edges: list[GraphEdge] = []
    for spec in edge_specs:
        key = (spec.left_observation_id, spec.right_observation_id)
        left = by_id[spec.left_observation_id]
        right = by_id[spec.right_observation_id]
        allowed_left = {item.id for item in left.candidates if item.kind != "null"}
        allowed_right = {item.id for item in right.candidates if item.kind != "null"}
        seen_pairs: set[tuple[str, str]] = set()
        potentials: list[PairwisePotential] = []

        for item in generated_by_key.get(key, []):
            pair = (item.left_candidate_id, item.right_candidate_id)
            if pair[0] not in allowed_left or pair[1] not in allowed_right or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            potentials.append(
                PairwisePotential(
                    left_candidate_id=pair[0],
                    right_candidate_id=pair[1],
                    score=float(item.score),
                    relation=item.relation,
                    reason=item.reason,
                )
            )

        # Guided output can omit a pair. Missing scores are explicit zero/uncertain factors rather
        # than an implicit failure that changes the graph topology.
        for left_candidate in left.candidates:
            for right_candidate in right.candidates:
                pair = (left_candidate.id, right_candidate.id)
                if pair in seen_pairs:
                    continue
                if left_candidate.kind == "null" or right_candidate.kind == "null":
                    relation: RelationKind | Literal["null"] = "null"
                else:
                    relation = "uncertain"
                potentials.append(
                    PairwisePotential(
                        left_candidate_id=pair[0],
                        right_candidate_id=pair[1],
                        score=0.0,
                        relation=relation,
                        reason="No explicit pairwise evidence.",
                    )
                )

        edges.append(
            GraphEdge(
                left_observation_id=spec.left_observation_id,
                right_observation_id=spec.right_observation_id,
                reasons=list(spec.reasons),
                potentials=potentials,
            )
        )

    return edges


def _potential_lookup(edges: list[GraphEdge]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for edge in edges:
        for potential in edge.potentials:
            result[(potential.left_candidate_id, potential.right_candidate_id)] = potential.score
    return result


def _incoming_edges(
    candidate_sets: list[CandidateSet],
    edges: list[GraphEdge],
) -> dict[int, list[tuple[int, GraphEdge]]]:
    index = {item.observation_id: i for i, item in enumerate(candidate_sets)}
    incoming: dict[int, list[tuple[int, GraphEdge]]] = defaultdict(list)
    for edge in edges:
        left = index.get(edge.left_observation_id)
        right = index.get(edge.right_observation_id)
        if left is None or right is None or left >= right:
            continue
        incoming[right].append((left, edge))
    return incoming


def beam_map_inference(
    candidate_sets: list[CandidateSet],
    edges: list[GraphEdge],
    *,
    beam_width: int,
    top_k: int,
    pairwise_weight: float,
) -> list[GlobalHypothesis]:
    """Approximate top-K MAP for an ordered sparse pairwise graph.

    Exact inference on a general loopy discrete graph is NP-hard. The lecture graph is naturally
    time ordered and sparse, so a left-to-right max-sum beam is a useful prototype backend: every
    factor is scored exactly once when its right endpoint is assigned, while only the global state
    pruning is approximate.
    """

    if not candidate_sets:
        return []

    incoming = _incoming_edges(candidate_sets, edges)
    potentials = _potential_lookup(edges)
    # (score, tuple(candidate_id in observation order))
    beam: list[tuple[float, tuple[str, ...]]] = [(0.0, tuple())]

    for right_index, candidate_set in enumerate(candidate_sets):
        expanded: list[tuple[float, tuple[str, ...]]] = []
        for score, assignment in beam:
            for candidate in candidate_set.candidates:
                total = score + float(candidate.unary_score)
                for left_index, _edge in incoming.get(right_index, []):
                    left_candidate_id = assignment[left_index]
                    total += pairwise_weight * potentials.get(
                        (left_candidate_id, candidate.id),
                        0.0,
                    )
                expanded.append((total, (*assignment, candidate.id)))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beam = expanded[:beam_width]

    hypotheses: list[GlobalHypothesis] = []
    for score, assignment in beam[:top_k]:
        hypotheses.append(
            GlobalHypothesis(
                score=score,
                assignments={
                    candidate_sets[index].observation_id: candidate_id
                    for index, candidate_id in enumerate(assignment)
                },
            )
        )
    return hypotheses


def _local_assignment(candidate_sets: list[CandidateSet]) -> dict[str, str]:
    return {
        item.observation_id: max(item.candidates, key=lambda candidate: candidate.unary_score).id
        for item in candidate_sets
    }


def _assignment_score(
    assignment: dict[str, str],
    candidate_sets: list[CandidateSet],
    edges: list[GraphEdge],
    *,
    pairwise_weight: float,
) -> tuple[float, float, float]:
    candidate_by_id = {
        candidate.id: candidate
        for item in candidate_sets
        for candidate in item.candidates
    }
    unary = sum(
        candidate_by_id[candidate_id].unary_score
        for candidate_id in assignment.values()
    )
    pairwise = 0.0
    for edge in edges:
        left = assignment.get(edge.left_observation_id)
        right = assignment.get(edge.right_observation_id)
        if left is None or right is None:
            continue
        potential = next(
            (
                item
                for item in edge.potentials
                if item.left_candidate_id == left and item.right_candidate_id == right
            ),
            None,
        )
        if potential is not None:
            pairwise += pairwise_weight * potential.score
    return unary + pairwise, unary, pairwise


def _include_local_baseline(
    hypotheses: list[GlobalHypothesis],
    local_assignment: dict[str, str],
    candidate_sets: list[CandidateSet],
    edges: list[GraphEdge],
    *,
    pairwise_weight: float,
    top_k: int,
) -> list[GlobalHypothesis]:
    local_score, _, _ = _assignment_score(
        local_assignment,
        candidate_sets,
        edges,
        pairwise_weight=pairwise_weight,
    )
    by_assignment = {
        tuple(sorted(item.assignments.items())): item
        for item in hypotheses
    }
    key = tuple(sorted(local_assignment.items()))
    previous = by_assignment.get(key)
    if previous is None or local_score > previous.score:
        by_assignment[key] = GlobalHypothesis(
            score=local_score,
            assignments=dict(local_assignment),
        )
    return sorted(by_assignment.values(), key=lambda item: item.score, reverse=True)[:top_k]


def _top_k_marginals(
    hypotheses: list[GlobalHypothesis],
    candidate_sets: list[CandidateSet],
) -> dict[str, dict[str, float]]:
    if not hypotheses:
        return {}
    top = hypotheses[0].score
    weights = [math.exp(max(-60.0, hypothesis.score - top)) for hypothesis in hypotheses]
    normalizer = sum(weights) or 1.0
    result: dict[str, dict[str, float]] = {
        item.observation_id: {candidate.id: 0.0 for candidate in item.candidates}
        for item in candidate_sets
    }
    for hypothesis, weight in zip(hypotheses, weights, strict=True):
        for observation_id, candidate_id in hypothesis.assignments.items():
            result[observation_id][candidate_id] += weight / normalizer
    return result


def _selected_edges(
    hypothesis: GlobalHypothesis,
    edges: list[GraphEdge],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for edge in edges:
        left_candidate = hypothesis.assignments.get(edge.left_observation_id)
        right_candidate = hypothesis.assignments.get(edge.right_observation_id)
        if left_candidate is None or right_candidate is None:
            continue
        potential = next(
            (
                item
                for item in edge.potentials
                if item.left_candidate_id == left_candidate
                and item.right_candidate_id == right_candidate
            ),
            None,
        )
        if potential is None:
            continue
        if (
            potential.relation in {"independent", "uncertain", "null"}
            and abs(potential.score) < 0.5
        ):
            continue
        selected.append(
            {
                "left_observation_id": edge.left_observation_id,
                "right_observation_id": edge.right_observation_id,
                "left_candidate_id": left_candidate,
                "right_candidate_id": right_candidate,
                "relation": potential.relation,
                "score": potential.score,
                "reason": potential.reason,
                "edge_reasons": edge.reasons,
            }
        )
    return selected


def run_graph_reconstruction(
    *,
    config: AppConfig,
    lecture: LectureConfig,
    llm,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_observations: int | None = None,
    candidate_count: int = 4,
    candidate_batch_size: int = 6,
    edge_batch_size: int = 5,
    neighbor_span: int = 2,
    max_gap_seconds: float = 90.0,
    symbol_gap_seconds: float = 240.0,
    beam_width: int = 256,
    top_k: int = 8,
    pairwise_weight: float = 1.0,
    force: bool = False,
) -> Path:
    if candidate_count < 3:
        raise ValueError("candidate_count must be >= 3 (source + alternative + null)")
    if beam_width < top_k:
        raise ValueError("beam_width must be >= top_k")

    work = config.runtime.work_dir / lecture.id
    state, state_path = _load_source_state(work)
    observations = sorted(state.observations, key=lambda item: (item.start, item.end, item.id))
    if start_seconds is not None:
        observations = [item for item in observations if item.end >= start_seconds]
    if end_seconds is not None:
        observations = [item for item in observations if item.start <= end_seconds]
    if max_observations is not None and max_observations > 0:
        observations = observations[:max_observations]
    if not observations:
        raise ValueError("No observations selected for graph reconstruction")

    root = work / "graph_reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    raw_context = _load_raw_context_index(work)
    orchestrator = KnowledgeOrchestrator(
        llm=llm,
        config=config.notes,
        output_language=config.llm.output_language,
    )

    candidate_sets = generate_candidate_sets(
        orchestrator,
        observations=observations,
        raw_context=raw_context,
        cache_dir=root / "candidates",
        llm_config=config.llm.model_dump(mode="json"),
        candidate_count=candidate_count,
        batch_size=candidate_batch_size,
        force=force,
    )
    edge_specs = build_sparse_edges(
        candidate_sets,
        neighbor_span=neighbor_span,
        max_gap_seconds=max_gap_seconds,
        symbol_gap_seconds=symbol_gap_seconds,
    )
    edges = score_sparse_edges(
        orchestrator,
        candidate_sets=candidate_sets,
        edge_specs=edge_specs,
        cache_dir=root / "edges",
        llm_config=config.llm.model_dump(mode="json"),
        batch_size=edge_batch_size,
        force=force,
    )
    hypotheses = beam_map_inference(
        candidate_sets,
        edges,
        beam_width=beam_width,
        top_k=top_k,
        pairwise_weight=pairwise_weight,
    )
    if not hypotheses:
        raise RuntimeError("Graph inference returned no hypotheses")

    local = _local_assignment(candidate_sets)
    hypotheses = _include_local_baseline(
        hypotheses,
        local,
        candidate_sets,
        edges,
        pairwise_weight=pairwise_weight,
        top_k=top_k,
    )
    best = hypotheses[0]
    changed = [
        observation_id
        for observation_id, candidate_id in best.assignments.items()
        if local.get(observation_id) != candidate_id
    ]
    candidate_by_id = {
        candidate.id: candidate
        for item in candidate_sets
        for candidate in item.candidates
    }
    marginals = _top_k_marginals(hypotheses, candidate_sets)
    score_margin = (
        hypotheses[0].score - hypotheses[1].score
        if len(hypotheses) > 1
        else None
    )
    global_score, global_unary, global_pairwise = _assignment_score(
        best.assignments,
        candidate_sets,
        edges,
        pairwise_weight=pairwise_weight,
    )
    local_score, local_unary, local_pairwise = _assignment_score(
        local,
        candidate_sets,
        edges,
        pairwise_weight=pairwise_weight,
    )
    selected_sources = {
        source: sum(
            candidate_by_id[candidate_id].source == source
            for candidate_id in best.assignments.values()
        )
        for source in ("source", "model", "null")
    }

    artifact = {
        "version": GRAPH_RECONSTRUCTION_VERSION,
        "lecture_id": lecture.id,
        "source_state": str(state_path),
        "selection": {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "max_observations": max_observations,
        },
        "parameters": {
            "candidate_count": candidate_count,
            "neighbor_span": neighbor_span,
            "max_gap_seconds": max_gap_seconds,
            "symbol_gap_seconds": symbol_gap_seconds,
            "beam_width": beam_width,
            "top_k": top_k,
            "pairwise_weight": pairwise_weight,
        },
        "summary": {
            "observations": len(candidate_sets),
            "candidate_states": sum(len(item.candidates) for item in candidate_sets),
            "edges": len(edges),
            "local_vs_global_changes": len(changed),
            "changed_observation_ids": changed,
            "top1_score": best.score,
            "top1_top2_margin": score_margin,
            "global_unary_score": global_unary,
            "global_pairwise_score": global_pairwise,
            "local_joint_score": local_score,
            "local_unary_score": local_unary,
            "local_pairwise_score": local_pairwise,
            "joint_score_gain": global_score - local_score,
            "selected_sources": selected_sources,
        },
        "candidate_sets": [item.model_dump(mode="json") for item in candidate_sets],
        "edges": [item.model_dump(mode="json") for item in edges],
        "local_assignment": local,
        "top_hypotheses": [item.model_dump(mode="json") for item in hypotheses],
        "top_k_marginals": marginals,
    }
    artifact_path = root / "graph_reconstruction.json"
    atomic_json_dump(artifact_path, artifact)

    canonical = {
        "version": GRAPH_RECONSTRUCTION_VERSION,
        "lecture_id": lecture.id,
        "score": best.score,
        "nodes": [
            {
                "observation_id": item.observation_id,
                "start": item.start,
                "end": item.end,
                "selected": candidate_by_id[best.assignments[item.observation_id]].model_dump(
                    mode="json"
                ),
                "top_k_mass": marginals.get(item.observation_id, {}).get(
                    best.assignments[item.observation_id],
                    0.0,
                ),
                "alternatives": [
                    {
                        "candidate": candidate.model_dump(mode="json"),
                        "top_k_mass": marginals.get(item.observation_id, {}).get(candidate.id, 0.0),
                    }
                    for candidate in item.candidates
                    if candidate.id != best.assignments[item.observation_id]
                ],
            }
            for item in candidate_sets
        ],
        "edges": _selected_edges(best, edges),
    }
    atomic_json_dump(root / "canonical_lecture_graph.json", canonical)
    return artifact_path
