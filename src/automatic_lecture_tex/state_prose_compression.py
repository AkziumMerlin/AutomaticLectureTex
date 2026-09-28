from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .schemas import (
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    ObservationKind,
    OutlineSection,
)
from .util import atomic_json_dump, stable_hash


STATE_PROSE_COMPRESSION_VERSION = 3
_SUMMARY_KINDS = {ObservationKind.REMARK, ObservationKind.NOTATION}
_DEDUPE_KINDS = {ObservationKind.REMARK, ObservationKind.NOTATION}
_FORMAL_TEXT_MARKERS = (
    "\\",
    "$",
    "=",
    "<",
    ">",
    "≤",
    "≥",
    "≠",
    "∈",
    "∉",
    "⊂",
    "⊆",
    "→",
    "↦",
    "⇒",
    "⇔",
    "∑",
    "∫",
    "∥",
)
_META_WORDS = (
    "лектор",
    "доск",
    "asr",
    "ocr",
    "кадр",
    "реконструк",
    "распозна",
)
_SINGLE_LATIN_SYMBOL = re.compile(r"(?<![A-Za-z0-9])[A-Za-z](?![A-Za-z0-9])")
_NUMBER_TOKEN = re.compile(r"\d+")
_CYRILLIC_LETTER = re.compile(r"[А-Яа-яЁё]")
_ALPHA_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
_INLINE_EQUALITY = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9_()]*\s*=\s*[−-]?[A-Za-z][A-Za-z0-9_()]*)"
)


class ProseCompressionSentence(BaseModel):
    text: str
    source_observation_ids: list[str] = Field(min_length=1)


class ProseSummaryGroupProposal(BaseModel):
    source_observation_ids: list[str] = Field(min_length=2)
    sentences: list[ProseCompressionSentence] = Field(min_length=1)


class ProseRedundancyGroupProposal(BaseModel):
    """Selection-only semantic deduplication: no generated replacement text."""

    source_observation_ids: list[str] = Field(min_length=2)
    representative_observation_id: str


class ProseCompressionPlan(BaseModel):
    summary_groups: list[ProseSummaryGroupProposal] = Field(default_factory=list)
    redundant_text_groups: list[ProseRedundancyGroupProposal] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class ProseCompressionRenderGroup(BaseModel):
    section_id: str
    anchor_observation_id: str
    source_observation_ids: list[str]
    summary_text: str


class ProseCompressionPolicy(BaseModel):
    """Render-only text policy. Mathematical state and LaTeX channels are untouched."""

    groups: list[ProseCompressionRenderGroup] = Field(default_factory=list)
    suppress_text_ids: list[str] = Field(default_factory=list)


def _section_observations(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
) -> list[LectureObservation]:
    episode_ids = set(section.episode_ids)
    observation_ids = {
        observation_id
        for episode in kb.episodes
        if episode.id in episode_ids
        for observation_id in episode.observation_ids
    }
    selected = [
        item
        for item in kb.observations
        if item.id in observation_ids or item.episode_id in episode_ids
    ]
    unique: dict[str, LectureObservation] = {}
    for item in sorted(selected, key=lambda obs: (obs.start, obs.end, obs.id)):
        unique.setdefault(item.id, item)
    return list(unique.values())


def _contains_formal_text(value: str) -> bool:
    return any(marker in value for marker in _FORMAL_TEXT_MARKERS)


def _summary_candidate(item: LectureObservation) -> bool:
    if item.kind not in _SUMMARY_KINDS or item.latex:
        return False
    text = item.text.strip()
    return bool(text) and not _contains_formal_text(text)


def _summary_runs(
    observations: list[LectureObservation],
    *,
    min_group_size: int,
) -> list[list[LectureObservation]]:
    """Build prose runs while treating transitions as transparent structural evidence."""

    runs: list[list[LectureObservation]] = []
    current: list[LectureObservation] = []
    for item in observations:
        if item.kind == ObservationKind.TRANSITION:
            # Hierarchy already consumed this event. It should neither be rewritten nor split
            # otherwise adjacent prose that belongs to one final-note setup.
            continue
        if _summary_candidate(item):
            current.append(item)
            continue
        if len(current) >= min_group_size:
            runs.append(current)
        current = []
    if len(current) >= min_group_size:
        runs.append(current)
    return runs


def _redundancy_candidates(
    observations: list[LectureObservation],
) -> list[LectureObservation]:
    return [
        item
        for item in observations
        if item.kind in _DEDUPE_KINDS and item.text.strip()
    ]


def _compact_summary_run(index: int, run: list[LectureObservation]) -> dict[str, Any]:
    return {
        "run_id": f"run_{index:03d}",
        "observations": [
            {"id": item.id, "kind": str(item.kind), "text": item.text.strip()}
            for item in run
        ],
    }


def _compact_redundancy_candidates(
    observations: list[LectureObservation],
) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "kind": str(item.kind),
            "text": item.text.strip(),
            "has_separate_latex": bool(item.latex),
            "contains_formal_text": _contains_formal_text(item.text),
        }
        for item in observations
    ]


def _load_plan(path: Path, fingerprint: str) -> ProseCompressionPlan | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return ProseCompressionPlan.model_validate(payload["plan"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _sentence_count(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    return max(1, len(re.findall(r"[.!?](?=\s|$)", text)))


def _symbol_tokens(value: str) -> set[str]:
    return set(_SINGLE_LATIN_SYMBOL.findall(value))


def _number_tokens(value: str) -> set[str]:
    return set(_NUMBER_TOKEN.findall(value))


def _language_ok(value: str, output_language: str) -> bool:
    if not output_language.casefold().startswith("ru"):
        return True
    letters = _ALPHA_LETTER.findall(value)
    if not letters:
        return True
    cyrillic = _CYRILLIC_LETTER.findall(value)
    return len(cyrillic) / len(letters) >= 0.55


def _formal_snippets(value: str) -> set[str]:
    """Extract simple literal inline equalities for host-side redundancy checks."""

    normalized = value.replace("−", "-")
    return {
        re.sub(r"\s+", "", match)
        for match in _INLINE_EQUALITY.findall(normalized)
    }


def _validate_summary_group(
    proposal: ProseSummaryGroupProposal,
    *,
    section_id: str,
    runs: list[list[LectureObservation]],
    used_ids: set[str],
    min_group_size: int,
    max_sentences: int,
    max_ratio: float,
    max_summary_chars: int,
) -> tuple[ProseCompressionRenderGroup | None, str]:
    source_ids = list(dict.fromkeys(proposal.source_observation_ids))
    if len(source_ids) < min_group_size:
        return None, f"summary group has fewer than {min_group_size} observations"
    if any(item in used_ids for item in source_ids):
        return None, "summary group overlaps another accepted prose group"

    matched_run: list[LectureObservation] | None = None
    for run in runs:
        index = {item.id: i for i, item in enumerate(run)}
        if all(item in index for item in source_ids):
            positions = [index[item] for item in source_ids]
            if positions != sorted(positions):
                return None, "source IDs are not in lecture order"
            if positions != list(range(positions[0], positions[-1] + 1)):
                return None, "source IDs are not a contiguous slice of one prose run"
            matched_run = run
            break
    if matched_run is None:
        return None, "summary group crosses a protected formal-content boundary"

    if len(proposal.sentences) > max_sentences:
        return None, f"group exceeds max_sentences={max_sentences}"

    group_set = set(source_ids)
    cited: set[str] = set()
    sentence_texts: list[str] = []
    for sentence in proposal.sentences:
        text = re.sub(r"\s+", " ", sentence.text.strip())
        sentence_sources = list(dict.fromkeys(sentence.source_observation_ids))
        if not text:
            return None, "empty summary sentence"
        if not sentence_sources or any(item not in group_set for item in sentence_sources):
            return None, "sentence cites an observation outside its group"
        if _contains_formal_text(text):
            return None, "generated summary contains formal mathematical syntax"
        lowered = text.casefold()
        if any(word in lowered for word in _META_WORDS):
            return None, "generated summary contains lecture/reconstruction narration"
        cited.update(sentence_sources)
        sentence_texts.append(text)

    if cited != group_set:
        return None, "sentence provenance does not cover the summary group exactly"

    summary = " ".join(sentence_texts).strip()
    if _sentence_count(summary) > max_sentences:
        return None, "rendered summary contains too many sentences"

    by_id = {item.id: item for item in matched_run}
    source_text = " ".join(by_id[item].text for item in source_ids)
    allowed_chars = min(
        max_summary_chars,
        max(80, int(len(source_text.strip()) * max_ratio)),
    )
    if len(summary) > allowed_chars:
        return None, (
            f"summary is not compact enough ({len(summary)} chars > {allowed_chars} allowed)"
        )

    new_symbols = _symbol_tokens(summary) - _symbol_tokens(source_text)
    if new_symbols:
        return None, "summary introduces new standalone Latin symbols: " + ", ".join(
            sorted(new_symbols)
        )
    new_numbers = _number_tokens(summary) - _number_tokens(source_text)
    if new_numbers:
        return None, "summary introduces new numeric tokens: " + ", ".join(
            sorted(new_numbers)
        )

    return (
        ProseCompressionRenderGroup(
            section_id=section_id,
            anchor_observation_id=source_ids[0],
            source_observation_ids=source_ids,
            summary_text=summary,
        ),
        "",
    )


def _validate_redundancy_group(
    proposal: ProseRedundancyGroupProposal,
    *,
    candidates: list[LectureObservation],
    used_ids: set[str],
    max_candidate_span: int = 8,
) -> tuple[set[str], str]:
    source_ids = list(dict.fromkeys(proposal.source_observation_ids))
    if len(source_ids) < 2:
        return set(), "redundancy group has fewer than two observations"
    if proposal.representative_observation_id not in source_ids:
        return set(), "representative is not a member of the redundancy group"
    if any(item in used_ids for item in source_ids):
        return set(), "redundancy group overlaps an accepted summary/dedup group"

    by_id = {item.id: item for item in candidates}
    if any(item not in by_id for item in source_ids):
        return set(), "redundancy group contains a non-remark/non-notation observation"
    positions = [next(i for i, item in enumerate(candidates) if item.id == oid) for oid in source_ids]
    if positions != sorted(positions):
        return set(), "redundancy IDs are not in lecture order"
    if positions[-1] - positions[0] > max_candidate_span:
        return set(), "redundancy group spans too many semantic text events"

    suppress: set[str] = set()
    for observation_id in source_ids:
        if observation_id == proposal.representative_observation_id:
            continue
        item = by_id[observation_id]
        # If formal content exists only inside prose, keep that text even when the model calls the
        # surrounding remark redundant. If a separate LaTeX channel exists, suppressing prose
        # cannot remove the formula itself.
        if _contains_formal_text(item.text) and not item.latex:
            continue
        suppress.add(observation_id)

    if not suppress:
        return set(), "no text channel in the proposed redundancy group is safe to suppress"
    return suppress, ""


def _compress_section(
    orchestrator,
    *,
    section: OutlineSection,
    observations: list[LectureObservation],
    runs: list[list[LectureObservation]],
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    min_group_size: int,
    max_sentences: int,
    max_ratio: float,
    max_summary_chars: int,
    force: bool,
) -> tuple[
    list[ProseCompressionRenderGroup],
    set[str],
    list[dict[str, Any]],
    list[str],
    int,
    int,
]:
    redundancy_candidates = _redundancy_candidates(observations)
    compact_runs = [_compact_summary_run(index, run) for index, run in enumerate(runs)]
    compact_redundancy = _compact_redundancy_candidates(redundancy_candidates)
    fingerprint = stable_hash(
        {
            "state_pipeline_version": pipeline_version,
            "prose_compression_version": STATE_PROSE_COMPRESSION_VERSION,
            "section": {
                "id": section.id,
                "title": section.title,
                "start": section.start,
                "end": section.end,
            },
            "summary_runs": compact_runs,
            "redundancy_candidates": compact_redundancy,
            "min_group_size": min_group_size,
            "max_sentences": max_sentences,
            "max_ratio": max_ratio,
            "max_summary_chars": max_summary_chars,
            "llm": llm_config,
        }
    )
    path = work / "state_prose_compression_sections" / f"{section.id}.json"
    plan = None if force else _load_plan(path, fingerprint)
    cache_hit = int(plan is not None)
    model_calls = 0

    if plan is None:
        prompt = f"""Compact repetitive lecture prose after mathematical repair is already finished.

Section title:
{section.title}

MATH-FREE SUMMARY RUNS:
{json.dumps(compact_runs, ensure_ascii=False, separators=(",", ":"))}

SELECTION-ONLY REMARK/NOTATION CANDIDATES:
{json.dumps(compact_redundancy, ensure_ascii=False, separators=(",", ":"))}

You have TWO strictly different operations.

1. summary_groups:
- only use observations from ONE MATH-FREE SUMMARY RUN;
- use at least {min_group_size} observations, preferably an entire repetitive run;
- write at most {max_sentences} short final-note sentences;
- every sentence cites exact source observation IDs and their union equals the group IDs;
- retain distinct facts, but remove narration about what the lecturer says/writes/repeats/points at;
- output plain prose only: no formulas, LaTeX, relation symbols, provenance or reconstruction talk.

2. redundant_text_groups:
- this is SELECTION ONLY; generate NO replacement text;
- group nearby remark/notation observations only when they repeat the same semantic point;
- choose ONE existing representative_observation_id from the group;
- be conservative: do not group merely related statements or successive proof developments;
- mathematical formulas are protected by the host. You are only deciding whether surrounding TEXT
  channels are repetitive.

Do not touch definitions, claims, equations, proof steps, examples, corrections or retractions.
Do not invent corrected mathematics or textbook material. If uncertain, omit a group.
"""
        try:
            plan = orchestrator._structured(
                prompt,
                ProseCompressionPlan,
                operation="state_prose_compress",
                max_tokens=2048,
                split_oversized_task=True,
                thinking=False,
                temperature=0.6,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )
            model_calls = 1
        except Exception as exc:
            return (
                [],
                set(),
                [],
                [
                    "State prose compression failed safely for "
                    f"{section.id}: {type(exc).__name__}: {exc}"
                ],
                cache_hit,
                model_calls,
            )
        atomic_json_dump(
            path,
            {"fingerprint": fingerprint, "plan": plan.model_dump(mode="json")},
        )

    accepted: list[ProseCompressionRenderGroup] = []
    suppress_text: set[str] = set()
    audit: list[dict[str, Any]] = []
    unresolved = list(plan.unresolved)
    used_ids: set[str] = set()

    for proposal in plan.summary_groups:
        group, reason = _validate_summary_group(
            proposal,
            section_id=section.id,
            runs=runs,
            used_ids=used_ids,
            min_group_size=min_group_size,
            max_sentences=max_sentences,
            max_ratio=max_ratio,
            max_summary_chars=max_summary_chars,
        )
        audit.append(
            {
                "mode": "summary",
                "proposal": proposal.model_dump(mode="json"),
                "accepted": group is not None,
                "reason": reason or "host-validated math-free summary",
                "render_group": group.model_dump(mode="json") if group is not None else None,
            }
        )
        if group is None:
            continue
        accepted.append(group)
        used_ids.update(group.source_observation_ids)

    for proposal in plan.redundant_text_groups:
        suppressed, reason = _validate_redundancy_group(
            proposal,
            candidates=redundancy_candidates,
            used_ids=used_ids,
        )
        audit.append(
            {
                "mode": "selection_dedup",
                "proposal": proposal.model_dump(mode="json"),
                "accepted": bool(suppressed),
                "reason": reason or "host-validated selection-only text deduplication",
                "suppressed_text_ids": sorted(suppressed),
            }
        )
        if not suppressed:
            continue
        suppress_text.update(suppressed)
        used_ids.update(proposal.source_observation_ids)

    return accepted, suppress_text, audit, unresolved, cache_hit, model_calls


def run_state_prose_compression(
    orchestrator,
    *,
    kb: LectureKnowledgeBase,
    outline: LectureOutline,
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    min_group_size: int,
    max_sentences: int,
    max_ratio: float,
    max_summary_chars: int,
    force: bool,
) -> tuple[ProseCompressionPolicy, dict[str, int], list[str]]:
    """Build render-only semantic prose compression without changing repaired state."""

    groups: list[ProseCompressionRenderGroup] = []
    suppress_text: set[str] = set()
    audit: list[dict[str, Any]] = []
    unresolved: list[str] = []
    stats = {
        "sections_with_candidates": 0,
        "model_calls": 0,
        "cache_hits": 0,
        "proposed_summary_groups": 0,
        "accepted_summary_groups": 0,
        "proposed_redundancy_groups": 0,
        "accepted_redundancy_groups": 0,
        "rejected_groups": 0,
        "summarized_observations": 0,
        "suppressed_redundant_text_blocks": 0,
    }

    for section in outline.sections:
        observations = _section_observations(kb, section)
        runs = _summary_runs(observations, min_group_size=min_group_size)
        redundancy_candidates = _redundancy_candidates(observations)
        if not runs and len(redundancy_candidates) < 2:
            continue
        stats["sections_with_candidates"] += 1
        (
            section_groups,
            section_suppressed,
            section_audit,
            section_unresolved,
            cache_hit,
            model_calls,
        ) = _compress_section(
            orchestrator,
            section=section,
            observations=observations,
            runs=runs,
            work=work,
            llm_config=llm_config,
            pipeline_version=pipeline_version,
            min_group_size=min_group_size,
            max_sentences=max_sentences,
            max_ratio=max_ratio,
            max_summary_chars=max_summary_chars,
            force=force,
        )
        groups.extend(section_groups)
        suppress_text.update(section_suppressed)
        audit.extend({"section_id": section.id, **item} for item in section_audit)
        unresolved.extend(section_unresolved)
        stats["cache_hits"] += cache_hit
        stats["model_calls"] += model_calls
        for item in section_audit:
            if item["mode"] == "summary":
                stats["proposed_summary_groups"] += 1
                stats["accepted_summary_groups"] += int(item["accepted"])
            else:
                stats["proposed_redundancy_groups"] += 1
                stats["accepted_redundancy_groups"] += int(item["accepted"])
            stats["rejected_groups"] += int(not item["accepted"])

    stats["summarized_observations"] = sum(
        len(group.source_observation_ids) for group in groups
    )
    stats["suppressed_redundant_text_blocks"] = len(suppress_text)
    policy = ProseCompressionPolicy(
        groups=groups,
        suppress_text_ids=sorted(suppress_text),
    )
    atomic_json_dump(
        work / "state_prose_compression.json",
        {
            "version": STATE_PROSE_COMPRESSION_VERSION,
            "policy": policy.model_dump(mode="json"),
            "stats": stats,
            "audit": audit,
            "unresolved": list(dict.fromkeys(unresolved)),
        },
    )
    return policy, stats, list(dict.fromkeys(unresolved))
