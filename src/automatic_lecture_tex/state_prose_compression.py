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


STATE_PROSE_COMPRESSION_VERSION = 1

_COMPRESSIBLE_KINDS = {
    ObservationKind.REMARK,
    ObservationKind.TRANSITION,
    ObservationKind.NOTATION,
}
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


class ProseCompressionSentence(BaseModel):
    """One math-free sentence supported by explicit repaired observations."""

    text: str
    source_observation_ids: list[str] = Field(min_length=1)


class ProseCompressionGroupProposal(BaseModel):
    """A proposed semantic merge of adjacent prose-only observations."""

    source_observation_ids: list[str] = Field(min_length=2)
    sentences: list[ProseCompressionSentence] = Field(min_length=1)


class ProseCompressionPlan(BaseModel):
    groups: list[ProseCompressionGroupProposal] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class ProseCompressionRenderGroup(BaseModel):
    section_id: str
    anchor_observation_id: str
    source_observation_ids: list[str]
    summary_text: str


class ProseCompressionPolicy(BaseModel):
    """Render-only prose replacements. Mathematical state is never changed."""

    groups: list[ProseCompressionRenderGroup] = Field(default_factory=list)


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


def _plain_prose_candidate(item: LectureObservation) -> bool:
    if item.kind not in _COMPRESSIBLE_KINDS:
        return False
    if item.latex:
        return False
    text = item.text.strip()
    if not text:
        return False
    return not any(marker in text for marker in _FORMAL_TEXT_MARKERS)


def _candidate_runs(
    observations: list[LectureObservation],
) -> list[list[LectureObservation]]:
    """Return contiguous prose-only runs; formal observations are hard boundaries."""

    runs: list[list[LectureObservation]] = []
    current: list[LectureObservation] = []
    for item in observations:
        if _plain_prose_candidate(item):
            current.append(item)
            continue
        if len(current) >= 2:
            runs.append(current)
        current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def _compact_run(index: int, run: list[LectureObservation]) -> dict[str, Any]:
    return {
        "run_id": f"run_{index:03d}",
        "observations": [
            {
                "id": item.id,
                "kind": str(item.kind),
                "text": item.text.strip(),
            }
            for item in run
        ],
    }


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
    endings = re.findall(r"[.!?](?=\s|$)", text)
    return max(1, len(endings))


def _symbol_tokens(value: str) -> set[str]:
    return set(_SINGLE_LATIN_SYMBOL.findall(value))


def _number_tokens(value: str) -> set[str]:
    return set(_NUMBER_TOKEN.findall(value))


def _validate_group(
    proposal: ProseCompressionGroupProposal,
    *,
    section_id: str,
    runs: list[list[LectureObservation]],
    used_ids: set[str],
    max_sentences: int,
    max_ratio: float,
    max_summary_chars: int,
) -> tuple[ProseCompressionRenderGroup | None, str]:
    source_ids = list(dict.fromkeys(proposal.source_observation_ids))
    if len(source_ids) < 2:
        return None, "group has fewer than two unique source observations"
    if any(item in used_ids for item in source_ids):
        return None, "group overlaps a previously accepted group"

    matched_run: list[LectureObservation] | None = None
    positions: list[int] = []
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
        return None, "group crosses a formal-math or section boundary"

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
        if any(marker in text for marker in _FORMAL_TEXT_MARKERS):
            return None, "generated prose contains formal mathematical syntax"
        lowered = text.casefold()
        if any(word in lowered for word in _META_WORDS):
            return None, "generated prose contains reconstruction/lecture narration"
        cited.update(sentence_sources)
        sentence_texts.append(text)

    if cited != group_set:
        return None, "not every compressed observation is cited by a summary sentence"

    summary = " ".join(sentence_texts).strip()
    if _sentence_count(summary) > max_sentences:
        return None, "rendered summary contains too many sentences"

    by_id = {item.id: item for item in matched_run}
    source_text = " ".join(by_id[item].text for item in source_ids)
    source_chars = len(source_text.strip())
    allowed_chars = min(max_summary_chars, max(80, int(source_chars * max_ratio)))
    if len(summary) > allowed_chars:
        return None, (
            f"summary is not compact enough ({len(summary)} chars > {allowed_chars} allowed)"
        )

    source_symbols = _symbol_tokens(source_text)
    new_symbols = _symbol_tokens(summary) - source_symbols
    if new_symbols:
        return None, "summary introduces new standalone Latin symbols: " + ", ".join(
            sorted(new_symbols)
        )
    source_numbers = _number_tokens(source_text)
    new_numbers = _number_tokens(summary) - source_numbers
    if new_numbers:
        return None, "summary introduces new numeric tokens: " + ", ".join(
            sorted(new_numbers)
        )

    source_items = [by_id[item] for item in source_ids]
    anchor = next(
        (item for item in source_items if item.kind != ObservationKind.TRANSITION),
        None,
    )
    if anchor is None:
        return None, "pure transition groups stay structural/audit-only"

    return (
        ProseCompressionRenderGroup(
            section_id=section_id,
            anchor_observation_id=anchor.id,
            source_observation_ids=source_ids,
            summary_text=summary,
        ),
        "",
    )


def _compress_section(
    orchestrator,
    *,
    section: OutlineSection,
    runs: list[list[LectureObservation]],
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    max_sentences: int,
    max_ratio: float,
    max_summary_chars: int,
    force: bool,
) -> tuple[list[ProseCompressionRenderGroup], list[dict[str, Any]], list[str], int, int]:
    compact_runs = [_compact_run(index, run) for index, run in enumerate(runs)]
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
            "runs": compact_runs,
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
        prompt = f"""Compress repetitive PROSE-ONLY lecture-state events into concise final-note prose.

Section title:
{section.title}

Candidate runs:
{json.dumps(compact_runs, ensure_ascii=False, separators=(",", ":"))}

The host has removed every observation containing a formal LaTeX payload or explicit mathematical
relation. You are NOT a mathematical writer. You are only deduplicating narration around already
protected mathematics.

Return zero or more groups. For each group:
- source_observation_ids must be a contiguous subsequence of ONE candidate run, in the given order;
- group only adjacent events that express the same point or a single compact setup;
- write at most {max_sentences} short final-note sentences;
- every sentence must cite the exact source observation IDs that support it;
- the union of sentence source IDs must equal the group's source_observation_ids exactly;
- preserve every distinct fact present in the grouped observations; if that cannot fit cleanly,
  split the group or leave observations ungrouped;
- aggressively merge repeated narration such as several variants of "X is a complex space";
- remove narration about what the lecturer writes, says, points at, repeats, or does on the board;
- do not mention ASR, OCR, frames, reconstruction, confidence, or provenance;
- do not introduce textbook facts, theorem names, assumptions, symbols, numbers, examples, or
  terminology absent from the supplied texts;
- output plain prose only: no LaTeX, equations, math delimiters, relation symbols, or formatting;
- unmentioned observations remain unchanged, so prefer no group when semantic equivalence is unclear.

This stage cannot modify any formula, definition, proof step, claim, example, correction, or
retraction: those observations are not present in the candidate runs.
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
            {
                "fingerprint": fingerprint,
                "plan": plan.model_dump(mode="json"),
            },
        )

    accepted: list[ProseCompressionRenderGroup] = []
    audit: list[dict[str, Any]] = []
    unresolved = list(plan.unresolved)
    used_ids: set[str] = set()

    for proposal in plan.groups:
        group, reason = _validate_group(
            proposal,
            section_id=section.id,
            runs=runs,
            used_ids=used_ids,
            max_sentences=max_sentences,
            max_ratio=max_ratio,
            max_summary_chars=max_summary_chars,
        )
        audit.append(
            {
                "proposal": proposal.model_dump(mode="json"),
                "accepted": group is not None,
                "reason": reason or "host-validated prose-only compression",
                "render_group": group.model_dump(mode="json") if group is not None else None,
            }
        )
        if group is None:
            continue
        accepted.append(group)
        used_ids.update(group.source_observation_ids)

    return accepted, audit, unresolved, cache_hit, model_calls


def run_state_prose_compression(
    orchestrator,
    *,
    kb: LectureKnowledgeBase,
    outline: LectureOutline,
    work: Path,
    llm_config: dict[str, Any],
    pipeline_version: int,
    max_sentences: int,
    max_ratio: float,
    max_summary_chars: int,
    force: bool,
) -> tuple[ProseCompressionPolicy, dict[str, int], list[str]]:
    """Build section-local prose replacements while leaving all mathematical state immutable."""

    groups: list[ProseCompressionRenderGroup] = []
    audit: list[dict[str, Any]] = []
    unresolved: list[str] = []
    stats = {
        "sections_with_candidates": 0,
        "model_calls": 0,
        "cache_hits": 0,
        "proposed_groups": 0,
        "accepted_groups": 0,
        "rejected_groups": 0,
        "compressed_observations": 0,
    }

    for section in outline.sections:
        observations = _section_observations(kb, section)
        runs = _candidate_runs(observations)
        if not runs:
            continue
        stats["sections_with_candidates"] += 1
        section_groups, section_audit, section_unresolved, cache_hit, model_calls = (
            _compress_section(
                orchestrator,
                section=section,
                runs=runs,
                work=work,
                llm_config=llm_config,
                pipeline_version=pipeline_version,
                max_sentences=max_sentences,
                max_ratio=max_ratio,
                max_summary_chars=max_summary_chars,
                force=force,
            )
        )
        groups.extend(section_groups)
        audit.extend(
            {
                "section_id": section.id,
                **item,
            }
            for item in section_audit
        )
        unresolved.extend(section_unresolved)
        stats["cache_hits"] += cache_hit
        stats["model_calls"] += model_calls
        stats["proposed_groups"] += len(section_audit)
        stats["accepted_groups"] += len(section_groups)
        stats["rejected_groups"] += len(section_audit) - len(section_groups)

    stats["compressed_observations"] = sum(
        len(group.source_observation_ids) for group in groups
    )
    policy = ProseCompressionPolicy(groups=groups)
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
