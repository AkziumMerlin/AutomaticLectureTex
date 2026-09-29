from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .latex import escape_tex
from .schemas import (
    BlockType,
    ChunkNotes,
    LectureKnowledgeBase,
    LectureObservation,
    NoteBlock,
    ObservationKind,
    OutlineSection,
)
from .state_canonicalization import CanonicalRenderPolicy
from .util import atomic_json_dump, stable_hash


STATE_DOCUMENT_ASSEMBLY_VERSION = 4

DocumentBlockKind = Literal[
    "subsection",
    "paragraph",
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "proof",
    "example",
    "remark",
    "formula",
]
DocumentOmissionChannel = Literal["text", "formula", "both"]
DocumentOmissionReason = Literal[
    "duplicate",
    "intermediate",
    "scratch",
    "transition",
    "incomplete",
    "routine",
    "meta",
]

_PROSE_BLOCK_TYPES = {
    "paragraph",
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "proof",
    "example",
    "remark",
}
_FORMAL_PROSE_MARKERS = ("$", "\\")
_RELATION_TOKENS = ("=", "→", "↦", "⇒", "⇔", "≤", "≥", "≠", "∈", "⊂", "⊆")
_IMPORTANT_TEXT_KINDS = {
    ObservationKind.DEFINITION,
    ObservationKind.CLAIM,
    ObservationKind.EXAMPLE,
    ObservationKind.NOTATION,
    ObservationKind.CORRECTION,
    ObservationKind.RETRACTION,
}
_IMPORTANT_FORMULA_KINDS = _IMPORTANT_TEXT_KINDS
_NARRATION_RE = re.compile(
    r"\b(?:"
    r"лектор|преподавател\w*|на\s+(?:левой|правой|средней\s+)?доске|доск\w*|"
    r"устно|записыва\w*|дописыва\w*|указывает|подч[её]ркива\w*|"
    r"комментиру\w*|поясня\w*|отмечает|говорит|произносит|обводит|"
    r"продолжая\s+(?:запись|объяснение)"
    r")\b",
    re.IGNORECASE,
)
_LATIN_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z](?![A-Za-z0-9])")
_NUMBER_RE = re.compile(r"\d+")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


class StateDocumentProseBlock(BaseModel):
    """Generated prose block whose required fields are visible to guided JSON decoding."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "subsection",
        "paragraph",
        "definition",
        "theorem",
        "lemma",
        "proposition",
        "corollary",
        "proof",
        "example",
        "remark",
    ]
    text: str = Field(min_length=1)
    title: str | None = None
    source_observation_ids: list[str] = Field(min_length=1)

    @field_validator("text", "title")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = re.sub(r"\s+", " ", value.strip())
        return value or None

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("prose block text must be non-empty")
        return value

    @field_validator("source_observation_ids")
    @classmethod
    def unique_sources(cls, value: list[str]) -> list[str]:
        unique = list(dict.fromkeys(item.strip() for item in value if item and item.strip()))
        if not unique:
            raise ValueError("prose block requires source_observation_ids")
        return unique


class StateDocumentFormulaBlock(BaseModel):
    """Formula block: the model may only point at existing repaired LaTeX."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["formula"]
    formula_observation_id: str = Field(min_length=1)


StateDocumentBlockProposal = Annotated[
    StateDocumentProseBlock | StateDocumentFormulaBlock,
    Field(discriminator="type"),
]


class StateDocumentOmission(BaseModel):
    observation_id: str
    channel: DocumentOmissionChannel
    reason: DocumentOmissionReason


class StateDocumentPlan(BaseModel):
    blocks: list[StateDocumentBlockProposal] = Field(default_factory=list)
    omissions: list[StateDocumentOmission] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


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


def _language_ok(value: str, output_language: str) -> bool:
    if not output_language.casefold().startswith("ru"):
        return True
    letters = _ALPHA_RE.findall(value)
    if not letters:
        return True
    cyrillic = _CYRILLIC_RE.findall(value)
    return len(cyrillic) / len(letters) >= 0.45


def _safe_generated_prose(
    value: str,
    *,
    source_items: list[LectureObservation],
    output_language: str,
) -> tuple[bool, str]:
    if any(marker in value for marker in _FORMAL_PROSE_MARKERS):
        return False, "generated prose contains unsupported mathematical syntax"
    if _NARRATION_RE.search(value):
        return False, "generated prose contains lecturer/board narration"
    if not _language_ok(value, output_language):
        return False, f"generated prose is not predominantly in {output_language}"

    source = " ".join(
        part
        for item in source_items
        for part in ((item.text or ""), (item.latex or ""))
        if part
    )
    new_symbols = set(_LATIN_SYMBOL_RE.findall(value)) - set(_LATIN_SYMBOL_RE.findall(source))
    if new_symbols:
        return False, "generated prose introduces new standalone Latin symbols: " + ", ".join(
            sorted(new_symbols)
        )
    new_numbers = set(_NUMBER_RE.findall(value)) - set(_NUMBER_RE.findall(source))
    if new_numbers:
        return False, "generated prose introduces new numeric tokens: " + ", ".join(
            sorted(new_numbers)
        )

    if "=" in value:
        normalized_source = re.sub(r"\s+", "", source).replace("−", "-")
        for snippet in _EQUALITY_SNIPPET_RE.findall(value):
            normalized = re.sub(r"\s+", "", snippet).replace("−", "-")
            if normalized not in normalized_source:
                return False, "generated prose introduces an equality not found in its sources"
    return True, ""


def _block_type(value: str) -> BlockType:
    return {
        "paragraph": BlockType.PARAGRAPH,
        "definition": BlockType.DEFINITION,
        "theorem": BlockType.THEOREM,
        "lemma": BlockType.LEMMA,
        "proposition": BlockType.PROPOSITION,
        "corollary": BlockType.COROLLARY,
        "proof": BlockType.PROOF,
        "example": BlockType.EXAMPLE,
        "remark": BlockType.REMARK,
    }[value]


def _compact_inputs(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    *,
    render_policy: CanonicalRenderPolicy,
) -> tuple[list[LectureObservation], list[dict[str, Any]]]:
    suppress_text = set(render_policy.suppress_text_ids)
    suppress_latex = set(render_policy.suppress_latex_ids)
    episode_kind = {episode.id: str(episode.kind) for episode in kb.episodes}

    observations = [
        item
        for item in _section_observations(kb, section)
        if item.kind not in {ObservationKind.UNRESOLVED}
    ]
    compact: list[dict[str, Any]] = []
    for item in observations:
        text = "" if item.id in suppress_text else item.text.strip()
        latex = "" if item.id in suppress_latex else (item.latex or "").strip()
        if item.kind == ObservationKind.TRANSITION and not text and not latex:
            continue
        compact.append(
            {
                "id": item.id,
                "episode_id": item.episode_id,
                "episode_kind": episode_kind.get(item.episode_id, "topic"),
                "kind": str(item.kind),
                "text": text,
                "latex": latex or None,
                "start": round(float(item.start), 3),
            }
        )
    return observations, compact


def _load_plan(path: Path, fingerprint: str) -> StateDocumentPlan | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return StateDocumentPlan.model_validate(payload["plan"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _omission_channels(plan: StateDocumentPlan) -> tuple[dict[str, set[str]], list[str]]:
    channels: dict[str, set[str]] = {}
    errors: list[str] = []
    for omission in plan.omissions:
        target = channels.setdefault(omission.observation_id, set())
        requested = {"text", "formula"} if omission.channel == "both" else {omission.channel}
        if target & requested:
            errors.append(f"duplicate omission channel for {omission.observation_id}")
        target.update(requested)
    return channels, errors


def _validate_and_render_plan(
    plan: StateDocumentPlan,
    *,
    section: OutlineSection,
    observations: list[LectureObservation],
    render_policy: CanonicalRenderPolicy,
    output_language: str,
    max_prose_ratio: float,
    max_remarks_fraction: float,
    max_blocks: int,
) -> tuple[ChunkNotes | None, list[str], dict[str, int]]:
    issues: list[str] = []
    stats = {
        "blocks": 0,
        "prose_blocks": 0,
        "formula_blocks": 0,
        "subsections": 0,
        "remarks": 0,
        "omitted_text_channels": 0,
        "omitted_formula_channels": 0,
    }

    if len(plan.blocks) > max_blocks:
        return None, [f"document plan has {len(plan.blocks)} blocks > max_blocks={max_blocks}"], stats

    by_id = {item.id: item for item in observations}
    suppress_text = set(render_policy.suppress_text_ids)
    suppress_latex = set(render_policy.suppress_latex_ids)
    omission_channels, omission_errors = _omission_channels(plan)
    issues.extend(omission_errors)

    for omission in plan.omissions:
        if omission.observation_id not in by_id:
            issues.append(f"omission references unknown observation {omission.observation_id}")

    rendered: list[NoteBlock] = []
    used_text_ids: set[str] = set()
    used_formula_ids: set[str] = set()
    source_prose_chars = sum(
        len(item.text.strip())
        for item in observations
        if item.id not in suppress_text
        and item.kind not in {ObservationKind.TRANSITION, ObservationKind.UNRESOLVED}
    )
    generated_prose_chars = 0

    for index, block in enumerate(plan.blocks):
        if block.type == "formula":
            observation_id = str(block.formula_observation_id)
            source = by_id.get(observation_id)
            if source is None:
                issues.append(f"block {index}: unknown formula observation {observation_id}")
                continue
            if observation_id in suppress_latex:
                issues.append(f"block {index}: formula {observation_id} is canonicalization-suppressed")
                continue
            latex = (source.latex or "").strip()
            if not latex:
                issues.append(f"block {index}: formula ref {observation_id} has no LaTeX")
                continue
            if observation_id in used_formula_ids:
                issues.append(f"block {index}: formula {observation_id} rendered more than once")
                continue
            used_formula_ids.add(observation_id)
            rendered.append(
                NoteBlock(
                    type=BlockType.EQUATION,
                    latex=latex,
                    source_evidence_ids=[observation_id],
                )
            )
            stats["formula_blocks"] += 1
            continue

        unknown_sources = [item for item in block.source_observation_ids if item not in by_id]
        if unknown_sources:
            issues.append(
                f"block {index}: unknown source observations " + ", ".join(unknown_sources)
            )
            continue
        source_items = [by_id[item] for item in block.source_observation_ids]
        safe, reason = _safe_generated_prose(
            block.text or "",
            source_items=source_items,
            output_language=output_language,
        )
        if not safe:
            issues.append(f"block {index}: {reason}")
            continue
        if block.title:
            safe_title, reason = _safe_generated_prose(
                block.title,
                source_items=source_items,
                output_language=output_language,
            )
            if not safe_title:
                issues.append(f"block {index} title: {reason}")
                continue

        used_text_ids.update(block.source_observation_ids)
        generated_prose_chars += len(block.text or "")
        if block.type == "subsection":
            rendered.append(
                NoteBlock(
                    type=BlockType.SUBSECTION,
                    latex=block.text or "",
                    source_evidence_ids=block.source_observation_ids,
                )
            )
            stats["subsections"] += 1
        else:
            rendered.append(
                NoteBlock(
                    type=_block_type(block.type),
                    title=block.title,
                    latex=escape_tex(block.text or ""),
                    source_evidence_ids=block.source_observation_ids,
                )
            )
            stats["prose_blocks"] += 1
            if block.type == "remark":
                stats["remarks"] += 1

    if issues:
        return None, issues, stats

    # Document editing is selective by design. Routine proof/remark/equation channels may be
    # omitted implicitly; requiring one JSON omission row per intermediate board state turns the
    # document planner back into an observation-ledger accountant. Important semantic channels
    # remain fail-closed: they must be represented or explicitly omitted as duplicate/incomplete.
    omission_by_id = {item.observation_id: item for item in plan.omissions}
    implicit_text_omissions = 0
    implicit_formula_omissions = 0
    for item in observations:
        if item.kind in {ObservationKind.TRANSITION, ObservationKind.UNRESOLVED}:
            continue
        channels = omission_channels.get(item.id, set())

        text_available = item.id not in suppress_text and bool(item.text.strip())
        text_unused = text_available and item.id not in used_text_ids and "text" not in channels
        if text_unused:
            if item.kind in _IMPORTANT_TEXT_KINDS:
                issues.append(f"important text channel for {item.id} is neither used nor omitted")
            else:
                implicit_text_omissions += 1

        formula_available = item.id not in suppress_latex and bool((item.latex or "").strip())
        formula_unused = (
            formula_available
            and item.id not in used_formula_ids
            and "formula" not in channels
        )
        if formula_unused:
            if item.kind in _IMPORTANT_FORMULA_KINDS:
                issues.append(f"important formula channel for {item.id} is neither used nor omitted")
            else:
                implicit_formula_omissions += 1

        if item.kind in _IMPORTANT_TEXT_KINDS and item.id not in used_text_ids:
            omission = omission_by_id.get(item.id)
            if omission is not None and omission.reason not in {"duplicate", "incomplete"}:
                issues.append(
                    f"important {item.kind} observation {item.id} omitted as {omission.reason}"
                )

    allowed_chars = max(600, math.ceil(source_prose_chars * max_prose_ratio))
    if generated_prose_chars > allowed_chars:
        issues.append(
            f"generated prose is too verbose ({generated_prose_chars} chars > {allowed_chars})"
        )

    prose_blocks = max(1, stats["prose_blocks"])
    if stats["remarks"] > max(3, math.ceil(prose_blocks * max_remarks_fraction)):
        issues.append(
            f"too many remark blocks ({stats['remarks']} of {stats['prose_blocks']} prose blocks)"
        )

    if issues:
        return None, issues, stats

    stats["blocks"] = len(rendered)
    stats["omitted_text_channels"] += implicit_text_omissions
    stats["omitted_formula_channels"] += implicit_formula_omissions
    for item in plan.omissions:
        if item.channel in {"text", "both"}:
            stats["omitted_text_channels"] += 1
        if item.channel in {"formula", "both"}:
            stats["omitted_formula_channels"] += 1

    return (
        ChunkNotes(
            chunk_id=section.id,
            start=section.start,
            end=section.end,
            section_title=section.title,
            blocks=rendered,
            unresolved=list(dict.fromkeys(plan.unresolved)),
        ),
        [],
        stats,
    )


def build_state_document_section(
    orchestrator,
    *,
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    render_policy: CanonicalRenderPolicy,
    work: Path,
    llm_config: dict[str, Any],
    max_prose_ratio: float,
    max_remarks_fraction: float,
    max_blocks: int,
    force: bool,
) -> tuple[ChunkNotes | None, dict[str, int], list[str]]:
    observations, compact = _compact_inputs(kb, section, render_policy=render_policy)
    stats = {
        "model_calls": 0,
        "retry_calls": 0,
        "cache_hits": 0,
        "blocks": 0,
        "prose_blocks": 0,
        "formula_blocks": 0,
        "subsections": 0,
        "remarks": 0,
        "omitted_text_channels": 0,
        "omitted_formula_channels": 0,
    }
    if not compact:
        return (
            ChunkNotes(
                chunk_id=section.id,
                start=section.start,
                end=section.end,
                section_title=section.title,
                blocks=[],
            ),
            stats,
            [],
        )

    fingerprint = stable_hash(
        {
            "document_assembly_version": STATE_DOCUMENT_ASSEMBLY_VERSION,
            "section": section.model_dump(mode="json"),
            "observations": compact,
            "render_policy": render_policy.model_dump(mode="json"),
            "max_prose_ratio": max_prose_ratio,
            "max_remarks_fraction": max_remarks_fraction,
            "max_blocks": max_blocks,
            "output_language": orchestrator.output_language,
            "llm": llm_config,
        }
    )
    path = work / "state_document_sections" / f"{section.id}.json"
    plan = None if force else _load_plan(path, fingerprint)
    if plan is not None:
        stats["cache_hits"] = 1
    else:
        prompt = f"""Edit one repaired section of a mathematical lecture into concise final notes.

Section title:
{section.title}

Canonical observations, in lecture order:
{json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}

The observations are an EVIDENCE LEDGER, not the desired document. Produce a semantic document
plan comparable to carefully edited university lecture notes.

Use blocks at mathematical-document granularity:
- definition / theorem / lemma / proposition / corollary / proof / example;
- ordinary paragraph for connective exposition;
- remark only for genuinely useful side information;
- subsection only for a real subtopic change inside this section;
- formula blocks must reference an EXISTING observation id with LaTeX.

Critical rules:
- do NOT make one block per observation;
- merge repeated remarks and successive board states into one coherent statement or proof;
- a proof should read as one argument, not as a transcript of each algebraic line;
- keep definitions, mathematical statements, hypotheses, conclusions and useful examples;
- retain only equations needed to state a result or follow a proof; intermediate/scratch/repeated
  formulas may be omitted explicitly;
- formula blocks contain ONLY formula_observation_id. Never transcribe, edit or regenerate LaTeX;
- prose is plain text only. Do not write LaTeX commands or math delimiters in prose; a short
  equality is allowed only when it is literally present in the cited source observations;
- every prose block must cite the observation ids supporting it;
- definitions/claims/examples/notation/corrections must be represented or explicitly omitted as
  duplicate/incomplete; routine proof-step/remark/equation channels may be omitted implicitly;
- do not mention lecturer/board/audio/OCR/reconstruction/timestamps;
- do not add textbook facts or silently correct the lecture from external knowledge. Preserve the
  repaired state's mathematical content even when the lecturer may have made a typo;
- write in language code {orchestrator.output_language};
- use concise stable subsection/block titles rather than chronological titles;
- use remarks sparingly: routine derivation commentary belongs in proof/paragraph blocks.

Compression target:
- generated prose should be at most about {max_prose_ratio:.0%} of the available observation prose;
- prefer roughly 2-4 source observations per prose block when the material permits;
- keep the total plan within {max_blocks} rendered blocks;
- the final document should be substantially shorter than the evidence ledger while remaining
  sufficient to study the mathematics without the original video.
"""
        try:
            plan = orchestrator._structured(
                prompt,
                StateDocumentPlan,
                operation="state_document_assembly",
                max_tokens=8192,
                split_oversized_task=True,
                thinking=False,
                temperature=0.5,
                top_p=0.85,
                top_k=20,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )
            stats["model_calls"] = 1
        except Exception as exc:
            return None, stats, [
                f"Document assembly failed for {section.id}: {type(exc).__name__}: {exc}"
            ]
        atomic_json_dump(
            path,
            {
                "fingerprint": fingerprint,
                "inputs": compact,
                "plan": plan.model_dump(mode="json"),
            },
        )

    notes, issues, render_stats = _validate_and_render_plan(
        plan,
        section=section,
        observations=observations,
        render_policy=render_policy,
        output_language=str(orchestrator.output_language or ""),
        max_prose_ratio=max_prose_ratio,
        max_remarks_fraction=max_remarks_fraction,
        max_blocks=max_blocks,
    )
    for key, value in render_stats.items():
        stats[key] = value

    if issues:
        retry_prompt = f"""Repair a rejected mathematical-document plan.

Section title:
{section.title}

Canonical observations:
{json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}

Rejected plan:
{json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

Host validation errors:
{json.dumps(issues, ensure_ascii=False)}

Return a COMPLETE corrected StateDocumentPlan, not a patch.

Do not weaken the document contract:
- formulas are existing formula_observation_id references only;
- generated prose contains no formulas/LaTeX and no lecturer/board narration;
- important definitions/claims/examples/notation/corrections are represented or explicitly
  omitted as duplicate/incomplete; routine proof-step/remark/equation channels may be unselected;
- keep document-level structure and aggressive semantic compression;
- write in language code {orchestrator.output_language}.
"""
        try:
            plan = orchestrator._structured(
                retry_prompt,
                StateDocumentPlan,
                operation="state_document_assembly_retry",
                max_tokens=8192,
                split_oversized_task=True,
                thinking=False,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            )
            stats["retry_calls"] = 1
            notes, issues, render_stats = _validate_and_render_plan(
                plan,
                section=section,
                observations=observations,
                render_policy=render_policy,
                output_language=str(orchestrator.output_language or ""),
                max_prose_ratio=max_prose_ratio,
                max_remarks_fraction=max_remarks_fraction,
                max_blocks=max_blocks,
            )
            for key, value in render_stats.items():
                stats[key] = value
            if not issues:
                atomic_json_dump(
                    path,
                    {
                        "fingerprint": fingerprint,
                        "inputs": compact,
                        "plan": plan.model_dump(mode="json"),
                        "repaired_after_validation": True,
                    },
                )
        except Exception as exc:
            issues = [*issues, f"document retry failed: {type(exc).__name__}: {exc}"]

    if issues:
        atomic_json_dump(
            work / "state_document_sections" / f"{section.id}.validation.json",
            {
                "issues": issues,
                "plan": plan.model_dump(mode="json"),
            },
        )
        return None, stats, issues
    return notes, stats, list(dict.fromkeys(plan.unresolved))
