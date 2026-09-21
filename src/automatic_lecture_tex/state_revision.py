from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .episode_graph import refresh_derived_anchors
from .knowledge import KnowledgeOrchestrator, _merge_unique
from .schemas import (
    ClaimStatus,
    LectureKnowledgeBase,
    MathStatus,
    SourceStatus,
    StateObservationRevision,
    StateReviewPlan,
    Transcript,
)
from .util import atomic_json_dump, stable_hash


STATE_REVISION_VERSION = 1


def _observation_catalog(kb: LectureKnowledgeBase, max_chars: int) -> list[dict[str, Any]]:
    rows = [
        {
            "id": obs.id,
            "episode_id": obs.episode_id,
            "start": round(obs.start, 3),
            "end": round(obs.end, 3),
            "kind": obs.kind,
            "text": obs.text[:240],
            "latex": obs.latex,
            "source_status": obs.source_status,
            "confidence": obs.confidence,
        }
        for obs in sorted(kb.observations, key=lambda item: (item.start, item.end, item.id))
    ]
    if len(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
        return rows

    for row in rows:
        row["text"] = str(row["text"])[:120]
        if row.get("latex"):
            row["latex"] = str(row["latex"])[:180]
    if len(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
        return rows

    # Preserve full-lecture coverage rather than truncating the tail.
    count = max(1, int(len(rows) * max_chars / max(
        1, len(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    )))
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[0]]
    indices = [
        round(index * (len(rows) - 1) / (count - 1))
        for index in range(count)
    ]
    return [rows[index] for index in dict.fromkeys(indices)]


def _deterministic_candidates(kb: LectureKnowledgeBase) -> list[str]:
    result: list[str] = []
    for obs in kb.observations:
        if obs.source_status == SourceStatus.INFERRED or obs.confidence < 0.72:
            result.append(obs.id)
    return result


def _review_plan(
    orchestrator: KnowledgeOrchestrator,
    kb: LectureKnowledgeBase,
    *,
    max_candidates: int,
    catalog_chars: int,
) -> StateReviewPlan:
    catalog = _observation_catalog(kb, catalog_chars)
    symbols = [
        item.model_dump(mode="json")
        for item in kb.symbols
        if item.active
    ][-40:]
    prompt = f"""Audit a compact catalog of canonical mathematical observations from one complete
lecture. This pass ONLY SELECTS observations that deserve re-reading from their original evidence;
it must not rewrite or repair anything itself.

Observation catalog:
{json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))}

Active notation:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Select at most {max_candidates} existing observation ids when there is a concrete reason to suspect
that the semantic reconstruction is unreliable. Useful signals include:
- contradiction with another observation elsewhere in the lecture;
- an elementary mathematical/type/quantifier inconsistency;
- a statement incompatible with a standard theorem the surrounding observations are clearly about;
- suspicious theorem/person terminology or notation drift;
- a claim marked inferred/low-confidence whose exact content matters;
- later material that appears to correct or clarify an earlier reconstruction.

Standard mathematics may be used to DETECT suspicious state, not to invent missing lecture content.
Do not flag merely stylistic, incomplete-but-correct, or pedagogically terse observations. Do not
propose replacement text in this pass. Every candidate id must occur verbatim in the catalog.
"""
    return orchestrator._structured(
        prompt,
        StateReviewPlan,
        operation="state_global_review",
        max_tokens=3072,
    )


def _load_window_payload(work: Path, window_id: str) -> dict[str, Any] | None:
    path = work / "knowledge_windows" / f"{window_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _target_evidence(
    kb: LectureKnowledgeBase,
    transcript: Transcript,
    work: Path,
    observation_id: str,
    *,
    context_seconds: float,
    image_limit: int = 5,
) -> tuple[dict[str, Any], list[Path]]:
    target = next(item for item in kb.observations if item.id == observation_id)
    center = 0.5 * (target.start + target.end)

    segments = [
        segment.model_dump(mode="json")
        for segment in transcript.segments
        if segment.end >= target.start - context_seconds
        and segment.start <= target.end + context_seconds
    ]
    same_episode = [
        item.model_dump(mode="json")
        for item in sorted(kb.observations, key=lambda obs: (obs.start, obs.end))
        if item.episode_id == target.episode_id and item.id != target.id
    ]
    nearby = sorted(
        (
            item
            for item in kb.observations
            if item.id != target.id
            and item.end >= target.start - context_seconds
            and item.start <= target.end + context_seconds
        ),
        key=lambda item: abs(0.5 * (item.start + item.end) - center),
    )[:12]

    window_ids = list(dict.fromkeys([target.window_id, *target.window_ids]))
    visual_metadata: list[dict[str, Any]] = []
    image_candidates: list[tuple[float, Path, str]] = []
    for window_id in window_ids:
        if not window_id:
            continue
        payload = _load_window_payload(work, window_id)
        if payload is None:
            continue
        for visual in payload.get("visual_evidence", []):
            metadata = dict(visual)
            frame_paths = list(metadata.pop("frame_paths", []) or [])
            frame_timestamps = list(metadata.get("frame_timestamps", []) or [])
            visual_metadata.append(metadata)
            for index, raw_path in enumerate(frame_paths):
                path = Path(raw_path)
                if not path.is_file():
                    continue
                timestamp = (
                    float(frame_timestamps[index])
                    if index < len(frame_timestamps)
                    else center
                )
                label = (
                    f"request_id={visual.get('request_id','')}, frame_index={index}, "
                    f"timestamp={timestamp:.3f}s"
                )
                image_candidates.append((abs(timestamp - center), path, label))

    images: list[Path] = []
    image_index: list[str] = []
    seen: set[Path] = set()
    for _distance, path, label in sorted(image_candidates, key=lambda item: item[0]):
        if path in seen:
            continue
        seen.add(path)
        image_index.append(f"Image {len(images)}: {label}")
        images.append(path)
        if len(images) >= image_limit:
            break

    payload = {
        "target": target.model_dump(mode="json"),
        "asr_context": segments,
        "same_episode_observations": same_episode,
        "nearby_observations": [item.model_dump(mode="json") for item in nearby],
        "active_symbols": [
            item.model_dump(mode="json")
            for item in kb.symbols
            if item.active
            and (item.episode_id == target.episode_id or item.introduced_at <= target.end)
        ][-30:],
        "visual_metadata": visual_metadata,
        "attached_image_index": image_index,
    }
    return payload, images


def _revise_one(
    orchestrator: KnowledgeOrchestrator,
    evidence: dict[str, Any],
    images: list[Path],
) -> StateObservationRevision:
    prompt = f"""Re-evaluate ONE suspicious canonical mathematical observation against its original
lecture evidence. The observation was flagged by a whole-lecture consistency pass; do not trust it
merely because it is already canonical.

Targeted evidence:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Choose exactly one action:
- keep: the existing observation is a faithful reconstruction of what the lecture communicated;
- replace: the existing reconstruction is wrong, but the intended content is strongly determined by
  the supplied ASR/board/local context; return the complete canonical replacement;
- unresolved: the evidence does not determine a unique reconstruction. This removes the observation
  from canonical state rather than leaking uncertainty into final notes.

Use the attached board images as direct sensor evidence in the order listed by attached_image_index.
Standard mathematics may be used as a bounded disambiguation/checking prior, never as a source for
material absent from the supplied lecture evidence. A genuine lecturer mistake supported by the
evidence should be kept; an ASR/VLM reconstruction mistake should not. A later explicit correction
may justify replacing or invalidating an earlier reconstruction.

For replace, preserve the original event's semantic scope and return only the corrected event text
and, when applicable, its complete LaTeX relation. Do not hedge inside replacement_text: if the
replacement itself needs 'probably', 'possibly', or alternatives, use unresolved instead.
"""
    return orchestrator._structured(
        prompt,
        StateObservationRevision,
        images=images or None,
        guided_json=not bool(images),
        operation="state_targeted_revision",
        max_tokens=2048,
    )


def _apply_revision(
    kb: LectureKnowledgeBase,
    observation_id: str,
    revision: StateObservationRevision,
    *,
    threshold: float,
) -> str:
    target = next((item for item in kb.observations if item.id == observation_id), None)
    if target is None:
        return "missing"
    if revision.confidence < threshold or revision.action == "keep":
        return "kept"

    related_claims = [
        claim for claim in kb.claims if observation_id in claim.evidence_ids
    ]

    if revision.action == "replace":
        old_text = target.text
        old_latex = target.latex
        if revision.replacement_text is not None:
            target.text = revision.replacement_text
        if revision.replacement_latex is not None:
            target.latex = revision.replacement_latex
        target.source_status = SourceStatus.RECONSTRUCTED
        target.confidence = revision.confidence
        for claim in related_claims:
            claim.content = target.text or (target.latex or "")
            claim.latex = target.latex
            claim.source_status = target.source_status
            claim.math_status = MathStatus.UNCHECKED
        kb.unresolved = _merge_unique(
            kb.unresolved,
            [
                "Global state revision replaced "
                f"{observation_id}: {revision.reason} "
                f"[old_text={old_text!r}, old_latex={old_latex!r}]"
            ],
        )
        return "replaced"

    # unresolved: remove this event from canonical state while retaining an audit message.
    kb.unresolved = _merge_unique(
        kb.unresolved,
        [
            f"Global state revision demoted {observation_id} to unresolved: "
            f"{revision.reason}"
        ],
    )
    for claim in related_claims:
        claim.status = ClaimStatus.UNRESOLVED
    for episode in kb.episodes:
        episode.observation_ids = [
            item for item in episode.observation_ids if item != observation_id
        ]
        episode.claim_ids = [
            claim_id
            for claim_id in episode.claim_ids
            if all(claim.id != claim_id or claim.status != ClaimStatus.UNRESOLVED for claim in related_claims)
        ]
    for symbol in kb.symbols:
        if observation_id in symbol.evidence_ids:
            symbol.evidence_ids = [item for item in symbol.evidence_ids if item != observation_id]
            if not symbol.evidence_ids:
                symbol.active = False
    kb.observations = [item for item in kb.observations if item.id != observation_id]
    refresh_derived_anchors(kb)
    return "unresolved"


def run_global_state_revision(
    orchestrator: KnowledgeOrchestrator,
    kb: LectureKnowledgeBase,
    transcript: Transcript,
    work: Path,
    *,
    force: bool,
) -> tuple[LectureKnowledgeBase, dict[str, Any]]:
    config = orchestrator.config
    if not config.state_revision_enabled:
        return kb, {"enabled": False, "candidates": 0, "replaced": 0, "unresolved": 0}

    input_fingerprint = stable_hash(
        {
            "version": STATE_REVISION_VERSION,
            "kb": kb.model_dump(mode="json"),
            "transcript": transcript.model_dump(mode="json"),
            "threshold": config.state_revision_apply_threshold,
            "max_candidates": config.state_revision_max_candidates,
            "catalog_chars": config.state_revision_catalog_chars,
            "context_seconds": config.state_revision_context_seconds,
        }
    )
    artifact = work / "state_revision.json"
    if artifact.exists() and not force:
        try:
            cached = json.loads(artifact.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == input_fingerprint:
                revised = LectureKnowledgeBase.model_validate(cached["revised_kb"])
                return revised, dict(cached.get("stats", {}))
        except (json.JSONDecodeError, KeyError, ValidationError):
            pass

    plan = _review_plan(
        orchestrator,
        kb,
        max_candidates=config.state_revision_max_candidates,
        catalog_chars=config.state_revision_catalog_chars,
    )
    valid_ids = {item.id for item in kb.observations}
    planned = [
        item.observation_id
        for item in plan.candidates
        if item.observation_id in valid_ids
    ]
    deterministic = _deterministic_candidates(kb)
    candidate_ids = list(dict.fromkeys([*planned, *deterministic]))[
        : config.state_revision_max_candidates
    ]

    revisions: list[dict[str, Any]] = []
    counts = {"kept": 0, "replaced": 0, "unresolved": 0, "missing": 0}
    for observation_id in candidate_ids:
        if observation_id not in {item.id for item in kb.observations}:
            continue
        evidence, images = _target_evidence(
            kb,
            transcript,
            work,
            observation_id,
            context_seconds=config.state_revision_context_seconds,
        )
        revision = _revise_one(orchestrator, evidence, images)
        outcome = _apply_revision(
            kb,
            observation_id,
            revision,
            threshold=config.state_revision_apply_threshold,
        )
        counts[outcome] = counts.get(outcome, 0) + 1
        revisions.append(
            {
                "observation_id": observation_id,
                "outcome": outcome,
                "revision": revision.model_dump(mode="json"),
                "attached_images": [str(path) for path in images],
            }
        )

    refresh_derived_anchors(kb)
    stats = {
        "enabled": True,
        "candidates": len(candidate_ids),
        "planned_candidates": len(planned),
        **counts,
    }
    atomic_json_dump(
        artifact,
        {
            "fingerprint": input_fingerprint,
            "review_plan": plan.model_dump(mode="json"),
            "revisions": revisions,
            "stats": stats,
            "revised_kb": kb.model_dump(mode="json"),
        },
    )
    return kb, stats
