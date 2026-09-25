from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .chunking import chunk_transcript
from .generated_notes import GeneratedChunkNotes, GeneratedObservationResolution
from .episode_graph import (
    apply_episode_tracking,
    build_outline_from_episodes,
    close_open_episodes,
)
from .episode_synthesis import (
    EPISODE_SYNTHESIS_CACHE_VERSION,
    HIERARCHY_CACHE_VERSION,
    apply_episode_validation,
    assemble_outline_sections,
    episode_evidence_batches,
    merge_episode_batches,
    plan_episode_hierarchy_bounded,
    previous_block_context,
    validate_episode_batch,
    write_episode_batch,
)
from .knowledge import (
    KnowledgeOrchestrator,
    compact_knowledge_state,
    evidence_for_section,
    make_lecture_state,
    merge_window_observations,
)
from .llm import LectureModelClient, StructuredTaskTooLargeError
from .media import copy_asset
from .schemas import (
    ChunkNotes,
    EpisodeHierarchyPlan,
    EpisodeTrackingUpdate,
    LectureIR,
    LectureKnowledgeBase,
    LectureOutline,
    OutlineSection,
    VisualEvidence,
    WindowObservations,
)
from .util import atomic_json_dump, stable_hash
from .vision import (
    dedupe_visual_requests,
    namespace_visual_requests,
    select_rule_based_visual_requests,
)

if TYPE_CHECKING:
    from .config import LectureConfig
    from .pipeline import Pipeline
    from .schemas import LectureChunk, Transcript

logger = logging.getLogger(__name__)

# Version 2 invalidates the former claim/anchor/free-form-outline cache. Old window artifacts cannot be
# replayed into the episode graph because they let an LLM create canonical claims independently.
KNOWLEDGE_CACHE_VERSION = 2
STATE_PIPELINE_VERSION = 3

# These settings affect only hierarchy/synthesis. Excluding them from the extraction fingerprint is
# intentional: changing downstream batching must not throw away expensive ASR/visual/evidence work.
_DOWNSTREAM_NOTE_FIELDS = {
    "hierarchy_batch_episodes",
    "episode_synthesis_max_evidence_chars",
    "episode_symbol_context_limit",
    "state_section_max_evidence_chars",
    "state_section_raw_context_seconds",
    "state_section_raw_evidence_chars",
    "state_observation_lookahead",
    "state_observation_history",
    "state_observation_max_raw_windows",
}


def _collect_visual_evidence(
    pipeline: Pipeline,
    lecture: LectureConfig,
    chunk: LectureChunk,
    transcript: Transcript,
    source: Any,
    work: Path,
    figures_root: Path,
    notation: dict[str, str],
) -> tuple[list, list[VisualEvidence], float]:
    requests = []
    if pipeline.config.notes.visual_rule_selector:
        requests.extend(
            select_rule_based_visual_requests(
                chunk,
                transcript,
                pipeline.config.notes.max_low_confidence_visual_requests,
            )
        )
    if pipeline.config.notes.visual_llm_selector:
        requests.extend(pipeline.llm.analyze_chunk(chunk, notation).visual_requests)
    requests = dedupe_visual_requests(
        requests,
        within_seconds=pipeline.config.notes.visual_dedupe_seconds,
        limit=pipeline.config.vision.max_requests_per_chunk,
    )
    requests = namespace_visual_requests(chunk.id, requests)

    started = time.perf_counter()
    prepared_visuals = []
    for request in requests:
        frame_times = [
            max(0.0, request.timestamp + offset)
            for offset in pipeline.config.vision.frame_offsets_seconds
        ]
        frame_dir = work / "frames" / request.id
        frames = source.extract_frames(frame_times, frame_dir)
        prepared_visuals.append((request, frames))

    evidence: list[VisualEvidence] = []
    if prepared_visuals:
        workers = min(pipeline.config.vision.max_workers, len(prepared_visuals))
        futures: list[Future[VisualEvidence]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for request, frames in prepared_visuals:
                futures.append(
                    executor.submit(
                        pipeline.llm.resolve_visual_request,
                        request,
                        chunk,
                        [frame.path for frame in frames],
                        [frame.timestamp for frame in frames],
                    )
                )
            for (request, frames), future in zip(prepared_visuals, futures, strict=True):
                try:
                    visual = future.result()
                except Exception as exc:
                    logger.warning(
                        "[%s] visual OCR failed for %s: %s",
                        lecture.id,
                        request.id,
                        exc,
                    )
                    visual = VisualEvidence(
                        request_id=request.id,
                        description=(
                            f"Visual OCR failed after retries: {type(exc).__name__}: {exc}"
                        ),
                    )
                if visual.requires_figure_in_notes and frames:
                    index = visual.best_frame_index if visual.best_frame_index is not None else 0
                    index = max(0, min(index, len(frames) - 1))
                    destination = figures_root / f"{request.id}.jpg"
                    copy_asset(frames[index].path, destination)
                    visual.asset_path = str(
                        destination.relative_to(pipeline.config.latex.output_dir)
                    )
                evidence.append(visual)
    return requests, evidence, time.perf_counter() - started


def _load_window_artifact(path: Path, fingerprint: str):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        batch = WindowObservations.model_validate(payload["observations"])
        ids = [item.id for item in batch.observations]
        expected_ids = [
            f"obs_{batch.window_id}_{index:03d}"
            for index in range(len(batch.observations))
        ]
        # Legacy artifacts could contain model-generated, duplicated, or gapped ids. Episode
        # tracking references make those caches unsafe to repair after the fact, so recompute only
        # the affected windows while keeping already canonical caches.
        if ids != expected_ids or len(ids) != len(set(ids)):
            logger.info(
                "[%s] invalidating legacy window cache with non-canonical observation ids",
                batch.window_id,
            )
            return None
        tracking = EpisodeTrackingUpdate.model_validate(payload["episode_update"])
        return payload, batch, tracking
    except (json.JSONDecodeError, KeyError, ValidationError):
        return None


def _load_episode_batch(path: Path, fingerprint: str) -> ChunkNotes | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return ChunkNotes.model_validate(payload["notes"])
    except (json.JSONDecodeError, KeyError, ValidationError):
        return None



def _load_observation_resolution(
    path: Path,
    fingerprint: str,
) -> GeneratedObservationResolution | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return GeneratedObservationResolution.model_validate(payload["resolution"])
    except (json.JSONDecodeError, KeyError, ValidationError):
        return None


def _state_outline_context(outline: LectureOutline) -> list[dict[str, Any]]:
    return [
        {
            "id": section.id,
            "title": section.title,
            "start": section.start,
            "end": section.end,
            "episode_ids": list(section.episode_ids),
        }
        for section in outline.sections
    ]


def _clip_state_raw_text(value: str | None, limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _load_state_raw_window_index(work: Path) -> list[dict[str, Any]]:
    """Load compact literal ASR/OCR evidence retained by the extraction stage.

    The final state writer is allowed to reinterpret the intermediate semantic state, therefore it
    needs access to the observations that state was reconstructed from. Keep this index compact:
    frame pixels are not sent again, only literal ASR and OCR candidates already saved in window
    artifacts.
    """

    root = work / "knowledge_windows"
    windows: list[dict[str, Any]] = []
    for path in sorted(root.glob("window_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        chunk = payload.get("chunk") or {}
        visual_latex: list[str] = []
        ocr_candidates: list[dict[str, Any]] = []
        seen_ocr: set[tuple[Any, str]] = set()

        for visual in payload.get("visual_evidence", []):
            for key in ("raw_latex", "latex"):
                value = _clip_state_raw_text(str(visual.get(key) or ""), 400)
                if value and value not in visual_latex:
                    visual_latex.append(value)

            for candidate in visual.get("math_ocr_candidates", []):
                value = _clip_state_raw_text(str(candidate.get("text") or ""), 400)
                if not value:
                    continue
                key = (candidate.get("timestamp"), value)
                if key in seen_ocr:
                    continue
                seen_ocr.add(key)
                ocr_candidates.append(
                    {
                        "timestamp": candidate.get("timestamp"),
                        "text": value,
                        "source_id": candidate.get("source_id"),
                    }
                )
                if len(ocr_candidates) >= 12:
                    break
            if len(ocr_candidates) >= 12:
                break

        start = float(chunk.get("start", 0.0))
        windows.append(
            {
                "window_id": str(chunk.get("id") or path.stem),
                "start": start,
                "end": float(chunk.get("end", start)),
                "asr": _clip_state_raw_text(
                    str(chunk.get("timestamped_text") or chunk.get("text") or ""),
                    1200,
                ),
                "visual_latex": visual_latex[:3],
                "math_ocr_candidates": ocr_candidates,
            }
        )

    return sorted(
        windows,
        key=lambda item: (item["start"], item["end"], item["window_id"]),
    )


def _state_raw_evidence_context(
    evidence: dict[str, Any],
    raw_windows: list[dict[str, Any]],
    config,
) -> list[dict[str, Any]]:
    """Select bounded bidirectional literal evidence around one writer batch.

    Forward raw evidence is useful for resolving handwriting scope or an incomplete formula, but it
    is deliberately distinct from future semantic state: the prompt forbids importing later
    material merely because it appears in the look-ahead.
    """

    if not raw_windows:
        return []

    observations = list(evidence.get("observations", []))
    episodes = list(evidence.get("episodes", []))
    if observations:
        start = min(float(item.get("start", 0.0)) for item in observations)
        end = max(float(item.get("end", start)) for item in observations)
    elif episodes:
        start = min(float(item.get("start", 0.0)) for item in episodes)
        end = max(float(item.get("end", start)) for item in episodes)
    else:
        section = evidence.get("section", {})
        start = float(section.get("start", 0.0))
        end = float(section.get("end", start))

    direct_window_ids: set[str] = set()
    for observation in observations:
        window_id = observation.get("window_id")
        if window_id:
            direct_window_ids.add(str(window_id))
        direct_window_ids.update(str(item) for item in observation.get("window_ids", []))
    if not observations:
        for episode in episodes:
            direct_window_ids.update(str(item) for item in episode.get("window_ids", []))

    radius = float(config.state_section_raw_context_seconds)
    lower = start - radius
    upper = end + radius
    center = 0.5 * (start + end)

    candidates: list[dict[str, Any]] = []
    for raw in raw_windows:
        is_direct = str(raw["window_id"]) in direct_window_ids
        overlaps_context = float(raw["end"]) >= lower and float(raw["start"]) <= upper
        if not is_direct and not overlaps_context:
            continue
        item = dict(raw)
        item["direct"] = is_direct
        candidates.append(item)

    # Prefer windows that directly generated current semantic evidence, then nearby context.
    candidates.sort(
        key=lambda item: (
            not item["direct"],
            abs(0.5 * (float(item["start"]) + float(item["end"])) - center),
            float(item["start"]),
        )
    )
    max_chars = int(config.state_section_raw_evidence_chars)
    selected: list[dict[str, Any]] = []
    used = 2
    for item in candidates:
        serialized = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        cost = len(serialized) + (1 if selected else 0)
        if selected and used + cost > max_chars:
            continue
        selected.append(item)
        used += cost
        if used >= max_chars:
            break

    selected.sort(
        key=lambda item: (float(item["start"]), float(item["end"]), item["window_id"])
    )
    return selected


def _observation_window_ids(observation: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    window_id = observation.get("window_id")
    if window_id:
        ids.add(str(window_id))
    ids.update(str(item) for item in observation.get("window_ids", []) if item)
    return ids


def _raw_windows_for_observation_sequence(
    current: dict[str, Any],
    lookahead: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    *,
    max_windows: int,
) -> list[dict[str, Any]]:
    """Return literal sensor windows attached only to current/look-ahead observations."""

    current_ids = _observation_window_ids(current)
    future_ids: set[str] = set()
    for item in lookahead:
        future_ids.update(_observation_window_ids(item))

    selected: list[dict[str, Any]] = []
    for raw in raw_windows:
        window_id = str(raw.get("window_id", ""))
        if window_id in current_ids:
            role = "current"
        elif window_id in future_ids:
            role = "lookahead"
        else:
            continue
        item = dict(raw)
        item["role"] = role
        selected.append(item)

    if not selected:
        center = 0.5 * (
            float(current.get("start", 0.0)) + float(current.get("end", 0.0))
        )
        nearest = sorted(
            raw_windows,
            key=lambda item: abs(
                0.5 * (float(item.get("start", 0.0)) + float(item.get("end", 0.0))) - center
            ),
        )
        for raw in nearest[:1]:
            item = dict(raw)
            item["role"] = "current_fallback"
            selected.append(item)

    selected.sort(
        key=lambda item: (
            item.get("role") != "current",
            float(item.get("start", 0.0)),
            str(item.get("window_id", "")),
        )
    )
    return selected[:max_windows]


def _claims_for_observation(
    evidence: dict[str, Any],
    observation_id: str,
) -> list[dict[str, Any]]:
    return [
        item
        for item in evidence.get("claims", [])
        if observation_id in {str(value) for value in item.get("evidence_ids", [])}
    ]


def _symbols_for_observation(
    evidence: dict[str, Any],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    cutoff = float(observation.get("end", observation.get("start", 0.0)))
    return [
        item
        for item in evidence.get("symbols", [])
        if float(item.get("introduced_at", 0.0)) <= cutoff
    ]


def _resolved_observation_from_result(
    original: dict[str, Any],
    resolution: GeneratedObservationResolution,
) -> dict[str, Any]:
    resolved = dict(original)
    if resolution.text.strip():
        resolved["text"] = resolution.text.strip()
    if resolution.latex is not None and resolution.latex.strip():
        resolved["latex"] = resolution.latex.strip()
    resolved["sequentially_resolved"] = True
    return resolved


def _resolve_single_state_observation(
    orchestrator: KnowledgeOrchestrator,
    *,
    section: OutlineSection,
    evidence: dict[str, Any],
    current: dict[str, Any],
    lookahead: list[dict[str, Any]],
    resolved_history: list[dict[str, Any]],
    previous_context: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    config,
) -> GeneratedObservationResolution:
    observation_id = str(current.get("id", ""))
    claims = _claims_for_observation(evidence, observation_id)
    symbols = _symbols_for_observation(evidence, current)
    raw = _raw_windows_for_observation_sequence(
        current,
        lookahead,
        raw_windows,
        max_windows=int(config.state_observation_max_raw_windows),
    )
    history_limit = int(config.state_observation_history)
    history = resolved_history[-history_limit:] if history_limit else []

    prompt = f"""Resolve exactly ONE chronological lecture observation into its best supported
mathematical meaning. This call is one step of a sequential state update; it must not rewrite
earlier accepted observations.

Current fixed section:
{json.dumps(section.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

Already resolved immutable history:
{json.dumps(history, ensure_ascii=False, separators=(",", ":"))}

Previously written section context (continuity only):
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

CURRENT observation to resolve:
{json.dumps(current, ensure_ascii=False, separators=(",", ":"))}

Claims currently derived from this observation (fallible):
{json.dumps(claims, ensure_ascii=False, separators=(",", ":"))}

Symbols established no later than this observation:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Next observations, provided only as fixed-lag look-ahead to disambiguate CURRENT:
{json.dumps(lookahead, ensure_ascii=False, separators=(",", ":"))}

Literal ASR/OCR windows attached only to CURRENT/look-ahead observations:
{json.dumps(raw, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Output the resolved form of CURRENT observation only.
- Earlier resolved history is immutable. Do not revise, summarize, or replace it.
- Look-ahead may clarify the scope, notation, sign, denominator, or role of CURRENT, but material
  belonging only to a later observation must not be moved into CURRENT.
- OCR, ASR, and the intermediate observation are all noisy. Interpret them jointly.
- Prefer repeated/consistent local evidence over a single cleaner-looking OCR fragment.
- Never delete a coefficient, denominator, quantifier, membership, subscript, or relation sign merely
  because one OCR candidate omitted it.
- Reject readings that contradict established history or elementary consequences of it, for example
  a zero denominator, incompatible kernel membership, or a violated linearity relation.
- Standard mathematics is a bounded consistency prior: it may reject an impossible reading but must
  not invent lecture-specific notation or a missing theorem statement.
- If the intermediate observation must change semantically, return a CorrectionRecord. If the
  ambiguity cannot be resolved, preserve only the common supported content and record the ambiguity
  in unresolved.
- Do not emit a no-op correction.
- Write prose in language code {orchestrator.output_language} and mathematics in LaTeX.
"""
    return orchestrator._structured(
        prompt,
        GeneratedObservationResolution,
        operation="state_observation_resolve",
        max_tokens=1536,
        split_oversized_task=True,
    )


def _section_observation_sequence(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
) -> list[dict[str, Any]]:
    episode_ids = set(section.episode_ids)
    return [
        item.model_dump(mode="json")
        for item in sorted(
            kb.observations,
            key=lambda item: (item.start, item.end, item.id),
        )
        if item.episode_id in episode_ids
    ]


def _resolve_state_batch_sequential(
    orchestrator: KnowledgeOrchestrator,
    *,
    section: OutlineSection,
    evidence: dict[str, Any],
    section_observations: list[dict[str, Any]],
    resolved_history: list[dict[str, Any]],
    previous_context: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    work: Path,
    config,
    llm_config: dict[str, Any],
    force: bool,
) -> tuple[dict[str, Any], list[Any], list[str], int]:
    """Resolve batch observations one-by-one with bounded future look-ahead."""

    by_id = {
        str(item.get("id", "")): index
        for index, item in enumerate(section_observations)
        if item.get("id")
    }
    batch_ids = {
        str(item.get("id", ""))
        for item in evidence.get("observations", [])
        if item.get("id")
    }
    resolved_batch: list[dict[str, Any]] = []
    corrections: list[Any] = []
    unresolved: list[str] = []
    cache_hits = 0
    lookahead_count = int(config.state_observation_lookahead)

    for original in sorted(
        evidence.get("observations", []),
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            str(item.get("id", "")),
        ),
    ):
        observation_id = str(original.get("id", ""))
        position = by_id.get(observation_id)
        lookahead = (
            section_observations[position + 1 : position + 1 + lookahead_count]
            if position is not None
            else []
        )

        raw = _raw_windows_for_observation_sequence(
            original,
            lookahead,
            raw_windows,
            max_windows=int(config.state_observation_max_raw_windows),
        )
        history_limit = int(config.state_observation_history)
        history = resolved_history[-history_limit:] if history_limit else []
        fingerprint = stable_hash(
            {
                "state_pipeline_version": STATE_PIPELINE_VERSION,
                "section_id": section.id,
                "current": original,
                "claims": _claims_for_observation(evidence, observation_id),
                "symbols": _symbols_for_observation(evidence, original),
                "lookahead": lookahead,
                "history": history,
                "previous_context": previous_context,
                "raw_windows": raw,
                "llm": llm_config,
            }
        )
        path = (
            work
            / "state_observation_resolutions"
            / section.id
            / f"{observation_id}.json"
        )
        resolution = None if force else _load_observation_resolution(path, fingerprint)
        if resolution is not None:
            cache_hits += 1
        else:
            try:
                resolution = _resolve_single_state_observation(
                    orchestrator,
                    section=section,
                    evidence=evidence,
                    current=original,
                    lookahead=lookahead,
                    resolved_history=resolved_history,
                    previous_context=previous_context,
                    raw_windows=raw_windows,
                    config=config,
                )
            except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
                resolution = GeneratedObservationResolution(
                    text=str(original.get("text") or ""),
                    latex=original.get("latex"),
                    unresolved=[
                        "Sequential observation resolution failed for "
                        f"{observation_id}: {type(exc).__name__}: {exc}"
                    ],
                )
            atomic_json_dump(
                path,
                {
                    "fingerprint": fingerprint,
                    "current": original,
                    "lookahead": lookahead,
                    "raw_windows": raw,
                    "resolution": resolution.model_dump(mode="json"),
                },
            )

        resolved = _resolved_observation_from_result(original, resolution)
        resolved_batch.append(resolved)
        resolved_history.append(resolved)
        if resolution.correction is not None:
            correction = resolution.correction
            if correction.original.strip() != correction.corrected.strip():
                corrections.append(correction)
        unresolved.extend(resolution.unresolved)

    payload = dict(evidence)
    payload["observations"] = resolved_batch
    # Claims are derived from pre-resolution observations and may now be stale. Ground final blocks
    # directly in resolved observation ids instead of exposing contradictory duplicate semantics.
    payload["claims"] = []
    payload["sequential_resolution"] = {
        "resolved_observation_ids": [
            str(item.get("id", "")) for item in resolved_batch
        ],
        "batch_observation_ids": sorted(batch_ids),
    }
    return payload, corrections, list(dict.fromkeys(unresolved)), cache_hits


def _state_section_payload(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    transcript: Transcript,
    config,
) -> dict[str, Any]:
    payload = evidence_for_section(kb, section, transcript, config)
    payload.pop("transcript", None)
    return payload


def _split_state_section_evidence_by_observations(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Split one semantic episode without falling back to raw ASR or recomputing state."""

    observations = list(evidence.get("observations", []))
    if len(observations) <= 1:
        return None

    midpoint = len(observations) // 2

    def build(selected: list[dict[str, Any]]) -> dict[str, Any]:
        selected_ids = {str(item.get("id", "")) for item in selected if item.get("id")}
        claims = [
            item
            for item in evidence.get("claims", [])
            if not item.get("evidence_ids")
            or selected_ids.intersection(str(value) for value in item.get("evidence_ids", []))
        ]
        claim_ids = {str(item.get("id", "")) for item in claims if item.get("id")}

        episodes = []
        for item in evidence.get("episodes", []):
            episode = dict(item)
            episode["observation_ids"] = [
                value
                for value in episode.get("observation_ids", [])
                if str(value) in selected_ids
            ]
            episode["claim_ids"] = [
                value
                for value in episode.get("claim_ids", [])
                if str(value) in claim_ids
            ]
            episodes.append(episode)

        child = dict(evidence)
        child["episodes"] = episodes
        child["claims"] = claims
        child["observations"] = selected
        return child

    return build(observations[:midpoint]), build(observations[midpoint:])



def _state_section_batches(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    transcript: Transcript,
    config,
) -> list[dict[str, Any]]:
    episode_ids = list(section.episode_ids)
    if not episode_ids:
        return []

    max_chars = int(config.state_section_max_evidence_chars)
    batches: list[dict[str, Any]] = []

    def append_bounded(payload: dict[str, Any]) -> None:
        pending = [payload]
        while pending:
            candidate = pending.pop(0)
            serialized = json.dumps(
                candidate,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(serialized) <= max_chars:
                batches.append(candidate)
                continue

            split = _split_state_section_evidence_by_observations(candidate)
            if split is None:
                logger.warning(
                    "[%s] smallest state-section evidence leaf still exceeds "
                    "notes.state_section_max_evidence_chars=%d (%d chars); "
                    "passing leaf to resilient writer",
                    section.id,
                    max_chars,
                    len(serialized),
                )
                batches.append(candidate)
                continue

            left, right = split
            logger.info(
                "[%s] pre-splitting oversized state evidence "
                "(%d chars, %d observations) into %d + %d observations",
                section.id,
                len(serialized),
                len(candidate.get("observations", [])),
                len(left.get("observations", [])),
                len(right.get("observations", [])),
            )
            pending[0:0] = [left, right]

    current: list[str] = []
    for episode_id in episode_ids:
        candidate_ids = [*current, episode_id]
        child = section.model_copy(
            update={
                "episode_ids": candidate_ids,
                "claim_ids": [],
                "evidence_ids": [],
                "anchor_ids": [],
                "subsections": [],
            }
        )
        payload = _state_section_payload(kb, child, transcript, config)
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if current and len(serialized) > max_chars:
            committed = section.model_copy(
                update={
                    "episode_ids": current,
                    "claim_ids": [],
                    "evidence_ids": [],
                    "anchor_ids": [],
                    "subsections": [],
                }
            )
            append_bounded(_state_section_payload(kb, committed, transcript, config))
            current = [episode_id]
        else:
            current = candidate_ids

    if current:
        committed = section.model_copy(
            update={
                "episode_ids": current,
                "claim_ids": [],
                "evidence_ids": [],
                "anchor_ids": [],
                "subsections": [],
            }
        )
        append_bounded(_state_section_payload(kb, committed, transcript, config))

    for index, payload in enumerate(batches):
        payload["batch"] = {"index": index, "count": len(batches)}
    return batches


def _write_state_section_batch(
    orchestrator: KnowledgeOrchestrator,
    section: OutlineSection,
    evidence: dict[str, Any],
    *,
    outline_context: list[dict[str, Any]],
    previous_context: list[dict[str, Any]],
    raw_evidence_context: list[dict[str, Any]] | None = None,
    guided_json: bool = True,
    max_tokens: int = 6144,
) -> ChunkNotes:
    prompt = f"""Write one contiguous part of a FINAL lecture-note section. You are the semantic
resolver at the end of a noisy multimodal reconstruction pipeline. Evidence extraction, episode
tracking and outline planning happened before this call, but their mathematical interpretation may
still be wrong.

Global lecture outline (read-only narrative context):
{json.dumps(outline_context, ensure_ascii=False, separators=(",", ":"))}

Current fixed section:
{json.dumps(section.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

Intermediate semantic reconstruction for this batch (useful but FALLIBLE):
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Nearby raw sensory evidence (bounded bidirectional context; literal/noisy, not authoritative):
{json.dumps(raw_evidence_context or [], ensure_ascii=False, separators=(",", ":"))}

Previously written blocks from THIS section only:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Interpret the lecture; do not merely paraphrase OCR or the intermediate reconstruction.
- Canonical observations/active claims/symbol records are fallible hypotheses produced earlier,
  NOT ground truth. Raw ASR/OCR candidates are also fallible observations.
- Compare all available evidence. Prefer the interpretation jointly supported by temporal
  continuity, notation/type information, neighboring board states, speech and mathematical
  consistency.
- A forward raw window may only disambiguate material already being written/discussed in this
  batch. Do not import a later theorem, definition or symbol merely because it appears in look-ahead
  sensory evidence.
- You MAY correct an intermediate claim or formula when raw evidence or an elementary consequence
  of accepted context shows that reading is inconsistent. When you do, add a CorrectionRecord with
  the original reading, corrected reading, reason, basis and confidence.
- Before finalizing, explicitly test your chosen reading for local contradictions. Reject readings
  that make a displayed denominator zero, contradict a simultaneous membership/equation, violate an
  already established type/linearity relation, or conflict with clearer adjacent board states.
- OCR token adjacency does not determine mathematical scope. A trailing membership/relation may
  apply to the whole left-hand expression rather than to the nearest symbol; resolve scope from the
  equation and surrounding evidence.
- If two readings remain genuinely ambiguous, write only their common supported content and record
  the ambiguity in unresolved. Do not invent a textbook completion.
- Follow the fixed episode order and the lecture's actual narrative line.
- Preserve lecturer corrections, notation evolution, theorem/proof continuity and level of detail.
- Never resurrect superseded/retracted content as a current fact.
- Do not introduce textbook material merely because it would make the exposition nicer.
- Avoid repeating a definition/proof step already present in previous_context unless this batch
  genuinely develops it further.
- Every substantive block must cite source_claim_ids and/or source_evidence_ids present in the
  supplied intermediate semantic evidence. A corrected interpretation should cite the evidence it
  corrects and uses.
- Return block bodies only; renderer owns section/theorem/proof wrappers.
- Write prose in language code {orchestrator.output_language} and mathematics in LaTeX.
"""
    generated = orchestrator._structured(
        prompt,
        GeneratedChunkNotes,
        operation="state_section_write",
        max_tokens=max_tokens,
        guided_json=guided_json,
        split_oversized_task=True,
    )
    notes = generated.to_chunk_notes()
    notes.chunk_id = section.id
    notes.start = section.start
    notes.end = section.end
    notes.section_title = section.title.replace("$", "")

    allowed_claims = {str(item["id"]) for item in evidence.get("claims", [])}
    allowed_observations = {str(item["id"]) for item in evidence.get("observations", [])}
    kept = []
    for block in notes.blocks:
        original_claims = list(block.source_claim_ids)
        original_evidence = list(block.source_evidence_ids)
        block.source_claim_ids = [item for item in original_claims if item in allowed_claims]
        block.source_evidence_ids = [
            item for item in original_evidence if item in allowed_observations
        ]
        if not block.source_claim_ids and not block.source_evidence_ids:
            notes.unresolved.append(
                f"Dropped ungrounded state-section block: {block.latex[:160]}"
            )
            continue
        kept.append(block)
    notes.blocks = kept
    notes.unresolved = list(dict.fromkeys(notes.unresolved))
    return notes


def _state_section_for_episode_ids(
    section: OutlineSection,
    episode_ids: list[str],
) -> OutlineSection:
    return section.model_copy(
        update={
            "episode_ids": list(episode_ids),
            "claim_ids": [],
            "evidence_ids": [],
            "anchor_ids": [],
            "subsections": [],
        }
    )



def _write_state_section_batch_resilient(
    orchestrator: KnowledgeOrchestrator,
    section: OutlineSection,
    evidence: dict[str, Any],
    *,
    outline_context: list[dict[str, Any]],
    previous_context: list[dict[str, Any]],
    kb: LectureKnowledgeBase,
    transcript: Transcript,
    config,
    raw_window_index: list[dict[str, Any]] | None = None,
    raw_evidence_context: list[dict[str, Any]] | None = None,
) -> ChunkNotes:
    """Recursively split final-writer work that cannot fit in one structured request."""

    if raw_evidence_context is None:
        raw_evidence_context = _state_raw_evidence_context(
            evidence,
            raw_window_index or [],
            config,
        )

    try:
        return _write_state_section_batch(
            orchestrator,
            section,
            evidence,
            outline_context=outline_context,
            previous_context=previous_context,
            raw_evidence_context=raw_evidence_context,
        )
    except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
        episode_ids = [
            str(item["id"])
            for item in evidence.get("episodes", [])
            if item.get("id")
        ]

        if len(episode_ids) > 1:
            midpoint = len(episode_ids) // 2
            left_ids = episode_ids[:midpoint]
            right_ids = episode_ids[midpoint:]
            logger.warning(
                "[%s] state-section task did not fit or failed structured retries; "
                "splitting %d episodes into %d + %d",
                section.id,
                len(episode_ids),
                len(left_ids),
                len(right_ids),
            )

            left_section = _state_section_for_episode_ids(section, left_ids)
            right_section = _state_section_for_episode_ids(section, right_ids)
            left_evidence = _state_section_payload(kb, left_section, transcript, config)
            right_evidence = _state_section_payload(kb, right_section, transcript, config)

            left_notes = _write_state_section_batch_resilient(
                orchestrator,
                left_section,
                left_evidence,
                outline_context=outline_context,
                previous_context=previous_context,
                kb=kb,
                transcript=transcript,
                config=config,
                raw_window_index=raw_window_index,
            )
            right_previous = [
                *previous_context,
                *previous_block_context([left_notes]),
            ][-2:]
            right_notes = _write_state_section_batch_resilient(
                orchestrator,
                right_section,
                right_evidence,
                outline_context=outline_context,
                previous_context=right_previous,
                kb=kb,
                transcript=transcript,
                config=config,
                raw_window_index=raw_window_index,
            )
            return _merge_state_section_batches(section, [left_notes, right_notes])

        observation_split = _split_state_section_evidence_by_observations(evidence)
        if observation_split is not None:
            left_evidence, right_evidence = observation_split
            left_count = len(left_evidence.get("observations", []))
            right_count = len(right_evidence.get("observations", []))
            logger.warning(
                "[%s] single-episode state-section task still too large/invalid; "
                "splitting canonical observations into %d + %d",
                section.id,
                left_count,
                right_count,
            )
            left_notes = _write_state_section_batch_resilient(
                orchestrator,
                section,
                left_evidence,
                outline_context=outline_context,
                previous_context=previous_context,
                kb=kb,
                transcript=transcript,
                config=config,
                raw_window_index=raw_window_index,
            )
            right_previous = [
                *previous_context,
                *previous_block_context([left_notes]),
            ][-2:]
            right_notes = _write_state_section_batch_resilient(
                orchestrator,
                section,
                right_evidence,
                outline_context=outline_context,
                previous_context=right_previous,
                kb=kb,
                transcript=transcript,
                config=config,
                raw_window_index=raw_window_index,
            )
            return _merge_state_section_batches(section, [left_notes, right_notes])

        if isinstance(exc, StructuredTaskTooLargeError):
            logger.warning(
                "[%s] state-section task cannot be split further after backend context limit: %s",
                section.id,
                exc,
            )
            return ChunkNotes(
                chunk_id=section.id,
                start=section.start,
                end=section.end,
                section_title=section.title.replace("$", ""),
                blocks=[],
                unresolved=[
                    "State-section writer reached the backend context limit after recursive "
                    "episode/observation splitting."
                ],
            )

        logger.warning(
            "[%s] state-section leaf remained invalid after structured retries; "
            "retrying once without guided JSON: %s",
            section.id,
            exc,
        )
        try:
            return _write_state_section_batch(
                orchestrator,
                section,
                evidence,
                outline_context=outline_context,
                previous_context=previous_context,
                raw_evidence_context=raw_evidence_context,
                guided_json=False,
                max_tokens=8192,
            )
        except (
            json.JSONDecodeError,
            ValidationError,
            StructuredTaskTooLargeError,
        ) as leaf_exc:
            logger.warning(
                "[%s] state-section leaf unresolved after unguided retry: %s",
                section.id,
                leaf_exc,
            )
            return ChunkNotes(
                chunk_id=section.id,
                start=section.start,
                end=section.end,
                section_title=section.title.replace("$", ""),
                blocks=[],
                unresolved=[
                    "State-section writer could not serialize the smallest canonical batch after "
                    f"structured retries: {type(leaf_exc).__name__}: {leaf_exc}"
                ],
            )


def _merge_state_section_batches(
    section: OutlineSection,
    batches: list[ChunkNotes],
) -> ChunkNotes:
    merged = ChunkNotes(
        chunk_id=section.id,
        start=section.start,
        end=section.end,
        section_title=section.title.replace("$", ""),
        blocks=[],
    )
    for notes in batches:
        merged.blocks.extend(notes.blocks)
        merged.notation.extend(notes.notation)
        merged.corrections.extend(notes.corrections)
        merged.unresolved = list(dict.fromkeys([*merged.unresolved, *notes.unresolved]))
    return merged


def run_knowledge_pipeline(
    pipeline: Pipeline,
    *,
    lecture: LectureConfig,
    transcript: Transcript,
    source: Any,
    source_identity: dict[str, Any],
    work: Path,
    ir_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    notation: dict[str, str],
    media_seconds: float,
    asr_seconds: float,
    run_started: float,
    force: bool,
) -> LectureIR:
    notes_started = time.perf_counter()
    pipeline.llm.reset_usage()
    orchestrator = KnowledgeOrchestrator(
        pipeline.llm,
        pipeline.config.notes,
        pipeline.config.llm.output_language,
    )
    kb = LectureKnowledgeBase(
        lecture_id=lecture.id,
        title=lecture.title or lecture.id,
    )
    chunks = chunk_transcript(
        transcript,
        pipeline.config.notes.chunk_target_seconds,
        pipeline.config.notes.chunk_overlap_seconds,
    )
    figures_root = pipeline.config.latex.output_dir / "figures" / lecture.id

    cache_hits = 0
    processed_windows = 0
    visual_requests_processed = 0
    visual_evidence_successful = 0
    vision_seconds = 0.0
    extract_seconds = 0.0
    episode_track_seconds = 0.0

    for chunk in chunks:
        state_before = compact_knowledge_state(kb, pipeline.config.notes)
        window_fingerprint = stable_hash(
            {
                "source": source_identity,
                "chunk": chunk.model_dump(mode="json"),
                "kb_state_before": state_before,
                "notes": pipeline.config.notes.model_dump(
                    mode="json", exclude=_DOWNSTREAM_NOTE_FIELDS
                ),
                "vision": pipeline.config.vision.model_dump(mode="json"),
                "llm": pipeline.config.llm.model_dump(mode="json"),
                "knowledge_cache_version": KNOWLEDGE_CACHE_VERSION,
            }
        )
        artifact = work / "knowledge_windows" / f"{chunk.id}.json"
        cached = None if force else _load_window_artifact(artifact, window_fingerprint)
        if cached is not None:
            payload, batch, tracking = cached
            added_ids = merge_window_observations(kb, batch)
            apply_episode_tracking(kb, tracking, added_ids, window_id=chunk.id)
            cache_hits += 1
            visual_requests_processed += len(payload.get("visual_requests", []))
            visual_evidence_successful += sum(
                item.get("kind") != "none" and float(item.get("confidence", 0.0)) >= 0.75
                for item in payload.get("visual_evidence", [])
            )
            logger.info("[%s] %s episode cache hit", lecture.id, chunk.id)
            if pipeline.config.notes.architecture == "state":
                atomic_json_dump(
                    work / "lecture_state.json",
                    make_lecture_state(kb).model_dump(mode="json"),
                )
            continue

        logger.info(
            "[%s] extracting evidence/episodes from %s (%d/%d)",
            lecture.id,
            chunk.id,
            processed_windows + cache_hits + 1,
            len(chunks),
        )
        requests, evidence, visual_elapsed = _collect_visual_evidence(
            pipeline,
            lecture,
            chunk,
            transcript,
            source,
            work,
            figures_root,
            notation,
        )
        vision_seconds += visual_elapsed
        visual_requests_processed += len(requests)
        visual_evidence_successful += sum(
            item.kind != "none" and item.confidence >= 0.75 for item in evidence
        )

        extract_started = time.perf_counter()
        batch = orchestrator.extract_observations(chunk, evidence, kb)
        extract_seconds += time.perf_counter() - extract_started
        added_ids = merge_window_observations(kb, batch)

        track_started = time.perf_counter()
        tracking = orchestrator.track_episodes(kb, batch, added_ids)
        episode_track_seconds += time.perf_counter() - track_started
        apply_episode_tracking(kb, tracking, added_ids, window_id=chunk.id)
        processed_windows += 1

        atomic_json_dump(
            artifact,
            {
                "fingerprint": window_fingerprint,
                "chunk": chunk.model_dump(mode="json"),
                "visual_requests": [item.model_dump(mode="json") for item in requests],
                "visual_evidence": [item.model_dump(mode="json") for item in evidence],
                "observations": batch.model_dump(mode="json"),
                "episode_update": tracking.model_dump(mode="json"),
            },
        )
        atomic_json_dump(work / "lecture_kb.json", kb.model_dump(mode="json"))
        if pipeline.config.notes.architecture == "state":
            atomic_json_dump(
                work / "lecture_state.json",
                make_lecture_state(kb).model_dump(mode="json"),
            )

    # A technical window never closes an episode. End-of-lecture is the only unconditional close.
    close_open_episodes(kb)
    kb_fingerprint = stable_hash(
        {
            "kb": kb.model_dump(mode="json"),
            "notes": pipeline.config.notes.model_dump(mode="json"),
            "llm": pipeline.config.llm.model_dump(mode="json"),
            "knowledge_cache_version": KNOWLEDGE_CACHE_VERSION,
        }
    )
    atomic_json_dump(work / "lecture_kb.json", kb.model_dump(mode="json"))

    hierarchy_path = work / "episode_hierarchy.json"
    hierarchy_fingerprint = stable_hash(
        {
            "kb_fingerprint": kb_fingerprint,
            "hierarchy_batch_episodes": pipeline.config.notes.hierarchy_batch_episodes,
            "hierarchy_cache_version": HIERARCHY_CACHE_VERSION,
        }
    )
    hierarchy: EpisodeHierarchyPlan | None = None
    if hierarchy_path.exists() and not force:
        try:
            payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == hierarchy_fingerprint:
                hierarchy = EpisodeHierarchyPlan.model_validate(payload["hierarchy"])
        except (json.JSONDecodeError, KeyError, ValidationError):
            hierarchy = None

    hierarchy_started = time.perf_counter()
    if hierarchy is None:
        hierarchy = plan_episode_hierarchy_bounded(orchestrator, kb)
        atomic_json_dump(
            hierarchy_path,
            {
                "fingerprint": hierarchy_fingerprint,
                "hierarchy": hierarchy.model_dump(mode="json"),
            },
        )
    hierarchy_seconds = time.perf_counter() - hierarchy_started

    # This is a deterministic projection of the episode graph. The hierarchy LLM only chooses
    # boundaries/titles; it cannot create, drop, reorder, resize, or populate a section independently.
    outline = LectureOutline(
        sections=build_outline_from_episodes(
            kb,
            hierarchy,
            lecture_title=lecture.title or lecture.id,
        ),
        unresolved=list(hierarchy.unresolved),
    )
    atomic_json_dump(
        work / "lecture_outline.json",
        {
            "fingerprint": hierarchy_fingerprint,
            "outline": outline.model_dump(mode="json"),
        },
    )
    if pipeline.config.notes.architecture == "state":
        atomic_json_dump(
            work / "lecture_state.json",
            make_lecture_state(kb, outline=outline).model_dump(mode="json"),
        )

    state_mode = pipeline.config.notes.architecture == "state"
    state_section_cache_hits = 0
    state_section_batches_total = 0
    state_synthesis_seconds = 0.0

    if state_mode:
        note_sections: list[ChunkNotes] = []
        outline_context = _state_outline_context(outline)
        raw_window_index = _load_state_raw_window_index(work)
        for section in outline.sections:
            evidence_batches = _state_section_batches(
                kb,
                section,
                transcript,
                pipeline.config.notes,
            )
            state_section_batches_total += len(evidence_batches)
            generated_batches: list[ChunkNotes] = []
            for batch_index, evidence_payload in enumerate(evidence_batches):
                previous_context = previous_block_context(generated_batches)
                raw_evidence_context = _state_raw_evidence_context(
                    evidence_payload,
                    raw_window_index,
                    pipeline.config.notes,
                )
                fingerprint = stable_hash(
                    {
                        "state_pipeline_version": STATE_PIPELINE_VERSION,
                        "section": section.model_dump(mode="json"),
                        "outline_context": outline_context,
                        "evidence": evidence_payload,
                        "raw_evidence_context": raw_evidence_context,
                        "previous_context": previous_context,
                        "llm": pipeline.config.llm.model_dump(mode="json"),
                    }
                )
                path = (
                    work
                    / "state_section_batches"
                    / section.id
                    / f"batch_{batch_index:03d}.json"
                )
                notes = None if force else _load_episode_batch(path, fingerprint)
                if notes is not None:
                    state_section_cache_hits += 1
                    generated_batches.append(notes)
                    continue

                started = time.perf_counter()
                notes = _write_state_section_batch_resilient(
                    orchestrator,
                    section,
                    evidence_payload,
                    outline_context=outline_context,
                    previous_context=previous_context,
                    kb=kb,
                    transcript=transcript,
                    config=pipeline.config.notes,
                    raw_window_index=raw_window_index,
                    raw_evidence_context=raw_evidence_context,
                )
                state_synthesis_seconds += time.perf_counter() - started
                atomic_json_dump(
                    path,
                    {
                        "fingerprint": fingerprint,
                        "evidence": evidence_payload,
                        "raw_evidence_context": raw_evidence_context,
                        "notes": notes.model_dump(mode="json"),
                    },
                )
                generated_batches.append(notes)

            note_sections.append(_merge_state_section_batches(section, generated_batches))

        episode_batch_cache_hits = 0
        episode_batches_total = 0
        episode_synthesis_seconds = 0.0
        episode_validation_seconds = 0.0
    else:
        episode_notes: dict[str, ChunkNotes] = {}
        episode_batch_cache_hits = 0
        episode_batches_total = 0
        episode_synthesis_seconds = 0.0
        episode_validation_seconds = 0.0

        episodes = sorted(
            [item for item in kb.episodes if item.observation_ids],
            key=lambda item: (item.start, item.end, item.id),
        )
        for episode in episodes:
            evidence_batches = episode_evidence_batches(kb, episode, pipeline.config.notes)
            episode_batches_total += len(evidence_batches)
            generated_batches: list[ChunkNotes] = []

            for batch_index, evidence_payload in enumerate(evidence_batches):
                previous_context = previous_block_context(generated_batches)
                batch_fingerprint = stable_hash(
                    {
                        "episode": episode.model_dump(mode="json"),
                        "evidence": evidence_payload,
                        "previous_context": previous_context,
                        "llm": pipeline.config.llm.model_dump(mode="json"),
                        "validation_enabled": pipeline.config.notes.global_validation,
                        "validation_threshold": (
                            pipeline.config.notes.global_validation_apply_threshold
                        ),
                        "episode_synthesis_cache_version": EPISODE_SYNTHESIS_CACHE_VERSION,
                    }
                )
                batch_path = (
                    work
                    / "knowledge_episode_batches"
                    / episode.id
                    / f"batch_{batch_index:03d}.json"
                )
                notes = None if force else _load_episode_batch(batch_path, batch_fingerprint)
                if notes is not None:
                    episode_batch_cache_hits += 1
                    generated_batches.append(notes)
                    logger.info(
                        "[%s] %s batch %d/%d cache hit",
                        lecture.id,
                        episode.id,
                        batch_index + 1,
                        len(evidence_batches),
                    )
                    continue

                logger.info(
                    "[%s] synthesizing %s batch %d/%d",
                    lecture.id,
                    episode.id,
                    batch_index + 1,
                    len(evidence_batches),
                )
                started = time.perf_counter()
                notes = write_episode_batch(
                    orchestrator,
                    episode,
                    evidence_payload,
                    previous_context,
                )
                episode_synthesis_seconds += time.perf_counter() - started

                validation_payload = None
                if pipeline.config.notes.global_validation and notes.blocks:
                    started = time.perf_counter()
                    validation_payload = validate_episode_batch(
                        orchestrator,
                        evidence_payload,
                        notes,
                    )
                    episode_validation_seconds += time.perf_counter() - started
                    apply_episode_validation(
                        notes,
                        validation_payload,
                        threshold=pipeline.config.notes.global_validation_apply_threshold,
                    )

                atomic_json_dump(
                    batch_path,
                    {
                        "fingerprint": batch_fingerprint,
                        "evidence": evidence_payload,
                        "notes": notes.model_dump(mode="json"),
                        "validation": (
                            validation_payload.model_dump(mode="json")
                            if validation_payload is not None
                            else None
                        ),
                    },
                )
                generated_batches.append(notes)

            episode_notes[episode.id] = merge_episode_batches(episode, generated_batches)

        # Sections are now a deterministic projection of validated episode notes. No section-level or
        # full-document synthesis/validation call can grow with lecture duration.
        note_sections = assemble_outline_sections(
            outline.sections,
            episode_notes,
            outline_unresolved=outline.unresolved,
        )
    ir = LectureIR(
        lecture_id=lecture.id,
        title=lecture.title or lecture.id,
        chunks=note_sections,
    )

    symbol_meanings: dict[str, set[str]] = {}
    for symbol in kb.symbols:
        if symbol.active and symbol.symbol and symbol.meaning:
            symbol_meanings.setdefault(symbol.symbol, set()).add(symbol.meaning)
    for symbol, meanings in symbol_meanings.items():
        # The course-level legacy registry is unscoped. Export only symbols whose meaning is
        # unambiguous across episode scopes.
        if len(meanings) == 1:
            notation.setdefault(symbol, next(iter(meanings)))
    pipeline._save_notation_registry(notation)

    atomic_json_dump(ir_path, ir.model_dump(mode="json"))
    manifest["ir_fingerprint"] = pipeline._ir_fingerprint(transcript, notation)
    atomic_json_dump(manifest_path, manifest)

    usage = pipeline.llm.usage_snapshot()
    atomic_json_dump(
        work / "run_metrics.json",
        {
            "lecture_id": lecture.id,
            "architecture": (
                "state_episode_graph_section_synthesis"
                if state_mode
                else "knowledge_episode_graph_bounded"
            ),
            "media_seconds": round(media_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "notes_seconds": round(time.perf_counter() - notes_started, 3),
            "vision_seconds": round(vision_seconds, 3),
            "knowledge_extract_seconds": round(extract_seconds, 3),
            "episode_track_seconds": round(episode_track_seconds, 3),
            "hierarchy_seconds": round(hierarchy_seconds, 3),
            "episode_synthesis_seconds": round(episode_synthesis_seconds, 3),
            "episode_validation_seconds": round(episode_validation_seconds, 3),
            "state_synthesis_seconds": round(state_synthesis_seconds, 3),
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "windows_total": len(chunks),
            "windows_processed": processed_windows,
            "window_cache_hits": cache_hits,
            "episodes_total": len(kb.episodes),
            "episode_batches_total": episode_batches_total,
            "episode_batch_cache_hits": episode_batch_cache_hits,
            "state_section_batches_total": state_section_batches_total,
            "state_section_cache_hits": state_section_cache_hits,
            "topic_sections_total": len(outline.sections),
            "subtopics_total": sum(len(item.subsections) for item in outline.sections),
            "sections_total": len(note_sections),
            "observations_total": len(kb.observations),
            "observation_aliases_total": len(kb.observation_aliases),
            "claims_total": len(kb.claims),
            "active_claims": sum(item.status == "active" for item in kb.claims),
            "superseded_claims": sum(item.status == "superseded" for item in kb.claims),
            "retracted_claims": sum(item.status == "retracted" for item in kb.claims),
            "symbols_total": len(kb.symbols),
            "anchors_total": len(kb.anchors),
            "visual_requests_processed": visual_requests_processed,
            "visual_evidence_successful": visual_evidence_successful,
            "corrections_total": sum(len(notes.corrections) for notes in note_sections),
            "unresolved_total": len(kb.unresolved)
            + sum(len(notes.unresolved) for notes in note_sections),
            "llm_usage": LectureModelClient.combine_usage([usage]),
        },
    )
    return ir
