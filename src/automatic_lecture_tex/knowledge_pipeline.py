from __future__ import annotations

import json
import logging
import time
from difflib import SequenceMatcher
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .chunking import chunk_transcript
from .generated_notes import (
    GeneratedChunkNotes,
    GeneratedObservationResolution,
    GeneratedObservationStatePatch,
)
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
STATE_PIPELINE_VERSION = 7

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
    "state_observation_max_images",
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
) -> GeneratedObservationStatePatch | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return GeneratedObservationStatePatch.model_validate(payload["patch"])
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
        formula_crops: list[dict[str, Any]] = []
        board_frames: list[dict[str, Any]] = []
        seen_ocr: set[tuple[Any, str]] = set()
        seen_crops: set[str] = set()
        seen_frames: set[str] = set()

        for visual in payload.get("visual_evidence", []):
            for key in ("raw_latex", "latex"):
                value = _clip_state_raw_text(str(visual.get(key) or ""), 400)
                if value and value not in visual_latex:
                    visual_latex.append(value)

            for crop in visual.get("formula_crops", []):
                crop_id = str(crop.get("id") or "")
                image_path = str(crop.get("image_path") or "")
                if not crop_id or not image_path or crop_id in seen_crops:
                    continue
                seen_crops.add(crop_id)
                formula_crops.append(
                    {
                        "id": crop_id,
                        "timestamp": crop.get("timestamp"),
                        "detector_confidence": crop.get("detector_confidence"),
                        "image_path": image_path,
                    }
                )

            frame_paths = list(visual.get("frame_paths", []))
            frame_timestamps = list(visual.get("frame_timestamps", []))
            for index, image_path in enumerate(frame_paths):
                image_path = str(image_path or "")
                if not image_path or image_path in seen_frames:
                    continue
                seen_frames.add(image_path)
                board_frames.append(
                    {
                        "image_path": image_path,
                        "timestamp": (
                            frame_timestamps[index]
                            if index < len(frame_timestamps)
                            else None
                        ),
                    }
                )

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
                "formula_crops": formula_crops,
                "board_frames": board_frames,
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


def _compact_formula_similarity_text(value: str) -> str:
    return (
        "".join(value.split())
        .replace(r"\parallel", "|")
        .replace(r"\|", "|")
        .replace(r"\left", "")
        .replace(r"\right", "")
    )


def _filter_current_window_ocr_candidates(
    current: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind OCR to CURRENT by formula structure and observation time."""

    anchor = str(current.get("latex") or "").strip()
    if not anchor:
        return list(candidates[:6])

    normalized_anchor = _compact_formula_similarity_text(anchor)
    start = float(current.get("start", 0.0))
    end = float(current.get("end", start))
    center = 0.5 * (start + end)
    ranked: list[tuple[bool, float, float, dict[str, Any]]] = []
    for candidate in candidates:
        text = str(candidate.get("text") or "").strip()
        if not text:
            continue
        score = SequenceMatcher(
            None,
            normalized_anchor,
            _compact_formula_similarity_text(text),
        ).ratio()
        if score < 0.55:
            continue
        timestamp = candidate.get("timestamp")
        if timestamp is None:
            in_interval = False
            distance = float("inf")
        else:
            value = float(timestamp)
            in_interval = start - 1.0 <= value <= end + 1.0
            distance = abs(value - center)
        ranked.append((in_interval, score, distance, candidate))

    if not ranked:
        return []
    has_local = any(item[0] for item in ranked)
    pool = [item for item in ranked if item[0]] if has_local else ranked
    pool.sort(key=lambda item: (-item[1], item[2]))
    return [dict(item[3]) for item in pool[:4]]

def _raw_windows_for_observation_sequence(
    current: dict[str, Any],
    lookahead: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    *,
    max_windows: int,
) -> list[dict[str, Any]]:
    """Return literal sensor evidence only from windows that generated CURRENT.

    Future observations are supplied semantically through fixed-lag look-ahead. Their raw windows
    are intentionally excluded because one 20 s board window can contain several adjacent proof
    steps; exposing all later OCR candidates lets the resolver replace CURRENT with the next formula.
    """

    del lookahead
    current_ids = _observation_window_ids(current)

    selected: list[dict[str, Any]] = []
    for raw in raw_windows:
        window_id = str(raw.get("window_id", ""))
        if window_id not in current_ids:
            continue
        item = dict(raw)
        item["role"] = "current"
        item["math_ocr_candidates"] = _filter_current_window_ocr_candidates(
            current,
            list(raw.get("math_ocr_candidates", [])),
        )
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
            item["math_ocr_candidates"] = _filter_current_window_ocr_candidates(
                current,
                list(raw.get("math_ocr_candidates", [])),
            )
            selected.append(item)

    selected.sort(
        key=lambda item: (
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


def _compact_resolver_observation(item: dict[str, Any]) -> dict[str, Any]:
    """Project one observation to the semantic fields the local resolver can actually use."""

    return {
        "id": str(item.get("id", "")),
        "kind": str(item.get("kind", "")),
        "text": str(item.get("resolved_text") or item.get("text") or "").strip(),
        "latex": (
            str(item.get("resolved_latex") or item.get("latex") or "").strip()
            or None
        ),
    }


def _resolver_symbol_context(
    evidence: dict[str, Any],
    current: dict[str, Any],
    lookahead: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Keep only notation that is lexically relevant to CURRENT/local look-ahead."""

    available = _symbols_for_observation(evidence, current)
    probe = " ".join(
        str(value)
        for item in [current, *lookahead]
        for value in (item.get("text"), item.get("latex"))
        if value
    )
    compact_probe = _compact_formula_similarity_text(probe)

    relevant: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for item in reversed(available):
        compact = {
            "symbol": item.get("symbol"),
            "meaning": item.get("meaning"),
            "type_hint": item.get("type_hint"),
        }
        fallback.append(compact)
        symbol = _compact_formula_similarity_text(str(item.get("symbol") or ""))
        if symbol and symbol in compact_probe:
            relevant.append(compact)
        if len(relevant) >= limit:
            break

    if relevant:
        return list(reversed(relevant[:limit]))
    return list(reversed(fallback[: min(limit, 3)]))


def _resolver_raw_prompt_windows(raw_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove host paths and retain only literal local sensor hypotheses."""

    result: list[dict[str, Any]] = []
    for raw in raw_windows:
        result.append(
            {
                "window_id": raw.get("window_id"),
                "asr": _clip_state_raw_text(str(raw.get("asr") or ""), 800),
                "visual_latex": [
                    _clip_state_raw_text(str(item), 300)
                    for item in raw.get("visual_latex", [])[:2]
                    if str(item).strip()
                ],
                "math_ocr_candidates": [
                    {
                        "timestamp": item.get("timestamp"),
                        "text": _clip_state_raw_text(str(item.get("text") or ""), 300),
                        "source_id": item.get("source_id"),
                    }
                    for item in raw.get("math_ocr_candidates", [])[:4]
                    if str(item.get("text") or "").strip()
                ],
            }
        )
    return result


def _resolver_visual_context(
    current: dict[str, Any],
    raw_windows: list[dict[str, Any]],
    *,
    max_images: int = 2,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Choose temporally bound visual evidence for CURRENT."""

    if max_images <= 0:
        return [], []

    images: list[Path] = []
    metadata: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def append(path_value: Any, ref: str, label: str, timestamp: Any = None) -> None:
        if len(images) >= max_images or not path_value:
            return
        path = Path(str(path_value))
        if path in seen or not path.is_file():
            return
        seen.add(path)
        images.append(path)
        metadata.append(
            {
                "ref": ref,
                "label": label,
                "timestamp": timestamp,
            }
        )

    source_ids = [
        str(item.get("source_id"))
        for raw in raw_windows
        for item in raw.get("math_ocr_candidates", [])
        if item.get("source_id")
    ]
    crops = [
        (str(raw.get("window_id", "")), crop)
        for raw in raw_windows
        for crop in raw.get("formula_crops", [])
    ]
    crops_by_id = {
        str(crop.get("id")): (window_id, crop)
        for window_id, crop in crops
        if crop.get("id")
    }

    for source_id in source_ids:
        matched = crops_by_id.get(source_id)
        if matched is None:
            continue
        window_id, crop = matched
        append(
            crop.get("image_path"),
            f"visual:crop:{source_id}",
            f"formula crop from {window_id}",
            crop.get("timestamp"),
        )
        if images:
            break

    center = 0.5 * (
        float(current.get("start", 0.0)) + float(current.get("end", 0.0))
    )
    if not images and crops:
        window_id, nearest_crop = min(
            crops,
            key=lambda pair: abs(float(pair[1].get("timestamp") or center) - center),
        )
        crop_id = str(nearest_crop.get("id") or "nearest")
        append(
            nearest_crop.get("image_path"),
            f"visual:crop:{crop_id}",
            f"nearest formula crop from {window_id}",
            nearest_crop.get("timestamp"),
        )

    board_frames = [
        (str(raw.get("window_id", "")), frame)
        for raw in raw_windows
        for frame in raw.get("board_frames", [])
        if frame.get("image_path")
    ]
    if len(images) < max_images and board_frames:
        window_id, nearest_frame = min(
            board_frames,
            key=lambda pair: abs(float(pair[1].get("timestamp") or center) - center),
        )
        timestamp = nearest_frame.get("timestamp")
        append(
            nearest_frame.get("image_path"),
            f"visual:board:{window_id}",
            f"local board state from {window_id}",
            timestamp,
        )

    return images, metadata

def _resolver_episode_context(
    evidence: dict[str, Any],
    current: dict[str, Any],
    section: OutlineSection,
) -> dict[str, Any]:
    episode_id = str(current.get("episode_id") or "")
    episode = next(
        (
            item
            for item in evidence.get("episodes", [])
            if str(item.get("id") or "") == episode_id
        ),
        None,
    )
    if episode is not None:
        return {
            "id": episode.get("id"),
            "title": episode.get("title"),
            "kind": episode.get("kind"),
        }
    return {"id": section.id, "title": section.title, "kind": "section"}


def _resolution_formula_supported(
    original: dict[str, Any],
    resolution: GeneratedObservationResolution,
    raw_windows: list[dict[str, Any]],
) -> bool:
    original_latex = str(original.get("latex") or "").strip()
    proposed_latex = str(resolution.latex or "").strip()
    if not original_latex or not proposed_latex:
        return proposed_latex == original_latex

    original_norm = _compact_formula_similarity_text(original_latex)
    proposed_norm = _compact_formula_similarity_text(proposed_latex)
    if proposed_norm == original_norm:
        return True

    candidates: list[str] = []
    for raw in raw_windows:
        candidates.extend(
            str(item.get("text") or "").strip()
            for item in raw.get("math_ocr_candidates", [])
            if str(item.get("text") or "").strip()
        )
        candidates.extend(
            str(item).strip()
            for item in raw.get("visual_latex", [])
            if str(item).strip()
        )

    return any(
        SequenceMatcher(
            None,
            proposed_norm,
            _compact_formula_similarity_text(candidate),
        ).ratio()
        >= 0.88
        for candidate in candidates
    )


def _resolution_text_correction_supported(
    resolution: GeneratedObservationResolution,
) -> bool:
    correction = resolution.correction
    if correction is None:
        return True
    return (
        str(correction.basis) in {"visual", "audio_context", "multimodal"}
        and float(correction.confidence) >= 0.8
    )


def _resolved_observation_from_result(
    original: dict[str, Any],
    resolution: GeneratedObservationResolution,
    raw_windows: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Attach a resolution overlay without mutating the source observation."""

    resolved = dict(original)
    original_text = str(original.get("text") or "").strip()
    original_latex = str(original.get("latex") or "").strip()
    proposed_latex = str(resolution.latex or "").strip()

    formula_supported = _resolution_formula_supported(original, resolution, raw_windows)
    text_supported = _resolution_text_correction_supported(resolution)

    accepted = True
    if original_latex:
        accepted = formula_supported
    elif resolution.correction is not None:
        accepted = text_supported

    if accepted:
        resolved["resolved_text"] = resolution.text.strip()
        resolved["resolved_latex"] = proposed_latex or original_latex or None
    else:
        resolved["resolved_text"] = original_text
        resolved["resolved_latex"] = original_latex or None

    resolved["sequentially_resolved"] = True
    resolved["resolution_accepted"] = accepted
    return resolved, accepted


def _resolve_single_state_observation(
    orchestrator: KnowledgeOrchestrator,
    *,
    section: OutlineSection,
    evidence: dict[str, Any],
    current: dict[str, Any],
    lookahead: list[dict[str, Any]],
    resolved_history: list[dict[str, Any]],
    raw_windows: list[dict[str, Any]],
    config,
) -> GeneratedObservationResolution:
    """Resolve one state transition from a small local state/evidence slice."""

    raw = _raw_windows_for_observation_sequence(
        current,
        lookahead,
        raw_windows,
        max_windows=int(config.state_observation_max_raw_windows),
    )
    raw_prompt = _resolver_raw_prompt_windows(raw)
    images, image_labels = _resolver_visual_context(
        current,
        raw,
        max_images=int(config.state_observation_max_images),
    )

    history_limit = int(config.state_observation_history)
    history = (
        [
            _compact_resolver_observation(item)
            for item in resolved_history[-history_limit:]
        ]
        if history_limit
        else []
    )
    current_compact = _compact_resolver_observation(current)
    lookahead_compact = [
        _compact_resolver_observation(item)
        for item in lookahead
    ]
    symbols = _resolver_symbol_context(
        evidence,
        current,
        lookahead,
    )
    episode = _resolver_episode_context(evidence, current, section)

    prompt = f"""Resolve CURRENT as one local update of an existing mathematical lecture state.

Local episode:
{json.dumps(episode, ensure_ascii=False, separators=(",", ":"))}

Immutable accepted state immediately before CURRENT:
{json.dumps(history, ensure_ascii=False, separators=(",", ":"))}

CURRENT:
{json.dumps(current_compact, ensure_ascii=False, separators=(",", ":"))}

Relevant established notation:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Next observations (disambiguation only; never move their content into CURRENT):
{json.dumps(lookahead_compact, ensure_ascii=False, separators=(",", ":"))}

Direct local ASR/OCR evidence for CURRENT:
{json.dumps(raw_prompt, ensure_ascii=False, separators=(",", ":"))}

Attached visual evidence:
{chr(10).join(image_labels) if image_labels else "No local image available."}

Rules:
- CURRENT is the default. Keep it unchanged unless direct local evidence or immutable state shows a
  concrete semantic error.
- Attached pixels are direct evidence. OCR/ASR are fallible hypotheses about those pixels/speech.
- If CURRENT has latex, return the complete final latex even when unchanged.
- Do not replace CURRENT with the next proof step from look-ahead or another board line.
- Preserve coefficients, denominators, signs, quantifiers, memberships, subscripts and relation signs
  unless direct evidence supports changing them.
- Standard mathematics may reject an impossible reading, but may not invent missing lecture content.
- Pure reformatting is not a correction. If meaning is unchanged, copy CURRENT text/latex.
- A semantic change requires CorrectionRecord; otherwise correction=null.
- If evidence is genuinely ambiguous, keep the supported common content and record unresolved.
- Return CURRENT only. Write prose in {orchestrator.output_language} and formulas in LaTeX.
"""
    schema = (
        GeneratedFormulaObservationResolution
        if str(current.get("latex") or "").strip()
        else GeneratedObservationResolution
    )
    return orchestrator._structured(
        prompt,
        schema,
        images=images or None,
        guided_json=not bool(images),
        operation="state_observation_resolve",
        max_tokens=768,
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
        current_window_ids = _observation_window_ids(original)
        support_raw = [
            item
            for item in raw_windows
            if str(item.get("window_id", "")) in current_window_ids
        ]
        history_limit = int(config.state_observation_history)
        history = (
            [
                _compact_resolver_observation(item)
                for item in resolved_history[-history_limit:]
            ]
            if history_limit
            else []
        )
        compact_lookahead = [
            _compact_resolver_observation(item)
            for item in lookahead
        ]
        compact_symbols = _resolver_symbol_context(
            evidence,
            original,
            lookahead,
        )
        raw_prompt = _resolver_raw_prompt_windows(raw)
        resolver_images, resolver_image_labels = _resolver_visual_context(
            original,
            raw,
            max_images=int(config.state_observation_max_images),
        )
        fingerprint = stable_hash(
            {
                "state_pipeline_version": STATE_PIPELINE_VERSION,
                "episode": _resolver_episode_context(evidence, original, section),
                "current": _compact_resolver_observation(original),
                "symbols": compact_symbols,
                "lookahead": compact_lookahead,
                "history": history,
                "raw_windows": raw_prompt,
                "images": [str(path) for path in resolver_images],
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
                    "current": _compact_resolver_observation(original),
                    "history_before": history,
                    "symbols": compact_symbols,
                    "lookahead": compact_lookahead,
                    "raw_windows": raw_prompt,
                    "images": [
                        {
                            "path": str(path),
                            "label": (
                                resolver_image_labels[index]
                                if index < len(resolver_image_labels)
                                else None
                            ),
                        }
                        for index, path in enumerate(resolver_images)
                    ],
                    "resolution": resolution.model_dump(mode="json"),
                },
            )

        resolved, resolution_accepted = _resolved_observation_from_result(
            original,
            resolution,
            support_raw or raw,
        )
        resolved_batch.append(resolved)
        resolved_history.append(resolved)
        if not resolution_accepted:
            unresolved.append(
                "Rejected unsupported sequential rewrite for "
                f"{observation_id}; preserved the source observation."
            )
        elif resolution.correction is not None:
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


def _math_atom_token(observation_id: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in observation_id)
    return f"MATHATOM__{safe}__"


def _writer_evidence_with_math_atoms(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Mask resolved LaTeX before prose synthesis so the LLM cannot rewrite it."""

    masked = dict(evidence)
    observations: list[dict[str, Any]] = []
    atoms: dict[str, str] = {}
    for item in evidence.get("observations", []):
        observation = dict(item)
        observation_id = str(observation.get("id", ""))
        resolved_text = str(
            observation.get("resolved_text") or observation.get("text") or ""
        ).strip()
        resolved_latex = str(
            observation.get("resolved_latex") or observation.get("latex") or ""
        ).strip()
        observation["text"] = resolved_text
        observation.pop("resolved_text", None)
        observation.pop("resolved_latex", None)
        if resolved_latex and observation_id:
            token = _math_atom_token(observation_id)
            observation["latex"] = token
            atoms[token] = resolved_latex
        observations.append(observation)
    masked["observations"] = observations
    return masked, atoms


def _restore_generated_math_atoms(
    generated: GeneratedChunkNotes,
    atoms: dict[str, str],
) -> None:
    for block in generated.blocks:
        for token, atom_latex in atoms.items():
            if token not in block.latex:
                continue
            replacement = atom_latex if str(block.type) == "equation" else f"${atom_latex}$"
            block.latex = block.latex.replace(token, replacement)

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
    writer_evidence, math_atoms = _writer_evidence_with_math_atoms(evidence)
    prompt = f"""Write one contiguous part of a FINAL lecture-note section from a chronological
state whose observations have already been resolved one-by-one against their local ASR/OCR evidence.

Global lecture outline (read-only narrative context):
{json.dumps(outline_context, ensure_ascii=False, separators=(",", ":"))}

Current fixed section:
{json.dumps(section.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

Sequentially resolved state evidence for this batch:
{json.dumps(writer_evidence, ensure_ascii=False, separators=(",", ":"))}

Previously written blocks from THIS section only:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Interpret and explain the resolved observations as coherent lecture notes; do not merely concatenate
  their wording.
- The mathematical content of a sequentially resolved observation is the primary local hypothesis.
- A non-empty latex field is represented by an opaque token such as
  MATHATOM__obs_window_0010_000__. The token is a CANONICAL MATH ATOM already resolved upstream.
  If you use that mathematical statement, copy the token EXACTLY and bare, without dollar signs,
  LaTeX commands, renaming, expansion, paraphrase, or translation. The host substitutes the exact
  resolved LaTeX after this call.
- Do not silently change a sign, coefficient, denominator, quantifier, membership, subscript, or
  relation merely to make the exposition look more familiar.
- You may still correct a resolved reading if it directly contradicts another supplied resolved fact
  or an elementary consequence of established context. Log every such semantic change in
  corrections. Do not emit no-op corrections.
- Preserve lecturer notation, theorem/proof continuity, order, and level of detail.
- Do not add material from future outline entries or unrelated textbook exposition.
- Avoid repeating a definition/proof step already present in previous_context unless this batch
  genuinely develops it further.
- Every substantive block must cite source_evidence_ids from the supplied resolved observations;
  source_claim_ids may be empty because pre-resolution claims are intentionally removed.
- If a remaining ambiguity cannot be resolved from this state, put it in unresolved instead of
  inventing a specific formula.
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
    _restore_generated_math_atoms(generated, math_atoms)
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
    notes.corrections = [
        item
        for item in notes.corrections
        if item.original.strip() != item.corrected.strip()
    ]
    notes.unresolved = list(dict.fromkeys(notes.unresolved))
    return notes


def _subset_state_evidence_by_episode_ids(
    evidence: dict[str, Any],
    episode_ids: list[str],
) -> dict[str, Any]:
    selected_episode_ids = set(episode_ids)
    episodes = [
        dict(item)
        for item in evidence.get("episodes", [])
        if str(item.get("id", "")) in selected_episode_ids
    ]
    observation_ids: set[str] = set()
    for episode in episodes:
        observation_ids.update(str(item) for item in episode.get("observation_ids", []))

    observations = [
        dict(item)
        for item in evidence.get("observations", [])
        if (
            str(item.get("episode_id", "")) in selected_episode_ids
            or str(item.get("id", "")) in observation_ids
        )
    ]
    selected_observation_ids = {
        str(item.get("id", "")) for item in observations if item.get("id")
    }
    claims = [
        dict(item)
        for item in evidence.get("claims", [])
        if not item.get("evidence_ids")
        or selected_observation_ids.intersection(
            str(value) for value in item.get("evidence_ids", [])
        )
    ]
    cutoff = max(
        (float(item.get("end", item.get("start", 0.0))) for item in observations),
        default=0.0,
    )
    symbols = [
        dict(item)
        for item in evidence.get("symbols", [])
        if (
            str(item.get("episode_id", "")) in selected_episode_ids
            or float(item.get("introduced_at", 0.0)) <= cutoff
        )
    ]

    child = dict(evidence)
    child["episodes"] = episodes
    child["observations"] = observations
    child["claims"] = claims
    child["symbols"] = symbols
    if "sequential_resolution" in child:
        child["sequential_resolution"] = {
            **dict(child["sequential_resolution"]),
            "resolved_observation_ids": [
                item
                for item in child["sequential_resolution"].get(
                    "resolved_observation_ids",
                    [],
                )
                if str(item) in selected_observation_ids
            ],
        }
    return child


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

    try:
        return _write_state_section_batch(
            orchestrator,
            section,
            evidence,
            outline_context=outline_context,
            previous_context=previous_context,
            raw_evidence_context=None,
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
            left_evidence = _subset_state_evidence_by_episode_ids(evidence, left_ids)
            right_evidence = _subset_state_evidence_by_episode_ids(evidence, right_ids)

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
                raw_evidence_context=None,
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
    state_observation_resolution_cache_hits = 0
    state_observations_resolved = 0
    state_resolution_seconds = 0.0
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
            section_observations = _section_observation_sequence(kb, section)
            resolved_history: list[dict[str, Any]] = []
            state_section_batches_total += len(evidence_batches)
            generated_batches: list[ChunkNotes] = []
            for batch_index, evidence_payload in enumerate(evidence_batches):
                previous_context = previous_block_context(generated_batches)
                resolution_started = time.perf_counter()
                (
                    resolved_evidence,
                    resolution_corrections,
                    resolution_unresolved,
                    resolution_cache_hits,
                ) = _resolve_state_batch_sequential(
                    orchestrator,
                    section=section,
                    evidence=evidence_payload,
                    section_observations=section_observations,
                    resolved_history=resolved_history,
                    raw_windows=raw_window_index,
                    work=work,
                    config=pipeline.config.notes,
                    llm_config=pipeline.config.llm.model_dump(mode="json"),
                    force=force,
                )
                state_resolution_seconds += time.perf_counter() - resolution_started
                state_observation_resolution_cache_hits += resolution_cache_hits
                state_observations_resolved += len(resolved_evidence.get("observations", []))
                fingerprint = stable_hash(
                    {
                        "state_pipeline_version": STATE_PIPELINE_VERSION,
                        "section": section.model_dump(mode="json"),
                        "outline_context": outline_context,
                        "resolved_evidence": resolved_evidence,
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
                    resolved_evidence,
                    outline_context=outline_context,
                    previous_context=previous_context,
                    kb=kb,
                    transcript=transcript,
                    config=pipeline.config.notes,
                    raw_window_index=None,
                    raw_evidence_context=None,
                )
                state_synthesis_seconds += time.perf_counter() - started
                notes.corrections.extend(resolution_corrections)
                notes.unresolved = list(
                    dict.fromkeys([*notes.unresolved, *resolution_unresolved])
                )
                atomic_json_dump(
                    path,
                    {
                        "fingerprint": fingerprint,
                        "evidence": evidence_payload,
                        "resolved_evidence": resolved_evidence,
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
            "state_resolution_seconds": round(state_resolution_seconds, 3),
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
            "state_observations_resolved": state_observations_resolved,
            "state_observation_resolution_cache_hits": (
                state_observation_resolution_cache_hits
            ),
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
