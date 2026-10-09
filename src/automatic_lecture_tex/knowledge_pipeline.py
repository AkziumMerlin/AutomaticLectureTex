from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .chunking import chunk_transcript
from .claim_compaction import compact_repaired_claims
from .graph_revision import metrics as graph_revision_metrics
from .graph_revision_pipeline import run_iterative_graph_revision
from .graph_revision_render import graph_state_to_ir
from .graph_surface_writer import write_graph_surface
from .generated_notes import (
    GeneratedObservationStatePatch,
    GeneratedSemanticTextCleanupBatch,
    GeneratedStateSectionNotes,
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
    GeneratedBoardStateWindow,
    KnowledgeOrchestrator,
    board_state_delta_to_observations,
    compact_knowledge_state,
    evidence_for_section,
    make_lecture_state,
    merge_window_observations,
)
from .latex import escape_tex
from .llm import LectureModelClient, StructuredTaskTooLargeError
from .media import copy_asset, extract_api_video_clip
from .schemas import (
    BlockType,
    ChunkNotes,
    ClaimStatus,
    EpisodeHierarchyPlan,
    EpisodeKind,
    EpisodeTrackingUpdate,
    LectureIR,
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    NoteBlock,
    ObservationKind,
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
KNOWLEDGE_CACHE_VERSION = 3
BOARD_STATE_EXTRACTION_VERSION = 1
STATE_PIPELINE_VERSION = 10
STATE_SEMANTIC_TEXT_VERSION = 1
STATE_SEMANTIC_TEXT_RETRY_VERSION = 1
STATE_SEMANTIC_GRAPH_VERSION = 1
STATE_SECTION_WRITER_CACHE_VERSION = 4

# These settings affect only hierarchy/synthesis. Excluding them from the extraction fingerprint is
# intentional: changing downstream batching must not throw away expensive ASR/visual/evidence work.
_DOWNSTREAM_NOTE_FIELDS = {
    "hierarchy_batch_episodes",
    "episode_synthesis_max_evidence_chars",
    "episode_symbol_context_limit",
    "state_section_max_evidence_chars",
    "state_section_assembly",
    "state_graph_revision_max_images",
    "state_graph_revision_max_tokens",
    "state_graph_revision_raw_context_chars",
    "state_graph_revision_catalog_chars",
    "state_graph_revision_frontier_width",
    "state_graph_revision_overlap_observations",
    "state_graph_revision_batch_observations",
    "state_graph_revision_rounds",
    "state_semantic_backend",
    "state_section_raw_context_seconds",
    "state_section_raw_evidence_chars",
    "state_section_writer_thinking",
    "state_section_writer_temperature",
    "state_section_writer_top_p",
    "state_section_writer_top_k",
    "state_section_writer_min_p",
    "state_section_writer_presence_penalty",
    "state_section_writer_repetition_penalty",
    "state_observation_lookahead",
    "state_observation_history",
    "state_observation_max_raw_windows",
    "state_observation_max_images",
    "state_repaired_episode_batch_observations",
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


def _load_board_state_window_artifact(path: Path, fingerprint: str):
    """Cache the visual snapshot itself; host-owned deltas are recomputed sequentially."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        board_state = GeneratedBoardStateWindow.model_validate(payload["board_state"])
        return payload, board_state
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


_SEMANTIC_NARRATION_RE = re.compile(
    r"\b(?:"
    r"лектор|преподавател\w*|на\s+(?:левой|правой|средней\s+)?доске|доск\w*|"
    r"устно|записыва\w*|дописыва\w*|указывает|подч[её]ркива\w*|"
    r"комментиру\w*|поясня\w*|отмечает|говорит|произносит|обводит|"
    r"переходя\s+к|продолжая\s+(?:запись|объяснение)|в\s+рамке"
    r")\b",
    re.IGNORECASE,
)
_SEMANTIC_LATIN_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z](?![A-Za-z0-9])")
_SEMANTIC_NUMBER_RE = re.compile(r"\d+")
_SEMANTIC_EQUALITY_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9_()]*\s*=\s*[−-]?[A-Za-z][A-Za-z0-9_()]*)"
)


def _semantic_text_needs_cleanup(value: str) -> bool:
    return bool(_SEMANTIC_NARRATION_RE.search(value))


def _semantic_text_signature(value: str) -> dict[str, set[str]]:
    normalized = value.replace("−", "-")
    return {
        "symbols": set(_SEMANTIC_LATIN_SYMBOL_RE.findall(normalized)),
        "numbers": set(_SEMANTIC_NUMBER_RE.findall(normalized)),
        "equalities": {
            re.sub(r"\s+", "", item)
            for item in _SEMANTIC_EQUALITY_RE.findall(normalized)
        },
    }


def _semantic_cleanup_safe(
    source: str,
    target: str,
    *,
    has_latex: bool,
) -> tuple[bool, str]:
    target = target.strip()
    if not target:
        return False, "cleanup returned empty text"
    if _semantic_text_needs_cleanup(target):
        return False, "cleanup still contains lecturer/board narration"
    if len(target) > max(len(source) + 40, int(len(source) * 1.15)):
        return False, "cleanup expanded the source instead of removing narration"

    source_sig = _semantic_text_signature(source)
    target_sig = _semantic_text_signature(target)
    new_symbols = target_sig["symbols"] - source_sig["symbols"]
    if new_symbols:
        return False, "cleanup introduced new Latin symbols: " + ", ".join(sorted(new_symbols))
    new_numbers = target_sig["numbers"] - source_sig["numbers"]
    if new_numbers:
        return False, "cleanup introduced new numbers: " + ", ".join(sorted(new_numbers))
    new_equalities = target_sig["equalities"] - source_sig["equalities"]
    if new_equalities:
        return False, "cleanup introduced new formal equalities"

    if not has_latex:
        missing_equalities = source_sig["equalities"] - target_sig["equalities"]
        if missing_equalities:
            return False, "cleanup dropped a formula that exists only in prose"
        missing_symbols = source_sig["symbols"] - target_sig["symbols"]
        if missing_symbols:
            return False, "cleanup dropped standalone mathematical symbols from prose"
        missing_numbers = source_sig["numbers"] - target_sig["numbers"]
        if missing_numbers:
            return False, "cleanup dropped numeric content from prose"
        if len(target) < max(20, int(len(source) * 0.22)):
            return False, "cleanup is too short to preserve a prose-only observation"

    return True, ""


def _load_semantic_cleanup(
    path: Path,
    fingerprint: str,
) -> GeneratedSemanticTextCleanupBatch | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return GeneratedSemanticTextCleanupBatch.model_validate(payload["cleanup"])
    except (json.JSONDecodeError, KeyError, ValidationError):
        return None


def _clean_repaired_semantic_prose(
    orchestrator: KnowledgeOrchestrator,
    *,
    repaired: LectureKnowledgeBase,
    work: Path,
    llm_config: dict[str, Any],
    force: bool,
) -> tuple[dict[str, int], list[str]]:
    """Remove lecture/board narration from repaired prose without touching mathematical LaTeX."""

    by_id = {item.id: item for item in repaired.observations}
    stats = {
        "candidates": 0,
        "model_calls": 0,
        "cache_hits": 0,
        "accepted": 0,
        "rejected": 0,
        "first_pass_rejected": 0,
        "retry_candidates": 0,
        "retry_model_calls": 0,
        "retry_cache_hits": 0,
        "retry_accepted": 0,
        "retry_rejected": 0,
    }
    unresolved: list[str] = []
    retry_candidates: list[dict[str, Any]] = []

    for episode in sorted(repaired.episodes, key=lambda item: (item.start, item.end, item.id)):
        candidates = [
            by_id[observation_id]
            for observation_id in episode.observation_ids
            if observation_id in by_id
            and by_id[observation_id].kind != ObservationKind.TRANSITION
            and _semantic_text_needs_cleanup(by_id[observation_id].text)
        ]
        if not candidates:
            continue

        stats["candidates"] += len(candidates)
        compact = [
            {
                "id": item.id,
                "kind": str(item.kind),
                "semantic_text": item.text,
                "has_separate_latex": bool(item.latex),
            }
            for item in candidates
        ]
        fingerprint = stable_hash(
            {
                "semantic_text_version": STATE_SEMANTIC_TEXT_VERSION,
                "episode_id": episode.id,
                "items": compact,
                "output_language": orchestrator.output_language,
                "llm": llm_config,
            }
        )
        path = work / "state_semantic_text_cleanup" / f"{episode.id}.json"
        cleanup = None if force else _load_semantic_cleanup(path, fingerprint)
        if cleanup is not None:
            stats["cache_hits"] += 1
        else:
            prompt = f"""Rewrite only narration-heavy semantic prose from an already repaired
mathematical lecture state into direct final-note prose.

Episode title:
{episode.title}

Items:
{json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}

Return one item for every input id, with the same observation_id.

Rules:
- remove references to the lecturer, teacher, board, speech/writing/pointing actions, frames, OCR,
  reconstruction, or where something was written;
- state the mathematical/course content directly, in language code {orchestrator.output_language};
- this is NOT mathematical repair: preserve the meaning and terminology already present;
- never invent a theorem name, assumption, symbol, number, example, or mathematical relation;
- preserve literal equalities/symbol names/numbers when they occur only in semantic_text;
- when has_separate_latex=true, do not repeat the displayed formula merely because the source prose
  narrates that it was written; keep only the role, conclusion, or explanation carried by the prose;
- prefer a short declarative sentence or phrase suitable for lecture notes;
- plain prose only: no LaTeX commands or math delimiters.
"""
            try:
                cleanup = orchestrator._structured(
                    prompt,
                    GeneratedSemanticTextCleanupBatch,
                    operation="state_semantic_text_cleanup",
                    max_tokens=4096,
                    split_oversized_task=True,
                    thinking=False,
                    temperature=0.2,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                )
                stats["model_calls"] += 1
            except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
                unresolved.append(
                    f"Semantic prose cleanup failed for {episode.id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            atomic_json_dump(
                path,
                {
                    "fingerprint": fingerprint,
                    "source_items": compact,
                    "cleanup": cleanup.model_dump(mode="json"),
                },
            )

        output_by_id: dict[str, str] = {}
        duplicate_ids: set[str] = set()
        for item in cleanup.items:
            if item.observation_id in output_by_id:
                duplicate_ids.add(item.observation_id)
            output_by_id[item.observation_id] = item.semantic_text

        for source in candidates:
            target = output_by_id.get(source.id)
            if target is None or source.id in duplicate_ids:
                reason = "semantic cleanup did not return exactly one item for the source"
                stats["first_pass_rejected"] += 1
                retry_candidates.append(
                    {
                        "source": source,
                        "first_attempt": target or "",
                        "reason": reason,
                    }
                )
                continue
            safe, reason = _semantic_cleanup_safe(
                source.text,
                target,
                has_latex=bool(source.latex),
            )
            if not safe:
                stats["first_pass_rejected"] += 1
                retry_candidates.append(
                    {
                        "source": source,
                        "first_attempt": target,
                        "reason": reason,
                    }
                )
                continue
            source.text = target.strip()
            stats["accepted"] += 1

    stats["retry_candidates"] = len(retry_candidates)
    if retry_candidates:
        retry_payload = [
            {
                "id": item["source"].id,
                "kind": str(item["source"].kind),
                "source_semantic_text": item["source"].text,
                "has_separate_latex": bool(item["source"].latex),
                "previous_attempt": item["first_attempt"],
                "host_rejection_reason": item["reason"],
            }
            for item in retry_candidates
        ]
        retry_fingerprint = stable_hash(
            {
                "semantic_text_retry_version": STATE_SEMANTIC_TEXT_RETRY_VERSION,
                "items": retry_payload,
                "output_language": orchestrator.output_language,
                "llm": llm_config,
            }
        )
        retry_path = work / "state_semantic_text_cleanup_retry.json"
        retry = None if force else _load_semantic_cleanup(retry_path, retry_fingerprint)
        if retry is not None:
            stats["retry_cache_hits"] = 1
        else:
            retry_prompt = f"""Repair ONLY the rejected semantic-prose cleanup attempts below.

Items:
{json.dumps(retry_payload, ensure_ascii=False, separators=(",", ":"))}

Return exactly one item for every input id, using the same observation_id.

The host rejected the previous attempt for the explicit reason shown in host_rejection_reason.
Fix that reason without weakening any content constraint.

Rules:
- remove lecturer/teacher/board/speech/writing/pointing narration completely;
- write direct final-note prose in language code {orchestrator.output_language};
- preserve the source meaning and terminology;
- do not invent any theorem name, assumption, symbol, number, example, or mathematical relation;
- if the rejection says a symbol/number/equality was dropped, retain it explicitly in plain prose;
- when has_separate_latex=true, the formula itself is already protected separately and need not be
  repeated in prose unless it is necessary to preserve the prose meaning;
- plain prose only: no LaTeX commands or math delimiters.
"""
            try:
                retry = orchestrator._structured(
                    retry_prompt,
                    GeneratedSemanticTextCleanupBatch,
                    operation="state_semantic_text_cleanup_retry",
                    max_tokens=3072,
                    split_oversized_task=True,
                    thinking=False,
                    temperature=0.2,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                )
                stats["retry_model_calls"] = 1
            except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
                retry = None
                unresolved.append(
                    "Semantic prose cleanup retry failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            if retry is not None:
                atomic_json_dump(
                    retry_path,
                    {
                        "fingerprint": retry_fingerprint,
                        "source_items": retry_payload,
                        "cleanup": retry.model_dump(mode="json"),
                    },
                )

        retry_by_id: dict[str, str] = {}
        retry_duplicates: set[str] = set()
        if retry is not None:
            for item in retry.items:
                if item.observation_id in retry_by_id:
                    retry_duplicates.add(item.observation_id)
                retry_by_id[item.observation_id] = item.semantic_text

        for item in retry_candidates:
            source = item["source"]
            target = retry_by_id.get(source.id)
            if retry is None or target is None or source.id in retry_duplicates:
                reason = (
                    "retry did not return exactly one item for the source"
                    if retry is not None
                    else item["reason"]
                )
                stats["rejected"] += 1
                stats["retry_rejected"] += 1
                unresolved.append(f"{source.id}: semantic cleanup rejected: {reason}")
                continue
            safe, reason = _semantic_cleanup_safe(
                source.text,
                target,
                has_latex=bool(source.latex),
            )
            if not safe:
                stats["rejected"] += 1
                stats["retry_rejected"] += 1
                unresolved.append(f"{source.id}: semantic cleanup retry rejected: {reason}")
                continue
            source.text = target.strip()
            stats["accepted"] += 1
            stats["retry_accepted"] += 1

    atomic_json_dump(
        work / "state_semantic_text_cleanup.json",
        {
            "version": STATE_SEMANTIC_TEXT_VERSION,
            "retry_version": STATE_SEMANTIC_TEXT_RETRY_VERSION,
            "stats": stats,
            "unresolved": list(dict.fromkeys(unresolved)),
        },
    )
    return stats, list(dict.fromkeys(unresolved))


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
    paths = sorted(
        [*root.glob("window_*.json"), *root.glob("chunk_*.json")]
    )
    for path in paths:
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
        extraction = payload.get("observations") or {}
        extraction_unresolved = [
            _clip_state_raw_text(str(item), 500)
            for item in extraction.get("unresolved", [])[:6]
            if str(item).strip()
        ]
        windows.append(
            {
                "window_id": str(chunk.get("id") or path.stem),
                "start": start,
                "end": float(chunk.get("end", start)),
                "asr": (
                    ""
                    if str(payload.get("evidence_backend") or "").startswith("native_video")
                    else _clip_state_raw_text(
                        str(chunk.get("timestamped_text") or chunk.get("text") or ""),
                        1200,
                    )
                ),
                "board_state": payload.get("board_state"),
                "board_delta": payload.get("board_delta"),
                "visual_latex": visual_latex[:3],
                "math_ocr_candidates": ocr_candidates,
                "extraction_unresolved": extraction_unresolved,
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
    """Select temporally local OCR; similarity only orders evidence, never hides corrections."""

    anchor = str(current.get("latex") or "").strip()
    normalized_anchor = _compact_formula_similarity_text(anchor)
    start = float(current.get("start", 0.0))
    end = float(current.get("end", start))
    center = 0.5 * (start + end)

    ranked: list[tuple[bool, float, float, dict[str, Any]]] = []
    for candidate in candidates:
        text = str(candidate.get("text") or "").strip()
        if not text:
            continue
        score = (
            SequenceMatcher(
                None,
                normalized_anchor,
                _compact_formula_similarity_text(text),
            ).ratio()
            if normalized_anchor
            else 0.0
        )
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

    # If the sensor provides observations from CURRENT's own time span, exclude other board steps.
    # Otherwise fall back to the nearest candidates from the same technical window. Formula
    # similarity is only a ranking hint: a badly reconstructed CURRENT must still be repairable.
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
        "text": str(
            item.get("semantic_text")
            or item.get("resolved_text")
            or item.get("text")
            or ""
        ).strip(),
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


def _resolver_evidence_catalog(
    raw_prompt: list[dict[str, Any]],
    visual_metadata: list[dict[str, Any]],
    history: list[dict[str, Any]],
    lookahead: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Enumerate exactly the provenance ids a state patch is allowed to cite."""

    catalog: list[dict[str, Any]] = []
    allowed: set[str] = set()
    direct: set[str] = set()

    def add(ref: str, kind: str, summary: str, *, is_direct: bool) -> None:
        if not ref or ref in allowed:
            return
        allowed.add(ref)
        if is_direct:
            direct.add(ref)
        catalog.append({"ref": ref, "kind": kind, "summary": summary})

    for raw in raw_prompt:
        window_id = str(raw.get("window_id") or "")
        asr = str(raw.get("asr") or "").strip()
        if asr and window_id:
            add(f"asr:{window_id}", "asr", asr, is_direct=True)
        for index, candidate in enumerate(raw.get("math_ocr_candidates", [])):
            source_id = str(candidate.get("source_id") or "")
            ref = f"ocr:{source_id}" if source_id else f"ocr:{window_id}:{index}"
            add(ref, "ocr", str(candidate.get("text") or ""), is_direct=True)
        for index, value in enumerate(raw.get("visual_latex", [])):
            add(
                f"visual_latex:{window_id}:{index}",
                "visual_latex",
                str(value),
                is_direct=True,
            )

    for item in visual_metadata:
        add(
            str(item.get("ref") or ""),
            "visual",
            str(item.get("label") or ""),
            is_direct=True,
        )

    for item in history:
        observation_id = str(item.get("id") or "")
        if observation_id:
            add(
                f"history:{observation_id}",
                "accepted_state",
                json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                is_direct=False,
            )
    for item in lookahead:
        observation_id = str(item.get("id") or "")
        if observation_id:
            add(
                f"lookahead:{observation_id}",
                "lookahead",
                json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                is_direct=False,
            )

    return catalog, allowed, direct


def _apply_observation_state_patch(
    original: dict[str, Any],
    patch: GeneratedObservationStatePatch,
    *,
    allowed_evidence_refs: set[str],
    direct_evidence_refs: set[str],
) -> tuple[dict[str, Any], bool, str | None]:
    """Apply one model patch while enforcing scope/provenance, not mathematical similarity."""

    resolved = dict(original)
    original_text = str(original.get("text") or "").strip()
    original_latex = str(original.get("latex") or "").strip() or None
    semantic_text = str(patch.semantic_text or "").strip()
    resolved["sequentially_resolved"] = True
    resolved["state_patch_action"] = patch.action

    unknown = [ref for ref in patch.evidence_refs if ref not in allowed_evidence_refs]

    def mark_unresolved(issue: str) -> tuple[dict[str, Any], bool, str]:
        resolved["resolution_status"] = "unresolved"
        resolved["resolution_accepted"] = False
        resolved["semantic_text"] = None
        resolved["resolved_text"] = None
        resolved["resolved_latex"] = None
        return resolved, False, issue

    if patch.action == "keep":
        resolved["resolution_status"] = "kept"
        resolved["resolution_accepted"] = True
        resolved["semantic_text"] = semantic_text
        # resolved_text is retained as a compatibility alias for cached/debug tooling.
        resolved["resolved_text"] = semantic_text
        resolved["resolved_latex"] = original_latex
        return resolved, True, None

    if unknown:
        return mark_unresolved(
            "State patch cited unknown evidence refs: " + ", ".join(unknown)
        )
    if not direct_evidence_refs.intersection(patch.evidence_refs):
        return mark_unresolved(
            "State patch attempted to change CURRENT without direct local evidence."
        )

    if patch.action == "reject":
        resolved["resolution_status"] = "rejected"
        resolved["resolution_accepted"] = True
        resolved["semantic_text"] = None
        resolved["resolved_text"] = None
        resolved["resolved_latex"] = None
        return resolved, True, None

    replacement_latex = (
        str(patch.replacement_latex).strip()
        if patch.replacement_latex is not None
        else original_latex
    )
    if original_latex is not None and not replacement_latex:
        return mark_unresolved(
            "State patch removed an existing formula instead of replacing or rejecting CURRENT."
        )

    resolved["resolution_status"] = "replaced"
    resolved["resolution_accepted"] = True
    resolved["semantic_text"] = semantic_text
    resolved["resolved_text"] = semantic_text
    resolved["resolved_latex"] = replacement_latex
    return resolved, True, None


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
) -> GeneratedObservationStatePatch:
    """Propose one bounded transaction against CURRENT; accepted history is immutable."""

    raw = _raw_windows_for_observation_sequence(
        current,
        lookahead,
        raw_windows,
        max_windows=int(config.state_observation_max_raw_windows),
    )
    raw_prompt = _resolver_raw_prompt_windows(raw)
    images, visual_metadata = _resolver_visual_context(
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
    symbols = _resolver_symbol_context(evidence, current, lookahead)
    episode = _resolver_episode_context(evidence, current, section)
    evidence_catalog, _, _ = _resolver_evidence_catalog(
        raw_prompt,
        visual_metadata,
        history,
        lookahead_compact,
    )
    image_index = "\n".join(
        f"Image {index}: ref={item['ref']}; {item['label']}; timestamp={item.get('timestamp')}"
        for index, item in enumerate(visual_metadata)
    )

    prompt = f"""Repair exactly one pending event in a chronological mathematical lecture state.
Accepted history is immutable. Return a TRANSACTION for CURRENT and a clean semantic_text for every
accepted event.

Local episode:
{json.dumps(episode, ensure_ascii=False, separators=(",", ":"))}

Accepted state immediately before CURRENT:
{json.dumps(history, ensure_ascii=False, separators=(",", ":"))}

CURRENT pending event:
{json.dumps(current_compact, ensure_ascii=False, separators=(",", ":"))}

Relevant established notation:
{json.dumps(symbols, ensure_ascii=False, separators=(",", ":"))}

Fixed-lag look-ahead (context only):
{json.dumps(lookahead_compact, ensure_ascii=False, separators=(",", ":"))}

Direct local sensor hypotheses:
{json.dumps(raw_prompt, ensure_ascii=False, separators=(",", ":"))}

Allowed evidence refs:
{json.dumps(evidence_catalog, ensure_ascii=False, separators=(",", ":"))}

Attached images:
{image_index or "No local image available."}

Choose exactly one action:
- keep: CURRENT is mathematically/semantically faithful. Return semantic_text as the clean lecture
  statement that should appear in final notes. Do not return replacement_latex.
- replace: CURRENT has a concrete semantic error and direct local evidence supports a corrected
  canonical event. Return clean semantic_text and, if CURRENT has latex, the COMPLETE corrected
  replacement_latex. Cite evidence_refs from the allowed catalog.
- reject: CURRENT itself is unsupported/contradictory and no defensible replacement is locally
  evidenced. Cite evidence_refs. The host keeps source/audit artifacts but suppresses this event.

semantic_text contract:
- It contains ONLY lecture content suitable for final notes, not an explanation of reconstruction
  and not a chronological description of the lecturer's actions.
- Never write "лектор/преподаватель говорит, пишет, записывает, указывает, подчёркивает, поясняет",
  "на доске записано", "устно", or analogous narration. State the content directly instead.
- Examples: "Лектор записывает определение X" -> "Определяется X"; "На доске получено A=B" ->
  state the conclusion directly (or omit it from semantic_text when latex already contains it).
- Never mention ASR, OCR, frames, board visibility, crops, confidence, evidence refs, or phrases such
  as "recovered from the board", "audio is degenerate", "the frame shows", etc.
- Put all such diagnostic reasoning in reason/unresolved, never semantic_text.
- Keep it concise: normally one sentence or a short mathematical statement.
- Do not duplicate the full displayed formula in semantic_text when latex already carries it.
- Do not emit LaTeX delimiters or commands in semantic_text; mathematical formulas belong in latex.
- Cleaning provenance language out of CURRENT does not count as a semantic correction.

Constraints:
- replace/reject MUST cite at least one direct CURRENT sensor ref: asr:, ocr:, visual_latex:, or
  visual:. history:/lookahead: may support disambiguation but cannot alone authorize a state change.
- Do not alter accepted history and do not import a later proof step into CURRENT.
- Pixels are direct evidence; OCR/ASR are fallible hypotheses about them.
- Do not change notation merely for style or textbook convention. Pure reformatting => keep.
- Preserve coefficients, denominators, signs, quantifiers, memberships, subscripts and relation
  signs unless local evidence supports changing them.
- Standard mathematics is only a consistency prior. It cannot by itself authorize replace.
- If CURRENT is wrong but the correct statement is not locally recoverable, reject rather than guess.
- Do not emit confidence scores. Write prose in {orchestrator.output_language} and formulas in LaTeX.
"""
    return orchestrator._structured(
        prompt,
        GeneratedObservationStatePatch,
        images=images or None,
        guided_json=not bool(images),
        operation="state_observation_resolve",
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
    """Resolve pending observations into canonical state via bounded transactions."""

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
        resolver_images, resolver_visual_metadata = _resolver_visual_context(
            original,
            raw,
            max_images=int(config.state_observation_max_images),
        )
        evidence_catalog, allowed_refs, direct_refs = _resolver_evidence_catalog(
            raw_prompt,
            resolver_visual_metadata,
            history,
            compact_lookahead,
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
                "visual_metadata": resolver_visual_metadata,
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
        patch = None if force else _load_observation_resolution(path, fingerprint)
        if patch is not None:
            cache_hits += 1
        else:
            try:
                patch = _resolve_single_state_observation(
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
                patch = GeneratedObservationStatePatch(
                    action="keep",
                    semantic_text=str(original.get("text") or "").strip() or "Неразрешённое утверждение.",
                    unresolved=[
                        "State repair failed for "
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
                    "evidence_catalog": evidence_catalog,
                    "images": [
                        {
                            "path": str(image_path),
                            **(
                                resolver_visual_metadata[index]
                                if index < len(resolver_visual_metadata)
                                else {}
                            ),
                        }
                        for index, image_path in enumerate(resolver_images)
                    ],
                    "patch": patch.model_dump(mode="json"),
                },
            )

        resolved, _patch_accepted, issue = _apply_observation_state_patch(
            original,
            patch,
            allowed_evidence_refs=allowed_refs,
            direct_evidence_refs=direct_refs,
        )
        resolved_batch.append(resolved)

        if resolved.get("resolution_status") in {"kept", "replaced"}:
            resolved_history.append(resolved)
        if issue:
            unresolved.append(f"{observation_id}: {issue}")
        unresolved.extend(patch.unresolved)

    canonical = [
        item
        for item in resolved_batch
        if item.get("resolution_status") in {"kept", "replaced"}
    ]
    payload = dict(evidence)
    payload["observations"] = canonical
    payload["claims"] = []

    status_ids = {
        status: [
            str(item.get("id", ""))
            for item in resolved_batch
            if item.get("resolution_status") == status
        ]
        for status in ("kept", "replaced", "rejected", "unresolved")
    }
    payload["sequential_resolution"] = {
        "resolved_observation_ids": [
            str(item.get("id", "")) for item in canonical
        ],
        "batch_observation_ids": sorted(batch_ids),
        "kept_observation_ids": status_ids["kept"],
        "replaced_observation_ids": status_ids["replaced"],
        "rejected_observation_ids": status_ids["rejected"],
        "unresolved_observation_ids": status_ids["unresolved"],
    }
    return payload, [], list(dict.fromkeys(unresolved)), cache_hits


def _canonical_observation_from_repaired(item: dict[str, Any]) -> LectureObservation:
    payload = {
        name: item.get(name)
        for name in LectureObservation.model_fields
        if name in item
    }
    payload["text"] = str(
        item.get("semantic_text")
        or item.get("resolved_text")
        or item.get("text")
        or ""
    ).strip()
    payload["latex"] = (
        item.get("resolved_latex")
        if "resolved_latex" in item
        else item.get("latex")
    )
    return LectureObservation.model_validate(payload)


def _repair_lecture_state(
    orchestrator: KnowledgeOrchestrator,
    *,
    kb: LectureKnowledgeBase,
    raw_windows: list[dict[str, Any]],
    work: Path,
    config,
    llm_config: dict[str, Any],
    force: bool,
) -> tuple[LectureKnowledgeBase, dict[str, int], list[str]]:
    """Resolve every pending observation once, then persist a canonical repaired state."""

    repaired = kb.model_copy(deep=True)
    all_observations = [
        item.model_dump(mode="json")
        for item in sorted(
            kb.observations,
            key=lambda item: (item.start, item.end, item.id),
        )
    ]
    by_id = {str(item.get("id", "")): item for item in all_observations}
    resolved_history: list[dict[str, Any]] = []
    canonical_by_id: dict[str, LectureObservation] = {}
    unresolved: list[str] = []
    stats = {
        "processed": 0,
        "cache_hits": 0,
        "kept": 0,
        "replaced": 0,
        "rejected": 0,
        "unresolved": 0,
    }

    for episode in sorted(kb.episodes, key=lambda item: (item.start, item.end, item.id)):
        episode_observations = [
            by_id[observation_id]
            for observation_id in episode.observation_ids
            if observation_id in by_id
        ]
        if not episode_observations:
            continue

        section = OutlineSection(
            id=f"repair_{episode.id}",
            title=episode.title,
            start=episode.start,
            end=episode.end,
            episode_ids=[episode.id],
        )
        evidence = {
            "section": section.model_dump(mode="json"),
            "episodes": [episode.model_dump(mode="json")],
            "observations": episode_observations,
            "claims": [],
            "symbols": [
                item.model_dump(mode="json")
                for item in kb.symbols
                if item.active and item.introduced_at <= episode.end
            ],
        }
        (
            repaired_evidence,
            _corrections,
            batch_unresolved,
            cache_hits,
        ) = _resolve_state_batch_sequential(
            orchestrator,
            section=section,
            evidence=evidence,
            section_observations=all_observations,
            resolved_history=resolved_history,
            raw_windows=raw_windows,
            work=work,
            config=config,
            llm_config=llm_config,
            force=force,
        )
        summary = repaired_evidence.get("sequential_resolution", {})
        stats["processed"] += len(summary.get("batch_observation_ids", []))
        stats["cache_hits"] += cache_hits
        stats["kept"] += len(summary.get("kept_observation_ids", []))
        stats["replaced"] += len(summary.get("replaced_observation_ids", []))
        stats["rejected"] += len(summary.get("rejected_observation_ids", []))
        stats["unresolved"] += len(summary.get("unresolved_observation_ids", []))
        unresolved.extend(batch_unresolved)

        for item in repaired_evidence.get("observations", []):
            canonical = _canonical_observation_from_repaired(item)
            canonical_by_id[canonical.id] = canonical

    # Episode tracking already established structural boundaries. Repair changes semantic event
    # content/acceptance, not past boundary decisions; empty episodes naturally disappear downstream.
    repaired.observations = sorted(
        canonical_by_id.values(),
        key=lambda item: (item.start, item.end, item.id),
    )
    canonical_ids = {item.id for item in repaired.observations}
    for episode in repaired.episodes:
        episode.observation_ids = [
            observation_id
            for observation_id in episode.observation_ids
            if observation_id in canonical_ids
        ]

    for symbol in repaired.symbols:
        if not symbol.evidence_ids:
            continue
        symbol.evidence_ids = [
            evidence_id
            for evidence_id in symbol.evidence_ids
            if evidence_id in canonical_ids
        ]
        if not symbol.evidence_ids:
            symbol.active = False

    semantic_stats, semantic_unresolved = _clean_repaired_semantic_prose(
        orchestrator,
        repaired=repaired,
        work=work,
        llm_config=llm_config,
        force=force,
    )
    stats.update(
        {
            "semantic_cleanup_candidates": semantic_stats["candidates"],
            "semantic_cleanup_calls": semantic_stats["model_calls"],
            "semantic_cleanup_cache_hits": semantic_stats["cache_hits"],
            "semantic_cleanup_accepted": semantic_stats["accepted"],
            "semantic_cleanup_rejected": semantic_stats["rejected"],
            "semantic_cleanup_first_pass_rejected": semantic_stats["first_pass_rejected"],
            "semantic_cleanup_retry_candidates": semantic_stats["retry_candidates"],
            "semantic_cleanup_retry_calls": semantic_stats["retry_model_calls"],
            "semantic_cleanup_retry_cache_hits": semantic_stats["retry_cache_hits"],
            "semantic_cleanup_retry_accepted": semantic_stats["retry_accepted"],
            "semantic_cleanup_retry_rejected": semantic_stats["retry_rejected"],
        }
    )
    unresolved.extend(semantic_unresolved)

    repaired.unresolved = list(dict.fromkeys([*repaired.unresolved, *unresolved]))
    return repaired, stats, list(dict.fromkeys(unresolved))

def _rebind_symbols_to_repaired_graph(
    rebuilt: LectureKnowledgeBase,
    source_symbols,
) -> None:
    """Preserve symbol evidence while deriving scope from the rebuilt repaired episode graph."""

    by_observation = {item.id: item for item in rebuilt.observations}
    by_episode = {item.id: item for item in rebuilt.episodes}
    rebuilt.symbols = []
    for raw in source_symbols:
        if not raw.active:
            continue
        symbol = raw.model_copy(deep=True)
        symbol.evidence_ids = [
            evidence_id
            for evidence_id in symbol.evidence_ids
            if evidence_id in by_observation
        ]
        if not symbol.evidence_ids:
            continue
        episode_ids = [
            by_observation[evidence_id].episode_id
            for evidence_id in symbol.evidence_ids
            if by_observation[evidence_id].episode_id
        ]
        symbol.episode_id = episode_ids[0] if episode_ids else ""
        symbol.scope = symbol.episode_id or "lecture"
        rebuilt.symbols.append(symbol)
        episode = by_episode.get(symbol.episode_id)
        if episode is not None and symbol.id and symbol.id not in episode.symbol_ids:
            episode.symbol_ids.append(symbol.id)


def _rebuild_repaired_semantic_graph(
    orchestrator: KnowledgeOrchestrator,
    *,
    repaired: LectureKnowledgeBase,
    work: Path,
    llm_config: dict[str, Any],
    batch_observations: int,
    force: bool,
) -> tuple[LectureKnowledgeBase, dict[str, Any]]:
    """Re-derive claims and semantic episodes from repaired observations.

    The pre-repair graph is evidence-tracking state. Once repair has replaced/rejected observations,
    its derived claims and episode labels are stale. Rebuild them from the repaired sequence before
    hierarchy or note realization.
    """

    ordered = sorted(repaired.observations, key=lambda item: (item.start, item.end, item.id))
    fingerprint = stable_hash(
        {
            "semantic_graph_version": STATE_SEMANTIC_GRAPH_VERSION,
            "observations": [item.model_dump(mode="json") for item in ordered],
            "symbols": [
                item.model_dump(mode="json")
                for item in repaired.symbols
                if item.active
            ],
            "batch_observations": batch_observations,
            "llm": llm_config,
        }
    )
    path = work / "repaired_semantic_graph.json"
    if path.exists() and not force:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint:
                cached = LectureKnowledgeBase.model_validate(payload["kb"])
                return cached, {
                    "cache_hits": 1,
                    "model_calls": 0,
                    "observations": len(cached.observations),
                    "claims": len(cached.claims),
                    "episodes": len(cached.episodes),
                }
        except (json.JSONDecodeError, KeyError, ValidationError):
            pass

    rebuilt_observations = []
    for item in ordered:
        observation = item.model_copy(deep=True)
        observation.episode_id = ""
        rebuilt_observations.append(observation)

    rebuilt = LectureKnowledgeBase(
        lecture_id=repaired.lecture_id,
        title=repaired.title,
        observations=rebuilt_observations,
        observation_aliases=dict(repaired.observation_aliases),
        claims=[],
        symbols=[],
        episodes=[],
        anchors=[],
        unresolved=list(repaired.unresolved),
    )

    model_calls = 0
    for batch_index, start in enumerate(range(0, len(rebuilt_observations), batch_observations)):
        selected = rebuilt_observations[start : start + batch_observations]
        if not selected:
            continue
        batch = WindowObservations(
            window_id=f"repaired_semantic_{batch_index:04d}",
            start=selected[0].start,
            end=selected[-1].end,
            observations=[item.model_copy(deep=True) for item in selected],
        )
        ids = [item.id for item in selected]
        update = orchestrator.track_repaired_episodes(rebuilt, batch, ids)
        model_calls += 1
        apply_episode_tracking(
            rebuilt,
            update,
            ids,
            window_id=batch.window_id,
        )

    close_open_episodes(rebuilt)
    _rebind_symbols_to_repaired_graph(rebuilt, repaired.symbols)

    stats: dict[str, Any] = {
        "cache_hits": 0,
        "model_calls": model_calls,
        "observations": len(rebuilt.observations),
        "claims": len(rebuilt.claims),
        "episodes": len(rebuilt.episodes),
        "episode_kinds": {},
    }
    for episode in rebuilt.episodes:
        key = str(episode.kind)
        stats["episode_kinds"][key] = int(stats["episode_kinds"].get(key, 0)) + 1

    atomic_json_dump(
        path,
        {
            "fingerprint": fingerprint,
            "version": STATE_SEMANTIC_GRAPH_VERSION,
            "stats": stats,
            "kb": rebuilt.model_dump(mode="json"),
        },
    )
    return rebuilt, stats


def _state_section_payload(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    transcript: Transcript,
    config,
) -> dict[str, Any]:
    payload = evidence_for_section(kb, section, transcript, config)
    payload.pop("transcript", None)
    # Claims and episodes have been re-derived from the repaired observations, so they are once
    # again valid canonical semantic state rather than stale pre-repair metadata.
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


def _state_section_observations(
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


def _episode_block_type(kind: EpisodeKind) -> BlockType:
    return {
        EpisodeKind.DEFINITION: BlockType.DEFINITION,
        EpisodeKind.THEOREM: BlockType.THEOREM,
        EpisodeKind.PROOF: BlockType.PROOF,
        EpisodeKind.EXAMPLE: BlockType.EXAMPLE,
        EpisodeKind.REMARK: BlockType.REMARK,
    }.get(kind, BlockType.PARAGRAPH)


def _episode_body_from_repaired_state(
    kb: LectureKnowledgeBase,
    episode,
) -> tuple[str, list[str]]:
    """Serialize one semantic leaf from ACTIVE repaired claims.

    Repaired observations remain immutable provenance. Corrections/retractions are interpreted when
    the repaired semantic graph derives KnowledgeClaim status, so rendering observations directly
    would resurrect superseded content.
    """

    claim_by_id = {item.id: item for item in kb.claims}
    observation_by_id = {item.id: item for item in kb.observations}
    claims = [
        claim_by_id[claim_id]
        for claim_id in episode.claim_ids
        if claim_id in claim_by_id and claim_by_id[claim_id].status == ClaimStatus.ACTIVE
    ]
    # episode.claim_ids is the canonical semantic order. Before compaction it is chronological;
    # after compaction it is the explicit logical order returned for the fixed episode.
    pieces: list[str] = []
    unresolved: list[str] = []
    seen_text: set[str] = set()
    seen_latex: set[str] = set()

    for claim in claims:
        text = claim.content.strip()
        latex = (claim.latex or "").strip()
        if text and text != latex:
            normalized_text = re.sub(r"\s+", " ", text).strip().casefold()
            if normalized_text and normalized_text not in seen_text:
                pieces.append(escape_tex(text))
                seen_text.add(normalized_text)
        if latex:
            normalized_latex = re.sub(r"\s+", "", latex)
            if normalized_latex and normalized_latex not in seen_latex:
                pieces.append("\\[\n" + latex + "\n\\]")
                seen_latex.add(normalized_latex)

    for observation_id in episode.observation_ids:
        observation = observation_by_id.get(observation_id)
        if (
            observation is not None
            and observation.kind == ObservationKind.UNRESOLVED
            and observation.text.strip()
        ):
            unresolved.append(observation.text.strip())

    return "\n\n".join(pieces), unresolved


def _assemble_state_section_deterministically(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
) -> ChunkNotes:
    """Realize repaired semantic episodes directly; observations are evidence, not document blocks."""

    blocks: list[NoteBlock] = []
    unresolved: list[str] = []
    by_episode = {item.id: item for item in kb.episodes}
    active_claims = {
        item.id
        for item in kb.claims
        if item.status == ClaimStatus.ACTIVE
    }

    subsection_before: dict[str, str] = {}
    if len(section.subsections) > 1:
        for subsection in section.subsections:
            if subsection.episode_ids:
                subsection_before[subsection.episode_ids[0]] = subsection.title

    for episode_id in section.episode_ids:
        episode = by_episode.get(episode_id)
        if episode is None or not episode.observation_ids:
            continue

        subsection_title = subsection_before.get(episode_id)
        if subsection_title:
            blocks.append(
                NoteBlock(
                    type=BlockType.SUBSECTION,
                    latex=subsection_title,
                    source_claim_ids=[
                        claim_id
                        for claim_id in episode.claim_ids
                        if claim_id in active_claims
                    ],
                    source_evidence_ids=list(episode.observation_ids),
                )
            )

        body, episode_unresolved = _episode_body_from_repaired_state(kb, episode)
        unresolved.extend(episode_unresolved)
        if not body.strip():
            continue

        blocks.append(
            NoteBlock(
                type=_episode_block_type(episode.kind),
                latex=body,
                source_claim_ids=[
                    claim_id
                    for claim_id in episode.claim_ids
                    if claim_id in active_claims
                ],
                source_evidence_ids=list(episode.observation_ids),
            )
        )

    return ChunkNotes(
        chunk_id=section.id,
        start=section.start,
        end=section.end,
        section_title=section.title,
        blocks=blocks,
        unresolved=list(dict.fromkeys(unresolved)),
    )


def _hierarchy_fingerprint(
    kb: LectureKnowledgeBase,
    *,
    llm_config: dict[str, Any],
    hierarchy_batch_episodes: int,
) -> str:
    """Hierarchy depends on canonical KB + hierarchy LLM policy, not writer configuration."""

    return stable_hash(
        {
            "kb": kb.model_dump(mode="json"),
            "llm": llm_config,
            "hierarchy_batch_episodes": hierarchy_batch_episodes,
            "hierarchy_cache_version": HIERARCHY_CACHE_VERSION,
        }
    )


def _writer_canonical_observations(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Project repaired state to the only fields final prose synthesis is allowed to use."""

    observations: list[dict[str, Any]] = []
    for item in evidence.get("observations", []):
        text = str(
            item.get("semantic_text")
            or item.get("resolved_text")
            or item.get("text")
            or ""
        ).strip()
        latex = (
            item.get("resolved_latex")
            if "resolved_latex" in item
            else item.get("latex")
        )
        projected = {
            "id": str(item.get("id", "")),
            "kind": str(item.get("kind", "claim")),
            "text": text,
        }
        if latex:
            projected["latex"] = str(latex).strip()
        observations.append(projected)
    return observations


def _writer_previous_tail(previous_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only a short continuity hint from the preceding generated prose."""

    tail: list[dict[str, Any]] = []
    for item in previous_context[-2:]:
        compact = {
            "type": str(item.get("type", "paragraph")),
            "latex_tail": str(item.get("latex_tail", ""))[-800:],
        }
        title = item.get("title")
        if title:
            compact["title"] = str(title)
        tail.append(compact)
    return tail


def _write_state_section_batch(
    orchestrator: KnowledgeOrchestrator,
    section: OutlineSection,
    evidence: dict[str, Any],
    *,
    previous_context: list[dict[str, Any]],
    guided_json: bool = True,
) -> ChunkNotes:
    observations = _writer_canonical_observations(evidence)
    previous_tail = _writer_previous_tail(previous_context)
    prompt = f"""Convert already-canonical lecture state into coherent final lecture notes.

Section title:
{section.title}

Canonical observations, in lecture order:
{json.dumps(observations, ensure_ascii=False, separators=(",", ":"))}

Short tail of the preceding generated prose, for continuity only:
{json.dumps(previous_tail, ensure_ascii=False, separators=(",", ":"))}

Requirements:
- The observations are FINAL. Do not correct, reinterpret, or add mathematical content.
- Preserve their order. Merge adjacent observations into readable exposition when useful.
- If an observation has a non-empty `latex` field and you restate that formula, copy its LaTeX
  exactly. Do not rename symbols, translate commands, simplify, expand, or repair it.
- Use `text` only to write the surrounding explanation. Do not include provenance or reconstruction
  commentary about ASR, OCR, frames, the board, confidence, or how the observation was recovered.
- Do not repeat material already present in the preceding-prose tail.
- Return block bodies only. The deterministic renderer owns section/theorem/proof wrappers.
- Write prose in language code {orchestrator.output_language}.
"""
    generated = orchestrator._structured(
        prompt,
        GeneratedStateSectionNotes,
        operation="state_section_write",
        guided_json=guided_json,
        split_oversized_task=True,
        temperature=float(orchestrator.config.state_section_writer_temperature),
        thinking=bool(orchestrator.config.state_section_writer_thinking),
        top_p=float(orchestrator.config.state_section_writer_top_p),
        top_k=int(orchestrator.config.state_section_writer_top_k),
        min_p=float(orchestrator.config.state_section_writer_min_p),
        presence_penalty=float(orchestrator.config.state_section_writer_presence_penalty),
        repetition_penalty=float(orchestrator.config.state_section_writer_repetition_penalty),
    )

    source_evidence_ids = [
        item["id"]
        for item in observations
        if item.get("id")
    ]
    blocks = [
        block.to_note_block(source_evidence_ids=source_evidence_ids)
        for block in generated.blocks
    ]
    return ChunkNotes(
        chunk_id=section.id,
        start=section.start,
        end=section.end,
        section_title=section.title.replace("$", ""),
        blocks=blocks,
    )



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
    previous_context: list[dict[str, Any]],
) -> ChunkNotes:
    """Recursively split final-writer work that cannot fit in one structured request."""

    try:
        return _write_state_section_batch(
            orchestrator,
            section,
            evidence,
            previous_context=previous_context,
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
                previous_context=previous_context,
            )
            right_previous = [
                *previous_context,
                *previous_block_context([left_notes]),
            ][-2:]
            right_notes = _write_state_section_batch_resilient(
                orchestrator,
                right_section,
                right_evidence,
                previous_context=right_previous,
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
                previous_context=previous_context,
            )
            right_previous = [
                *previous_context,
                *previous_block_context([left_notes]),
            ][-2:]
            right_notes = _write_state_section_batch_resilient(
                orchestrator,
                section,
                right_evidence,
                previous_context=right_previous,
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
                previous_context=previous_context,
                guided_json=False,
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
    board_state_mode = (
        pipeline.config.notes.window_evidence_backend == "native_video_board_state"
    )
    chunks = chunk_transcript(
        transcript,
        (
            pipeline.config.notes.native_video_board_chunk_seconds
            if board_state_mode
            else pipeline.config.notes.chunk_target_seconds
        ),
        0.0 if board_state_mode else pipeline.config.notes.chunk_overlap_seconds,
    )
    figures_root = pipeline.config.latex.output_dir / "figures" / lecture.id

    cache_hits = 0
    processed_windows = 0
    visual_requests_processed = 0
    visual_evidence_successful = 0
    vision_seconds = 0.0
    extract_seconds = 0.0
    episode_track_seconds = 0.0
    native_video_mode = pipeline.config.notes.window_evidence_backend in {
        "native_video",
        "native_video_board_state",
    }
    previous_board_state: GeneratedBoardStateWindow | None = None
    native_source_video: Path | None = None
    native_video_prepare_seconds = 0.0
    native_video_clip_seconds = 0.0
    native_video_windows_processed = 0
    native_video_provider_retries = 0

    for chunk in chunks:
        state_before = (
            {}
            if board_state_mode
            else compact_knowledge_state(kb, pipeline.config.notes)
        )
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
                "board_state_extraction_version": (
                    BOARD_STATE_EXTRACTION_VERSION if board_state_mode else None
                ),
            }
        )
        artifact = work / "knowledge_windows" / f"{chunk.id}.json"
        cached = None
        if not force:
            cached = (
                _load_board_state_window_artifact(artifact, window_fingerprint)
                if board_state_mode
                else _load_window_artifact(artifact, window_fingerprint)
            )
        if cached is not None:
            if board_state_mode:
                payload, current_board_state = cached
                batch, removed_board_lines = board_state_delta_to_observations(
                    chunk,
                    previous=previous_board_state,
                    current=current_board_state,
                )
                kb.observations.extend(
                    item.model_copy(deep=True) for item in batch.observations
                )
                previous_board_state = current_board_state
                payload["observations"] = batch.model_dump(mode="json")
                payload["board_delta"] = {
                    "added_observation_ids": [
                        item.id for item in batch.observations
                    ],
                    "removed": [
                        item.model_dump(mode="json")
                        for item in removed_board_lines
                    ],
                }
                atomic_json_dump(artifact, payload)
            else:
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
        native_video_clip: Path | None = None
        if native_video_mode:
            requests = []
            evidence = []
            if native_source_video is None:
                prepare_started = time.perf_counter()
                native_source_video = source.prepare_video(
                    work / "native_video_source",
                    max_height=pipeline.config.notes.native_video_height,
                )
                native_video_prepare_seconds += time.perf_counter() - prepare_started

            clip_started = time.perf_counter()
            native_video_clip = extract_api_video_clip(
                pipeline.config.runtime,
                native_source_video,
                start=chunk.start,
                end=chunk.end,
                output_path=work / "native_video_windows" / f"{chunk.id}.mp4",
                max_height=pipeline.config.notes.native_video_height,
                video_bitrate_kbps=(
                    pipeline.config.notes.native_video_video_bitrate_kbps
                ),
                audio_bitrate_kbps=(
                    pipeline.config.notes.native_video_audio_bitrate_kbps
                ),
                max_bytes=pipeline.config.notes.native_video_max_bytes,
            )
            native_video_clip_seconds += time.perf_counter() - clip_started
            native_video_windows_processed += 1

            extract_started = time.perf_counter()
            current_board_state: GeneratedBoardStateWindow | None = None
            removed_board_lines = []
            try:
                if board_state_mode:
                    current_board_state = orchestrator.extract_board_state_from_video(
                        chunk,
                        native_video_clip,
                        model=pipeline.config.notes.native_video_model,
                        thinking=pipeline.config.notes.native_video_thinking,
                        temperature=pipeline.config.notes.native_video_temperature,
                    )
                    batch, removed_board_lines = board_state_delta_to_observations(
                        chunk,
                        previous=previous_board_state,
                        current=current_board_state,
                    )
                else:
                    batch = orchestrator.extract_observations_from_video(
                        chunk,
                        native_video_clip,
                        kb,
                        model=pipeline.config.notes.native_video_model,
                        thinking=pipeline.config.notes.native_video_thinking,
                        temperature=pipeline.config.notes.native_video_temperature,
                    )
            except Exception as exc:
                if "invalid video file" not in str(exc).lower():
                    raise

                logger.warning(
                    "[%s] provider rejected %s as an invalid video; rebuilding a conservative "
                    "H.264/AAC clip with normalized timestamps and retrying once",
                    lecture.id,
                    chunk.id,
                )
                native_video_provider_retries += 1
                clip_started = time.perf_counter()
                native_video_clip = extract_api_video_clip(
                    pipeline.config.runtime,
                    native_source_video,
                    start=chunk.start,
                    end=chunk.end,
                    output_path=work / "native_video_windows" / f"{chunk.id}.mp4",
                    max_height=pipeline.config.notes.native_video_height,
                    video_bitrate_kbps=(
                        pipeline.config.notes.native_video_video_bitrate_kbps
                    ),
                    audio_bitrate_kbps=(
                        pipeline.config.notes.native_video_audio_bitrate_kbps
                    ),
                    max_bytes=pipeline.config.notes.native_video_max_bytes,
                    force=True,
                    conservative=True,
                )
                native_video_clip_seconds += time.perf_counter() - clip_started
                if board_state_mode:
                    current_board_state = orchestrator.extract_board_state_from_video(
                        chunk,
                        native_video_clip,
                        model=pipeline.config.notes.native_video_model,
                        thinking=pipeline.config.notes.native_video_thinking,
                        temperature=pipeline.config.notes.native_video_temperature,
                    )
                    batch, removed_board_lines = board_state_delta_to_observations(
                        chunk,
                        previous=previous_board_state,
                        current=current_board_state,
                    )
                else:
                    batch = orchestrator.extract_observations_from_video(
                        chunk,
                        native_video_clip,
                        kb,
                        model=pipeline.config.notes.native_video_model,
                        thinking=pipeline.config.notes.native_video_thinking,
                        temperature=pipeline.config.notes.native_video_temperature,
                    )
            extract_seconds += time.perf_counter() - extract_started
        else:
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

        if board_state_mode:
            kb.observations.extend(
                item.model_copy(deep=True) for item in batch.observations
            )
            added_ids = [item.id for item in batch.observations]
        else:
            added_ids = merge_window_observations(kb, batch)

        if board_state_mode:
            tracking = EpisodeTrackingUpdate()
            if current_board_state is None:
                raise RuntimeError("board-state extraction did not produce a snapshot")
            previous_board_state = current_board_state
        else:
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
                "evidence_backend": pipeline.config.notes.window_evidence_backend,
                "native_video_clip": (
                    str(native_video_clip) if native_video_clip is not None else None
                ),
                "board_state": (
                    current_board_state.model_dump(mode="json")
                    if board_state_mode and current_board_state is not None
                    else None
                ),
                "board_delta": (
                    {
                        "added_observation_ids": [item.id for item in batch.observations],
                        "removed": [
                            item.model_dump(mode="json")
                            for item in removed_board_lines
                        ],
                    }
                    if board_state_mode
                    else None
                ),
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

    state_mode = pipeline.config.notes.architecture == "state"
    state_section_cache_hits = 0
    state_section_batches_total = 0
    state_observation_resolution_cache_hits = 0
    state_observations_resolved = 0
    state_patches_kept = 0
    state_patches_replaced = 0
    state_patches_rejected = 0
    state_patches_unresolved = 0
    state_resolution_seconds = 0.0
    state_semantic_graph_seconds = 0.0
    state_semantic_graph_stats: dict[str, Any] = {}
    state_claim_compaction_seconds = 0.0
    state_claim_compaction_stats: dict[str, Any] = {}
    state_claim_compaction_unresolved: list[str] = []
    state_synthesis_seconds = 0.0
    state_repair_unresolved: list[str] = []

    if state_mode and pipeline.config.notes.state_semantic_backend == "mutable_graph":
        # The mutable graph backend consumes the extraction-level observations directly. The
        # legacy sequential repair, repaired episode graph, hierarchy and claim compaction are
        # intentionally bypassed: graph revision is the semantic inference step.
        source_state = make_lecture_state(kb)
        atomic_json_dump(
            work / "lecture_state_pre_graph_revision.json",
            source_state.model_dump(mode="json"),
        )
        raw_window_index = _load_state_raw_window_index(work)
        logger.info(
            "[graph_revision] semantic backend start: lecture=%s observations=%d raw_windows=%d",
            lecture.id,
            len(source_state.observations),
            len(raw_window_index),
        )
        graph_started = time.perf_counter()
        graph_run = run_iterative_graph_revision(
            orchestrator,
            lecture_state=source_state,
            raw_windows=raw_window_index,
            work=work,
            llm_config=pipeline.config.llm.model_dump(mode="json"),
            rounds=pipeline.config.notes.state_graph_revision_rounds,
            batch_observations=(
                pipeline.config.notes.state_graph_revision_batch_observations
            ),
            overlap_observations=(
                pipeline.config.notes.state_graph_revision_overlap_observations
            ),
            frontier_width=pipeline.config.notes.state_graph_revision_frontier_width,
            catalog_chars=pipeline.config.notes.state_graph_revision_catalog_chars,
            raw_context_chars=(
                pipeline.config.notes.state_graph_revision_raw_context_chars
            ),
            max_images=pipeline.config.notes.state_graph_revision_max_images,
            max_tokens=(
                pipeline.config.notes.state_graph_revision_max_tokens
                or pipeline.config.llm.max_tokens
            ),
            force=force,
        )
        graph_seconds = time.perf_counter() - graph_started
        consensus = graph_run.consensus
        logger.info(
            "[graph_revision] semantic search finished: lecture=%s seconds=%.1f frontier=%d",
            lecture.id,
            graph_seconds,
            len(graph_run.frontier),
        )
        atomic_json_dump(
            work / "lecture_graph.json",
            consensus.model_dump(mode="json"),
        )
        logger.info(
            "[graph_revision] rendering consensus graph to deterministic LectureIR fallback: "
            "lecture=%s active_nodes=%d",
            lecture.id,
            len(consensus.active_nodes()),
        )
        fallback_ir = graph_state_to_ir(
            consensus,
            lecture_id=lecture.id,
            title=lecture.title or lecture.id,
        )
        surface_writer_seconds = 0.0
        if pipeline.config.notes.state_section_assembly == "llm":
            surface_started = time.perf_counter()
            logger.info(
                "[graph_surface_writer] start: lecture=%s sections=%d",
                lecture.id,
                len(fallback_ir.chunks),
            )
            ir = write_graph_surface(
                orchestrator,
                state=consensus,
                lecture_id=lecture.id,
                lecture_title=lecture.title or lecture.id,
                fallback_ir=fallback_ir,
                work=work,
                llm_config=pipeline.config.llm.model_dump(mode="json"),
                force=force,
            )
            surface_writer_seconds = time.perf_counter() - surface_started
            logger.info(
                "[graph_surface_writer] complete: lecture=%s seconds=%.1f",
                lecture.id,
                surface_writer_seconds,
            )
        else:
            ir = fallback_ir

        logger.info(
            "[graph_revision] LectureIR ready: lecture=%s sections=%d blocks=%d unresolved=%d",
            lecture.id,
            len(ir.chunks),
            sum(len(chunk.blocks) for chunk in ir.chunks),
            len({item for chunk in ir.chunks for item in chunk.unresolved}),
        )

        pipeline._save_notation_registry(notation)
        atomic_json_dump(ir_path, ir.model_dump(mode="json"))
        manifest["ir_fingerprint"] = pipeline._ir_fingerprint(transcript, notation)
        atomic_json_dump(manifest_path, manifest)

        unique_unresolved = {
            item
            for notes in ir.chunks
            for item in notes.unresolved
        }
        usage = pipeline.llm.usage_snapshot()
        graph_metric = graph_revision_metrics(consensus)
        atomic_json_dump(
            work / "run_metrics.json",
            {
                "lecture_id": lecture.id,
                "architecture": "state_mutable_graph",
                "media_seconds": round(media_seconds, 3),
                "asr_seconds": round(asr_seconds, 3),
                "notes_seconds": round(time.perf_counter() - notes_started, 3),
                "vision_seconds": round(vision_seconds, 3),
                "native_video_prepare_seconds": round(native_video_prepare_seconds, 3),
                "native_video_clip_seconds": round(native_video_clip_seconds, 3),
                "native_video_windows_processed": native_video_windows_processed,
                "native_video_provider_retries": native_video_provider_retries,
                "window_evidence_backend": pipeline.config.notes.window_evidence_backend,
                "native_video_model": (
                    pipeline.config.notes.native_video_model if native_video_mode else None
                ),
                "knowledge_extract_seconds": round(extract_seconds, 3),
                "episode_track_seconds": round(episode_track_seconds, 3),
                "graph_revision_seconds": round(graph_seconds, 3),
                "graph_surface_writer_seconds": round(surface_writer_seconds, 3),
                "graph_revision": {
                    **graph_run.stats,
                    "frontier_states": len(graph_run.frontier),
                    "consensus_metrics": graph_metric.model_dump(mode="json"),
                },
                "total_seconds": round(time.perf_counter() - run_started, 3),
                "windows_total": len(chunks),
                "windows_processed": processed_windows,
                "window_cache_hits": cache_hits,
                "observations_total": len(kb.observations),
                "graph_nodes_total": len(consensus.nodes),
                "graph_active_nodes": len(consensus.active_nodes()),
                "graph_edges_total": len(consensus.edges),
                "topic_sections_total": len(ir.chunks),
                "sections_total": len(ir.chunks),
                "visual_requests_processed": visual_requests_processed,
                "visual_evidence_successful": visual_evidence_successful,
                "unresolved_total": len(unique_unresolved),
                "llm_usage": LectureModelClient.combine_usage([usage]),
            },
        )
        return ir

    if state_mode:
        # Preserve the extraction/episode-tracking state for audit, then repair it exactly once.
        atomic_json_dump(
            work / "lecture_state_pre_repair.json",
            make_lecture_state(kb).model_dump(mode="json"),
        )
        raw_window_index = _load_state_raw_window_index(work)
        repair_started = time.perf_counter()
        kb, repair_stats, state_repair_unresolved = _repair_lecture_state(
            orchestrator,
            kb=kb,
            raw_windows=raw_window_index,
            work=work,
            config=pipeline.config.notes,
            llm_config=pipeline.config.llm.model_dump(mode="json"),
            force=force,
        )
        state_resolution_seconds = time.perf_counter() - repair_started
        state_observations_resolved = int(repair_stats["processed"])
        state_observation_resolution_cache_hits = int(repair_stats["cache_hits"])
        state_patches_kept = int(repair_stats["kept"])
        state_patches_replaced = int(repair_stats["replaced"])
        state_patches_rejected = int(repair_stats["rejected"])
        state_patches_unresolved = int(repair_stats["unresolved"])

        # PR #3 made claims/episodes derived semantic state. Transactional repair (#83) changes the
        # observations those objects were derived from, so the pre-repair graph is now audit-only.
        atomic_json_dump(
            work / "lecture_state_pre_semantic_graph.json",
            make_lecture_state(kb).model_dump(mode="json"),
        )
        semantic_graph_started = time.perf_counter()
        kb, state_semantic_graph_stats = _rebuild_repaired_semantic_graph(
            orchestrator,
            repaired=kb,
            work=work,
            llm_config=pipeline.config.llm.model_dump(mode="json"),
            batch_observations=(
                pipeline.config.notes.state_repaired_episode_batch_observations
            ),
            force=force,
        )
        state_semantic_graph_seconds = time.perf_counter() - semantic_graph_started
        atomic_json_dump(
            work / "lecture_state.json",
            make_lecture_state(kb).model_dump(mode="json"),
        )

    atomic_json_dump(work / "lecture_kb.json", kb.model_dump(mode="json"))

    hierarchy_path = work / "episode_hierarchy.json"
    hierarchy_fingerprint = _hierarchy_fingerprint(
        kb,
        llm_config=pipeline.config.llm.model_dump(mode="json"),
        hierarchy_batch_episodes=pipeline.config.notes.hierarchy_batch_episodes,
    )
    # Accept the immediately preceding cache identity once so existing runs migrate without
    # paying for another hierarchy LLM call. Future writer-only config changes use only the new
    # dependency-minimal fingerprint above.
    legacy_kb_fingerprint = stable_hash(
        {
            "kb": kb.model_dump(mode="json"),
            "notes": pipeline.config.notes.model_dump(mode="json"),
            "llm": pipeline.config.llm.model_dump(mode="json"),
            "knowledge_cache_version": KNOWLEDGE_CACHE_VERSION,
            "state_pipeline_version": STATE_PIPELINE_VERSION if state_mode else None,
        }
    )
    legacy_hierarchy_fingerprint = stable_hash(
        {
            "kb_fingerprint": legacy_kb_fingerprint,
            "hierarchy_batch_episodes": pipeline.config.notes.hierarchy_batch_episodes,
            "hierarchy_cache_version": HIERARCHY_CACHE_VERSION,
        }
    )

    hierarchy: EpisodeHierarchyPlan | None = None
    cached_hierarchy_fingerprint: str | None = None
    if hierarchy_path.exists() and not force:
        try:
            payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
            cached_hierarchy_fingerprint = str(payload.get("fingerprint") or "")
            if cached_hierarchy_fingerprint in {
                hierarchy_fingerprint,
                legacy_hierarchy_fingerprint,
            }:
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
    elif cached_hierarchy_fingerprint != hierarchy_fingerprint:
        atomic_json_dump(
            hierarchy_path,
            {
                "fingerprint": hierarchy_fingerprint,
                "hierarchy": hierarchy.model_dump(mode="json"),
            },
        )
    hierarchy_seconds = time.perf_counter() - hierarchy_started

    if state_mode:
        # Hierarchy depends only on repaired semantic leaves. Canonical claim compaction happens
        # afterwards, so changing the semantic wording/content density does not require replanning
        # episode boundaries or section grouping.
        atomic_json_dump(
            work / "lecture_state_pre_claim_compaction.json",
            make_lecture_state(kb).model_dump(mode="json"),
        )
        claim_compaction_started = time.perf_counter()
        (
            kb,
            state_claim_compaction_stats,
            state_claim_compaction_unresolved,
        ) = compact_repaired_claims(
            orchestrator,
            kb=kb,
            work=work,
            llm_config=pipeline.config.llm.model_dump(mode="json"),
            force=force,
        )
        state_claim_compaction_seconds = time.perf_counter() - claim_compaction_started
        atomic_json_dump(work / "lecture_kb.json", kb.model_dump(mode="json"))

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

    if state_mode:
        note_sections: list[ChunkNotes] = []
        if pipeline.config.notes.state_section_assembly == "deterministic":
            note_sections = [
                _assemble_state_section_deterministically(kb, section)
                for section in outline.sections
            ]
        else:
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
                    writer_observations = _writer_canonical_observations(evidence_payload)
                    writer_previous_tail = _writer_previous_tail(previous_context)
                    fingerprint = stable_hash(
                        {
                            "state_pipeline_version": STATE_PIPELINE_VERSION,
                            "state_section_writer_cache_version": STATE_SECTION_WRITER_CACHE_VERSION,
                            "state_section_writer_policy": {
                                "thinking": pipeline.config.notes.state_section_writer_thinking,
                                "temperature": pipeline.config.notes.state_section_writer_temperature,
                                "top_p": pipeline.config.notes.state_section_writer_top_p,
                                "top_k": pipeline.config.notes.state_section_writer_top_k,
                                "min_p": pipeline.config.notes.state_section_writer_min_p,
                                "presence_penalty": (
                                    pipeline.config.notes.state_section_writer_presence_penalty
                                ),
                                "repetition_penalty": (
                                    pipeline.config.notes.state_section_writer_repetition_penalty
                                ),
                            },
                            "section_title": section.title,
                            "canonical_observations": writer_observations,
                            "previous_tail": writer_previous_tail,
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
                        previous_context=previous_context,
                    )
                    state_synthesis_seconds += time.perf_counter() - started
                    atomic_json_dump(
                        path,
                        {
                            "fingerprint": fingerprint,
                            "repaired_evidence": evidence_payload,
                            "notes": notes.model_dump(mode="json"),
                        },
                    )
                    generated_batches.append(notes)

                note_sections.append(_merge_state_section_batches(section, generated_batches))

        state_pipeline_unresolved = list(
            dict.fromkeys([
                *state_repair_unresolved,
                *state_claim_compaction_unresolved,
            ])
        )
        if state_pipeline_unresolved and note_sections:
            note_sections[-1].unresolved = list(
                dict.fromkeys([*note_sections[-1].unresolved, *state_pipeline_unresolved])
            )

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

    unique_unresolved = set(kb.unresolved)
    for notes in note_sections:
        unique_unresolved.update(notes.unresolved)

    usage = pipeline.llm.usage_snapshot()
    atomic_json_dump(
        work / "run_metrics.json",
        {
            "lecture_id": lecture.id,
            "architecture": (
                "state_repaired_semantic_graph"
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
            "state_semantic_graph_seconds": round(state_semantic_graph_seconds, 3),
            "state_semantic_graph": state_semantic_graph_stats,
            "state_claim_compaction_seconds": round(state_claim_compaction_seconds, 3),
            "state_claim_compaction": state_claim_compaction_stats,
            "state_synthesis_seconds": round(state_synthesis_seconds, 3),
            "state_section_assembly": (
                pipeline.config.notes.state_section_assembly if state_mode else None
            ),
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
            "state_patches_kept": state_patches_kept,
            "state_patches_replaced": state_patches_replaced,
            "state_patches_rejected": state_patches_rejected,
            "state_patches_unresolved": state_patches_unresolved,
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
            "unresolved_total": len(unique_unresolved),
            "llm_usage": LectureModelClient.combine_usage([usage]),
        },
    )
    return ir
