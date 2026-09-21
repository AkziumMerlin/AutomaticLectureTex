from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .chunking import chunk_transcript
from .generated_notes import GeneratedChunkNotes
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
from .llm import LectureModelClient
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
STATE_PIPELINE_VERSION = 1

# These settings affect only hierarchy/synthesis. Excluding them from the extraction fingerprint is
# intentional: changing downstream batching must not throw away expensive ASR/visual/evidence work.
_DOWNSTREAM_NOTE_FIELDS = {
    "hierarchy_batch_episodes",
    "episode_synthesis_max_evidence_chars",
    "episode_symbol_context_limit",
    "state_section_max_evidence_chars",
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


def _state_section_payload(
    kb: LectureKnowledgeBase,
    section: OutlineSection,
    transcript: Transcript,
    config,
) -> dict[str, Any]:
    payload = evidence_for_section(kb, section, transcript, config)
    payload.pop("transcript", None)
    return payload


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
            batches.append(_state_section_payload(kb, committed, transcript, config))
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
        payload = _state_section_payload(kb, committed, transcript, config)
        if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) > max_chars:
            raise ValueError(
                f"single state-section episode batch exceeds "
                f"notes.state_section_max_evidence_chars={max_chars}"
            )
        batches.append(payload)

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
) -> ChunkNotes:
    prompt = f"""Write one contiguous part of a FINAL lecture-note section from an already assembled
persistent LectureState. All local evidence extraction, episode tracking, notation tracking and
global outline planning happened BEFORE this call.

Global lecture outline (read-only narrative context):
{json.dumps(outline_context, ensure_ascii=False, separators=(",", ":"))}

Current fixed section:
{json.dumps(section.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))}

Canonical state evidence for this batch:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Previously written blocks from THIS section only:
{json.dumps(previous_context, ensure_ascii=False, separators=(",", ":"))}

Rules:
- Follow the fixed episode order and the lecture's actual narrative line.
- Canonical observations/active claims/symbol records are the source of truth for final writing.
- Do not recreate material from raw ASR wording; raw ASR was intentionally removed at this stage.
- Preserve lecturer corrections, notation evolution, theorem/proof continuity and level of detail.
- Never resurrect superseded/retracted or unresolved content as a current fact.
- Do not introduce textbook material merely because it would make the exposition nicer.
- Avoid repeating a definition/proof step already present in previous_context unless this batch
  genuinely develops it further.
- Every substantive block must cite source_claim_ids and/or source_evidence_ids present in the
  supplied canonical state evidence.
- Return block bodies only; renderer owns section/theorem/proof wrappers.
- Write prose in language code {orchestrator.output_language} and mathematics in LaTeX.
"""
    generated = orchestrator._structured(
        prompt,
        GeneratedChunkNotes,
        operation="state_section_write",
        max_tokens=6144,
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
                "omni": pipeline.config.omni.model_dump(mode="json"),
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
                fingerprint = stable_hash(
                    {
                        "state_pipeline_version": STATE_PIPELINE_VERSION,
                        "section": section.model_dump(mode="json"),
                        "outline_context": outline_context,
                        "evidence": evidence_payload,
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
                notes = _write_state_section_batch(
                    orchestrator,
                    section,
                    evidence_payload,
                    outline_context=outline_context,
                    previous_context=previous_context,
                )
                state_synthesis_seconds += time.perf_counter() - started
                atomic_json_dump(
                    path,
                    {
                        "fingerprint": fingerprint,
                        "evidence": evidence_payload,
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
