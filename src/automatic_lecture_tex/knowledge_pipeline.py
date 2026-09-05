from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .chunking import chunk_transcript
from .knowledge import (
    KnowledgeOrchestrator,
    apply_global_validation,
    apply_knowledge_update,
    compact_knowledge_state,
    merge_window_observations,
)
from .llm import LectureModelClient
from .media import copy_asset
from .schemas import (
    ChunkNotes,
    GlobalValidation,
    KnowledgeUpdate,
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

KNOWLEDGE_CACHE_VERSION = 1


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
        update = KnowledgeUpdate.model_validate(payload["knowledge_update"])
        return payload, batch, update
    except (json.JSONDecodeError, KeyError, ValidationError):
        return None


def _ensure_outline(
    outline: LectureOutline,
    kb: LectureKnowledgeBase,
    transcript: Transcript,
    lecture_title: str,
) -> LectureOutline:
    if outline.sections:
        return outline
    if not transcript.segments:
        return outline
    outline.sections = [
        OutlineSection(
            id="section_000",
            title=lecture_title,
            start=transcript.segments[0].start,
            end=transcript.segments[-1].end,
            claim_ids=[
                item.id
                for item in kb.claims
                if item.status == "active"
            ],
            evidence_ids=[item.id for item in kb.observations],
        )
    ]
    outline.unresolved.append(
        "Семантический планировщик не вернул секции; использована одна секция на всю лекцию."
    )
    return outline


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
    update_seconds = 0.0

    for chunk in chunks:
        state_before = compact_knowledge_state(kb, pipeline.config.notes)
        window_fingerprint = stable_hash(
            {
                "source": source_identity,
                "chunk": chunk.model_dump(mode="json"),
                "kb_state_before": state_before,
                "notes": pipeline.config.notes.model_dump(mode="json"),
                "vision": pipeline.config.vision.model_dump(mode="json"),
                "llm": pipeline.config.llm.model_dump(mode="json"),
                "knowledge_cache_version": KNOWLEDGE_CACHE_VERSION,
            }
        )
        artifact = work / "knowledge_windows" / f"{chunk.id}.json"
        cached = None if force else _load_window_artifact(artifact, window_fingerprint)
        if cached is not None:
            payload, batch, update = cached
            added_ids = merge_window_observations(kb, batch)
            apply_knowledge_update(kb, update, window_id=chunk.id)
            cache_hits += 1
            visual_requests_processed += len(payload.get("visual_requests", []))
            visual_evidence_successful += sum(
                item.get("kind") != "none" and float(item.get("confidence", 0.0)) >= 0.75
                for item in payload.get("visual_evidence", [])
            )
            logger.info("[%s] %s knowledge cache hit", lecture.id, chunk.id)
            continue

        logger.info(
            "[%s] extracting knowledge from %s (%d/%d)",
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

        update_started = time.perf_counter()
        update = orchestrator.update_knowledge(kb, batch, added_ids)
        update_seconds += time.perf_counter() - update_started
        apply_knowledge_update(kb, update, window_id=chunk.id)
        processed_windows += 1

        atomic_json_dump(
            artifact,
            {
                "fingerprint": window_fingerprint,
                "chunk": chunk.model_dump(mode="json"),
                "visual_requests": [item.model_dump(mode="json") for item in requests],
                "visual_evidence": [item.model_dump(mode="json") for item in evidence],
                "observations": batch.model_dump(mode="json"),
                "knowledge_update": update.model_dump(mode="json"),
            },
        )
        atomic_json_dump(work / "lecture_kb.json", kb.model_dump(mode="json"))

    kb_fingerprint = stable_hash(
        {
            "kb": kb.model_dump(mode="json"),
            "notes": pipeline.config.notes.model_dump(mode="json"),
            "llm": pipeline.config.llm.model_dump(mode="json"),
            "knowledge_cache_version": KNOWLEDGE_CACHE_VERSION,
        }
    )
    atomic_json_dump(work / "lecture_kb.json", kb.model_dump(mode="json"))

    outline_path = work / "lecture_outline.json"
    outline_fingerprint = stable_hash(
        {
            "kb_fingerprint": kb_fingerprint,
            "transcript": transcript.model_dump(mode="json"),
            "boundary_context_seconds": pipeline.config.notes.boundary_context_seconds,
            "max_outline_sections": pipeline.config.notes.max_outline_sections,
        }
    )
    outline: LectureOutline | None = None
    if outline_path.exists() and not force:
        try:
            payload = json.loads(outline_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == outline_fingerprint:
                outline = LectureOutline.model_validate(payload["outline"])
        except (json.JSONDecodeError, KeyError, ValidationError):
            outline = None
    outline_started = time.perf_counter()
    if outline is None:
        outline = orchestrator.plan_outline(kb, transcript)
        outline = _ensure_outline(outline, kb, transcript, lecture.title or lecture.id)
        atomic_json_dump(
            outline_path,
            {
                "fingerprint": outline_fingerprint,
                "outline": outline.model_dump(mode="json"),
            },
        )
    outline_seconds = time.perf_counter() - outline_started

    note_sections: list[ChunkNotes] = []
    section_cache_hits = 0
    section_write_seconds = 0.0
    for section in outline.sections:
        section_path = work / "knowledge_sections" / f"{section.id}.json"
        section_fingerprint = stable_hash(
            {
                "section": section.model_dump(mode="json"),
                "kb_fingerprint": kb_fingerprint,
                "llm": pipeline.config.llm.model_dump(mode="json"),
            }
        )
        notes = None
        if section_path.exists() and not force:
            try:
                payload = json.loads(section_path.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == section_fingerprint:
                    notes = ChunkNotes.model_validate(payload["notes"])
                    section_cache_hits += 1
            except (json.JSONDecodeError, KeyError, ValidationError):
                notes = None
        if notes is None:
            started = time.perf_counter()
            notes = orchestrator.write_section(section, kb, transcript)
            section_write_seconds += time.perf_counter() - started
            atomic_json_dump(
                section_path,
                {
                    "fingerprint": section_fingerprint,
                    "notes": notes.model_dump(mode="json"),
                },
            )
        note_sections.append(notes)

    ir = LectureIR(
        lecture_id=lecture.id,
        title=lecture.title or lecture.id,
        chunks=note_sections,
    )

    validation_seconds = 0.0
    validation: GlobalValidation | None = None
    if pipeline.config.notes.global_validation and ir.chunks:
        validation_path = work / "global_validation.json"
        validation_fingerprint = stable_hash(
            {
                "kb_fingerprint": kb_fingerprint,
                "draft": ir.model_dump(mode="json"),
                "llm": pipeline.config.llm.model_dump(mode="json"),
            }
        )
        if validation_path.exists() and not force:
            try:
                payload = json.loads(validation_path.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == validation_fingerprint:
                    validation = GlobalValidation.model_validate(payload["validation"])
            except (json.JSONDecodeError, KeyError, ValidationError):
                validation = None
        if validation is None:
            started = time.perf_counter()
            validation = orchestrator.validate_lecture(ir, kb)
            validation_seconds = time.perf_counter() - started
            atomic_json_dump(
                validation_path,
                {
                    "fingerprint": validation_fingerprint,
                    "validation": validation.model_dump(mode="json"),
                },
            )
        apply_global_validation(
            ir,
            validation,
            threshold=pipeline.config.notes.global_validation_apply_threshold,
        )

    symbol_meanings: dict[str, set[str]] = {}
    for symbol in kb.symbols:
        if symbol.active and symbol.symbol and symbol.meaning:
            symbol_meanings.setdefault(symbol.symbol, set()).add(symbol.meaning)
    for symbol, meanings in symbol_meanings.items():
        # The course-level legacy registry is unscoped. Export only unambiguous symbols; scoped
        # collisions remain represented faithfully in lecture_kb.json.
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
            "architecture": "knowledge",
            "media_seconds": round(media_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "notes_seconds": round(time.perf_counter() - notes_started, 3),
            "vision_seconds": round(vision_seconds, 3),
            "knowledge_extract_seconds": round(extract_seconds, 3),
            "knowledge_update_seconds": round(update_seconds, 3),
            "outline_seconds": round(outline_seconds, 3),
            "section_write_seconds": round(section_write_seconds, 3),
            "global_validation_seconds": round(validation_seconds, 3),
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "windows_total": len(chunks),
            "windows_processed": processed_windows,
            "window_cache_hits": cache_hits,
            "sections_total": len(note_sections),
            "section_cache_hits": section_cache_hits,
            "observations_total": len(kb.observations),
            "claims_total": len(kb.claims),
            "active_claims": sum(item.status == "active" for item in kb.claims),
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
