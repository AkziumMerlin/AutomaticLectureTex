from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pydantic_core import ValidationError

from .chunking import chunk_transcript
from .linear_llm_policy import has_explicit_correction_signal
from .linear_notes import (
    LinearCorrectionScan,
    LinearPatch,
    block_id,
    provenance_claim_ids,
    scan_linear_corrections,
)
from .linear_visual_fallback import (
    clean_stale_architecture_artifacts,
    inject_unresolved_board_snapshots,
    is_board_snapshot,
)
from .llm import LectureModelClient
from .schemas import ChunkNotes, CorrectionRecord, LectureIR, NoteBlock, Transcript, VisualEvidence
from .sensory_evidence import collect_visual_evidence
from .util import atomic_json_dump, stable_hash

logger = logging.getLogger(__name__)

# Version 5 makes board fallbacks block-linked and suppresses source-grounded unsafe blocks while
# preserving the restored pre-PR2 chronological writer.
LINEAR_PIPELINE_VERSION = 5


def _all_blocks(note_chunks: list[ChunkNotes]) -> list[NoteBlock]:
    return [block for notes in note_chunks for block in notes.blocks]


def _stamp_chunk_provenance(notes: ChunkNotes, chunk_index: int, segment_ids: list[str]) -> None:
    """Assign stable host-owned block ids after note generation."""

    for block_index, block in enumerate(notes.blocks):
        stable_id = f"block_{chunk_index:04d}_{block_index:03d}"
        block.source_claim_ids = provenance_claim_ids(stable_id, list(segment_ids))


def _writer_context_notes(notes: ChunkNotes | None) -> ChunkNotes | None:
    """Return the pre-PR2 view of previous notes used by ``finalize_chunk``.

    Stable ids and evidence provenance were added much later. Passing those arrays back through
    ``previous_notes`` polluted the old writer context and encouraged the model to reproduce large
    bookkeeping lists in structured output. Keep the stored IR rich, but expose only the fields the
    old writer actually knew about.
    """

    if notes is None:
        return None
    return ChunkNotes(
        chunk_id=notes.chunk_id,
        start=notes.start,
        end=notes.end,
        section_title=notes.section_title,
        blocks=[
            NoteBlock(
                type=block.type,
                title=block.title,
                latex=block.latex,
                asset_path=block.asset_path,
                caption=block.caption,
            )
            for block in notes.blocks
        ],
        notation=list(notes.notation),
        corrections=list(notes.corrections),
        unresolved=list(notes.unresolved),
    )


def _apply_patch(
    patch: LinearPatch,
    *,
    block_map: dict[str, NoteBlock],
    owner_map: dict[str, ChunkNotes],
    allowed_targets: set[str],
    allowed_evidence_ids: set[str],
    apply_threshold: float,
) -> tuple[bool, str | None]:
    if patch.target_block_id not in allowed_targets:
        return False, f"Rejected correction: unknown/out-of-scope target {patch.target_block_id!r}."
    if not set(patch.evidence_segment_ids).issubset(allowed_evidence_ids):
        return False, (
            f"Rejected correction for {patch.target_block_id}: evidence ids must come from the "
            "current transcript chunk."
        )
    if patch.confidence < apply_threshold:
        return False, (
            f"Unapplied correction for {patch.target_block_id} "
            f"(confidence={patch.confidence:.2f}): {patch.reason}"
        )

    block = block_map.get(patch.target_block_id)
    owner = owner_map.get(patch.target_block_id)
    if block is None or owner is None:
        return False, f"Rejected correction: target {patch.target_block_id!r} no longer exists."

    original = block.latex
    if patch.action == "retract":
        owner.blocks = [item for item in owner.blocks if block_id(item) != patch.target_block_id]
        block_map.pop(patch.target_block_id, None)
        owner_map.pop(patch.target_block_id, None)
        owner.corrections.append(
            CorrectionRecord(
                original=original,
                corrected="[RETRACTED]",
                reason=patch.reason,
                basis="audio_context",
                confidence=patch.confidence,
            )
        )
        return True, None

    replacement = (patch.replacement_latex or "").strip()
    if replacement == original.strip():
        return False, None
    try:
        checked = NoteBlock(
            type=block.type,
            title=block.title,
            latex=replacement,
            asset_path=block.asset_path,
            caption=block.caption,
            source_claim_ids=list(block.source_claim_ids),
            source_evidence_ids=list(block.source_evidence_ids),
        )
    except ValidationError as exc:
        return False, f"Rejected correction for {patch.target_block_id}: invalid replacement ({exc})."

    block.latex = checked.latex
    owner.corrections.append(
        CorrectionRecord(
            original=original,
            corrected=checked.latex,
            reason=patch.reason,
            basis="audio_context",
            confidence=patch.confidence,
        )
    )
    return True, None


def _chunk_cache_fingerprint(
    pipeline,
    *,
    chunk,
    previous_notes: ChunkNotes | None,
    notation: dict[str, str],
    source_identity,
) -> str:
    return stable_hash(
        {
            "version": LINEAR_PIPELINE_VERSION,
            "writer": "pre_pr2_finalize_chunk_clean_context",
            "chunk": chunk.model_dump(mode="json"),
            "previous_notes": (
                previous_notes.model_dump(mode="json") if previous_notes is not None else None
            ),
            "notation": notation,
            "source": source_identity,
            "notes": pipeline.config.notes.model_dump(mode="json"),
            "vision": pipeline.config.vision.model_dump(mode="json"),
            "llm": pipeline.config.llm.model_dump(mode="json"),
        }
    )


def _register_blocks(
    notes: ChunkNotes,
    block_map: dict[str, NoteBlock],
    owner_map: dict[str, ChunkNotes],
) -> None:
    for block in notes.blocks:
        stable_id = block_id(block)
        if not stable_id:
            continue
        block_map[stable_id] = block
        owner_map[stable_id] = notes


def run_linear_pipeline(
    pipeline,
    *,
    lecture,
    transcript: Transcript,
    source,
    source_identity,
    work: Path,
    ir_path: Path,
    manifest_path: Path,
    manifest: dict,
    notation: dict[str, str],
    media_seconds: float,
    asr_seconds: float,
    run_started: float,
    force: bool,
) -> LectureIR:
    """The original chronological writer plus narrow evidence-preserving fail-safes."""

    clean_stale_architecture_artifacts(work)
    pipeline.llm.reset_usage()
    notes_started = time.perf_counter()
    config = pipeline.config.notes
    chunks = chunk_transcript(transcript, config.chunk_target_seconds, overlap_seconds=0.0)
    note_chunks: list[ChunkNotes] = []
    evidence_by_chunk: dict[str, list[VisualEvidence]] = {}
    block_map: dict[str, NoteBlock] = {}
    owner_map: dict[str, ChunkNotes] = {}

    chunk_cache_hits = 0
    processed_chunks = 0
    visual_seconds = 0.0
    finalize_seconds = 0.0
    correction_seconds = 0.0
    visual_requests_processed = 0
    visual_evidence_successful = 0
    correction_patches_applied = 0
    finalize_contextless_retries = 0
    finalize_failures = 0
    llm_usages: list[dict] = []

    chunks_dir = work / "linear_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, chunk in enumerate(chunks):
        previous_notes = note_chunks[-1] if note_chunks else None
        writer_previous_notes = _writer_context_notes(previous_notes)
        artifact = chunks_dir / f"{chunk.id}.json"
        fingerprint = _chunk_cache_fingerprint(
            pipeline,
            chunk=chunk,
            previous_notes=writer_previous_notes,
            notation=notation,
            source_identity=source_identity,
        )

        cached = None
        if artifact.exists() and not force:
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == fingerprint:
                    cached = payload
            except (json.JSONDecodeError, ValueError):
                cached = None

        if cached is not None:
            evidence = [
                VisualEvidence.model_validate(item) for item in cached.get("visual_evidence", [])
            ]
            notes = ChunkNotes.model_validate(cached["notes"])
            chunk_cache_hits += 1
            logger.info("[%s] %s pre-PR2 writer cache hit", lecture.id, chunk.id)
            llm_usages.append(cached.get("llm_usage", {}))
        else:
            logger.info(
                "[%s] pre-PR2 finalize %s (%d/%d)",
                lecture.id,
                chunk.id,
                chunk_index + 1,
                len(chunks),
            )
            usage_before = pipeline.llm.usage_snapshot()

            visual_started = time.perf_counter()
            requests, evidence, _elapsed = collect_visual_evidence(
                pipeline,
                lecture,
                chunk,
                transcript,
                source,
                work,
                pipeline.config.latex.output_dir / "figures" / lecture.id,
                notation,
            )
            visual_seconds += time.perf_counter() - visual_started
            visual_requests_processed += len(requests)
            visual_evidence_successful += sum(
                item.kind != "none" and item.confidence >= 0.75 for item in evidence
            )

            finalize_started = time.perf_counter()
            try:
                notes = pipeline.llm.finalize_chunk(
                    chunk, evidence, notation, writer_previous_notes
                )
            except (json.JSONDecodeError, ValidationError) as first_exc:
                finalize_contextless_retries += 1
                logger.warning(
                    "[%s] %s finalize_chunk failed with previous context; retrying contextless: %s",
                    lecture.id,
                    chunk.id,
                    first_exc,
                )
                try:
                    notes = pipeline.llm.finalize_chunk(chunk, evidence, notation, None)
                except (json.JSONDecodeError, ValidationError) as exc:
                    finalize_failures += 1
                    logger.warning("[%s] %s finalize_chunk failed: %s", lecture.id, chunk.id, exc)
                    notes = ChunkNotes(
                        chunk_id=chunk.id,
                        start=chunk.start,
                        end=chunk.end,
                        section_title=(
                            previous_notes.section_title if previous_notes else "Без названия"
                        ),
                        blocks=[],
                        unresolved=[
                            f"Не удалось разобрать structured output для {chunk.id}; "
                            "фрагмент сохранён только в transcript evidence."
                        ],
                    )
            finalize_seconds += time.perf_counter() - finalize_started

            inject_unresolved_board_snapshots(
                notes,
                evidence,
                pipeline.config.vision,
                output_language=pipeline.config.llm.output_language,
            )
            _stamp_chunk_provenance(notes, chunk_index, chunk.segment_ids)

            usage = LectureModelClient.usage_delta(pipeline.llm.usage_snapshot(), usage_before)
            llm_usages.append(usage)
            processed_chunks += 1
            atomic_json_dump(
                artifact,
                {
                    "fingerprint": fingerprint,
                    "chunk": chunk.model_dump(mode="json"),
                    "visual_evidence": [item.model_dump(mode="json") for item in evidence],
                    "notes": notes.model_dump(mode="json"),
                    "llm_usage": usage,
                },
            )

        if any(not block_id(block) for block in notes.blocks):
            _stamp_chunk_provenance(notes, chunk_index, chunk.segment_ids)

        evidence_by_chunk[chunk.id] = evidence
        note_chunks.append(notes)
        _register_blocks(notes, block_map, owner_map)
        for item in notes.notation:
            notation.setdefault(item.latex, item.meaning)

    if config.linear_correction_scan_enabled:
        scan_dir = work / "linear_correction_scans"
        scan_dir.mkdir(parents=True, exist_ok=True)
        triggered_ids = {
            chunk.id
            for chunk in chunks
            if has_explicit_correction_signal(chunk.timestamped_text or chunk.text)
        }
        for stale in scan_dir.glob("*.json"):
            if stale.stem not in triggered_ids:
                stale.unlink()

        for chunk_index, chunk in enumerate(chunks):
            if chunk.id not in triggered_ids:
                continue
            earlier = [block for notes in note_chunks[:chunk_index] for block in notes.blocks]
            if not earlier:
                continue
            evidence = evidence_by_chunk.get(chunk.id, [])
            scan_fingerprint = stable_hash(
                {
                    "version": LINEAR_PIPELINE_VERSION,
                    "chunk": chunk.model_dump(mode="json"),
                    "earlier_blocks": [block.model_dump(mode="json") for block in earlier],
                    "visual_evidence": [item.model_dump(mode="json") for item in evidence],
                    "llm": pipeline.config.llm.model_dump(mode="json"),
                    "catalog_chars": config.linear_correction_catalog_chars,
                }
            )
            scan_path = scan_dir / f"{chunk.id}.json"
            scan = None
            if scan_path.exists() and not force:
                try:
                    payload = json.loads(scan_path.read_text(encoding="utf-8"))
                    if payload.get("fingerprint") == scan_fingerprint:
                        scan = LinearCorrectionScan.model_validate(payload["scan"])
                except (json.JSONDecodeError, ValueError):
                    scan = None

            if scan is None:
                usage_before = pipeline.llm.usage_snapshot()
                started = time.perf_counter()
                try:
                    scan = scan_linear_corrections(
                        pipeline.llm,
                        chunk=chunk,
                        evidence=evidence,
                        earlier_blocks=earlier,
                        output_language=pipeline.config.llm.output_language,
                        catalog_chars=config.linear_correction_catalog_chars,
                    )
                except (json.JSONDecodeError, ValidationError) as exc:
                    logger.warning(
                        "[%s] correction scan failed for %s: %s", lecture.id, chunk.id, exc
                    )
                    scan = LinearCorrectionScan(
                        unresolved=[f"Correction scan failed for {chunk.id}: {type(exc).__name__}"]
                    )
                correction_seconds += time.perf_counter() - started
                usage = LectureModelClient.usage_delta(pipeline.llm.usage_snapshot(), usage_before)
                llm_usages.append(usage)
                atomic_json_dump(
                    scan_path,
                    {"fingerprint": scan_fingerprint, "scan": scan.model_dump(mode="json")},
                )

            note_chunks[chunk_index].unresolved.extend(scan.unresolved)
            allowed_targets = {block_id(block) for block in earlier if block_id(block)}
            allowed_evidence = set(chunk.segment_ids)
            for patch in scan.patches:
                applied, issue = _apply_patch(
                    patch,
                    block_map=block_map,
                    owner_map=owner_map,
                    allowed_targets=allowed_targets,
                    allowed_evidence_ids=allowed_evidence,
                    apply_threshold=config.linear_patch_apply_threshold,
                )
                if applied:
                    correction_patches_applied += 1
                if issue:
                    note_chunks[chunk_index].unresolved.append(issue)
            note_chunks[chunk_index].unresolved = list(
                dict.fromkeys(note_chunks[chunk_index].unresolved)
            )

    ir = LectureIR(
        lecture_id=lecture.id,
        title=lecture.title or lecture.id,
        chunks=note_chunks,
    )
    atomic_json_dump(ir_path, ir.model_dump(mode="json"))
    pipeline._save_notation_registry(notation)
    manifest["ir_fingerprint"] = pipeline._ir_fingerprint(transcript, notation)
    atomic_json_dump(manifest_path, manifest)
    atomic_json_dump(
        work / "run_metrics.json",
        {
            "lecture_id": lecture.id,
            "architecture": "linear_pre_pr2_with_corrections",
            "linear_pipeline_version": LINEAR_PIPELINE_VERSION,
            "media_seconds": round(media_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "notes_seconds": round(time.perf_counter() - notes_started, 3),
            "vision_seconds": round(visual_seconds, 3),
            "finalize_and_math_audit_seconds": round(finalize_seconds, 3),
            "finalize_contextless_retries": finalize_contextless_retries,
            "finalize_failures": finalize_failures,
            "correction_scan_seconds": round(correction_seconds, 3),
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "chunks_total": len(chunks),
            "chunks_processed": processed_chunks,
            "chunk_cache_hits": chunk_cache_hits,
            "blocks_total": sum(len(notes.blocks) for notes in note_chunks),
            "board_snapshots_total": sum(
                is_board_snapshot(block) for block in _all_blocks(note_chunks)
            ),
            "correction_patches_applied": correction_patches_applied,
            "visual_requests_processed": visual_requests_processed,
            "visual_evidence_successful": visual_evidence_successful,
            "corrections_total": sum(len(notes.corrections) for notes in note_chunks),
            "unresolved_total": sum(len(notes.unresolved) for notes in note_chunks),
            "llm_usage": LectureModelClient.combine_usage(llm_usages),
        },
    )
    return ir
