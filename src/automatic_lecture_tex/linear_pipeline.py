from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pydantic_core import ValidationError

from .chunking import chunk_transcript
from .linear_notes import (
    LinearChunkDraft,
    LinearCorrectionScan,
    LinearPatch,
    block_id,
    draft_linear_chunk,
    provenance_claim_ids,
    scan_linear_corrections,
)
from .llm import LectureModelClient
from .schemas import ChunkNotes, CorrectionRecord, LectureIR, NoteBlock, Transcript, VisualEvidence
from .sensory_evidence import collect_visual_evidence
from .util import atomic_json_dump, stable_hash

logger = logging.getLogger(__name__)

LINEAR_PIPELINE_VERSION = 1


def _all_blocks(note_chunks: list[ChunkNotes]) -> list[NoteBlock]:
    return [block for notes in note_chunks for block in notes.blocks]


def _previous_transcript_tail(transcript: Transcript, chunk, count: int) -> list[dict]:
    if not chunk.segment_ids or count <= 0:
        return []
    index_by_id = {segment.id: index for index, segment in enumerate(transcript.segments)}
    first = index_by_id.get(chunk.segment_ids[0], 0)
    start = max(0, first - count)
    return [
        {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in transcript.segments[start:first]
    ]


def _recent_blocks(note_chunks: list[ChunkNotes], limit: int) -> list[NoteBlock]:
    blocks = _all_blocks(note_chunks)
    return blocks[-limit:] if limit > 0 else []


def _build_note_block(
    generated,
    *,
    stable_block_id: str,
    allowed_segment_ids: set[str],
    visual_by_id: dict[str, VisualEvidence],
) -> tuple[NoteBlock | None, str | None]:
    sources = [item for item in generated.source_segment_ids if item in allowed_segment_ids]
    if not sources:
        return None, f"Dropped {stable_block_id}: no valid CURRENT source_segment_ids."
    visual_ids = [item for item in generated.visual_evidence_ids if item in visual_by_id]
    asset_path = generated.asset_path
    if asset_path is not None:
        allowed_assets = {
            item.asset_path
            for item in visual_by_id.values()
            if item.asset_path is not None and item.request_id in visual_ids
        }
        if asset_path not in allowed_assets:
            return None, (
                f"Dropped {stable_block_id}: figure asset_path is not backed by cited visual evidence."
            )
    try:
        block = NoteBlock(
            type=generated.type,
            title=generated.title,
            latex=generated.latex,
            asset_path=asset_path,
            caption=generated.caption,
            source_claim_ids=provenance_claim_ids(stable_block_id, sources),
            source_evidence_ids=visual_ids,
        )
    except ValidationError as exc:
        return None, f"Dropped {stable_block_id}: invalid renderable block ({exc})."
    return block, None


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
    recent_blocks: list[NoteBlock],
    notation: dict[str, str],
    source_identity,
) -> str:
    return stable_hash(
        {
            "version": LINEAR_PIPELINE_VERSION,
            "chunk": chunk.model_dump(mode="json"),
            "recent_blocks": [block.model_dump(mode="json") for block in recent_blocks],
            "notation": notation,
            "source": source_identity,
            "notes": pipeline.config.notes.model_dump(mode="json"),
            "vision": pipeline.config.vision.model_dump(mode="json"),
            "llm": pipeline.config.llm.model_dump(mode="json"),
        }
    )


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
    """Chronological note writing plus explicit lecturer-correction patches.

    There is deliberately no canonical observation graph, episode graph, synthesis tree, formula
    gate, or mathematical rewrite validator in this path.
    """

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
    write_seconds = 0.0
    correction_seconds = 0.0
    visual_requests_processed = 0
    visual_evidence_successful = 0
    recent_patch_count = 0
    global_patch_count = 0
    llm_usages: list[dict] = []

    chunks_dir = work / "linear_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, chunk in enumerate(chunks):
        recent = _recent_blocks(note_chunks, config.linear_recent_blocks)
        artifact = chunks_dir / f"{chunk.id}.json"
        fingerprint = _chunk_cache_fingerprint(
            pipeline,
            chunk=chunk,
            recent_blocks=recent,
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
            draft = LinearChunkDraft.model_validate(cached["draft"])
            chunk_cache_hits += 1
            logger.info("[%s] %s linear cache hit", lecture.id, chunk.id)
            usage = cached.get("llm_usage", {})
            llm_usages.append(usage)
        else:
            logger.info(
                "[%s] linear write %s (%d/%d)",
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
            write_started = time.perf_counter()
            try:
                draft = draft_linear_chunk(
                    pipeline.llm,
                    chunk=chunk,
                    evidence=evidence,
                    known_notation=notation,
                    recent_blocks=recent,
                    previous_transcript_tail=_previous_transcript_tail(
                        transcript, chunk, config.linear_previous_transcript_segments
                    ),
                    output_language=pipeline.config.llm.output_language,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning("[%s] %s linear write failed: %s", lecture.id, chunk.id, exc)
                draft = LinearChunkDraft(
                    section_title=note_chunks[-1].section_title if note_chunks else "Без названия",
                    unresolved=[
                        f"Не удалось разобрать structured output для {chunk.id}; "
                        "фрагмент сохранён только в transcript evidence."
                    ],
                )
            write_seconds += time.perf_counter() - write_started
            usage = LectureModelClient.usage_delta(pipeline.llm.usage_snapshot(), usage_before)
            llm_usages.append(usage)
            processed_chunks += 1
            atomic_json_dump(
                artifact,
                {
                    "fingerprint": fingerprint,
                    "chunk": chunk.model_dump(mode="json"),
                    "visual_evidence": [item.model_dump(mode="json") for item in evidence],
                    "draft": draft.model_dump(mode="json"),
                    "llm_usage": usage,
                },
            )

        evidence_by_chunk[chunk.id] = evidence
        current_unresolved = list(draft.unresolved)
        recent_target_ids = {block_id(block) for block in recent if block_id(block)}
        current_segment_ids = set(chunk.segment_ids)
        for patch in draft.recent_patches:
            applied, issue = _apply_patch(
                patch,
                block_map=block_map,
                owner_map=owner_map,
                allowed_targets=recent_target_ids,
                allowed_evidence_ids=current_segment_ids,
                apply_threshold=config.linear_patch_apply_threshold,
            )
            if applied:
                recent_patch_count += 1
            if issue:
                current_unresolved.append(issue)

        visual_by_id = {item.request_id: item for item in evidence if item.request_id}
        notes = ChunkNotes(
            chunk_id=chunk.id,
            start=chunk.start,
            end=chunk.end,
            section_title=draft.section_title.replace("$", ""),
            blocks=[],
            notation=list(draft.notation),
            unresolved=current_unresolved,
        )
        for generated_index, generated in enumerate(draft.blocks):
            stable_block_id = f"block_{chunk_index:04d}_{generated_index:03d}"
            block, issue = _build_note_block(
                generated,
                stable_block_id=stable_block_id,
                allowed_segment_ids=current_segment_ids,
                visual_by_id=visual_by_id,
            )
            if issue:
                notes.unresolved.append(issue)
                continue
            assert block is not None
            notes.blocks.append(block)
            block_map[stable_block_id] = block
            owner_map[stable_block_id] = notes

        notes.unresolved = list(dict.fromkeys(notes.unresolved))
        note_chunks.append(notes)
        for item in notes.notation:
            notation.setdefault(item.latex, item.meaning)

    if config.linear_correction_scan_enabled:
        scan_dir = work / "linear_correction_scans"
        scan_dir.mkdir(parents=True, exist_ok=True)
        for chunk_index, chunk in enumerate(chunks):
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
                    global_patch_count += 1
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
            "architecture": "linear_correction",
            "linear_pipeline_version": LINEAR_PIPELINE_VERSION,
            "media_seconds": round(media_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "notes_seconds": round(time.perf_counter() - notes_started, 3),
            "vision_seconds": round(visual_seconds, 3),
            "linear_write_seconds": round(write_seconds, 3),
            "correction_scan_seconds": round(correction_seconds, 3),
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "chunks_total": len(chunks),
            "chunks_processed": processed_chunks,
            "chunk_cache_hits": chunk_cache_hits,
            "blocks_total": sum(len(notes.blocks) for notes in note_chunks),
            "recent_patches_applied": recent_patch_count,
            "global_patches_applied": global_patch_count,
            "visual_requests_processed": visual_requests_processed,
            "visual_evidence_successful": visual_evidence_successful,
            "corrections_total": sum(len(notes.corrections) for notes in note_chunks),
            "unresolved_total": sum(len(notes.unresolved) for notes in note_chunks),
            "llm_usage": LectureModelClient.combine_usage(llm_usages),
        },
    )
    return ir
