from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pydantic_core import ValidationError

from .chunking import chunk_transcript
from .linear_llm_policy import has_explicit_correction_signal
from .linear_notes import (
    GlobalLectureEditPlan,
    LinearCorrectionScan,
    LinearPatch,
    block_id,
    block_segment_ids,
    plan_global_lecture_edit,
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
from .tex_safety import (
    assert_balanced_math_delimiters,
    normalize_heading_math,
    normalize_math_spans,
)
from .util import atomic_json_dump, stable_hash

logger = logging.getLogger(__name__)

# Final-IR version. Chunk reconstruction keeps its own cache version so adding the global editor
# does not force expensive multimodal chunk recomputation.
LINEAR_PIPELINE_VERSION = 10
LINEAR_CHUNK_CACHE_VERSION = 6
GLOBAL_LECTURE_EDITOR_VERSION = 4


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
            "version": LINEAR_CHUNK_CACHE_VERSION,
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


def _apply_global_edit_plan(
    draft_ir: LectureIR,
    plan: GlobalLectureEditPlan,
    *,
    apply_threshold: float,
) -> LectureIR:
    """Apply a compact global edit plan while preventing silent block loss or invention."""

    block_map: dict[str, NoteBlock] = {}
    owner_map: dict[str, ChunkNotes] = {}
    for chunk in draft_ir.chunks:
        for block in chunk.blocks:
            stable_id = block_id(block)
            if not stable_id:
                raise ValueError("global editor requires stable ids on every draft block")
            if stable_id in block_map:
                raise ValueError(f"duplicate draft block id {stable_id!r}")
            block_map[stable_id] = block
            owner_map[stable_id] = chunk

    applied_patches = {}
    dropped: set[str] = set()
    provenance_merges: dict[str, list[str]] = {}
    seen_patch_targets: set[str] = set()
    for patch in plan.patches:
        if patch.target_block_id not in block_map:
            raise ValueError(f"global edit references unknown block {patch.target_block_id!r}")
        if patch.target_block_id in seen_patch_targets:
            raise ValueError(f"multiple global edits target {patch.target_block_id!r}")
        seen_patch_targets.add(patch.target_block_id)
        if patch.confidence < apply_threshold:
            continue
        if patch.action == "drop":
            dropped.add(patch.target_block_id)
            if patch.merge_into_block_id is not None:
                if patch.merge_into_block_id not in block_map:
                    raise ValueError(
                        f"global provenance merge references unknown block {patch.merge_into_block_id!r}"
                    )
                provenance_merges.setdefault(patch.merge_into_block_id, []).append(
                    patch.target_block_id
                )
        else:
            applied_patches[patch.target_block_id] = patch

    ordered_ids = [stable_id for section in plan.sections for stable_id in section.block_ids]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("global edit plan assigns a block to multiple sections")
    unknown = set(ordered_ids) - set(block_map)
    if unknown:
        raise ValueError(f"global edit plan contains unknown block ids: {sorted(unknown)!r}")
    if dropped.intersection(ordered_ids):
        raise ValueError("global edit plan both drops and retains the same block")

    expected = set(block_map) - dropped
    retained = set(ordered_ids)
    if retained != expected:
        missing = sorted(expected - retained)
        extra = sorted(retained - expected)
        raise ValueError(
            f"global edit coverage mismatch; missing={missing[:8]!r}, extra={extra[:8]!r}"
        )

    for merge_target, merge_sources in provenance_merges.items():
        if merge_target in dropped:
            raise ValueError("global provenance merge target cannot itself be dropped")
        if merge_target not in retained:
            raise ValueError("global provenance merge target must be retained")
        for merge_source in merge_sources:
            if merge_source not in dropped:
                raise ValueError("global provenance merge source must be dropped")

    final_chunks: list[ChunkNotes] = []
    for section_index, section in enumerate(plan.sections):
        blocks: list[NoteBlock] = []
        corrections: list[CorrectionRecord] = []
        starts: list[float] = []
        ends: list[float] = []
        for stable_id in section.block_ids:
            source_block = block_map[stable_id]
            source_owner = owner_map[stable_id]
            block = source_block.model_copy(deep=True)
            starts.append(source_owner.start)
            ends.append(source_owner.end)
            merge_sources = provenance_merges.get(stable_id, [])
            if merge_sources:
                merged_segments = list(block_segment_ids(block))
                merged_evidence = list(block.source_evidence_ids)
                for merge_source_id in merge_sources:
                    duplicate = block_map[merge_source_id]
                    duplicate_owner = owner_map[merge_source_id]
                    starts.append(duplicate_owner.start)
                    ends.append(duplicate_owner.end)
                    for segment_id in block_segment_ids(duplicate):
                        if segment_id not in merged_segments:
                            merged_segments.append(segment_id)
                    for evidence_id in duplicate.source_evidence_ids:
                        if evidence_id not in merged_evidence:
                            merged_evidence.append(evidence_id)
                block.source_claim_ids = provenance_claim_ids(stable_id, merged_segments)
                block.source_evidence_ids = merged_evidence

            patch = applied_patches.get(stable_id)
            if patch is not None:
                original = block.latex
                replacement = (patch.replacement_latex or "").strip()
                checked = NoteBlock(
                    type=block.type,
                    title=block.title,
                    latex=replacement,
                    asset_path=block.asset_path,
                    caption=block.caption,
                    source_claim_ids=list(block.source_claim_ids),
                    source_evidence_ids=list(block.source_evidence_ids),
                )
                block.latex = checked.latex
                corrections.append(
                    CorrectionRecord(
                        original=original,
                        corrected=checked.latex,
                        reason=patch.reason,
                        basis="mathematical_consistency",
                        confidence=patch.confidence,
                    )
                )
            blocks.append(block)

        final_chunks.append(
            ChunkNotes(
                chunk_id=f"global_section_{section_index:03d}",
                start=min(starts) if starts else 0.0,
                end=max(ends) if ends else 0.0,
                section_title=section.title.strip(),
                blocks=blocks,
                corrections=corrections,
                unresolved=list(plan.unresolved) if section_index == 0 else [],
            )
        )

    return LectureIR(
        lecture_id=draft_ir.lecture_id,
        title=draft_ir.title,
        chunks=final_chunks,
    )


def _sanitize_final_ir_tex(ir: LectureIR) -> LectureIR:
    """Normalize common model TeX damage and reject residual delimiter corruption."""

    result = ir.model_copy(deep=True)
    result.title = normalize_heading_math(result.title)
    assert_balanced_math_delimiters(result.title)
    for chunk in result.chunks:
        chunk.section_title = normalize_heading_math(chunk.section_title)
        assert_balanced_math_delimiters(chunk.section_title)
        for block in chunk.blocks:
            block.latex = normalize_math_spans(block.latex).strip()
            assert_balanced_math_delimiters(block.latex)
            if block.title:
                block.title = normalize_heading_math(block.title)
                assert_balanced_math_delimiters(block.title)
            if block.caption:
                block.caption = normalize_heading_math(block.caption)
                assert_balanced_math_delimiters(block.caption)
    return result


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

    draft_ir = LectureIR(
        lecture_id=lecture.id,
        title=lecture.title or lecture.id,
        chunks=note_chunks,
    )
    atomic_json_dump(work / "draft_lecture_ir.json", draft_ir.model_dump(mode="json"))

    global_edit_seconds = 0.0
    global_edit_cache_hit = False
    global_edit_applied = False
    exact_dedup_blocks = 0
    reconciled_blocks = 0
    ir = draft_ir
    if config.global_validation and _all_blocks(note_chunks):
        global_path = work / "global_lecture_edit.json"
        global_fingerprint = stable_hash(
            {
                "version": GLOBAL_LECTURE_EDITOR_VERSION,
                "draft_ir": draft_ir.model_dump(mode="json"),
                "llm": pipeline.config.llm.model_dump(mode="json"),
                "apply_threshold": config.global_validation_apply_threshold,
                "batch_chars": config.linear_global_editor_batch_chars,
                "catalog_excerpt_chars": config.linear_global_editor_catalog_excerpt_chars,
                "course_conventions": config.linear_global_editor_conventions,
            }
        )
        global_plan = None
        if global_path.exists() and not force:
            try:
                payload = json.loads(global_path.read_text(encoding="utf-8"))
                if payload.get("fingerprint") == global_fingerprint:
                    global_plan = GlobalLectureEditPlan.model_validate(payload["plan"])
                    global_edit_cache_hit = True
                    llm_usages.append(payload.get("llm_usage", {}))
            except (json.JSONDecodeError, ValueError):
                global_plan = None

        if global_plan is None:
            usage_before = pipeline.llm.usage_snapshot()
            started = time.perf_counter()
            try:
                global_plan = plan_global_lecture_edit(
                    pipeline.llm,
                    draft_ir=draft_ir,
                    output_language=pipeline.config.llm.output_language,
                    apply_threshold=config.global_validation_apply_threshold,
                    batch_chars=config.linear_global_editor_batch_chars,
                    catalog_excerpt_chars=config.linear_global_editor_catalog_excerpt_chars,
                    course_conventions=config.linear_global_editor_conventions,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                logger.warning("[%s] global lecture editor failed: %s", lecture.id, exc)
                global_plan = None
            global_edit_seconds += time.perf_counter() - started
            usage = LectureModelClient.usage_delta(pipeline.llm.usage_snapshot(), usage_before)
            llm_usages.append(usage)
            if global_plan is not None:
                atomic_json_dump(
                    global_path,
                    {
                        "fingerprint": global_fingerprint,
                        "plan": global_plan.model_dump(mode="json"),
                        "llm_usage": usage,
                    },
                )

        if global_plan is not None:
            exact_dedup_blocks = sum(
                patch.action == "drop" and patch.merge_kind == "exact_dedup"
                for patch in global_plan.patches
            )
            reconciled_blocks = sum(
                patch.action == "drop" and patch.merge_kind == "reconciliation"
                for patch in global_plan.patches
            )
            try:
                ir = _apply_global_edit_plan(
                    draft_ir,
                    global_plan,
                    apply_threshold=config.global_validation_apply_threshold,
                )
                global_edit_applied = True
            except (ValueError, ValidationError) as exc:
                logger.warning("[%s] global lecture edit plan rejected: %s", lecture.id, exc)
                ir = draft_ir

    ir = _sanitize_final_ir_tex(ir)
    atomic_json_dump(ir_path, ir.model_dump(mode="json"))
    pipeline._save_notation_registry(notation)
    manifest["ir_fingerprint"] = pipeline._ir_fingerprint(transcript, notation)
    atomic_json_dump(manifest_path, manifest)
    atomic_json_dump(
        work / "run_metrics.json",
        {
            "lecture_id": lecture.id,
            "architecture": "linear_multimodal_with_global_editor",
            "linear_pipeline_version": LINEAR_PIPELINE_VERSION,
            "media_seconds": round(media_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "notes_seconds": round(time.perf_counter() - notes_started, 3),
            "vision_seconds": round(visual_seconds, 3),
            "finalize_and_math_audit_seconds": round(finalize_seconds, 3),
            "finalize_contextless_retries": finalize_contextless_retries,
            "finalize_failures": finalize_failures,
            "correction_scan_seconds": round(correction_seconds, 3),
            "global_edit_seconds": round(global_edit_seconds, 3),
            "global_edit_cache_hit": global_edit_cache_hit,
            "global_edit_applied": global_edit_applied,
            "exact_dedup_blocks": exact_dedup_blocks,
            "reconciled_blocks": reconciled_blocks,
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "chunks_total": len(chunks),
            "chunks_processed": processed_chunks,
            "chunk_cache_hits": chunk_cache_hits,
            "draft_blocks_total": sum(len(notes.blocks) for notes in note_chunks),
            "final_sections_total": len(ir.chunks),
            "blocks_total": sum(len(notes.blocks) for notes in ir.chunks),
            "board_snapshots_total": sum(
                is_board_snapshot(block) for block in _all_blocks(ir.chunks)
            ),
            "correction_patches_applied": correction_patches_applied,
            "visual_requests_processed": visual_requests_processed,
            "visual_evidence_successful": visual_evidence_successful,
            "corrections_total": sum(len(notes.corrections) for notes in ir.chunks),
            "unresolved_total": sum(len(notes.unresolved) for notes in ir.chunks),
            "llm_usage": LectureModelClient.combine_usage(llm_usages),
        },
    )
    return ir
