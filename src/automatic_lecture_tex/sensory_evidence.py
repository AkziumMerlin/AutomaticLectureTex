from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .board import build_temporal_board_composite, temporal_sample_offsets
from .board_crop import generate_board_crops
from .frame_selection import select_board_state_frames, select_least_occluded_frame
from .math_ocr import make_math_ocr_backend
from .media import copy_asset
from .omni import make_omni_backend
from .schemas import ExtractedFrame, MathOCRCandidate, VisualEvidence, VisualKind
from .vision import (
    CHUNK_BOARD_SCAN_REASON,
    dedupe_visual_requests,
    make_chunk_board_scan_request,
    namespace_visual_requests,
    board_change_probe_times,
    select_rule_based_visual_requests,
    uniform_chunk_sample_times,
)

if TYPE_CHECKING:
    from .config import LectureConfig
    from .pipeline import Pipeline
    from .schemas import LectureChunk, Transcript

logger = logging.getLogger(__name__)


def _unique_times(values: list[float]) -> list[float]:
    result: list[float] = []
    seen: set[int] = set()
    for value in values:
        value = max(0.0, float(value))
        key = round(value * 1000)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _omni_backend(pipeline: Pipeline):
    config = getattr(pipeline.config, "omni", None)
    if config is None or not config.enabled:
        return None
    if not getattr(pipeline, "_omni_backend_initialized", False):
        pipeline._omni_backend = make_omni_backend(
            config,
            output_language=pipeline.config.llm.output_language,
        )
        pipeline._omni_backend_initialized = True
    return pipeline._omni_backend


def _math_ocr_backend(pipeline: Pipeline):
    if not getattr(pipeline, "_math_ocr_backend_initialized", False):
        pipeline._math_ocr_backend = make_math_ocr_backend(pipeline.config.vision.math_ocr)
        pipeline._math_ocr_backend_initialized = True
    return pipeline._math_ocr_backend


def _append_unique(frames: list[ExtractedFrame], frame: ExtractedFrame) -> None:
    if all(frame.path != existing.path for existing in frames):
        frames.append(frame)


def _prefer_board_views(
    raw_views: list[ExtractedFrame],
    board_views: list[ExtractedFrame],
    *,
    limit: int,
) -> list[ExtractedFrame]:
    """Keep one full-context frame, then spend the image budget on readable board crops."""

    selected: list[ExtractedFrame] = []
    if raw_views:
        _append_unique(selected, raw_views[0])
    for frame in board_views:
        if len(selected) >= limit:
            break
        _append_unique(selected, frame)
    for frame in raw_views[1:]:
        if len(selected) >= limit:
            break
        _append_unique(selected, frame)
    return selected[:limit]


def collect_visual_evidence(
    pipeline: Pipeline,
    lecture: LectureConfig,
    chunk: LectureChunk,
    transcript: Transcript,
    source: Any,
    work: Path,
    figures_root: Path,
    notation: dict[str, str],
) -> tuple[list, list[VisualEvidence], float]:
    """Collect visual sensor inputs and OCR evidence.

    The mandatory whole-chunk board scan is deliberately *not* OCR'd by a separate VLM call. Its
    selected board-state frames are retained as raw multimodal inputs for semantic reconstruction.
    Supplemental local visual requests keep the legacy isolated OCR path. This avoids compressing the
    main board channel into an intermediate text representation before reconstruction.
    """

    scan_requests = []
    if (
        getattr(pipeline.config.notes, "visual_chunk_board_scan", False)
        and pipeline.config.vision.max_requests_per_chunk > 0
    ):
        scan_requests.append(make_chunk_board_scan_request(chunk))

    local_requests = []
    if pipeline.config.notes.visual_rule_selector:
        local_requests.extend(
            select_rule_based_visual_requests(
                chunk,
                transcript,
                pipeline.config.notes.max_low_confidence_visual_requests,
            )
        )
    if pipeline.config.notes.visual_llm_selector:
        local_requests.extend(pipeline.llm.analyze_chunk(chunk, notation).visual_requests)

    local_requests = dedupe_visual_requests(
        local_requests,
        within_seconds=pipeline.config.notes.visual_dedupe_seconds,
        limit=max(0, pipeline.config.vision.max_requests_per_chunk - len(scan_requests)),
    )
    requests = namespace_visual_requests(chunk.id, [*scan_requests, *local_requests])

    started = time.perf_counter()
    ocr_backend = _math_ocr_backend(pipeline)
    evidence: list[VisualEvidence] = []
    prepared_visuals: list[
        tuple[Any, list[ExtractedFrame], list[ExtractedFrame], list[MathOCRCandidate]]
    ] = []

    for request in requests:
        is_chunk_board_scan = request.reason == CHUNK_BOARD_SCAN_REASON
        if is_chunk_board_scan:
            if pipeline.config.vision.board_sampling_mode == "change":
                raw_times = board_change_probe_times(
                    chunk,
                    probe_seconds=pipeline.config.vision.board_change_probe_seconds,
                    max_probe_frames=pipeline.config.vision.board_change_max_probe_frames,
                )
            else:
                sample_count = min(
                    pipeline.config.vision.board_uniform_samples,
                    pipeline.config.vision.board_crop_max_vlm_images,
                )
                raw_times = uniform_chunk_sample_times(chunk, sample_count)
        else:
            raw_times = _unique_times(
                [request.timestamp + offset for offset in pipeline.config.vision.frame_offsets_seconds]
            )

        temporal_times: list[float] = []
        if pipeline.config.vision.temporal_composite_enabled and not is_chunk_board_scan:
            temporal_times = _unique_times(
                [
                    request.timestamp + offset
                    for offset in temporal_sample_offsets(pipeline.config.vision)
                ]
            )

        all_times = [*temporal_times]
        raw_indices: list[int] = []
        for value in raw_times:
            key = round(value * 1000)
            existing = next(
                (
                    index
                    for index, item in enumerate(all_times)
                    if round(item * 1000) == key
                ),
                None,
            )
            if existing is None:
                raw_indices.append(len(all_times))
                all_times.append(value)
            else:
                raw_indices.append(existing)

        frame_dir = work / "frames" / request.id
        try:
            frames = source.extract_frames(all_times, frame_dir)
        except Exception as exc:
            logger.warning(
                "[%s] frame extraction failed for %s; continuing without this visual evidence: %s",
                lecture.id,
                request.id,
                exc,
            )
            evidence.append(
                VisualEvidence(
                    request_id=request.id,
                    description=(
                        "Visual frame extraction unavailable; semantic reconstruction must rely on "
                        f"other evidence. {type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue

        raw_views: list[ExtractedFrame] = []
        ocr_image: Path | None = None

        if temporal_times and len(temporal_times) >= 3:
            temporal_frames = frames[: len(temporal_times)]
            try:
                composite_path = build_temporal_board_composite(
                    temporal_frames,
                    frame_dir / "board_composite.jpg",
                )
                primary = select_least_occluded_frame(
                    temporal_frames,
                    composite_path,
                    target_timestamp=request.timestamp,
                )
                _append_unique(raw_views, primary)
                _append_unique(
                    raw_views,
                    ExtractedFrame(timestamp=request.timestamp, path=composite_path),
                )
                ocr_image = primary.path
            except Exception as exc:
                logger.warning(
                    "[%s] temporal board reconstruction/selection failed for %s: %s",
                    lecture.id,
                    request.id,
                    exc,
                )

        raw_frames = [frames[index] for index in raw_indices]
        if is_chunk_board_scan:
            if pipeline.config.vision.board_sampling_mode == "change":
                raw_views = select_board_state_frames(
                    raw_frames,
                    max_states=pipeline.config.vision.board_crop_max_vlm_images,
                    change_threshold=pipeline.config.vision.board_change_threshold,
                    min_gap_seconds=pipeline.config.vision.board_change_min_gap_seconds,
                )
            else:
                raw_views = raw_frames[: pipeline.config.vision.board_crop_max_vlm_images]
        else:
            raw_frames.sort(key=lambda frame: abs(frame.timestamp - request.timestamp))
            for frame in raw_frames:
                if len(raw_views) >= 4:
                    break
                _append_unique(raw_views, frame)

        if not raw_views:
            if is_chunk_board_scan:
                raw_views = frames[: pipeline.config.vision.board_crop_max_vlm_images]
            else:
                raw_views = sorted(
                    frames, key=lambda frame: abs(frame.timestamp - request.timestamp)
                )[:4]
        if ocr_image is None and raw_views:
            ocr_image = raw_views[0].path

        board_views: list[ExtractedFrame] = []
        if is_chunk_board_scan:
            display_frames: list[ExtractedFrame] = []
            for index, frame in enumerate(raw_views):
                selected = frame
                if pipeline.config.vision.board_auto_crop_enabled:
                    try:
                        crop_result = generate_board_crops(
                            frame,
                            frame_dir / "board_crops" / f"state_{index:02d}",
                            pipeline.config.vision,
                        )
                        if crop_result is not None and crop_result.frames:
                            selected = crop_result.frames[0]
                            board_views.append(selected)
                    except Exception as exc:
                        logger.warning(
                            "[%s] board auto-crop failed for %s state %d; using raw frame: %s",
                            lecture.id,
                            request.id,
                            index,
                            exc,
                        )
                display_frames.append(selected)
            display_frames = display_frames[: pipeline.config.vision.board_crop_max_vlm_images]
            if display_frames:
                ocr_image = display_frames[0].path
        else:
            if raw_views and pipeline.config.vision.board_auto_crop_enabled:
                try:
                    crop_result = generate_board_crops(
                        raw_views[0],
                        frame_dir / "board_crops",
                        pipeline.config.vision,
                    )
                    if crop_result is not None:
                        board_views = crop_result.frames
                        # Specialized OCR benefits from the board-only full crop as well.
                        if board_views:
                            ocr_image = board_views[0].path
                except Exception as exc:
                    logger.warning(
                        "[%s] board auto-crop failed for %s; using raw frames: %s",
                        lecture.id,
                        request.id,
                        exc,
                    )

            display_frames = _prefer_board_views(
                raw_views,
                board_views,
                limit=pipeline.config.vision.board_crop_max_vlm_images,
            )

        candidates: list[MathOCRCandidate] = []
        if not is_chunk_board_scan and ocr_backend is not None and ocr_image is not None:
            try:
                candidate = ocr_backend.recognize(ocr_image)
                if candidate is not None and (
                    candidate.confidence is None
                    or candidate.confidence >= pipeline.config.vision.math_ocr.min_confidence
                ):
                    candidates.append(candidate)
            except Exception as exc:
                logger.warning(
                    "[%s] specialized math OCR failed for %s: %s",
                    lecture.id,
                    request.id,
                    exc,
                )

        prepared_visuals.append((request, display_frames, board_views, candidates))

    if prepared_visuals:
        workers = min(pipeline.config.vision.max_workers, len(prepared_visuals))
        futures: list[Future[VisualEvidence] | None] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for request, frames, _board_views, _candidates in prepared_visuals:
                if request.reason == CHUNK_BOARD_SCAN_REASON:
                    # The board-state scan is a raw sensor bundle. Do not spend a separate VLM call
                    # converting it to OCR before the multimodal writer sees the images.
                    futures.append(None)
                    continue
                futures.append(
                    executor.submit(
                        pipeline.llm.resolve_visual_request,
                        request,
                        chunk,
                        [frame.path for frame in frames],
                        [frame.timestamp for frame in frames],
                    )
                )

            for (request, frames, board_views, candidates), future in zip(
                prepared_visuals, futures, strict=True
            ):
                if future is None:
                    visual = VisualEvidence(
                        request_id=request.id,
                        kind=VisualKind.BOARD_SCAN,
                        description=(
                            "Chronologically ordered board states selected by the configured "
                            "host-side sampler and attached directly to reconstruction; no "
                            "intermediate VLM OCR was performed."
                        ),
                        confidence=1.0,
                        frame_paths=[str(frame.path) for frame in frames],
                        frame_timestamps=[frame.timestamp for frame in frames],
                    )
                else:
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
                            description=f"Visual OCR failed after retries: {type(exc).__name__}: {exc}",
                        )
                    visual.math_ocr_candidates = candidates

                # Persist one wide board state only as a possible unresolved-content fallback.
                # The selected scan frames remain working sensor inputs and are not figures
                # in the final lecture unless a fallback explicitly references this asset.
                asset_frame: ExtractedFrame | None = None
                if request.reason == CHUNK_BOARD_SCAN_REASON and frames:
                    asset_frame = frames[len(frames) // 2]
                elif board_views and visual.kind != "none":
                    asset_frame = board_views[0]
                elif visual.requires_figure_in_notes and frames:
                    index = visual.best_frame_index if visual.best_frame_index is not None else 0
                    index = max(0, min(index, len(frames) - 1))
                    asset_frame = frames[index]

                if asset_frame is not None:
                    destination = figures_root / f"{request.id}_board.jpg"
                    copy_asset(asset_frame.path, destination)
                    visual.asset_path = str(destination.relative_to(pipeline.config.latex.output_dir))
                evidence.append(visual)

    omni_backend = _omni_backend(pipeline)
    if omni_backend is not None:
        request_id = f"{chunk.id}_omni_av"
        clip_path = work / "omni_clips" / f"{chunk.id}.mp4"
        try:
            source.extract_clip(chunk.start, chunk.end, clip_path)
            description = omni_backend.analyze(clip_path)
            evidence.append(
                VisualEvidence(
                    request_id=request_id,
                    kind=VisualKind.AUDIO_VIDEO,
                    description=description,
                    # This is a generated sensor interpretation, not a calibrated confidence score.
                    confidence=0.5,
                )
            )
        except Exception as exc:
            logger.warning(
                "[%s] native AV evidence failed for %s; continuing with ASR/board evidence: %s",
                lecture.id,
                chunk.id,
                exc,
            )
            evidence.append(
                VisualEvidence(
                    request_id=request_id,
                    kind=VisualKind.AUDIO_VIDEO,
                    description=(
                        "Native audio-video sensor unavailable for this window: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    confidence=0.0,
                )
            )

    return requests, evidence, time.perf_counter() - started
