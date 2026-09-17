from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .board import build_temporal_board_composite, temporal_sample_offsets
from .board_crop import generate_board_crops
from .frame_selection import select_least_occluded_frame
from .math_ocr import make_math_ocr_backend
from .media import copy_asset
from .schemas import ExtractedFrame, MathOCRCandidate, VisualEvidence
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
    """Collect literal visual evidence, preferring high-resolution board crops when available.

    The VLM always retains one uncropped frame for context. Host-side board detection then contributes
    a full board ROI plus overlapping horizontal tiles, up to the configured image budget. If crop
    detection fails, behavior falls back to the existing raw-frame path. The best crop is also copied
    into the TeX figure tree so an unresolved mathematical fragment can be represented by source
    evidence rather than an invented reconstruction.
    """

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
    ocr_backend = _math_ocr_backend(pipeline)
    evidence: list[VisualEvidence] = []
    prepared_visuals: list[
        tuple[Any, list[ExtractedFrame], list[ExtractedFrame], list[MathOCRCandidate]]
    ] = []

    for request in requests:
        raw_times = _unique_times(
            [request.timestamp + offset for offset in pipeline.config.vision.frame_offsets_seconds]
        )
        temporal_times: list[float] = []
        if pipeline.config.vision.temporal_composite_enabled:
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
        raw_frames.sort(key=lambda frame: abs(frame.timestamp - request.timestamp))
        for frame in raw_frames:
            if len(raw_views) >= 4:
                break
            _append_unique(raw_views, frame)

        if not raw_views:
            raw_views = sorted(frames, key=lambda frame: abs(frame.timestamp - request.timestamp))[:4]
        if ocr_image is None and raw_views:
            ocr_image = raw_views[0].path

        board_views: list[ExtractedFrame] = []
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
        if ocr_backend is not None and ocr_image is not None:
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
        futures: list[Future[VisualEvidence]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for request, frames, _board_views, _candidates in prepared_visuals:
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

                # Preserve a readable board crop for both ordinary figure requests and unresolved
                # fallbacks. Prefer the VLM-selected crop when it selected one; otherwise keep the
                # full board ROI. Raw frames remain the fallback for non-board diagrams/slides.
                asset_frame: ExtractedFrame | None = None
                if board_views and visual.kind != "none":
                    index = visual.best_frame_index if visual.best_frame_index is not None else 0
                    index = max(0, min(index, len(frames) - 1)) if frames else 0
                    selected = frames[index] if frames else None
                    if selected is not None and any(selected.path == item.path for item in board_views):
                        asset_frame = selected
                    else:
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

    return requests, evidence, time.perf_counter() - started
