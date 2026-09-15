from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .board import build_temporal_board_composite, temporal_sample_offsets
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
    """Collect visual evidence with temporal board reconstruction before VLM OCR.

    The VLM sees a robust temporal-median board image first, followed by the original sparse frames.
    An optional specialized OCR backend reads the same composite independently; its output is stored
    as a hypothesis rather than replacing literal VLM evidence.
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
    prepared_visuals: list[
        tuple[Any, list[ExtractedFrame], list[MathOCRCandidate]]
    ] = []

    for request in requests:
        raw_times = _unique_times(
            [
                request.timestamp + offset
                for offset in pipeline.config.vision.frame_offsets_seconds
            ]
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
        frames = source.extract_frames(all_times, frame_dir)
        display_frames: list[ExtractedFrame] = []
        ocr_image: Path | None = None

        if temporal_times and len(temporal_times) >= 3:
            temporal_frames = frames[: len(temporal_times)]
            try:
                composite_path = build_temporal_board_composite(
                    temporal_frames,
                    frame_dir / "board_composite.jpg",
                )
                composite = ExtractedFrame(timestamp=request.timestamp, path=composite_path)
                display_frames.append(composite)
                ocr_image = composite_path
            except Exception as exc:
                logger.warning(
                    "[%s] temporal board reconstruction failed for %s: %s",
                    lecture.id,
                    request.id,
                    exc,
                )

        for index in raw_indices:
            frame = frames[index]
            if all(frame.path != existing.path for existing in display_frames):
                display_frames.append(frame)
        if not display_frames:
            display_frames = frames
        if ocr_image is None and display_frames:
            ocr_image = display_frames[0].path

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

        prepared_visuals.append((request, display_frames, candidates))

    evidence: list[VisualEvidence] = []
    if prepared_visuals:
        workers = min(pipeline.config.vision.max_workers, len(prepared_visuals))
        futures: list[Future[VisualEvidence]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for request, frames, _candidates in prepared_visuals:
                futures.append(
                    executor.submit(
                        pipeline.llm.resolve_visual_request,
                        request,
                        chunk,
                        [frame.path for frame in frames],
                        [frame.timestamp for frame in frames],
                    )
                )
            for (request, frames, candidates), future in zip(
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
                        description=(
                            f"Visual OCR failed after retries: {type(exc).__name__}: {exc}"
                        ),
                    )
                visual.math_ocr_candidates = candidates
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
