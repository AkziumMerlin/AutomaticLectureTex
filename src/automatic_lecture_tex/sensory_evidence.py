from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .board import build_temporal_board_composite, temporal_sample_offsets
from .board_crop import generate_board_crops
from .formula_detection import (
    DetectedFormulaCrop,
    build_formula_contact_sheet,
    detect_temporal_formula_crops,
    make_formula_detector,
    merge_formula_crops,
    split_oversized_formula_crops,
)
from .frame_selection import select_board_state_frames, select_least_occluded_frame
from .math_ocr import make_math_ocr_backend
from .media import copy_asset
from .schemas import (
    ExtractedFrame,
    FormulaVisualCrop,
    MathOCRCandidate,
    VisualEvidence,
    VisualKind,
)
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


def _math_ocr_backend(pipeline: Pipeline):
    if not getattr(pipeline, "_math_ocr_backend_initialized", False):
        pipeline._math_ocr_backend = make_math_ocr_backend(
            pipeline.config.vision.math_ocr,
            pipeline.config.llm,
        )
        pipeline._math_ocr_backend_initialized = True
    return pipeline._math_ocr_backend


def _formula_detector(pipeline: Pipeline):
    if not getattr(pipeline, "_formula_detector_initialized", False):
        pipeline._formula_detector = make_formula_detector(
            pipeline.config.vision.formula_detection
        )
        pipeline._formula_detector_initialized = True
    return pipeline._formula_detector


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


def _select_diverse_formula_crops(
    crops: list[DetectedFormulaCrop],
    *,
    limit: int,
) -> list[DetectedFormulaCrop]:
    """Spend crop budget across board states before adding same-state extras."""

    if limit <= 0 or not crops:
        return []
    groups: dict[int, list[DetectedFormulaCrop]] = {}
    for crop in crops:
        key = round(crop.frame.timestamp * 1000)
        groups.setdefault(key, []).append(crop)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                "_t" in item.id,  # MFD/line-refined proposals before temporal attention proposals.
                -item.confidence,
            )
        )

    selected: list[DetectedFormulaCrop] = []
    selected_ids: set[str] = set()
    # First pass: one strongest formula per chronological board state.
    for key in sorted(groups):
        crop = groups[key][0]
        selected.append(crop)
        selected_ids.add(crop.id)
        if len(selected) >= limit:
            return selected

    # Second pass: fill remaining capacity with the strongest unused detections globally.
    remaining = sorted(
        (crop for crop in crops if crop.id not in selected_ids),
        key=lambda item: (
            "_t" in item.id,
            -item.confidence,
        ),
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return sorted(
        selected,
        key=lambda item: (
            item.frame.timestamp,
            item.bbox[1],
            item.bbox[0],
        ),
    )


def _subsample_ocr_frames(
    frames: list[ExtractedFrame],
    *,
    limit: int,
) -> list[ExtractedFrame]:
    """Keep chronological coverage while bounding expensive specialized OCR calls."""

    if limit <= 0 or not frames:
        return []
    if len(frames) <= limit:
        return list(frames)
    if limit == 1:
        return [frames[-1]]

    indices = [
        round(index * (len(frames) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [frames[index] for index in dict.fromkeys(indices)]


def _run_math_ocr(
    backend,
    inputs: list[tuple[str, ExtractedFrame]],
    *,
    min_confidence: float,
    lecture_id: str,
    request_id: str,
) -> list[MathOCRCandidate]:
    candidates: list[MathOCRCandidate] = []
    for source_id, frame in inputs:
        try:
            candidate = backend.recognize(frame.path)
        except Exception as exc:
            logger.warning(
                "[%s] specialized math OCR failed for %s/%s at %.3fs: %s",
                lecture_id,
                request_id,
                source_id,
                frame.timestamp,
                exc,
            )
            continue
        if candidate is None:
            continue
        if candidate.confidence is not None and candidate.confidence < min_confidence:
            continue
        candidates.append(
            candidate.model_copy(
                update={"timestamp": frame.timestamp, "source_id": source_id}
            )
        )
    return candidates


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

    The mandatory whole-chunk board scan is deliberately *not* OCR'd by a separate general VLM
    call. Its selected board-state frames remain raw multimodal inputs for semantic reconstruction,
    while an optional specialized image-to-LaTeX backend may attach literal formula candidates.
    Supplemental local visual requests keep the legacy isolated VLM OCR path. This preserves the
    raw board channel instead of replacing it with an intermediate textual interpretation.
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
    formula_detector = _formula_detector(pipeline)
    evidence: list[VisualEvidence] = []
    prepared_visuals: list[
        tuple[
            Any,
            list[ExtractedFrame],
            list[ExtractedFrame],
            list[MathOCRCandidate],
            list[DetectedFormulaCrop],
            Path | None,
        ]
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
        formula_crops: list[DetectedFormulaCrop] = []
        formula_contact_sheet: Path | None = None

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
                if formula_detector is not None:
                    try:
                        state_dir = frame_dir / "formula_crops" / f"state_{index:02d}"
                        detected = formula_detector.detect(
                            selected,
                            state_dir,
                            id_prefix=f"{request.id}_s{index:02d}",
                        )
                        detected = split_oversized_formula_crops(
                            detected,
                            state_dir / "line_crops",
                            pipeline.config.vision.formula_detection,
                        )
                        formula_crops.extend(detected)
                    except Exception as exc:
                        logger.warning(
                            "[%s] formula detection failed for %s state %d: %s",
                            lecture.id,
                            request.id,
                            index,
                            exc,
                        )

            if (
                formula_detector is not None
                and pipeline.config.vision.formula_detection.temporal_proposals_enabled
                and len(display_frames) >= 3
            ):
                try:
                    temporal_dir = frame_dir / "formula_crops" / "temporal"
                    temporal_crops = detect_temporal_formula_crops(
                        display_frames,
                        temporal_dir,
                        pipeline.config.vision.formula_detection,
                        id_prefix=request.id,
                    )
                    temporal_crops = split_oversized_formula_crops(
                        temporal_crops,
                        temporal_dir / "line_crops",
                        pipeline.config.vision.formula_detection,
                    )
                    formula_crops.extend(temporal_crops)
                except Exception as exc:
                    logger.warning(
                        "[%s] temporal formula proposal extraction failed for %s: %s",
                        lecture.id,
                        request.id,
                        exc,
                    )

            formula_crops = merge_formula_crops(formula_crops)
            display_frames = display_frames[: pipeline.config.vision.board_crop_max_vlm_images]
            if display_frames:
                ocr_image = display_frames[0].path

            if formula_crops:
                max_crops = pipeline.config.vision.formula_detection.max_crops_per_chunk
                formula_crops = _select_diverse_formula_crops(
                    formula_crops,
                    limit=max_crops,
                )
                if pipeline.config.vision.formula_detection.contact_sheet_enabled:
                    formula_contact_sheet = build_formula_contact_sheet(
                        formula_crops,
                        frame_dir / "formula_crops" / "contact_sheet.jpg",
                        columns=pipeline.config.vision.formula_detection.contact_sheet_columns,
                        max_items=max_crops,
                    )
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
        if ocr_backend is not None:
            if (
                is_chunk_board_scan
                and pipeline.config.vision.math_ocr.board_scan_enabled
            ):
                selected_crops = _select_diverse_formula_crops(
                    formula_crops,
                    limit=pipeline.config.vision.math_ocr.board_scan_max_images,
                )
                ocr_inputs = [(item.id, item.frame) for item in selected_crops]
                if ocr_inputs:
                    candidates = _run_math_ocr(
                        ocr_backend,
                        ocr_inputs,
                        min_confidence=pipeline.config.vision.math_ocr.min_confidence,
                        lecture_id=lecture.id,
                        request_id=request.id,
                    )
            elif not is_chunk_board_scan and ocr_image is not None:
                candidates = _run_math_ocr(
                    ocr_backend,
                    [
                        (
                            f"{request.id}_local",
                            ExtractedFrame(timestamp=request.timestamp, path=ocr_image),
                        )
                    ],
                    min_confidence=pipeline.config.vision.math_ocr.min_confidence,
                    lecture_id=lecture.id,
                    request_id=request.id,
                )

        prepared_visuals.append(
            (
                request,
                display_frames,
                board_views,
                candidates,
                formula_crops,
                formula_contact_sheet,
            )
        )

    if prepared_visuals:
        workers = min(pipeline.config.vision.max_workers, len(prepared_visuals))
        futures: list[Future[VisualEvidence] | None] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for (
                request,
                frames,
                _board_views,
                _candidates,
                _formula_crops,
                _formula_contact_sheet,
            ) in prepared_visuals:
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

            for (
                request,
                frames,
                board_views,
                candidates,
                formula_crops,
                formula_contact_sheet,
            ), future in zip(prepared_visuals, futures, strict=True):
                if future is None:
                    visual = VisualEvidence(
                        request_id=request.id,
                        kind=VisualKind.BOARD_SCAN,
                        description=(
                            "Chronologically ordered board states selected by the configured "
                            "host-side sampler and attached directly to reconstruction. "
                            "Specialized formula OCR candidates, when present, are literal "
                            "transcription hypotheses rather than semantic corrections."
                        ),
                        confidence=1.0,
                        frame_paths=[str(frame.path) for frame in frames],
                        frame_timestamps=[frame.timestamp for frame in frames],
                        math_ocr_candidates=candidates,
                        formula_crops=[
                            FormulaVisualCrop(
                                id=item.id,
                                timestamp=item.frame.timestamp,
                                bbox=item.bbox,
                                detector_confidence=item.confidence,
                                image_path=str(item.frame.path),
                            )
                            for item in formula_crops
                        ],
                        formula_contact_sheet_path=(
                            str(formula_contact_sheet)
                            if formula_contact_sheet is not None
                            else None
                        ),
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

    return requests, evidence, time.perf_counter() - started
