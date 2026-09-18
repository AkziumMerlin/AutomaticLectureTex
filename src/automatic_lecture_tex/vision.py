from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import LectureChunk, Transcript, VisualRequest

CHUNK_BOARD_SCAN_REASON = "chunk_board_scan"

_RULES: list[tuple[re.Pattern[str], str, str, int]] = [
    (
        re.compile(r"\b(как видно|на рисунк|на картинк|на слайд|на доске)\w*", re.I),
        "explicit_visual_reference",
        "Extract the referenced visual information that is necessary for the notes.",
        5,
    ),
    (
        re.compile(r"\b(вот этот|вот эта|вот это|сюда|слева|справа)\b", re.I),
        "deictic_reference",
        "Resolve what the lecturer is pointing to or referring to visually.",
        4,
    ),
    (
        re.compile(r"\b(обозначим|обозначается|запишем|перепишем)\b", re.I),
        "notation_may_be_visual",
        "Check exact mathematical notation written by the lecturer.",
        3,
    ),
    (
        re.compile(r"\b(нарисуем|изобразим|график|диаграмм|схем|стрелк)\w*", re.I),
        "diagram_or_graph",
        "Extract the diagram, graph, labels, arrows, and their mathematical meaning.",
        5,
    ),
]


def uniform_chunk_sample_times(chunk: LectureChunk, count: int) -> list[float]:
    """Return evenly spaced interior timestamps for a whole-chunk board scan."""

    if count <= 0:
        return []
    duration = max(0.0, chunk.end - chunk.start)
    if duration == 0.0:
        return [chunk.start]
    return [
        chunk.start + duration * (index + 0.5) / count
        for index in range(count)
    ]


def make_chunk_board_scan_request(chunk: LectureChunk) -> VisualRequest:
    """Create one multimodal request representing the board state across the whole chunk."""

    return VisualRequest(
        id="board_scan",
        timestamp=(chunk.start + chunk.end) / 2,
        reason=CHUNK_BOARD_SCAN_REASON,
        question=(
            "The attached frames are evenly sampled board states from this entire lecture chunk. "
            "Transcribe the mathematically relevant board content across them literally, preserving "
            "symbols, signs, indices, and visible changes between states. In raw_latex and latex, "
            "separate distinct states with Frame 0:, Frame 1:, etc. Do not infer missing steps."
        ),
        priority=5,
    )


def select_rule_based_visual_requests(
    chunk: LectureChunk,
    transcript: Transcript,
    max_low_confidence_requests: int = 2,
) -> list[VisualRequest]:
    segment_by_id = {segment.id: segment for segment in transcript.segments}
    requests: list[VisualRequest] = []
    low_confidence_segments = []
    for segment_id in chunk.segment_ids:
        segment = segment_by_id[segment_id]
        for pattern, reason, question, priority in _RULES:
            if pattern.search(segment.text):
                requests.append(
                    VisualRequest(
                        id=f"rule_{segment.id}_{len(requests):02d}",
                        timestamp=(segment.start + segment.end) / 2,
                        reason=reason,
                        question=question,
                        priority=priority,
                    )
                )
                break
        if segment.confidence is not None and segment.confidence < 0.6:
            low_confidence_segments.append(segment)
    low_confidence_segments.sort(key=lambda segment: (segment.confidence, segment.start))
    for segment in low_confidence_segments[:max_low_confidence_requests]:
        requests.append(
            VisualRequest(
                id=f"asr_{segment.id}_{len(requests):02d}",
                timestamp=(segment.start + segment.end) / 2,
                reason="low_asr_confidence",
                question=(
                    "Transcribe all clearly visible board or slide content near this moment, "
                    "especially exact formulas, variable names, and signs."
                ),
                priority=3,
            )
        )
    return requests


def dedupe_visual_requests(
    requests: Iterable[VisualRequest],
    within_seconds: float,
    limit: int,
) -> list[VisualRequest]:
    if limit <= 0:
        return []
    ordered = sorted(requests, key=lambda r: (-r.priority, r.timestamp, r.id))
    selected: list[VisualRequest] = []
    for request in ordered:
        if any(
            abs(request.timestamp - existing.timestamp) <= within_seconds for existing in selected
        ):
            continue
        selected.append(request)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda r: r.timestamp)


def namespace_visual_requests(
    chunk_id: str, requests: Iterable[VisualRequest]
) -> list[VisualRequest]:
    result: list[VisualRequest] = []
    for index, request in enumerate(requests):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.id).strip("._-") or "request"
        result.append(request.model_copy(update={"id": f"{chunk_id}_{safe_id}_{index:02d}"}))
    return result
