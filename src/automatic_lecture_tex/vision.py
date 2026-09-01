from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import LectureChunk, Transcript, VisualRequest


_RULES: list[tuple[re.Pattern[str], str, str, int]] = [
    (
        re.compile(r"\b(как видно|на рисунк|на картинк|на слайд|на доске)\w*", re.I),
        "explicit_visual_reference",
        "Extract the referenced visual information that is necessary for the notes.",
        5,
    ),
    (
        re.compile(r"\b(вот этот|вот эта|вот это|здесь|сюда|отсюда|слева|справа)\b", re.I),
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


def select_rule_based_visual_requests(
    chunk: LectureChunk,
    transcript: Transcript,
) -> list[VisualRequest]:
    segment_by_id = {segment.id: segment for segment in transcript.segments}
    requests: list[VisualRequest] = []
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
        if segment.confidence is not None and segment.confidence < 0.55:
            requests.append(
                VisualRequest(
                    id=f"asr_{segment.id}_{len(requests):02d}",
                    timestamp=(segment.start + segment.end) / 2,
                    reason="low_asr_confidence",
                    question="Use the board or slide to resolve mathematical terms or notation missed by ASR.",
                    priority=4,
                )
            )
    return requests


def dedupe_visual_requests(
    requests: Iterable[VisualRequest],
    within_seconds: float,
    limit: int,
) -> list[VisualRequest]:
    ordered = sorted(requests, key=lambda r: (-r.priority, r.timestamp, r.id))
    selected: list[VisualRequest] = []
    for request in ordered:
        if any(abs(request.timestamp - existing.timestamp) <= within_seconds for existing in selected):
            continue
        selected.append(request)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda r: r.timestamp)
