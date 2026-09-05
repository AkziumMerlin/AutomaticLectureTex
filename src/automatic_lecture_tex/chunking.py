from __future__ import annotations

from .schemas import LectureChunk, Transcript
from .util import format_timestamp


def _make_chunk(segments, chunk_id: str) -> LectureChunk:
    confidence_values = [s.confidence for s in segments if s.confidence is not None]
    return LectureChunk(
        id=chunk_id,
        start=segments[0].start,
        end=segments[-1].end,
        segment_ids=[s.id for s in segments],
        text="\n".join(s.text for s in segments if s.text),
        timestamped_text="\n".join(
            f"[{format_timestamp(s.start)}-{format_timestamp(s.end)}] {s.text}"
            for s in segments
            if s.text
        ),
        asr_confidence=(
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
        low_confidence_fraction=(
            sum(value < 0.6 for value in confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
    )


def chunk_transcript(
    transcript: Transcript,
    target_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[LectureChunk]:
    """Split a transcript on ASR segment boundaries, optionally with temporal overlap.

    The overlap is evidence overlap, not a semantic boundary: a segment whose end falls inside the
    overlap interval is repeated in the next window. This deliberately favors duplicated evidence
    over losing a proof/correction that straddles a technical chunk boundary.
    """
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= target_seconds:
        raise ValueError("overlap_seconds must satisfy 0 <= overlap_seconds < target_seconds")
    if not transcript.segments:
        return []

    segments = transcript.segments
    chunks: list[LectureChunk] = []
    start_idx = 0

    while start_idx < len(segments):
        end_idx = start_idx
        while end_idx + 1 < len(segments):
            candidate = segments[end_idx + 1]
            if candidate.end - segments[start_idx].start > target_seconds:
                break
            end_idx += 1

        current = segments[start_idx : end_idx + 1]
        prefix = "chunk" if overlap_seconds == 0 else "window"
        chunks.append(_make_chunk(current, f"{prefix}_{len(chunks):04d}"))

        if end_idx == len(segments) - 1:
            break
        if overlap_seconds == 0:
            start_idx = end_idx + 1
            continue

        overlap_threshold = current[-1].end - overlap_seconds
        next_idx = end_idx + 1
        for idx in range(start_idx + 1, end_idx + 1):
            if segments[idx].end > overlap_threshold:
                next_idx = idx
                break

        # Always make progress, including the case of one very long ASR segment.
        start_idx = max(start_idx + 1, next_idx)

    return chunks
