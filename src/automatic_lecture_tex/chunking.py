from __future__ import annotations

from .schemas import LectureChunk, Transcript


def chunk_transcript(transcript: Transcript, target_seconds: float) -> list[LectureChunk]:
    if not transcript.segments:
        return []
    chunks: list[LectureChunk] = []
    current = []
    chunk_start = transcript.segments[0].start

    def flush() -> None:
        nonlocal current, chunk_start
        if not current:
            return
        chunks.append(
            LectureChunk(
                id=f"chunk_{len(chunks):04d}",
                start=chunk_start,
                end=current[-1].end,
                segment_ids=[s.id for s in current],
                text="\n".join(s.text for s in current if s.text),
            )
        )
        current = []

    for segment in transcript.segments:
        if current and segment.end - chunk_start > target_seconds:
            flush()
            chunk_start = segment.start
        current.append(segment)
    flush()
    return chunks
