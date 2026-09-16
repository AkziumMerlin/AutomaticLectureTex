from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from .knowledge_integrity import IntegrityKnowledgeOrchestrator
from .schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    TranscriptSegment,
    VisualEvidence,
    WindowObservations,
)
from .util import stable_hash

logger = logging.getLogger(__name__)


class ResilientIntegrityKnowledgeOrchestrator(IntegrityKnowledgeOrchestrator):
    """Keep malformed structured output local to the affected reconstruction window.

    The underlying client already retries malformed/truncated JSON and increases its bounded output
    budget. If those retries are exhausted, a lecture-sized semantic window can still be too hard for
    the backend to serialize reliably. Split only the ASR evidence, preserve the same bounded visual
    and knowledge context, and retry each half independently. A one-segment leaf is recorded as
    unresolved rather than aborting the full lecture run.
    """

    def extract_observations(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        kb: LectureKnowledgeBase,
    ) -> WindowObservations:
        return self._extract_resilient(chunk, evidence, kb, parent_window_id=chunk.id)

    def _extract_resilient(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        kb: LectureKnowledgeBase,
        *,
        parent_window_id: str,
    ) -> WindowObservations:
        try:
            result = super().extract_observations(chunk, evidence, kb)
        except (json.JSONDecodeError, ValidationError) as exc:
            if len(chunk.segment_ids) <= 1:
                segment_label = chunk.segment_ids[0] if chunk.segment_ids else "<empty>"
                logger.warning(
                    "[%s] semantic reconstruction leaf unresolved after structured-output failure: %s",
                    chunk.id,
                    exc,
                )
                return WindowObservations(
                    window_id=parent_window_id,
                    start=chunk.start,
                    end=chunk.end,
                    observations=[],
                    unresolved=[
                        "Semantic reconstruction structured output remained invalid for "
                        f"segment {segment_label}: {type(exc).__name__}: {exc}"
                    ],
                )

            left, right = self._split_chunk(chunk)
            logger.warning(
                "[%s] structured semantic reconstruction failed after client retries; "
                "splitting %d ASR segments into %d + %d",
                chunk.id,
                len(chunk.segment_ids),
                len(left.segment_ids),
                len(right.segment_ids),
            )
            first = self._extract_resilient(
                left,
                evidence,
                kb,
                parent_window_id=parent_window_id,
            )
            second = self._extract_resilient(
                right,
                evidence,
                kb,
                parent_window_id=parent_window_id,
            )
            observations = [*first.observations, *second.observations]
            for observation in observations:
                observation.window_id = parent_window_id
                observation.window_ids = [parent_window_id]
            return WindowObservations(
                window_id=parent_window_id,
                start=chunk.start,
                end=chunk.end,
                observations=observations,
                unresolved=list(dict.fromkeys([*first.unresolved, *second.unresolved])),
            )

        if chunk.id != parent_window_id:
            for observation in result.observations:
                observation.window_id = parent_window_id
                observation.window_ids = [parent_window_id]
            result.window_id = parent_window_id
        return result

    def _split_chunk(self, chunk: LectureChunk) -> tuple[LectureChunk, LectureChunk]:
        midpoint = len(chunk.segment_ids) // 2
        left_ids = chunk.segment_ids[:midpoint]
        right_ids = chunk.segment_ids[midpoint:]
        return (
            self._chunk_from_segment_ids(chunk, left_ids, "a"),
            self._chunk_from_segment_ids(chunk, right_ids, "b"),
        )

    def _chunk_from_segment_ids(
        self,
        parent: LectureChunk,
        segment_ids: list[str],
        suffix: str,
    ) -> LectureChunk:
        allowed = set(segment_ids)
        segments: list[TranscriptSegment] = [
            segment for segment in self.transcript.segments if segment.id in allowed
        ]
        if not segments:
            return LectureChunk(
                id=f"{parent.id}__{suffix}_{stable_hash(segment_ids)[:8]}",
                start=parent.start,
                end=parent.end,
                segment_ids=list(segment_ids),
                text="",
                timestamped_text="",
            )
        text = "\n".join(segment.text for segment in segments)
        timestamped = "\n".join(
            f"[{segment.start:.3f}-{segment.end:.3f}] {segment.text}" for segment in segments
        )
        confidences = [segment.confidence for segment in segments if segment.confidence is not None]
        return LectureChunk(
            id=f"{parent.id}__{suffix}_{stable_hash(segment_ids)[:8]}",
            start=min(segment.start for segment in segments),
            end=max(segment.end for segment in segments),
            segment_ids=list(segment_ids),
            text=text,
            timestamped_text=timestamped,
            asr_confidence=(sum(confidences) / len(confidences) if confidences else None),
        )
