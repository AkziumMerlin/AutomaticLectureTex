from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from .episode_graph import apply_episode_tracking
from .knowledge_integrity import IntegrityKnowledgeOrchestrator
from .llm import StructuredTaskTooLargeError
from .schemas import (
    EpisodeTrackingUpdate,
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
        return self._extract_resilient(
            chunk,
            evidence,
            kb,
            parent_window_id=chunk.id,
            allow_visual_compaction=True,
        )

    def track_episodes(
        self,
        kb: LectureKnowledgeBase,
        batch: WindowObservations,
        added_observation_ids: list[str],
    ) -> EpisodeTrackingUpdate:
        return self._track_episodes_resilient(kb, batch, added_observation_ids)

    def _track_episodes_resilient(
        self,
        kb: LectureKnowledgeBase,
        batch: WindowObservations,
        added_observation_ids: list[str],
    ) -> EpisodeTrackingUpdate:
        try:
            return super().track_episodes(kb, batch, added_observation_ids)
        except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
            if len(added_observation_ids) <= 1:
                observation_id = (
                    added_observation_ids[0] if added_observation_ids else "<none>"
                )
                logger.warning(
                    "[%s] episode tracking leaf unresolved for %s: %s",
                    batch.window_id,
                    observation_id,
                    exc,
                )
                return EpisodeTrackingUpdate(
                    unresolved=[
                        "Episode tracking failed for indivisible canonical observation "
                        f"{observation_id}: {type(exc).__name__}: {exc}"
                    ]
                )

            midpoint = len(added_observation_ids) // 2
            left_ids = added_observation_ids[:midpoint]
            right_ids = added_observation_ids[midpoint:]
            logger.warning(
                "[%s] episode tracking task too large/invalid; splitting %d observations "
                "into %d + %d",
                batch.window_id,
                len(added_observation_ids),
                len(left_ids),
                len(right_ids),
            )

            # Track the right half against a shadow state containing the left-half structural
            # decisions. The real KB is mutated only once by the caller with the merged update.
            shadow = kb.model_copy(deep=True)
            left = self._track_episodes_resilient(shadow, batch, left_ids)
            apply_episode_tracking(
                shadow,
                left,
                left_ids,
                window_id=batch.window_id,
            )
            right = self._track_episodes_resilient(shadow, batch, right_ids)

            return EpisodeTrackingUpdate(
                boundaries=[*left.boundaries, *right.boundaries],
                close_after_observation_ids=list(
                    dict.fromkeys(
                        [
                            *left.close_after_observation_ids,
                            *right.close_after_observation_ids,
                        ]
                    )
                ),
                symbols=[*left.symbols, *right.symbols],
                unresolved=list(
                    dict.fromkeys([*left.unresolved, *right.unresolved])
                ),
            )

    def _extract_resilient(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
        kb: LectureKnowledgeBase,
        *,
        parent_window_id: str,
        allow_visual_compaction: bool,
    ) -> WindowObservations:
        try:
            result = super().extract_observations(chunk, evidence, kb)
        except (json.JSONDecodeError, ValidationError, StructuredTaskTooLargeError) as exc:
            if len(chunk.segment_ids) <= 1:
                if allow_visual_compaction and self._visual_complexity(evidence) > 1:
                    compact = self._compact_visual_evidence(chunk, evidence)
                    logger.warning(
                        "[%s] semantic reconstruction leaf still oversized/invalid with one ASR "
                        "segment; compacting visual evidence from %d to %d sensor atoms",
                        chunk.id,
                        self._visual_complexity(evidence),
                        self._visual_complexity(compact),
                    )
                    return self._extract_resilient(
                        chunk,
                        compact,
                        kb,
                        parent_window_id=parent_window_id,
                        allow_visual_compaction=False,
                    )
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
            left_evidence = self._slice_visual_evidence(left, evidence)
            right_evidence = self._slice_visual_evidence(right, evidence)
            first = self._extract_resilient(
                left,
                left_evidence,
                kb,
                parent_window_id=parent_window_id,
                allow_visual_compaction=True,
            )
            second = self._extract_resilient(
                right,
                right_evidence,
                kb,
                parent_window_id=parent_window_id,
                allow_visual_compaction=True,
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

    @staticmethod
    def _visual_complexity(evidence: list[VisualEvidence]) -> int:
        return sum(
            len(item.frame_paths)
            + len(item.formula_crops)
            + len(item.math_ocr_candidates)
            + (1 if item.formula_contact_sheet_path else 0)
            for item in evidence
        )

    @staticmethod
    def _nearest_index(values: list[float], midpoint: float) -> int | None:
        if not values:
            return None
        return min(range(len(values)), key=lambda index: abs(values[index] - midpoint))

    def _slice_visual_evidence(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
    ) -> list[VisualEvidence]:
        """Keep only sensor atoms temporally local to one reconstruction child."""

        midpoint = (chunk.start + chunk.end) / 2
        sliced: list[VisualEvidence] = []
        for item in evidence:
            frame_pairs = [
                (path, item.frame_timestamps[index])
                for index, path in enumerate(item.frame_paths)
                if index < len(item.frame_timestamps)
            ]
            local_frames = [
                pair for pair in frame_pairs if chunk.start <= pair[1] <= chunk.end
            ]
            if not local_frames and frame_pairs:
                nearest = min(frame_pairs, key=lambda pair: abs(pair[1] - midpoint))
                local_frames = [nearest]

            local_crops = [
                crop
                for crop in item.formula_crops
                if chunk.start <= crop.timestamp <= chunk.end
            ]
            if not local_crops and item.formula_crops:
                local_crops = [
                    min(
                        item.formula_crops,
                        key=lambda crop: abs(crop.timestamp - midpoint),
                    )
                ]
            crop_ids = {crop.id for crop in local_crops}

            local_ocr = [
                candidate
                for candidate in item.math_ocr_candidates
                if (
                    candidate.source_id in crop_ids
                    or (
                        candidate.timestamp is not None
                        and chunk.start <= candidate.timestamp <= chunk.end
                    )
                )
            ]
            if not local_ocr and item.math_ocr_candidates:
                timestamped = [
                    candidate
                    for candidate in item.math_ocr_candidates
                    if candidate.timestamp is not None
                ]
                if timestamped:
                    local_ocr = [
                        min(
                            timestamped,
                            key=lambda candidate: abs(
                                float(candidate.timestamp) - midpoint
                            ),
                        )
                    ]

            sliced.append(
                item.model_copy(
                    deep=True,
                    update={
                        "frame_paths": [pair[0] for pair in local_frames],
                        "frame_timestamps": [pair[1] for pair in local_frames],
                        "formula_crops": local_crops,
                        "math_ocr_candidates": local_ocr,
                        # Parent contact sheets mix formula regions from both temporal halves.
                        "formula_contact_sheet_path": None,
                        "best_frame_index": None,
                    },
                )
            )
        return sliced

    def _compact_visual_evidence(
        self,
        chunk: LectureChunk,
        evidence: list[VisualEvidence],
    ) -> list[VisualEvidence]:
        """Reduce a one-segment leaf to one board state and one formula/OCR hypothesis."""

        midpoint = (chunk.start + chunk.end) / 2
        compact: list[VisualEvidence] = []
        for item in evidence:
            frame_pairs = [
                (path, item.frame_timestamps[index])
                for index, path in enumerate(item.frame_paths)
                if index < len(item.frame_timestamps)
            ]
            if frame_pairs:
                frame_pairs = [
                    min(frame_pairs, key=lambda pair: abs(pair[1] - midpoint))
                ]

            crops = list(item.formula_crops)
            if crops:
                crops = [
                    min(crops, key=lambda crop: abs(crop.timestamp - midpoint))
                ]
            crop_ids = {crop.id for crop in crops}

            ocr = [
                candidate
                for candidate in item.math_ocr_candidates
                if candidate.source_id in crop_ids
            ]
            if not ocr:
                timestamped = [
                    candidate
                    for candidate in item.math_ocr_candidates
                    if candidate.timestamp is not None
                ]
                if timestamped:
                    ocr = [
                        min(
                            timestamped,
                            key=lambda candidate: abs(
                                float(candidate.timestamp) - midpoint
                            ),
                        )
                    ]

            compact.append(
                item.model_copy(
                    deep=True,
                    update={
                        "frame_paths": [pair[0] for pair in frame_pairs],
                        "frame_timestamps": [pair[1] for pair in frame_pairs],
                        "formula_crops": crops,
                        "math_ocr_candidates": ocr[:1],
                        "formula_contact_sheet_path": None,
                        "best_frame_index": None,
                    },
                )
            )
        return compact

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
