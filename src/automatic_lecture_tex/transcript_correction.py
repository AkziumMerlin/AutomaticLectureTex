from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .asr import ASRBackend, make_asr_backend
from .config import RuntimeConfig, TranscriptCorrectionConfig
from .schemas import Transcript, TranscriptSegment
from .util import run_checked

logger = logging.getLogger(__name__)

TRANSCRIPT_CORRECTION_CACHE_VERSION = 1


class SegmentCorrection(BaseModel):
    id: str
    status: Literal["unchanged", "reconstructed", "ambiguous"]
    corrected_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    uncertain_spans: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def require_text_for_reconstruction(self) -> SegmentCorrection:
        if self.status == "reconstructed" and not (self.corrected_text or "").strip():
            raise ValueError("reconstructed transcript segment requires corrected_text")
        return self


class TranscriptCorrectionBatch(BaseModel):
    segments: list[SegmentCorrection]


class TranscriptCorrectionResult(BaseModel):
    transcript: Transcript
    audit: dict
    metrics: dict


class TranscriptReconstructor:
    def __init__(
        self,
        llm,
        config: TranscriptCorrectionConfig,
        runtime: RuntimeConfig,
        *,
        course_title: str,
        language: str,
        hotwords: list[str],
    ) -> None:
        self.llm = llm
        self.config = config
        self.runtime = runtime
        self.course_title = course_title
        self.language = language
        self.glossary = list(dict.fromkeys([*hotwords, *config.glossary]))
        self._fallback: ASRBackend | None = None
        self._llm_calls = 0
        self._fallback_spans = 0

    def _target_batches(self, transcript: Transcript) -> list[list[TranscriptSegment]]:
        batches: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        for segment in transcript.segments:
            if current:
                duration = segment.end - current[0].start
                if (
                    duration > self.config.window_seconds
                    or len(current) >= self.config.max_segments_per_batch
                ):
                    batches.append(current)
                    current = []
            current.append(segment)
        if current:
            batches.append(current)
        return batches

    def _context_segments(
        self,
        transcript: Transcript,
        targets: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        start = targets[0].start - self.config.context_seconds
        end = targets[-1].end + self.config.context_seconds
        target_ids = {item.id for item in targets}
        return [
            item
            for item in transcript.segments
            if item.id not in target_ids and item.end >= start and item.start <= end
        ]

    @staticmethod
    def _segment_payload(segment: TranscriptSegment) -> dict:
        return {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "raw_text": segment.text,
            "asr_confidence": segment.confidence,
        }

    def _correct_batch(
        self,
        transcript: Transcript,
        targets: list[TranscriptSegment],
        *,
        previous: dict[str, SegmentCorrection] | None = None,
        alternate_asr: dict[str, str] | None = None,
    ) -> dict[str, SegmentCorrection]:
        target_ids = [item.id for item in targets]
        context = self._context_segments(transcript, targets)
        previous = previous or {}
        alternate_asr = alternate_asr or {}
        payload = {
            "targets": [self._segment_payload(item) for item in targets],
            "context": [self._segment_payload(item) for item in context],
            "previous_reconstruction": {
                item_id: item.model_dump(mode="json")
                for item_id, item in previous.items()
                if item_id in target_ids
            },
            "alternate_asr": {
                item_id: text for item_id, text in alternate_asr.items() if item_id in target_ids
            },
        }
        prompt = f"""Reconstruct noisy ASR from ONE bounded window of a university lecture.

Course: {self.course_title}
Language: {self.language}
Course glossary / likely terminology:
{json.dumps(self.glossary, ensure_ascii=False)}

Window payload:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

Return exactly one item for every target segment id, in the same order. Never invent or remove ids.
You may use neighboring context and an alternate ASR hypothesis only to resolve what was actually
said. Correct phonetic ASR errors, malformed technical terms, punctuation, grammar, and obvious
word-boundary errors. Preserve the lecturer's mathematical claim even if it is false; do NOT replace
it with textbook knowledge and do NOT add missing mathematics merely because it would be standard.

Use status:
- unchanged: raw_text is already usable; corrected_text may be omitted;
- reconstructed: the intended wording is locally recoverable; return complete corrected_text;
- ambiguous: the audio hypotheses/context do not determine a safe reconstruction. Do not guess.

A semantically natural sentence is not enough evidence for reconstruction: it must remain plausible
from the raw phonetics, neighboring speech, glossary, or alternate ASR. Confidence measures the
reconstruction itself, not whether the mathematical statement is true. Put genuinely uncertain
words/phrases in uncertain_spans. Write corrected prose in {self.language}.
"""
        self._llm_calls += 1
        result = self.llm._structured(  # noqa: SLF001
            prompt,
            TranscriptCorrectionBatch,
            operation="transcript_reconstruction",
            max_tokens=4096,
        )
        returned_ids = [item.id for item in result.segments]
        if returned_ids != target_ids:
            raise ValidationError.from_exception_data(
                "TranscriptCorrectionBatch",
                [
                    {
                        "type": "value_error",
                        "loc": ("segments",),
                        "input": returned_ids,
                        "ctx": {
                            "error": ValueError(
                                f"expected exact target ids {target_ids}, got {returned_ids}"
                            )
                        },
                    }
                ],
            )
        return {item.id: item for item in result.segments}

    def _first_pass(self, transcript: Transcript) -> dict[str, SegmentCorrection]:
        corrections: dict[str, SegmentCorrection] = {}
        for targets in self._target_batches(transcript):
            try:
                corrections.update(self._correct_batch(transcript, targets))
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    "transcript reconstruction failed for %s..%s: %s",
                    targets[0].id,
                    targets[-1].id,
                    exc,
                )
                for segment in targets:
                    corrections[segment.id] = SegmentCorrection(
                        id=segment.id,
                        status="ambiguous",
                        confidence=0.0,
                        reason=f"structured transcript reconstruction failed: {type(exc).__name__}",
                    )
        return corrections

    def _suspicious_ids(
        self,
        transcript: Transcript,
        corrections: dict[str, SegmentCorrection],
    ) -> set[str]:
        suspicious: set[str] = set()
        for segment in transcript.segments:
            correction = corrections[segment.id]
            if correction.status == "ambiguous":
                suspicious.add(segment.id)
                continue
            if correction.confidence < self.config.reconstruction_confidence_threshold:
                suspicious.add(segment.id)
                continue
            if (
                segment.confidence is not None
                and segment.confidence < self.config.suspicious_asr_confidence
            ):
                suspicious.add(segment.id)
        return suspicious

    def _fallback_groups(
        self,
        transcript: Transcript,
        suspicious_ids: set[str],
    ) -> list[list[TranscriptSegment]]:
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        for segment in transcript.segments:
            if segment.id not in suspicious_ids:
                if current:
                    groups.append(current)
                    current = []
                continue
            if current and segment.end - current[0].start > self.config.fallback_max_group_seconds:
                groups.append(current)
                current = []
            current.append(segment)
        if current:
            groups.append(current)
        return groups

    def _fallback_backend(self) -> ASRBackend:
        if self._fallback is None:
            if self.config.fallback_asr is None:
                raise RuntimeError("fallback ASR requested without fallback_asr config")
            self._fallback = make_asr_backend(self.config.fallback_asr, self.runtime)
        return self._fallback

    def _extract_audio_span(self, audio_path: Path, start: float, end: float, output: Path) -> None:
        run_checked(
            [
                self.runtime.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{max(0.1, end - start):.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ]
        )

    @staticmethod
    def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
        return max(0.0, min(left_end, right_end) - max(left_start, right_start))

    def _fallback_hypotheses(
        self,
        transcript: Transcript,
        suspicious_ids: set[str],
        audio_path: Path,
    ) -> dict[str, str]:
        if not suspicious_ids or not self.config.fallback_enabled:
            return {}
        backend = self._fallback_backend()
        hypotheses: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="automatic-lecture-tex-fallback-asr-") as tmp_name:
            tmp = Path(tmp_name)
            for group_index, group in enumerate(self._fallback_groups(transcript, suspicious_ids)):
                span_start = max(0.0, group[0].start - self.config.fallback_context_seconds)
                span_end = group[-1].end + self.config.fallback_context_seconds
                path = tmp / f"span_{group_index:04d}.wav"
                self._extract_audio_span(audio_path, span_start, span_end, path)
                self._fallback_spans += 1
                fallback = backend.transcribe(f"fallback_{group_index:04d}", path)
                shifted = [
                    (
                        item.start + span_start,
                        item.end + span_start,
                        item.text,
                    )
                    for item in fallback.segments
                    if item.text.strip()
                ]
                for target in group:
                    pieces = [
                        text
                        for start, end, text in shifted
                        if self._overlap(start, end, target.start, target.end) > 0.0
                    ]
                    if pieces:
                        hypotheses[target.id] = " ".join(pieces).strip()
        return hypotheses

    def _second_pass(
        self,
        transcript: Transcript,
        corrections: dict[str, SegmentCorrection],
        suspicious_ids: set[str],
        alternate_asr: dict[str, str],
    ) -> dict[str, SegmentCorrection]:
        if not suspicious_ids:
            return corrections
        by_id = {item.id: item for item in transcript.segments}
        for batch in self._target_batches(transcript):
            targets = [item for item in batch if item.id in suspicious_ids]
            if not targets:
                continue
            previous = {item.id: corrections[item.id] for item in targets}
            try:
                revised = self._correct_batch(
                    transcript,
                    targets,
                    previous=previous,
                    alternate_asr=alternate_asr,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    "second-pass transcript reconstruction failed for %s: %s",
                    [item.id for item in targets],
                    exc,
                )
                continue
            for item_id, proposal in revised.items():
                if item_id in by_id:
                    corrections[item_id] = proposal
        return corrections

    def reconstruct(self, transcript: Transcript, audio_path: Path) -> TranscriptCorrectionResult:
        first = self._first_pass(transcript)
        suspicious = self._suspicious_ids(transcript, first)
        alternate = self._fallback_hypotheses(transcript, suspicious, audio_path)
        corrections = self._second_pass(transcript, first, suspicious, alternate)

        corrected_segments: list[TranscriptSegment] = []
        audit_segments: list[dict] = []
        for segment in transcript.segments:
            proposal = corrections[segment.id]
            corrected_text = segment.text
            corrected_confidence = segment.confidence
            if proposal.status == "reconstructed" and (proposal.corrected_text or "").strip():
                corrected_text = proposal.corrected_text.strip()
                corrected_confidence = proposal.confidence
            elif proposal.status == "ambiguous":
                raw_confidence = segment.confidence if segment.confidence is not None else 1.0
                corrected_confidence = min(raw_confidence, self.config.ambiguous_segment_confidence)

            corrected_segments.append(
                segment.model_copy(
                    update={
                        "text": corrected_text,
                        "confidence": corrected_confidence,
                    },
                    deep=True,
                )
            )
            audit_segments.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "raw_text": segment.text,
                    "corrected_text": corrected_text,
                    "raw_confidence": segment.confidence,
                    "corrected_confidence": corrected_confidence,
                    "status": proposal.status,
                    "reconstruction_confidence": proposal.confidence,
                    "uncertain_spans": proposal.uncertain_spans,
                    "reason": proposal.reason,
                    "alternate_asr": alternate.get(segment.id),
                }
            )

        counts = {status: 0 for status in ("unchanged", "reconstructed", "ambiguous")}
        for item in corrections.values():
            counts[item.status] += 1
        metrics = {
            "llm_calls": self._llm_calls,
            "fallback_spans": self._fallback_spans,
            "suspicious_segments": len(suspicious),
            **{f"segments_{key}": value for key, value in counts.items()},
        }
        return TranscriptCorrectionResult(
            transcript=Transcript(
                lecture_id=transcript.lecture_id,
                language=transcript.language,
                segments=corrected_segments,
            ),
            audit={
                "version": TRANSCRIPT_CORRECTION_CACHE_VERSION,
                "segments": audit_segments,
            },
            metrics=metrics,
        )


def reconstruct_transcript(
    transcript: Transcript,
    *,
    llm,
    config: TranscriptCorrectionConfig,
    runtime: RuntimeConfig,
    course_title: str,
    language: str,
    hotwords: Iterable[str],
    audio_path: Path,
) -> TranscriptCorrectionResult:
    reconstructor = TranscriptReconstructor(
        llm,
        config,
        runtime,
        course_title=course_title,
        language=language,
        hotwords=list(hotwords),
    )
    return reconstructor.reconstruct(transcript, audio_path)
