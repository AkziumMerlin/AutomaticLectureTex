from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .asr import GigaAMBackend, _extract_audio_chunk
from .schemas import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


def _split_interval(start: float, end: float, max_seconds: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    cursor = start
    while end - cursor > max_seconds:
        result.append((cursor, cursor + max_seconds))
        cursor += max_seconds
    if end - cursor > 0.05:
        result.append((cursor, end))
    return result


def merge_speech_intervals(
    intervals: list[tuple[float, float]],
    *,
    duration: float,
    max_seconds: float,
    merge_gap_seconds: float,
    pad_seconds: float,
) -> list[tuple[float, float]]:
    """Pad and merge VAD utterances without exceeding GigaAM's short-form limit."""

    padded: list[tuple[float, float]] = []
    for start, end in intervals:
        start = max(0.0, float(start) - pad_seconds)
        end = min(duration, float(end) + pad_seconds)
        if end <= start:
            continue
        padded.extend(_split_interval(start, end, max_seconds))

    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        gap = max(0.0, start - previous_end)
        candidate_end = max(previous_end, end)
        if gap <= merge_gap_seconds and candidate_end - previous_start <= max_seconds:
            merged[-1] = (previous_start, candidate_end)
        else:
            merged.append((start, end))
    return merged


class VadGigaAMBackend(GigaAMBackend):
    """GigaAM short-form ASR driven by faster-whisper's standalone Silero VAD.

    No Whisper acoustic model is instantiated. Faster-whisper is used only for 16 kHz decoding and
    its built-in Silero VAD, so cuts follow speech/silence boundaries instead of arbitrary 24-second
    wall-clock boundaries.
    """

    def _speech_intervals(self, audio_path: Path, duration: float) -> list[tuple[float, float]]:
        try:
            from faster_whisper.audio import decode_audio
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError as exc:
            raise RuntimeError(
                "VAD-aware GigaAM requires faster-whisper for standalone Silero VAD. "
                "Install with: pip install -e '.[whisper]'"
            ) from exc

        max_seconds = min(
            float(self.config.gigaam_vad_max_speech_seconds),
            self._MAX_SHORTFORM_SECONDS,
        )
        audio = decode_audio(str(audio_path), sampling_rate=_SAMPLE_RATE)
        options = VadOptions(
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            max_speech_duration_s=max(1.0, max_seconds - 2 * self.config.gigaam_vad_pad_seconds),
            speech_pad_ms=0,
        )
        speech = get_speech_timestamps(audio, options)
        raw = [
            (float(item["start"]) / _SAMPLE_RATE, float(item["end"]) / _SAMPLE_RATE)
            for item in speech
        ]
        return merge_speech_intervals(
            raw,
            duration=duration,
            max_seconds=max_seconds,
            merge_gap_seconds=self.config.gigaam_vad_merge_gap_seconds,
            pad_seconds=self.config.gigaam_vad_pad_seconds,
        )

    def transcribe(self, lecture_id: str, audio_path: Path) -> Transcript:
        from .media import probe_duration

        duration = probe_duration(audio_path, self.runtime)
        intervals = self._speech_intervals(audio_path, duration)
        if not intervals:
            logger.warning(
                "[%s] Silero VAD returned no speech; falling back to the legacy bounded GigaAM path",
                lecture_id,
            )
            return super().transcribe(lecture_id, audio_path)

        segments: list[TranscriptSegment] = []
        with tempfile.TemporaryDirectory(prefix="automatic-lecture-tex-gigaam-vad-") as tmp_name:
            tmp = Path(tmp_name)
            for source_index, (start, end) in enumerate(intervals):
                length = end - start
                if length <= 0.05:
                    continue
                chunk_path = tmp / f"utterance_{source_index:05d}.wav"
                _extract_audio_chunk(self.runtime, audio_path, start, length, chunk_path)
                result = self.model.transcribe(str(chunk_path), word_timestamps=True)
                text = str(result.text).strip()
                if not text:
                    continue
                words = self._shift_words(getattr(result, "words", None), start)
                seg_start = words[0].start if words else start
                seg_end = words[-1].end if words else end
                segments.append(
                    TranscriptSegment(
                        id=f"seg_{len(segments):05d}",
                        start=seg_start,
                        end=seg_end,
                        text=text,
                        words=words,
                    )
                )

        return Transcript(
            lecture_id=lecture_id,
            language=self.config.language or "ru",
            segments=segments,
        )
