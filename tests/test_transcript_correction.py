from types import SimpleNamespace

import pytest

from automatic_lecture_tex.config import RuntimeConfig, TranscriptCorrectionConfig
from automatic_lecture_tex.correcting_asr import CorrectingASRBackend
from automatic_lecture_tex.schemas import Transcript, TranscriptSegment
from automatic_lecture_tex.transcript_correction import TranscriptReconstructor
from automatic_lecture_tex.util import atomic_json_dump


class _QueuedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        assert operation == "transcript_reconstruction"
        self.prompts.append(prompt)
        return schema.model_validate(self.responses.pop(0))


def _transcript() -> Transcript:
    return Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_00000",
                start=0.0,
                end=8.0,
                text="линейный функцеонал",
                confidence=0.91,
            ),
            TranscriptSegment(
                id="seg_00001",
                start=8.0,
                end=16.0,
                text="совсем непонятный кусок",
                confidence=0.35,
            ),
        ],
    )


def test_reconstruction_preserves_ids_and_timestamps_and_marks_ambiguity(tmp_path):
    llm = _QueuedLLM(
        [
            {
                "segments": [
                    {
                        "id": "seg_00000",
                        "status": "reconstructed",
                        "corrected_text": "линейный функционал",
                        "confidence": 0.97,
                    },
                    {
                        "id": "seg_00001",
                        "status": "ambiguous",
                        "confidence": 0.2,
                        "uncertain_spans": ["непонятный кусок"],
                    },
                ]
            },
            {
                "segments": [
                    {
                        "id": "seg_00001",
                        "status": "ambiguous",
                        "confidence": 0.25,
                        "uncertain_spans": ["непонятный кусок"],
                    }
                ]
            },
        ]
    )
    reconstructor = TranscriptReconstructor(
        llm,
        TranscriptCorrectionConfig(
            enabled=True,
            fallback_enabled=False,
            suspicious_asr_confidence=0.5,
            reconstruction_confidence_threshold=0.7,
            ambiguous_segment_confidence=0.2,
        ),
        RuntimeConfig(work_dir=tmp_path),
        course_title="Функциональный анализ",
        language="ru",
        hotwords=["линейный функционал"],
    )

    result = reconstructor.reconstruct(_transcript(), tmp_path / "audio.wav")

    first, second = result.transcript.segments
    assert first.id == "seg_00000"
    assert (first.start, first.end) == (0.0, 8.0)
    assert first.text == "линейный функционал"
    assert first.confidence == pytest.approx(0.97)
    assert second.id == "seg_00001"
    assert (second.start, second.end) == (8.0, 16.0)
    assert second.text == "совсем непонятный кусок"
    assert second.confidence == pytest.approx(0.2)
    assert result.metrics["segments_reconstructed"] == 1
    assert result.metrics["segments_ambiguous"] == 1


class _FakeFallback:
    def transcribe(self, lecture_id, audio_path):
        return Transcript(
            lecture_id=lecture_id,
            language="ru",
            segments=[
                TranscriptSegment(
                    id="fallback",
                    start=8.0,
                    end=18.0,
                    text="теорема Хана Банаха",
                    confidence=0.99,
                )
            ],
        )


def test_selective_fallback_is_reconciled_only_for_suspicious_segment(tmp_path, monkeypatch):
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_00000",
                start=100.0,
                end=110.0,
                text="теорема хана банана",
                confidence=0.3,
            )
        ],
    )
    llm = _QueuedLLM(
        [
            {
                "segments": [
                    {
                        "id": "seg_00000",
                        "status": "ambiguous",
                        "confidence": 0.3,
                    }
                ]
            },
            {
                "segments": [
                    {
                        "id": "seg_00000",
                        "status": "reconstructed",
                        "corrected_text": "теорема Хана — Банаха",
                        "confidence": 0.98,
                    }
                ]
            },
        ]
    )
    config = TranscriptCorrectionConfig(
        enabled=True,
        fallback_enabled=True,
        fallback_asr={"backend": "qwen3"},
        fallback_context_seconds=8.0,
    )
    reconstructor = TranscriptReconstructor(
        llm,
        config,
        RuntimeConfig(work_dir=tmp_path),
        course_title="Функциональный анализ",
        language="ru",
        hotwords=[],
    )
    monkeypatch.setattr(reconstructor, "_fallback_backend", lambda: _FakeFallback())
    monkeypatch.setattr(reconstructor, "_extract_audio_span", lambda *args, **kwargs: None)

    result = reconstructor.reconstruct(transcript, tmp_path / "audio.wav")

    assert result.transcript.segments[0].text == "теорема Хана — Банаха"
    assert result.metrics["fallback_spans"] == 1
    assert "теорема Хана Банаха" in llm.prompts[1]


def test_correcting_backend_reuses_raw_only_for_matching_fingerprint(tmp_path):
    runtime = RuntimeConfig(work_dir=tmp_path)
    work = tmp_path / "lecture"
    work.mkdir(parents=True)
    raw = _transcript()
    atomic_json_dump(work / "raw_transcript.json", raw.model_dump(mode="json"))
    atomic_json_dump(work / "raw_transcript_meta.json", {"fingerprint": "right"})

    backend = CorrectingASRBackend(
        SimpleNamespace(),
        TranscriptCorrectionConfig(enabled=True),
        runtime,
        llm=SimpleNamespace(),
        course_title="Course",
        language="ru",
        expected_raw_fingerprint="right",
        reuse_raw=True,
    )

    loaded = backend._load_reusable_raw("lecture")
    assert loaded is not None
    assert loaded.segments[0].text == raw.segments[0].text

    backend.expected_raw_fingerprint = "wrong"
    assert backend._load_reusable_raw("lecture") is None
