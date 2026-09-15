from pathlib import Path
from types import SimpleNamespace

from automatic_lecture_tex.config import MathOCRConfig, RuntimeConfig, VisionConfig
from automatic_lecture_tex.media import YouTubeMediaSource
from automatic_lecture_tex.schemas import LectureChunk, Transcript, TranscriptSegment
from automatic_lecture_tex.sensory_evidence import collect_visual_evidence


def test_remote_frame_extraction_falls_back_to_direct_stream(tmp_path, monkeypatch):
    source = YouTubeMediaSource(
        "https://example.invalid/video",
        RuntimeConfig(work_dir=tmp_path),
        VisionConfig(),
    )
    calls = []

    def fail_section(**kwargs):
        calls.append(("section", kwargs["force_keyframes"]))
        raise RuntimeError("ffmpeg exited with code 69")

    def direct_stream(safe_times, targets):
        calls.append(("stream", tuple(safe_times)))
        frames = []
        from automatic_lecture_tex.schemas import ExtractedFrame

        for timestamp, path in zip(safe_times, targets, strict=True):
            Path(path).write_bytes(b"jpeg")
            frames.append(ExtractedFrame(timestamp=timestamp, path=path))
        return frames

    monkeypatch.setattr(source, "_download_section", fail_section)
    monkeypatch.setattr(source, "_extract_frames_from_stream", direct_stream)

    frames = source.extract_frames([10.0, 12.0], tmp_path / "frames")

    assert [item.timestamp for item in frames] == [10.0, 12.0]
    assert calls[0] == ("section", True)
    assert calls[1][0] == "stream"


class _FailingSource:
    def extract_frames(self, timestamps, output_dir):
        del timestamps, output_dir
        raise RuntimeError("remote frame extraction failed")


class _NeverCalledLLM:
    def resolve_visual_request(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("VLM must not be called when frame acquisition failed")


def test_visual_frame_failure_is_nonfatal(tmp_path):
    vision = VisionConfig(
        temporal_composite_enabled=True,
        math_ocr=MathOCRConfig(backend="none"),
    )
    pipeline = SimpleNamespace(
        config=SimpleNamespace(
            notes=SimpleNamespace(
                visual_rule_selector=True,
                visual_llm_selector=False,
                max_low_confidence_visual_requests=0,
                visual_dedupe_seconds=8.0,
            ),
            vision=vision,
            latex=SimpleNamespace(output_dir=tmp_path / "tex"),
        ),
        llm=_NeverCalledLLM(),
    )
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_0",
                start=10.0,
                end=12.0,
                text="Запишем это на доске",
            )
        ],
    )
    chunk = LectureChunk(
        id="window_0",
        start=10.0,
        end=12.0,
        segment_ids=["seg_0"],
        text="Запишем это на доске",
    )

    requests, evidence, _elapsed = collect_visual_evidence(
        pipeline,
        SimpleNamespace(id="lecture"),
        chunk,
        transcript,
        _FailingSource(),
        tmp_path / "work",
        tmp_path / "figures",
        {},
    )

    assert len(requests) == 1
    assert len(evidence) == 1
    assert evidence[0].request_id == requests[0].id
    assert evidence[0].confidence == 0.0
    assert "Visual frame extraction unavailable" in (evidence[0].description or "")
