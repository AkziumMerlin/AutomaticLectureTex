from __future__ import annotations

from types import SimpleNamespace

from automatic_lecture_tex.config import NotesConfig, RuntimeConfig
from automatic_lecture_tex.knowledge import KnowledgeOrchestrator
from automatic_lecture_tex.media import extract_api_video_clip
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    WindowObservations,
)


class CapturingLLM:
    def __init__(self) -> None:
        self.prompt = ""
        self.kwargs = {}

    def _structured(self, prompt, schema, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        assert schema is WindowObservations
        return WindowObservations(
            observations=[
                LectureObservation(
                    start=-10.0,
                    end=1000.0,
                    kind=ObservationKind.CLAIM,
                    text="Наблюдаемое утверждение.",
                )
            ]
        )


def test_native_video_extractor_uses_video_without_transcript(tmp_path):
    llm = CapturingLLM()
    orchestrator = KnowledgeOrchestrator(
        llm=llm,
        config=NotesConfig(),
        output_language="ru",
    )
    chunk = LectureChunk(
        id="window_0001",
        start=20.0,
        end=40.0,
        segment_ids=["timing_0001"],
        text="THIS TRANSCRIPT MUST NOT BE USED",
        timestamped_text="THIS TRANSCRIPT MUST NOT BE USED",
    )
    video = tmp_path / "window_0001.mp4"
    video.write_bytes(b"video")

    result = orchestrator.extract_observations_from_video(
        chunk,
        video,
        LectureKnowledgeBase(lecture_id="lecture", title="Lecture"),
        model="qwen/qwen3.8-omni-flash",
        thinking=False,
        temperature=0.1,
    )

    assert "THIS TRANSCRIPT MUST NOT BE USED" not in llm.prompt
    assert llm.kwargs["videos"] == [video]
    assert llm.kwargs["model"] == "qwen/qwen3.8-omni-flash"
    assert llm.kwargs["operation"] == "knowledge_extract_native_video"
    assert llm.kwargs["thinking"] is False
    assert llm.kwargs["temperature"] == 0.1
    assert result.observations[0].start == 20.0
    assert result.observations[0].end == 40.0
    assert result.observations[0].evidence_refs == ["video:window_0001"]


def test_extract_api_video_clip_builds_compact_av_mp4(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "clip.mp4"
    captured = {}

    def fake_run_checked(args, **kwargs):
        del kwargs
        captured["args"] = args
        output.write_bytes(b"x" * 1024)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("automatic_lecture_tex.media.run_checked", fake_run_checked)

    result = extract_api_video_clip(
        RuntimeConfig(),
        source,
        start=10.0,
        end=30.0,
        output_path=output,
        max_height=720,
        video_bitrate_kbps=1000,
        audio_bitrate_kbps=64,
        max_bytes=7_000_000,
    )

    assert result == output
    args = captured["args"]
    assert "-ss" in args and args[args.index("-ss") + 1] == "10.000"
    assert "-t" in args and args[args.index("-t") + 1] == "20.000"
    assert "-map" in args
    assert "0:a:0?" in args
    assert "libx264" in args
    assert "aac" in args
    assert str(output) == args[-1]
