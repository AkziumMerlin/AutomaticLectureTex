from __future__ import annotations

from types import SimpleNamespace

from automatic_lecture_tex.config import NotesConfig, RuntimeConfig
from automatic_lecture_tex.knowledge import (
    GeneratedNativeVideoObservation,
    GeneratedNativeVideoWindow,
    KnowledgeOrchestrator,
)
from automatic_lecture_tex.media import (
    _api_video_clip_is_valid,
    extract_api_video_clip,
)
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    ObservationKind,
    SourceStatus,
)


class CapturingLLM:
    def __init__(self) -> None:
        self.prompt = ""
        self.kwargs = {}

    def _structured(self, prompt, schema, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        assert schema is GeneratedNativeVideoWindow
        return GeneratedNativeVideoWindow(
            observations=[
                GeneratedNativeVideoObservation(
                    start_offset_seconds=1.5,
                    end_offset_seconds=5.0,
                    kind=ObservationKind.CLAIM,
                    text="Наблюдаемое утверждение.",
                    confidence=0.9,
                    source_status=SourceStatus.OBSERVED,
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
    assert llm.kwargs["guided_json"] is False
    assert llm.kwargs["thinking"] is False
    assert llm.kwargs["temperature"] == 0.1
    assert result.observations[0].start == 21.5
    assert result.observations[0].end == 25.0
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
    monkeypatch.setattr(
        "automatic_lecture_tex.media._api_video_clip_is_valid",
        lambda *args, **kwargs: True,
    )

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



def test_native_video_host_resolves_local_correction_target(tmp_path):
    class LocalTargetLLM:
        def _structured(self, prompt, schema, **kwargs):
            del prompt, kwargs
            assert schema is GeneratedNativeVideoWindow
            return GeneratedNativeVideoWindow(
                observations=[
                    GeneratedNativeVideoObservation(
                        start_offset_seconds=1.0,
                        end_offset_seconds=2.0,
                        kind=ObservationKind.CLAIM,
                        text="Первое утверждение.",
                        confidence=0.9,
                    ),
                    GeneratedNativeVideoObservation(
                        start_offset_seconds=3.0,
                        end_offset_seconds=4.0,
                        kind=ObservationKind.CORRECTION,
                        text="Исправление первого утверждения.",
                        target_local_index=0,
                        confidence=0.95,
                    ),
                ]
            )

    orchestrator = KnowledgeOrchestrator(
        llm=LocalTargetLLM(),
        config=NotesConfig(),
        output_language="ru",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    chunk = LectureChunk(
        id="window_0002",
        start=100.0,
        end=120.0,
        segment_ids=["timing"],
        text="",
    )

    result = orchestrator.extract_observations_from_video(
        chunk,
        video,
        LectureKnowledgeBase(lecture_id="lecture", title="Lecture"),
        model="qwen/qwen3.8-omni-flash",
        thinking=False,
        temperature=0.1,
    )

    assert [item.id for item in result.observations] == [
        "obs_window_0002_000",
        "obs_window_0002_001",
    ]
    assert (
        result.observations[1].target_observation_id
        == "obs_window_0002_000"
    )



def test_invalid_cached_api_clip_is_rebuilt(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"stale-partial-clip")
    calls = []

    monkeypatch.setattr(
        "automatic_lecture_tex.media._api_video_clip_is_valid",
        lambda *args, **kwargs: False if len(calls) == 0 else True,
    )

    def fake_run_checked(args, **kwargs):
        del kwargs
        calls.append(args)
        output.write_bytes(b"rebuilt-valid-clip")
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
    assert output.read_bytes() == b"rebuilt-valid-clip"
    assert calls


def test_conservative_api_clip_normalizes_timestamps_and_framerate(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "clip.mp4"
    captured = {}

    def fake_run_checked(args, **kwargs):
        del kwargs
        captured["args"] = args
        output.write_bytes(b"safe-clip")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("automatic_lecture_tex.media.run_checked", fake_run_checked)
    monkeypatch.setattr(
        "automatic_lecture_tex.media._api_video_clip_is_valid",
        lambda *args, **kwargs: True,
    )

    extract_api_video_clip(
        RuntimeConfig(),
        source,
        start=100.0,
        end=120.0,
        output_path=output,
        max_height=720,
        video_bitrate_kbps=1000,
        audio_bitrate_kbps=64,
        max_bytes=7_000_000,
        force=True,
        conservative=True,
    )

    args = captured["args"]
    assert "fps=8,setpts=PTS-STARTPTS" in args[args.index("-vf") + 1]
    assert args[args.index("-af") + 1] == (
        "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS"
    )
    assert args[args.index("-profile:v") + 1] == "main"
    assert args[args.index("-avoid_negative_ts") + 1] == "make_zero"


def test_api_clip_validator_rejects_missing_video_stream(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    def fake_run_checked(args, **kwargs):
        del kwargs
        if args[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0,
                stdout='{"format":{"duration":"20.0"},"streams":[{"codec_type":"audio"}]}',
                stderr="",
            )
        raise AssertionError("ffmpeg decode probe must not run without a video stream")

    monkeypatch.setattr("automatic_lecture_tex.media.run_checked", fake_run_checked)

    assert not _api_video_clip_is_valid(
        RuntimeConfig(),
        clip,
        expected_duration=20.0,
    )
