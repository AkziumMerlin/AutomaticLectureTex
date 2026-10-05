import json

from automatic_lecture_tex.config import AppConfig
from automatic_lecture_tex.pipeline import Pipeline
from automatic_lecture_tex.schemas import ChunkNotes, LectureIR, NoteBlock, Transcript, TranscriptSegment
from automatic_lecture_tex.util import atomic_json_dump, stable_hash


class FakeLLM:
    def __init__(self) -> None:
        self.finalize_calls = 0

    def reset_usage(self):
        pass

    def usage_snapshot(self):
        return {"requests": self.finalize_calls, "total_tokens": 0, "by_operation": {}}

    def finalize_chunk(self, chunk, evidence, notation, previous_notes=None):
        self.finalize_calls += 1
        return ChunkNotes(
            chunk_id=chunk.id,
            start=chunk.start,
            end=chunk.end,
            section_title="Section",
            blocks=[NoteBlock(type="paragraph", latex=chunk.text)],
        )


class FakeSource:
    def __init__(self) -> None:
        self.prepare_calls = 0

    def identity(self):
        return {"type": "fake", "id": "stable-source"}

    def prepare_audio(self, output_path):
        self.prepare_calls += 1
        output_path.write_bytes(b"audio")


class FakeASR:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, lecture_id, audio_path):
        self.calls += 1
        return Transcript(
            lecture_id=lecture_id,
            language="ru",
            segments=[TranscriptSegment(id="s0", start=0, end=10, text="content")],
        )


def test_pipeline_reuses_completed_chunk_after_interruption(tmp_path):
    source_path = tmp_path / "lecture.mp4"
    source_path.write_bytes(b"not accessed because transcript is cached")
    cfg = AppConfig.model_validate(
        {
            "course": {
                "id": "course",
                "title": "Course",
                "lectures": [{"id": "lecture", "source": {"type": "file", "path": source_path}}],
            },
            "notes": {
                "architecture": "legacy",
                "visual_rule_selector": False,
                "visual_llm_selector": False,
            },
            "runtime": {"work_dir": tmp_path / "work"},
            "latex": {"output_dir": tmp_path / "tex"},
        }
    )
    lecture = cfg.course.lectures[0]
    transcript = Transcript(
        lecture_id=lecture.id,
        language="ru",
        segments=[TranscriptSegment(id="s0", start=0, end=10, text="content")],
    )
    work = cfg.runtime.work_dir / lecture.id
    work.mkdir(parents=True)
    transcript_path = work / "transcript.json"
    atomic_json_dump(transcript_path, transcript.model_dump(mode="json"))
    stat = source_path.resolve().stat()
    source_identity = {
        "type": "file",
        "path": str(source_path.resolve()),
        "size": str(stat.st_size),
        "mtime_ns": str(stat.st_mtime_ns),
    }
    atomic_json_dump(
        work / "manifest.json",
        {
            "transcript_fingerprint": stable_hash(
                {
                    "source": source_identity,
                    "asr": cfg.asr.model_dump(mode="json"),
                    "asr_cache_version": 2,
                }
            )
        },
    )

    first = Pipeline(cfg)
    first_llm = FakeLLM()
    first._llm = first_llm
    first.run_lecture(lecture)
    assert first_llm.finalize_calls == 1

    (work / "lecture_ir.json").unlink()
    second = Pipeline(cfg)
    second_llm = FakeLLM()
    second._llm = second_llm
    second.run_lecture(lecture)

    assert second_llm.finalize_calls == 0
    metrics = json.loads((work / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["chunk_cache_hits"] == 1


def test_pipeline_reuses_audio_when_asr_config_changes(tmp_path, monkeypatch):
    source_path = tmp_path / "lecture.mp4"
    source_path.write_bytes(b"source")
    cfg = AppConfig.model_validate(
        {
            "course": {
                "id": "course",
                "title": "Course",
                "lectures": [{"id": "lecture", "source": {"type": "file", "path": source_path}}],
            },
            "notes": {
                "architecture": "legacy",
                "visual_rule_selector": False,
                "visual_llm_selector": False,
            },
            "runtime": {"work_dir": tmp_path / "work"},
            "latex": {"output_dir": tmp_path / "tex"},
        }
    )
    source = FakeSource()
    asr = FakeASR()
    monkeypatch.setattr(
        "automatic_lecture_tex.pipeline.media_source_from_config", lambda *args: source
    )

    first = Pipeline(cfg)
    first._asr = asr
    first._llm = FakeLLM()
    first.run_lecture(cfg.course.lectures[0])

    cfg.asr.model = "a-different-asr-model"
    second = Pipeline(cfg)
    second._asr = asr
    second._llm = FakeLLM()
    second.run_lecture(cfg.course.lectures[0])

    assert source.prepare_calls == 1
    assert asr.calls == 2



def test_native_video_mode_skips_asr_and_builds_timing_transcript(tmp_path, monkeypatch):
    source_path = tmp_path / "lecture.mp4"
    source_path.write_bytes(b"source")
    cfg = AppConfig.model_validate(
        {
            "course": {
                "id": "course",
                "title": "Course",
                "lectures": [
                    {"id": "lecture", "source": {"type": "file", "path": source_path}}
                ],
            },
            "notes": {
                "architecture": "state",
                "window_evidence_backend": "native_video",
                "chunk_target_seconds": 10,
                "chunk_overlap_seconds": 5,
            },
            "runtime": {"work_dir": tmp_path / "work"},
            "latex": {"output_dir": tmp_path / "tex"},
        }
    )

    class NativeSource:
        def __init__(self):
            self.prepare_video_calls = 0

        def identity(self):
            return {"type": "fake", "id": "native-video-source"}

        def prepare_video(self, output_dir, *, max_height):
            self.prepare_video_calls += 1
            assert max_height == 720
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "source.mp4"
            path.write_bytes(b"video")
            return path

    class ForbiddenASR:
        def transcribe(self, *args, **kwargs):
            raise AssertionError("native_video mode must not invoke ASR")

    source = NativeSource()
    captured = {}

    monkeypatch.setattr(
        "automatic_lecture_tex.pipeline.media_source_from_config",
        lambda *args: source,
    )
    monkeypatch.setattr(
        "automatic_lecture_tex.pipeline.probe_duration",
        lambda *args: 12.0,
    )

    def fake_run_knowledge_pipeline(pipeline, **kwargs):
        del pipeline
        captured["transcript"] = kwargs["transcript"]
        captured["asr_seconds"] = kwargs["asr_seconds"]
        return LectureIR(lecture_id="lecture", title="Lecture", chunks=[])

    monkeypatch.setattr(
        "automatic_lecture_tex.pipeline.run_knowledge_pipeline",
        fake_run_knowledge_pipeline,
    )

    pipeline = Pipeline(cfg)
    pipeline._asr = ForbiddenASR()
    result = pipeline.run_lecture(cfg.course.lectures[0])

    assert result.lecture_id == "lecture"
    assert source.prepare_video_calls == 1
    assert captured["asr_seconds"] == 0.0
    transcript = captured["transcript"]
    assert [segment.text for segment in transcript.segments] == ["", "", ""]
    assert [(segment.start, segment.end) for segment in transcript.segments] == [
        (0.0, 2.5),
        (2.5, 5.0),
        (5.0, 7.5),
        (7.5, 10.0),
        (10.0, 12.0),
    ]
