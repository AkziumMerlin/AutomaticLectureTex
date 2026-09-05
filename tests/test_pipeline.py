import json

from automatic_lecture_tex.config import AppConfig
from automatic_lecture_tex.pipeline import Pipeline
from automatic_lecture_tex.schemas import ChunkNotes, NoteBlock, Transcript, TranscriptSegment
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
            "notes": {"visual_rule_selector": False, "visual_llm_selector": False},
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
            "notes": {"visual_rule_selector": False, "visual_llm_selector": False},
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
