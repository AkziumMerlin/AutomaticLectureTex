import json

from automatic_lecture_tex.config import AppConfig
from automatic_lecture_tex.generated_notes import GeneratedChunkNotes, GeneratedNoteBlock
from automatic_lecture_tex.knowledge_pipeline import _load_window_artifact
from automatic_lecture_tex.pipeline import Pipeline
from automatic_lecture_tex.schemas import (
    EpisodeBoundary,
    EpisodeHierarchyPlan,
    EpisodeKind,
    EpisodeTrackingUpdate,
    HierarchyBoundary,
    HierarchyLevel,
    LectureObservation,
    MathAudit,
    ObservationKind,
    Transcript,
    TranscriptSegment,
    WindowObservations,
)


class FakeSource:
    def identity(self):
        return {"type": "fake", "id": "knowledge-source"}

    def prepare_audio(self, output_path):
        output_path.write_bytes(b"audio")


class FakeASR:
    def transcribe(self, lecture_id, audio_path):
        return Transcript(
            lecture_id=lecture_id,
            language="ru",
            segments=[
                TranscriptSegment(id="s0", start=0, end=20, text="Определим функционал."),
                TranscriptSegment(id="s1", start=20, end=40, text="Получаем формулу."),
            ],
        )


class FakeKnowledgeLLM:
    def __init__(self):
        self.operations = []

    def reset_usage(self):
        self.operations.clear()

    def usage_snapshot(self):
        return {
            "requests": len(self.operations),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
            "by_operation": {},
        }

    def _structured(self, prompt, schema, images=None, max_tokens=None, **kwargs):
        operation = kwargs.get("operation", "structured")
        self.operations.append(operation)
        if schema is WindowObservations:
            return WindowObservations(
                observations=[
                    LectureObservation(
                        start=0,
                        end=20,
                        kind=ObservationKind.DEFINITION,
                        text="Определение функционала",
                        confidence=1.0,
                    )
                ]
            )
        if schema is EpisodeTrackingUpdate:
            return EpisodeTrackingUpdate(
                boundaries=[
                    EpisodeBoundary(
                        before_observation_id="obs_window_0000_000",
                        kind=EpisodeKind.DEFINITION,
                        title="Определение функционала",
                    )
                ]
            )
        if schema is EpisodeHierarchyPlan:
            return EpisodeHierarchyPlan(
                boundaries=[
                    HierarchyBoundary(
                        before_episode_id="episode_0000",
                        level=HierarchyLevel.TOPIC,
                        title="Линейные функционалы",
                    )
                ]
            )
        if schema is GeneratedChunkNotes:
            return GeneratedChunkNotes(
                section_title="ignored",
                blocks=[
                    GeneratedNoteBlock(
                        type="paragraph",
                        latex="Определение функционала.",
                        source_claim_ids=["claim_obs_window_0000_000"],
                        source_evidence_ids=["obs_window_0000_000"],
                    )
                ],
            )
        if schema is MathAudit:
            return MathAudit()
        raise AssertionError(f"unexpected schema: {schema}")


def test_knowledge_pipeline_builds_episode_graph_outline_and_ir(tmp_path, monkeypatch):
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
                "architecture": "knowledge",
                "chunk_target_seconds": 60,
                "chunk_overlap_seconds": 10,
                "visual_rule_selector": False,
                "visual_llm_selector": False,
            },
            "runtime": {"work_dir": tmp_path / "work"},
            "latex": {"output_dir": tmp_path / "tex"},
        }
    )
    monkeypatch.setattr(
        "automatic_lecture_tex.pipeline.media_source_from_config",
        lambda *args: FakeSource(),
    )

    pipeline = Pipeline(cfg)
    pipeline._asr = FakeASR()
    fake_llm = FakeKnowledgeLLM()
    pipeline._llm = fake_llm
    ir = pipeline.run_lecture(cfg.course.lectures[0])

    work = cfg.runtime.work_dir / "lecture"
    assert (work / "lecture_kb.json").exists()
    assert (work / "episode_hierarchy.json").exists()
    assert (work / "lecture_outline.json").exists()
    assert (work / "knowledge_episode_batches" / "episode_0000" / "batch_000.json").exists()
    assert not (work / "global_validation.json").exists()
    assert ir.chunks[0].section_title == "Линейные функционалы"
    assert ir.chunks[0].blocks[0].latex == "Определение функционала."
    assert fake_llm.operations == [
        "knowledge_extract",
        "episode_track",
        "episode_hierarchy",
        "episode_write",
        "episode_validation",
    ]


def test_window_cache_with_duplicate_observation_ids_is_invalidated(tmp_path):
    path = tmp_path / "window.json"
    payload = {
        "fingerprint": "fp",
        "observations": WindowObservations(
            window_id="window_0001",
            start=0.0,
            end=2.0,
            observations=[
                LectureObservation(
                    id="obs_window_0001_000",
                    window_id="window_0001",
                    start=0.0,
                    end=1.0,
                    kind=ObservationKind.CLAIM,
                    text="A",
                ),
                LectureObservation(
                    id="obs_window_0001_000",
                    window_id="window_0001",
                    start=1.0,
                    end=2.0,
                    kind=ObservationKind.CLAIM,
                    text="B",
                ),
            ],
        ).model_dump(mode="json"),
        "episode_update": EpisodeTrackingUpdate().model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_window_artifact(path, "fp") is None
