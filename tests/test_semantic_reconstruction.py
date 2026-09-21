from types import SimpleNamespace

from automatic_lecture_tex import pipeline as base_pipeline
from automatic_lecture_tex import pipeline_robust
from automatic_lecture_tex.config import (
    AppConfig,
    CourseConfig,
    LectureConfig,
    NotesConfig,
    RuntimeConfig,
    SourceConfig,
)
from automatic_lecture_tex.knowledge_integrity import IntegrityKnowledgeOrchestrator
from automatic_lecture_tex.pipeline_robust import Pipeline
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    Transcript,
    TranscriptSegment,
)
from automatic_lecture_tex.util import atomic_json_dump, stable_hash


class _SemanticLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""

    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        assert operation == "knowledge_extract"
        self.prompt = prompt
        return schema.model_validate(self.payload)


def _raw_input():
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_00001",
                start=10.0,
                end=14.0,
                text="комплексный линейный функцеонал",
                confidence=0.41,
            ),
            TranscriptSegment(
                id="seg_00002",
                start=14.0,
                end=19.0,
                text="эф икс это у икс минус и у от и икс",
                confidence=0.36,
            ),
        ],
    )
    chunk = LectureChunk(
        id="window_0000",
        start=10.0,
        end=19.0,
        segment_ids=["seg_00001", "seg_00002"],
        text="\n".join(item.text for item in transcript.segments),
    )
    kb = LectureKnowledgeBase(lecture_id="lecture", title="Lecture")
    return transcript, chunk, kb


def test_raw_asr_is_reconstructed_directly_into_canonical_math_event():
    transcript, chunk, kb = _raw_input()
    llm = _SemanticLLM(
        {
            "observations": [
                {
                    "kind": "definition",
                    "text": "Рассматривается комплексный линейный функционал.",
                    "confidence": 0.94,
                    "source_status": "reconstructed",
                    "source_segment_ids": ["seg_00001"],
                },
                {
                    "kind": "equation",
                    "text": "Функционал восстанавливается по вещественной части.",
                    "latex": r"f(x)=u(x)-iu(ix)",
                    "confidence": 0.91,
                    "source_status": "reconstructed",
                    "source_segment_ids": ["seg_00002"],
                },
            ],
            "unresolved": [],
        }
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(chunk, [], kb)

    assert [item.kind for item in result.observations] == ["definition", "equation"]
    assert result.observations[1].latex == r"f(x)=u(x)-iu(ix)"
    assert result.observations[1].start == 14.0
    assert result.observations[1].end == 19.0
    assert result.observations[1].evidence_refs == ["seg_00002"]
    assert "Raw timestamped ASR segments" in llm.prompt
    assert "mathematical consistency" in llm.prompt
    assert "not a license to complete the lecture" in llm.prompt


def test_unresolved_reconstruction_never_enters_canonical_observations():
    transcript, chunk, kb = _raw_input()
    llm = _SemanticLLM(
        {
            "observations": [
                {
                    "kind": "unresolved",
                    "text": "Не удаётся однозначно восстановить название конструкции.",
                    "confidence": 0.25,
                    "source_status": "inferred",
                    "source_segment_ids": ["seg_00001"],
                }
            ],
            "unresolved": [],
        }
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(chunk, [], kb)

    assert result.observations == []
    assert len(result.unresolved) == 1
    assert "Не удаётся однозначно" in result.unresolved[0]
    assert "seg_00001" in result.unresolved[0]


def test_pipeline_restores_saved_raw_asr_without_rerunning_whisper(tmp_path, monkeypatch):
    lecture = LectureConfig(
        id="lecture",
        source=SourceConfig(type="file", path=tmp_path / "source.mp4"),
    )
    config = AppConfig(
        course=CourseConfig(id="course", title="Course", lectures=[lecture]),
        runtime=RuntimeConfig(work_dir=tmp_path),
    )
    pipeline = Pipeline(config)
    source_identity = {"kind": "test-source", "id": "lecture"}
    monkeypatch.setattr(
        pipeline_robust,
        "media_source_from_config",
        lambda *args, **kwargs: SimpleNamespace(identity=lambda: source_identity),
    )

    work = tmp_path / "lecture"
    work.mkdir()
    raw, _, _ = _raw_input()
    raw.lecture_id = "lecture"
    raw_fingerprint = stable_hash(
        {
            "source": source_identity,
            "asr": config.asr.model_dump(mode="json"),
            "asr_cache_version": base_pipeline.ASR_CACHE_VERSION,
        }
    )
    atomic_json_dump(work / "raw_transcript.json", raw.model_dump(mode="json"))
    atomic_json_dump(work / "raw_transcript_meta.json", {"fingerprint": raw_fingerprint})
    atomic_json_dump(
        work / "transcript.json",
        Transcript(
            lecture_id="lecture",
            segments=[TranscriptSegment(id="bad", start=0, end=1, text="corrected stale text")],
        ).model_dump(mode="json"),
    )
    atomic_json_dump(work / "manifest.json", {"transcript_fingerprint": "old-corrected"})
    atomic_json_dump(work / "transcript_reconstruction.json", {"stale": True})

    pipeline._restore_raw_asr_cache(lecture)

    restored = Transcript.model_validate_json((work / "transcript.json").read_text(encoding="utf-8"))
    assert restored.segments[0].id == "seg_00001"
    assert restored.segments[0].text == "комплексный линейный функцеонал"
    manifest = __import__("json").loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["transcript_fingerprint"] == raw_fingerprint
    assert not (work / "transcript_reconstruction.json").exists()



def test_epistemic_hedging_is_demoted_to_unresolved():
    transcript, chunk, kb = _raw_input()
    llm = _SemanticLLM(
        {
            "observations": [
                {
                    "kind": "claim",
                    "text": "Вероятно, это утверждение о компактности шара.",
                    "confidence": 0.88,
                    "source_status": "reconstructed",
                    "source_segment_ids": ["seg_00001"],
                }
            ],
            "unresolved": [],
        }
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(chunk, [], kb)

    assert result.observations == []
    assert any("Host ambiguity gate suppressed" in item for item in result.unresolved)


def test_mathematical_possibility_word_is_not_treated_as_epistemic_hedging():
    transcript, chunk, kb = _raw_input()
    llm = _SemanticLLM(
        {
            "observations": [
                {
                    "kind": "claim",
                    "text": "Можно выбрать функционал так, чтобы норма сохранялась.",
                    "confidence": 0.9,
                    "source_status": "reconstructed",
                    "source_segment_ids": ["seg_00001"],
                }
            ],
            "unresolved": [],
        }
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(chunk, [], kb)

    assert len(result.observations) == 1
