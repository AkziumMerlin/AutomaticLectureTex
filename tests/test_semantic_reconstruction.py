from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.knowledge_integrity import IntegrityKnowledgeOrchestrator
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    Transcript,
    TranscriptSegment,
)


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
