from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.knowledge_integrity import IntegrityKnowledgeOrchestrator
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    Transcript,
    TranscriptSegment,
)


class _ExtractLLM:
    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        assert operation == "knowledge_extract"
        return schema.model_validate(
            {
                "observations": [
                    {
                        "kind": "claim",
                        "text": "Случайно восстановленное математическое утверждение.",
                        "confidence": 0.9,
                        "source_status": "reconstructed",
                        "source_segment_ids": ["seg_00000"],
                    }
                ],
                "unresolved": [],
            }
        )


def test_ambiguous_only_transcript_cannot_create_mandatory_substantive_evidence():
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_00000",
                start=0.0,
                end=10.0,
                text="неразборчивый шум",
                confidence=0.20,
            )
        ],
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        _ExtractLLM(),
        NotesConfig(),
        "ru",
        transcript=transcript,
        ambiguous_transcript_confidence=0.20,
    )
    chunk = LectureChunk(
        id="window_0000",
        start=0.0,
        end=10.0,
        segment_ids=["seg_00000"],
        text="неразборчивый шум",
    )

    result = orchestrator.extract_observations(
        chunk,
        [],
        LectureKnowledgeBase(lecture_id="lecture", title="Lecture"),
    )

    assert result.observations == []
    assert "ambiguous-only transcript evidence" in result.unresolved[0]
