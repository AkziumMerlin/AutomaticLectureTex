import json

from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.knowledge_reconstruction_resilient import (
    ResilientIntegrityKnowledgeOrchestrator,
)
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    Transcript,
    TranscriptSegment,
)


class _SplitRecoveryLLM:
    def __init__(self, *, always_fail: bool = False) -> None:
        self.always_fail = always_fail
        self.prompts: list[str] = []

    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        del max_tokens
        assert operation == "knowledge_extract"
        self.prompts.append(prompt)
        both_segments = "seg_00001" in prompt and "seg_00002" in prompt
        if self.always_fail or both_segments:
            raw = '{"observations":[{"kind":"claim","text":"unterminated'
            raise json.JSONDecodeError("Unterminated string", raw, len(raw) - 3)

        segment_id = "seg_00001" if "seg_00001" in prompt else "seg_00002"
        return schema.model_validate(
            {
                "observations": [
                    {
                        "kind": "claim",
                        "text": f"Содержательное утверждение из {segment_id}.",
                        "confidence": 0.9,
                        "source_status": "observed",
                        "source_segment_ids": [segment_id],
                    }
                ],
                "unresolved": [],
            }
        )


def _two_segment_input():
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(id="seg_00001", start=0.0, end=4.0, text="Первая часть."),
            TranscriptSegment(id="seg_00002", start=4.0, end=8.0, text="Вторая часть."),
        ],
    )
    chunk = LectureChunk(
        id="window_0000",
        start=0.0,
        end=8.0,
        segment_ids=["seg_00001", "seg_00002"],
        text="Первая часть.\nВторая часть.",
    )
    kb = LectureKnowledgeBase(lecture_id="lecture", title="Lecture")
    return transcript, chunk, kb


def test_structured_failure_splits_window_and_preserves_parent_window_identity():
    transcript, chunk, kb = _two_segment_input()
    llm = _SplitRecoveryLLM()
    orchestrator = ResilientIntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(chunk, [], kb)

    assert len(llm.prompts) == 3
    assert len(result.observations) == 2
    assert result.unresolved == []
    assert result.window_id == "window_0000"
    assert [item.evidence_refs for item in result.observations] == [
        ["seg_00001"],
        ["seg_00002"],
    ]
    assert all(item.window_id == "window_0000" for item in result.observations)
    assert all(item.window_ids == ["window_0000"] for item in result.observations)


def test_indivisible_structured_failure_becomes_unresolved_instead_of_raising():
    transcript, chunk, kb = _two_segment_input()
    leaf = LectureChunk(
        id="window_leaf",
        start=0.0,
        end=4.0,
        segment_ids=["seg_00001"],
        text="Первая часть.",
    )
    llm = _SplitRecoveryLLM(always_fail=True)
    orchestrator = ResilientIntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.extract_observations(leaf, [], kb)

    assert result.observations == []
    assert len(result.unresolved) == 1
    assert "seg_00001" in result.unresolved[0]
    assert "JSONDecodeError" in result.unresolved[0]
