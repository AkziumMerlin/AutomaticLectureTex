import json

from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.llm import StructuredTaskTooLargeError
from automatic_lecture_tex.knowledge_reconstruction_resilient import (
    ResilientIntegrityKnowledgeOrchestrator,
)
from automatic_lecture_tex.schemas import (
    FormulaVisualCrop,
    LectureChunk,
    LectureKnowledgeBase,
    LectureObservation,
    MathOCRCandidate,
    ObservationKind,
    Transcript,
    TranscriptSegment,
    VisualEvidence,
    WindowObservations,
)


class _SplitRecoveryLLM:
    def __init__(self, *, always_fail: bool = False) -> None:
        self.always_fail = always_fail
        self.prompts: list[str] = []

    def _structured(
        self,
        prompt,
        schema,
        *,
        operation,
        max_tokens=None,
        split_oversized_task=False,
    ):
        del max_tokens, split_oversized_task
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


class _TrackingSplitLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _structured(
        self,
        prompt,
        schema,
        *,
        operation,
        max_tokens=None,
        split_oversized_task=False,
    ):
        del max_tokens
        assert operation == "episode_track"
        assert split_oversized_task is True
        self.calls.append(prompt)
        if "obs_a" in prompt and "obs_b" in prompt:
            raise StructuredTaskTooLargeError("too large")
        return schema.model_validate({})


def test_episode_tracking_context_overflow_splits_canonical_observations():
    transcript = Transcript(lecture_id="lecture", language="ru", segments=[])
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=[
            LectureObservation(
                id="obs_a",
                window_id="window",
                window_ids=["window"],
                start=0.0,
                end=1.0,
                kind=ObservationKind.CLAIM,
                text="A",
            ),
            LectureObservation(
                id="obs_b",
                window_id="window",
                window_ids=["window"],
                start=1.0,
                end=2.0,
                kind=ObservationKind.CLAIM,
                text="B",
            ),
        ],
    )
    batch = WindowObservations(window_id="window", start=0.0, end=2.0)
    llm = _TrackingSplitLLM()
    orchestrator = ResilientIntegrityKnowledgeOrchestrator(
        llm,
        NotesConfig(),
        "ru",
        transcript=transcript,
    )

    result = orchestrator.track_episodes(kb, batch, ["obs_a", "obs_b"])

    assert len(llm.calls) == 3
    assert result.unresolved == []


def test_reconstruction_split_slices_visual_evidence_by_child_time():
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(id="s0", start=0.0, end=5.0, text="first"),
            TranscriptSegment(id="s1", start=5.0, end=10.0, text="second"),
        ],
    )
    orchestrator = ResilientIntegrityKnowledgeOrchestrator(
        _SplitRecoveryLLM(),
        NotesConfig(),
        "ru",
        transcript=transcript,
    )
    parent = LectureChunk(
        id="window",
        start=0.0,
        end=10.0,
        segment_ids=["s0", "s1"],
        text="first second",
    )
    left, right = orchestrator._split_chunk(parent)
    evidence = [
        VisualEvidence(
            request_id="board",
            frame_paths=["f1.jpg", "f2.jpg"],
            frame_timestamps=[2.0, 8.0],
            formula_contact_sheet_path="contact.jpg",
            formula_crops=[
                FormulaVisualCrop(
                    id="c1",
                    timestamp=2.0,
                    bbox=(0, 0, 1, 1),
                    detector_confidence=0.9,
                    image_path="c1.jpg",
                ),
                FormulaVisualCrop(
                    id="c2",
                    timestamp=8.0,
                    bbox=(0, 0, 1, 1),
                    detector_confidence=0.9,
                    image_path="c2.jpg",
                ),
            ],
            math_ocr_candidates=[
                MathOCRCandidate(
                    backend="unimernet",
                    text="x",
                    timestamp=2.0,
                    source_id="c1",
                ),
                MathOCRCandidate(
                    backend="unimernet",
                    text="y",
                    timestamp=8.0,
                    source_id="c2",
                ),
            ],
        )
    ]

    left_evidence = orchestrator._slice_visual_evidence(left, evidence)
    right_evidence = orchestrator._slice_visual_evidence(right, evidence)

    assert left_evidence[0].frame_paths == ["f1.jpg"]
    assert right_evidence[0].frame_paths == ["f2.jpg"]
    assert [item.id for item in left_evidence[0].formula_crops] == ["c1"]
    assert [item.id for item in right_evidence[0].formula_crops] == ["c2"]
    assert [item.source_id for item in left_evidence[0].math_ocr_candidates] == ["c1"]
    assert [item.source_id for item in right_evidence[0].math_ocr_candidates] == ["c2"]
    assert left_evidence[0].formula_contact_sheet_path is None
    assert right_evidence[0].formula_contact_sheet_path is None
