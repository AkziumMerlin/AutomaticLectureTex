from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.knowledge_integrity import (
    GeneratedLectureObservation,
    GeneratedWindowObservations,
    IntegrityKnowledgeOrchestrator,
)
from automatic_lecture_tex.schemas import (
    LectureChunk,
    LectureKnowledgeBase,
    ObservationKind,
    SourceStatus,
    Transcript,
    TranscriptSegment,
    VisualEvidence,
    VisualKind,
)


class _PromptCaptureLLM:
    def __init__(self):
        self.prompt = ""
        self.images = None
        self.guided_json = None

    def _structured(
        self,
        prompt,
        schema,
        images=None,
        *,
        operation,
        max_tokens=None,
        guided_json=True,
    ):
        assert operation == "knowledge_extract"
        self.prompt = prompt
        self.images = images
        self.guided_json = guided_json
        return GeneratedWindowObservations(
            observations=[
                GeneratedLectureObservation(
                    kind=ObservationKind.CLAIM,
                    text="Канонически восстановленное математическое утверждение.",
                    confidence=0.95,
                    source_status=SourceStatus.RECONSTRUCTED,
                    source_segment_ids=["seg_1"],
                )
            ]
        )


def test_semantic_reconstruction_treats_asr_as_phonetic_evidence():
    llm = _PromptCaptureLLM()
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_1",
                start=0.0,
                end=5.0,
                text="искажённое фонетическое распознавание математического термина",
            )
        ],
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        llm=llm,
        config=NotesConfig(),
        output_language="ru",
        transcript=transcript,
    )
    chunk = LectureChunk(
        id="window",
        start=0.0,
        end=5.0,
        segment_ids=["seg_1"],
        text=transcript.segments[0].text,
        timestamped_text="[0.000-5.000] " + transcript.segments[0].text,
    )
    kb = LectureKnowledgeBase(lecture_id="lecture", title="Lecture")

    result = orchestrator.extract_observations(chunk, [], kb)

    assert len(result.observations) == 1
    assert "Treat ASR as PHONETIC EVIDENCE" in llm.prompt
    assert "standard mathematical knowledge as a DISAMBIGUATION PRIOR" in llm.prompt
    assert "invented-looking proper names" in llm.prompt
    assert "must NEVER be expanded into an unrelated specific theorem/person" in llm.prompt
    assert "A lone garbled phrase" in llm.prompt
    assert "Preserve a lecturer mistake only when" in llm.prompt



def test_semantic_reconstruction_receives_actual_board_images(tmp_path):
    llm = _PromptCaptureLLM()
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_1",
                start=0.0,
                end=5.0,
                text="тогда на доске получаем следующую формулу",
            )
        ],
    )
    orchestrator = IntegrityKnowledgeOrchestrator(
        llm=llm,
        config=NotesConfig(),
        output_language="ru",
        transcript=transcript,
    )
    chunk = LectureChunk(
        id="window",
        start=0.0,
        end=5.0,
        segment_ids=["seg_1"],
        text=transcript.segments[0].text,
        timestamped_text="[0.000-5.000] " + transcript.segments[0].text,
    )
    image_a = tmp_path / "board_a.jpg"
    image_b = tmp_path / "board_b.jpg"
    image_a.write_bytes(b"board-a")
    image_b.write_bytes(b"board-b")
    evidence = [
        VisualEvidence(
            request_id="window__board_scan",
            kind=VisualKind.BOARD_SCAN,
            confidence=1.0,
            frame_paths=[str(image_a), str(image_b)],
            frame_timestamps=[1.0, 4.0],
        )
    ]
    kb = LectureKnowledgeBase(lecture_id="lecture", title="Lecture")

    orchestrator.extract_observations(chunk, evidence, kb)

    assert llm.images == [image_a, image_b]
    assert llm.guided_json is False
    assert "Attached board-image index" in llm.prompt
    assert "request_id=window__board_scan" in llm.prompt
    assert "timestamp=1.000s" in llm.prompt
    assert "DIRECT SENSOR EVIDENCE" in llm.prompt
