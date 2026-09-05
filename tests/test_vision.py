from automatic_lecture_tex.schemas import LectureChunk, Transcript, TranscriptSegment, VisualRequest
from automatic_lecture_tex.vision import (
    dedupe_visual_requests,
    namespace_visual_requests,
    select_rule_based_visual_requests,
)


def test_rule_selector_marks_board_reference():
    transcript = Transcript(
        lecture_id="l1",
        segments=[
            TranscriptSegment(
                id="s", start=10, end=20, text="Как видно на рисунке, здесь две стрелки"
            )
        ],
    )
    chunk = LectureChunk(
        id="c", start=10, end=20, segment_ids=["s"], text=transcript.segments[0].text
    )
    requests = select_rule_based_visual_requests(chunk, transcript)
    assert len(requests) == 1
    assert requests[0].priority == 5


def test_dedupe_prefers_priority():
    requests = [
        VisualRequest(id="low", timestamp=10, reason="x", question="x", priority=1),
        VisualRequest(id="high", timestamp=12, reason="y", question="y", priority=5),
    ]
    result = dedupe_visual_requests(requests, within_seconds=5, limit=4)
    assert [r.id for r in result] == ["high"]


def test_namespace_visual_requests_makes_ids_unique_and_path_safe():
    requests = [
        VisualRequest(id="req_1", timestamp=10, reason="x", question="x"),
        VisualRequest(id="../../req 1", timestamp=20, reason="y", question="y"),
    ]

    result = namespace_visual_requests("chunk_0003", requests)

    assert [request.id for request in result] == [
        "chunk_0003_req_1_00",
        "chunk_0003_req_1_01",
    ]


def test_rule_selector_caps_low_confidence_frame_requests():
    segments = [
        TranscriptSegment(
            id=f"s{index}", start=index * 10, end=index * 10 + 5, text="speech", confidence=0.1
        )
        for index in range(5)
    ]
    transcript = Transcript(lecture_id="l1", segments=segments)
    chunk = LectureChunk(
        id="c", start=0, end=50, segment_ids=[segment.id for segment in segments], text="speech"
    )

    requests = select_rule_based_visual_requests(chunk, transcript, max_low_confidence_requests=2)

    assert len(requests) == 2
