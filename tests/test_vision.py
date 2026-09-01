from automatic_lecture_tex.schemas import LectureChunk, Transcript, TranscriptSegment, VisualRequest
from automatic_lecture_tex.vision import dedupe_visual_requests, select_rule_based_visual_requests


def test_rule_selector_marks_board_reference():
    transcript = Transcript(
        lecture_id="l1",
        segments=[TranscriptSegment(id="s", start=10, end=20, text="Как видно на рисунке, здесь две стрелки")],
    )
    chunk = LectureChunk(id="c", start=10, end=20, segment_ids=["s"], text=transcript.segments[0].text)
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
