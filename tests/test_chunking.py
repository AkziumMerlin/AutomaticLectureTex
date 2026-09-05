from automatic_lecture_tex.chunking import chunk_transcript
from automatic_lecture_tex.schemas import Transcript, TranscriptSegment


def _transcript():
    return Transcript(
        lecture_id="l1",
        segments=[
            TranscriptSegment(id="s0", start=0, end=60, text="a"),
            TranscriptSegment(id="s1", start=60, end=120, text="b"),
            TranscriptSegment(id="s2", start=120, end=180, text="c"),
            TranscriptSegment(id="s3", start=180, end=240, text="d"),
        ],
    )


def test_chunk_transcript_respects_segment_boundaries():
    chunks = chunk_transcript(_transcript(), target_seconds=150)
    assert [c.segment_ids for c in chunks] == [["s0", "s1"], ["s2", "s3"]]
    assert chunks[0].id == "chunk_0000"
    assert chunks[0].text == "a\nb"
    assert chunks[0].timestamped_text == "[00:00.000-01:00.000] a\n[01:00.000-02:00.000] b"


def test_chunk_transcript_overlaps_evidence_windows():
    chunks = chunk_transcript(_transcript(), target_seconds=180, overlap_seconds=70)
    assert [c.segment_ids for c in chunks] == [
        ["s0", "s1", "s2"],
        ["s1", "s2", "s3"],
    ]
    assert [c.id for c in chunks] == ["window_0000", "window_0001"]


def test_overlap_must_be_smaller_than_window():
    try:
        chunk_transcript(_transcript(), target_seconds=120, overlap_seconds=120)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
