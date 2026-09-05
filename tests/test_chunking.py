from automatic_lecture_tex.chunking import chunk_transcript
from automatic_lecture_tex.schemas import Transcript, TranscriptSegment


def test_chunk_transcript_respects_segment_boundaries():
    transcript = Transcript(
        lecture_id="l1",
        segments=[
            TranscriptSegment(id="s0", start=0, end=60, text="a"),
            TranscriptSegment(id="s1", start=60, end=120, text="b"),
            TranscriptSegment(id="s2", start=120, end=180, text="c"),
            TranscriptSegment(id="s3", start=180, end=240, text="d"),
        ],
    )
    chunks = chunk_transcript(transcript, target_seconds=150)
    assert [c.segment_ids for c in chunks] == [["s0", "s1"], ["s2", "s3"]]
    assert chunks[0].text == "a\nb"
    assert chunks[0].timestamped_text == "[00:00.000-01:00.000] a\n[01:00.000-02:00.000] b"
