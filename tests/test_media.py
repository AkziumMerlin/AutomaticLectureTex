from subprocess import CompletedProcess

from automatic_lecture_tex.config import RuntimeConfig, VisionConfig
from automatic_lecture_tex.media import LocalMediaSource, YouTubeMediaSource


def test_playlist_url_resolves_first_item(monkeypatch):
    calls = []

    def fake_run_checked(args, **kwargs):
        calls.append(args)
        return CompletedProcess(
            args, 0, stdout="https://www.youtube.com/watch?v=abc123\n", stderr=""
        )

    monkeypatch.setattr("automatic_lecture_tex.media.run_checked", fake_run_checked)
    source = YouTubeMediaSource(
        "https://www.youtube.com/playlist?list=PL123",
        RuntimeConfig(),
        VisionConfig(),
    )

    assert source._media_url() == "https://www.youtube.com/watch?v=abc123"
    assert source._media_url() == "https://www.youtube.com/watch?v=abc123"
    assert len(calls) == 1
    assert "--playlist-items" in calls[0]
    assert calls[0][calls[0].index("--playlist-items") + 1] == "1"


def test_direct_youtube_url_does_not_resolve(monkeypatch):
    def fail_run_checked(*args, **kwargs):
        raise AssertionError("direct video URL must not invoke yt-dlp resolver")

    monkeypatch.setattr("automatic_lecture_tex.media.run_checked", fail_run_checked)
    url = "https://www.youtube.com/watch?v=abc123&list=PL123"
    source = YouTubeMediaSource(url, RuntimeConfig(), VisionConfig())

    assert source._media_url() == url


def test_local_frame_extraction_reuses_existing_frame(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "frames"
    output.mkdir()
    expected = output / "frame_00_12.500.jpg"
    expected.write_bytes(b"frame")

    def fail_run_checked(*args, **kwargs):
        raise AssertionError("cached frame must not invoke ffmpeg")

    monkeypatch.setattr("automatic_lecture_tex.media.run_checked", fail_run_checked)
    source = LocalMediaSource(video, RuntimeConfig(), VisionConfig())

    frames = source.extract_frames([12.5], output)

    assert frames[0].path == expected
