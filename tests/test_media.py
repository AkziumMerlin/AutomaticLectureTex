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



def test_ytdlp_proxy_is_added_only_to_ytdlp_commands() -> None:
    source = YouTubeMediaSource(
        "https://www.youtube.com/watch?v=abc123",
        RuntimeConfig(yt_dlp_proxy_url="socks5://127.0.0.1:1080"),
        VisionConfig(),
    )

    assert source._yt_dlp_command() == [
        "yt-dlp",
        "--proxy",
        "socks5://127.0.0.1:1080",
    ]


def test_ytdlp_proxy_skips_direct_ffmpeg_stream_fallback(tmp_path, monkeypatch) -> None:
    calls: list[bool] = []

    def fake_download(
        self,
        *,
        start,
        end,
        directory,
        force_keyframes,
    ):
        del self, start, end
        calls.append(force_keyframes)
        if force_keyframes:
            raise RuntimeError("forced-keyframe extraction failed")
        segment = directory / "segment.mp4"
        segment.write_bytes(b"segment")
        return segment

    def fail_direct_stream(*args, **kwargs):
        raise AssertionError("direct ffmpeg stream fallback must not bypass scoped yt-dlp proxy")

    def fake_extract_section(self, **kwargs):
        del self, kwargs
        return []

    monkeypatch.setattr(YouTubeMediaSource, "_download_section", fake_download)
    monkeypatch.setattr(YouTubeMediaSource, "_extract_frames_from_stream", fail_direct_stream)
    monkeypatch.setattr(YouTubeMediaSource, "_extract_frames_from_section", fake_extract_section)

    source = YouTubeMediaSource(
        "https://www.youtube.com/watch?v=abc123",
        RuntimeConfig(yt_dlp_proxy_url="socks5://127.0.0.1:1080"),
        VisionConfig(),
    )

    assert source.extract_frames([12.0], tmp_path / "frames") == []
    assert calls == [True, False]
