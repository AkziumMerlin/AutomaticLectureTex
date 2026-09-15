from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import RuntimeConfig, SourceConfig, VisionConfig
from .schemas import ExtractedFrame
from .util import run_checked

logger = logging.getLogger(__name__)


class MediaSource(ABC):
    def __init__(self, runtime: RuntimeConfig, vision: VisionConfig) -> None:
        self.runtime = runtime
        self.vision = vision

    @abstractmethod
    def prepare_audio(self, output_wav: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def extract_frames(self, timestamps: list[float], output_dir: Path) -> list[ExtractedFrame]:
        raise NotImplementedError

    @abstractmethod
    def identity(self) -> dict[str, str]:
        raise NotImplementedError

    def _normalize_audio(self, input_path: Path, output_wav: Path) -> Path:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                self.runtime.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ]
        )
        return output_wav


class LocalMediaSource(MediaSource):
    def __init__(self, path: Path, runtime: RuntimeConfig, vision: VisionConfig) -> None:
        super().__init__(runtime, vision)
        self.path = path.resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def prepare_audio(self, output_wav: Path) -> Path:
        return self._normalize_audio(self.path, output_wav)

    def extract_frames(self, timestamps: list[float], output_dir: Path) -> list[ExtractedFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ExtractedFrame] = []
        for index, timestamp in enumerate(timestamps):
            timestamp = max(0.0, timestamp)
            path = output_dir / f"frame_{index:02d}_{timestamp:.3f}.jpg"
            if not path.is_file() or path.stat().st_size == 0:
                run_checked(
                    [
                        self.runtime.ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        str(self.path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(path),
                    ]
                )
            frames.append(ExtractedFrame(timestamp=timestamp, path=path))
        return frames

    def identity(self) -> dict[str, str]:
        stat = self.path.stat()
        return {
            "type": "file",
            "path": str(self.path),
            "size": str(stat.st_size),
            "mtime_ns": str(stat.st_mtime_ns),
        }


class YouTubeMediaSource(MediaSource):
    def __init__(self, url: str, runtime: RuntimeConfig, vision: VisionConfig) -> None:
        super().__init__(runtime, vision)
        self.url = url
        self._resolved_url: str | None = None
        self._stream_info: tuple[str, dict[str, str]] | None = None

    def _is_playlist_url(self) -> bool:
        parsed = urlparse(self.url)
        query = parse_qs(parsed.query)
        return parsed.path.rstrip("/").endswith("playlist") or (
            "list" in query and "v" not in query
        )

    def _media_url(self) -> str:
        if not self._is_playlist_url():
            return self.url
        if self._resolved_url is not None:
            return self._resolved_url
        proc = run_checked(
            [
                self.runtime.yt_dlp,
                "--flat-playlist",
                "--playlist-items",
                "1",
                "--no-warnings",
                "--print",
                "webpage_url",
                self.url,
            ]
        )
        urls = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(urls) != 1:
            raise RuntimeError(
                f"yt-dlp resolved {len(urls)} playlist entries, expected exactly one"
            )
        self._resolved_url = urls[0]
        return self._resolved_url

    def prepare_audio(self, output_wav: Path) -> Path:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="automatic-lecture-tex-audio-") as tmp_name:
            tmp = Path(tmp_name)
            template = tmp / "audio.%(ext)s"
            run_checked(
                [
                    self.runtime.yt_dlp,
                    "--no-playlist",
                    "--no-progress",
                    "--quiet",
                    "-f",
                    "bestaudio",
                    "-o",
                    str(template),
                    self._media_url(),
                ]
            )
            candidates = [p for p in tmp.glob("audio.*") if p.is_file()]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"yt-dlp produced {len(candidates)} audio files, expected exactly one"
                )
            return self._normalize_audio(candidates[0], output_wav)

    def _resolve_video_stream(self) -> tuple[str, dict[str, str]]:
        """Resolve one directly seekable video stream without downloading the full lecture."""

        if self._stream_info is not None:
            return self._stream_info
        proc = run_checked(
            [
                self.runtime.yt_dlp,
                "--no-playlist",
                "--no-progress",
                "--quiet",
                "--dump-single-json",
                "-f",
                self.vision.youtube_video_format,
                self._media_url(),
            ]
        )
        payload = json.loads(proc.stdout)
        stream_url = str(payload.get("url") or "").strip()
        if not stream_url:
            raise RuntimeError("yt-dlp did not expose a direct video stream URL")
        headers = {
            str(key): str(value)
            for key, value in (payload.get("http_headers") or {}).items()
            if key and value
        }
        self._stream_info = (stream_url, headers)
        return self._stream_info

    @staticmethod
    def _ffmpeg_headers(headers: dict[str, str]) -> str:
        return "".join(f"{key}: {value}\r\n" for key, value in headers.items())

    def _extract_frames_from_stream(
        self,
        safe_times: list[float],
        targets: list[Path],
    ) -> list[ExtractedFrame]:
        stream_url, headers = self._resolve_video_stream()
        header_blob = self._ffmpeg_headers(headers)
        frames: list[ExtractedFrame] = []
        for timestamp, path in zip(safe_times, targets, strict=True):
            if not path.is_file() or path.stat().st_size == 0:
                command = [
                    self.runtime.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                ]
                if header_blob:
                    command.extend(["-headers", header_blob])
                command.extend(
                    [
                        "-i",
                        stream_url,
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(path),
                    ]
                )
                run_checked(command)
            frames.append(ExtractedFrame(timestamp=timestamp, path=path))
        return frames

    def _download_section(
        self,
        *,
        start: float,
        end: float,
        directory: Path,
        force_keyframes: bool,
    ) -> Path:
        template = directory / "segment.%(ext)s"
        section = f"*{start:.3f}-{end:.3f}"
        command = [
            self.runtime.yt_dlp,
            "--no-playlist",
            "--no-progress",
            "--quiet",
            "--download-sections",
            section,
        ]
        if force_keyframes:
            command.append("--force-keyframes-at-cuts")
        command.extend(
            [
                "-f",
                self.vision.youtube_video_format,
                "-o",
                str(template),
                self._media_url(),
            ]
        )
        run_checked(command)
        candidates = [p for p in directory.glob("segment.*") if p.is_file()]
        if not candidates:
            raise RuntimeError("yt-dlp did not produce a video segment")
        return max(candidates, key=lambda p: p.stat().st_size)

    def _extract_frames_from_section(
        self,
        *,
        safe_times: list[float],
        targets: list[Path],
        start: float,
        segment: Path,
    ) -> list[ExtractedFrame]:
        frames: list[ExtractedFrame] = []
        for timestamp, path in zip(safe_times, targets, strict=True):
            relative = max(0.0, timestamp - start)
            if not path.is_file() or path.stat().st_size == 0:
                run_checked(
                    [
                        self.runtime.ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{relative:.3f}",
                        "-i",
                        str(segment),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(path),
                    ]
                )
            frames.append(ExtractedFrame(timestamp=timestamp, path=path))
        return frames

    def extract_frames(self, timestamps: list[float], output_dir: Path) -> list[ExtractedFrame]:
        if not timestamps:
            return []
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_times = [max(0.0, value) for value in timestamps]
        targets = [
            output_dir / f"frame_{index:02d}_{timestamp:.3f}.jpg"
            for index, timestamp in enumerate(safe_times)
        ]
        if all(path.is_file() and path.stat().st_size > 0 for path in targets):
            return [
                ExtractedFrame(timestamp=timestamp, path=path)
                for timestamp, path in zip(safe_times, targets, strict=True)
            ]

        start = max(0.0, min(safe_times) - 2.0)
        end = max(safe_times) + 2.0
        exact_error: Exception | None = None
        with tempfile.TemporaryDirectory(prefix="automatic-lecture-tex-video-") as tmp_name:
            tmp = Path(tmp_name)
            try:
                segment = self._download_section(
                    start=start,
                    end=end,
                    directory=tmp,
                    force_keyframes=True,
                )
                return self._extract_frames_from_section(
                    safe_times=safe_times,
                    targets=targets,
                    start=start,
                    segment=segment,
                )
            except Exception as exc:
                exact_error = exc
                logger.warning(
                    "exact yt-dlp section extraction failed; retrying direct stream seek: %s",
                    exc,
                )

        try:
            return self._extract_frames_from_stream(safe_times, targets)
        except Exception as stream_error:
            logger.warning(
                "direct stream frame extraction failed; retrying section extraction without "
                "forced keyframes: %s",
                stream_error,
            )

        # Last media-level fallback: avoid re-encoding and ask for a wider section. The cut may land
        # on a neighboring keyframe, which is acceptable for board OCR and temporal aggregation.
        fallback_start = max(0.0, start - 6.0)
        fallback_end = end + 6.0
        with tempfile.TemporaryDirectory(prefix="automatic-lecture-tex-video-fallback-") as tmp_name:
            tmp = Path(tmp_name)
            try:
                segment = self._download_section(
                    start=fallback_start,
                    end=fallback_end,
                    directory=tmp,
                    force_keyframes=False,
                )
                return self._extract_frames_from_section(
                    safe_times=safe_times,
                    targets=targets,
                    start=fallback_start,
                    segment=segment,
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "all remote frame extraction strategies failed; "
                    f"exact-section={exact_error}; fallback-section={fallback_error}"
                ) from fallback_error

    def identity(self) -> dict[str, str]:
        return {"type": "youtube", "url": self.url}


def media_source_from_config(
    source: SourceConfig, runtime: RuntimeConfig, vision: VisionConfig
) -> MediaSource:
    if source.type == "file":
        assert source.path is not None
        return LocalMediaSource(source.path, runtime, vision)
    assert source.url is not None
    return YouTubeMediaSource(source.url, runtime, vision)


def probe_duration(path: Path, runtime: RuntimeConfig) -> float:
    proc = run_checked(
        [
            runtime.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(proc.stdout)
    duration = float(payload["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"invalid media duration: {duration}")
    return duration


def copy_asset(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
