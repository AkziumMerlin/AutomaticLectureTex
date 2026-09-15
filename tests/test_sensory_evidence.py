from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from automatic_lecture_tex import asr as asr_module
from automatic_lecture_tex.asr import GigaAMBackend
from automatic_lecture_tex.board import build_temporal_board_composite, temporal_sample_offsets
from automatic_lecture_tex.config import ASRConfig, MathOCRConfig, RuntimeConfig, VisionConfig
from automatic_lecture_tex.schemas import (
    ExtractedFrame,
    LectureChunk,
    Transcript,
    TranscriptSegment,
    VisualEvidence,
)
from automatic_lecture_tex.sensory_evidence import collect_visual_evidence


class _FakeGigaAMModel:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, path: str, word_timestamps: bool = False):
        del path
        assert word_timestamps is True
        self.calls += 1
        return SimpleNamespace(
            text=f"фрагмент {self.calls}",
            words=[SimpleNamespace(text="слово", start=1.0, end=2.0)],
        )


def test_gigaam_host_chunking_preserves_global_word_timestamps(tmp_path, monkeypatch):
    model = _FakeGigaAMModel()
    captured = {}

    def load_model(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return model

    monkeypatch.setitem(sys.modules, "gigaam", SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(asr_module, "probe_duration", lambda *_args: 50.0)

    def fake_extract(_runtime, _audio, _start, _duration, output):
        Path(output).write_bytes(b"wav")

    monkeypatch.setattr(asr_module, "_extract_audio_chunk", fake_extract)
    config = ASRConfig(
        backend="gigaam",
        model="v3_e2e_rnnt",
        language="ru",
        chunk_seconds=60,
        device="cuda",
        gigaam_fp16_encoder=True,
        gigaam_use_flash=False,
    )
    backend = GigaAMBackend(config, RuntimeConfig(work_dir=tmp_path))

    transcript = backend.transcribe("lecture", tmp_path / "audio.wav")

    assert captured["name"] == "v3_e2e_rnnt"
    assert captured["device"] == "cuda"
    assert model.calls == 3
    assert [segment.start for segment in transcript.segments] == [1.0, 25.0, 49.0]
    assert [segment.end for segment in transcript.segments] == [2.0, 26.0, 50.0]
    assert [segment.words[0].start for segment in transcript.segments] == [1.0, 25.0, 49.0]


def _write_board_frame(path: Path, moving_column: int) -> None:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[8, :, :] = 0  # persistent writing
    image[:, moving_column, :] = 0  # moving lecturer/occlusion proxy
    Image.fromarray(image).save(path)


def test_temporal_median_removes_moving_occlusion_and_keeps_persistent_writing(tmp_path):
    frames = []
    for index, column in enumerate([1, 3, 5, 7, 9]):
        path = tmp_path / f"frame_{index}.png"
        _write_board_frame(path, column)
        frames.append(ExtractedFrame(timestamp=float(index), path=path))

    output = build_temporal_board_composite(frames, tmp_path / "composite.jpg", autocontrast=False)
    composite = np.asarray(Image.open(output).convert("RGB"))

    assert int(composite[8].mean()) < 40
    # Each moving column was dark in only one of five frames, so the median should remove it away
    # from the persistent horizontal line.
    assert int(composite[2, 1].mean()) > 220
    assert int(composite[2, 9].mean()) > 220


def test_temporal_offsets_are_bounded_and_symmetric():
    config = VisionConfig(
        temporal_composite_enabled=True,
        temporal_window_seconds=10,
        temporal_sample_period_seconds=2,
        temporal_max_frames=11,
    )
    offsets = temporal_sample_offsets(config)
    assert len(offsets) == 11
    assert offsets[0] == -10
    assert offsets[-1] == 10
    assert offsets[len(offsets) // 2] == 0


class _FakeSource:
    def extract_frames(self, timestamps, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, timestamp in enumerate(timestamps):
            path = output_dir / f"{index}.png"
            _write_board_frame(path, 1 + (index % 10))
            frames.append(ExtractedFrame(timestamp=timestamp, path=path))
        return frames


class _FakeLLM:
    def __init__(self) -> None:
        self.first_image = None
        self.image_names = []

    def resolve_visual_request(self, request, chunk, frame_paths, frame_timestamps):
        del request, chunk, frame_timestamps
        self.first_image = frame_paths[0]
        self.image_names = [path.name for path in frame_paths]
        return VisualEvidence(kind="equation", raw_latex="x=1", latex="x=1", confidence=0.9)


def test_visual_collector_sends_raw_primary_then_temporal_composite(tmp_path):
    llm = _FakeLLM()
    vision = VisionConfig(
        frame_offsets_seconds=[-3, 2, 7],
        temporal_composite_enabled=True,
        temporal_window_seconds=4,
        temporal_sample_period_seconds=2,
        temporal_max_frames=5,
        math_ocr=MathOCRConfig(backend="none"),
    )
    pipeline = SimpleNamespace(
        config=SimpleNamespace(
            notes=SimpleNamespace(
                visual_rule_selector=True,
                visual_llm_selector=False,
                max_low_confidence_visual_requests=0,
                visual_dedupe_seconds=8.0,
            ),
            vision=vision,
            latex=SimpleNamespace(output_dir=tmp_path / "tex"),
        ),
        llm=llm,
    )
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_0",
                start=10,
                end=12,
                text="Запишем это на доске",
            )
        ],
    )
    chunk = LectureChunk(
        id="window_0",
        start=10,
        end=12,
        segment_ids=["seg_0"],
        text="Запишем это на доске",
    )

    requests, evidence, _elapsed = collect_visual_evidence(
        pipeline,
        SimpleNamespace(id="lecture"),
        chunk,
        transcript,
        _FakeSource(),
        tmp_path / "work",
        tmp_path / "figures",
        {},
    )

    assert len(requests) == 1
    assert len(evidence) == 1
    assert llm.first_image is not None
    assert llm.first_image.name != "board_composite.jpg"
    assert llm.image_names[1] == "board_composite.jpg"
    assert llm.first_image.is_file()
