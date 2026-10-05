from pathlib import Path

import pytest
from pydantic import ValidationError

from automatic_lecture_tex.config import AppConfig, load_config


def test_minimal_config_parses():
    cfg = AppConfig.model_validate(
        {
            "course": {
                "id": "c",
                "title": "Course",
                "lectures": [{"id": "l", "source": {"type": "file", "path": "lecture.mp4"}}],
            }
        }
    )
    assert cfg.course.lectures[0].source.path == Path("lecture.mp4")
    assert cfg.asr.backend == "qwen3"
    assert cfg.notes.architecture == "knowledge"
    assert cfg.notes.chunk_overlap_seconds < cfg.notes.chunk_target_seconds


def test_lecture_id_cannot_escape_work_directory():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "course": {
                    "id": "course",
                    "title": "Course",
                    "lectures": [
                        {"id": "../escape", "source": {"type": "file", "path": "lecture.mp4"}}
                    ],
                }
            }
        )


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "course": {
                    "id": "course",
                    "title": "Course",
                    "lectures": [
                        {"id": "lecture", "source": {"type": "file", "path": "lecture.mp4"}}
                    ],
                },
                "notes": {
                    "chunk_target_seconds": 100,
                    "chunk_overlap_seconds": 100,
                },
            }
        )



def test_openrouter_deepseek_config_parses():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "functional_analysis_vk_lecture01_state_20s_openrouter.yaml"
    )

    cfg = load_config(config_path)

    assert cfg.llm.base_url == "https://openrouter.ai/api/v1"
    assert cfg.llm.proxy_url == "socks5://127.0.0.1:1080"
    assert cfg.llm.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.llm.compatibility_mode == "generic"
    assert cfg.llm.reasoning_transport == "openrouter"
    assert cfg.llm.model == "deepseek/deepseek-v4.1-flash"
    assert cfg.asr.backend == "openai_compatible"
    assert cfg.asr.model == "qwen/qwen3-asr-1.7b"
    assert cfg.asr.transcription_response_format == "verbose_json"
    assert cfg.asr.extra_body["timestamp_granularities"] == ["word"]
    assert cfg.vision.math_ocr.backend == "openai_compatible"
    assert cfg.vision.math_ocr.openai_model == "deepseek/deepseek-v4.1-flash"
    assert cfg.runtime.yt_dlp_proxy_url == "socks5://127.0.0.1:1080"



def test_openrouter_omni_video_ablation_config_parses():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "functional_analysis_vk_lecture01_state_20s_openrouter_omni_video.yaml"
    )

    cfg = load_config(config_path)

    assert cfg.notes.window_evidence_backend == "native_video"
    assert cfg.notes.native_video_model == "qwen/qwen3.8-omni-flash"
    assert cfg.notes.native_video_height == 720
    assert cfg.notes.native_video_max_bytes == 7_000_000
    assert cfg.notes.visual_chunk_board_scan is False
    assert cfg.vision.formula_detection.enabled is False
    assert cfg.vision.math_ocr.backend == "none"
    assert cfg.course.lectures[0].source.url == (
        "https://www.youtube.com/watch?v=H3KIzsh8a0Q"
    )
    assert cfg.llm.model == "deepseek/deepseek-v4.1-flash"
    assert cfg.llm.proxy_url == "socks5://127.0.0.1:1080"
    assert cfg.runtime.yt_dlp_proxy_url == "socks5://127.0.0.1:1080"
