from pathlib import Path

import pytest
from pydantic import ValidationError

from automatic_lecture_tex.config import AppConfig


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
