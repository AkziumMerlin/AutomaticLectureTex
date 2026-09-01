from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceConfig(BaseModel):
    type: Literal["file", "youtube"]
    path: Path | None = None
    url: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if self.type == "file" and self.path is None:
            raise ValueError("file source requires path")
        if self.type == "youtube" and not self.url:
            raise ValueError("youtube source requires url")
        return self


class LectureConfig(BaseModel):
    id: str
    title: str | None = None
    source: SourceConfig


class CourseConfig(BaseModel):
    id: str
    title: str
    language: str = "ru"
    lectures: list[LectureConfig]


class ASRConfig(BaseModel):
    backend: Literal["qwen3", "faster_whisper"] = "qwen3"
    model: str = "Qwen/Qwen3-ASR-1.7B-hf"
    aligner_model: str | None = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    language: str | None = "ru"
    hotwords: list[str] = Field(default_factory=list)
    chunk_seconds: float = 60.0
    max_new_tokens: int = 768
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    whisper_compute_type: str = "float16"


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: float = 300.0
    thinking: bool = False


class NotesConfig(BaseModel):
    chunk_target_seconds: float = 180.0
    visual_rule_selector: bool = True
    visual_llm_selector: bool = True
    visual_dedupe_seconds: float = 8.0


class VisionConfig(BaseModel):
    frame_offsets_seconds: list[float] = Field(default_factory=lambda: [-3.0, 2.0, 7.0])
    youtube_video_format: str = "bestvideo[height<=1080]/best[height<=1080]"
    max_requests_per_chunk: int = 4


class LiteratureConfig(BaseModel):
    enabled: bool = False
    directory: Path = Path("literature")
    retrieval_top_k: int = 4
    chunk_chars: int = 1800


class LatexConfig(BaseModel):
    output_dir: Path = Path("tex")
    compiler: str = "latexmk"
    compile: bool = False


class RuntimeConfig(BaseModel):
    work_dir: Path = Path("work")
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    yt_dlp: str = "yt-dlp"


class AppConfig(BaseModel):
    course: CourseConfig
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    latex: LatexConfig = Field(default_factory=LatexConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = AppConfig.model_validate(raw)
    base = path.parent

    for lecture in cfg.course.lectures:
        if lecture.source.path is not None and not lecture.source.path.is_absolute():
            lecture.source.path = (base / lecture.source.path).resolve()
    if not cfg.runtime.work_dir.is_absolute():
        cfg.runtime.work_dir = (base / cfg.runtime.work_dir).resolve()
    if not cfg.latex.output_dir.is_absolute():
        cfg.latex.output_dir = (base / cfg.latex.output_dir).resolve()
    if not cfg.literature.directory.is_absolute():
        cfg.literature.directory = (base / cfg.literature.directory).resolve()
    return cfg
