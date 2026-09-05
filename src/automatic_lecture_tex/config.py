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
    def validate_source(self) -> SourceConfig:
        if self.type == "file" and self.path is None:
            raise ValueError("file source requires path")
        if self.type == "youtube" and not self.url:
            raise ValueError("youtube source requires url")
        return self


class LectureConfig(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str | None = None
    source: SourceConfig


class CourseConfig(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str
    language: str = "ru"
    lectures: list[LectureConfig]


class ASRConfig(BaseModel):
    backend: Literal["qwen3", "qwen3_hf", "faster_whisper"] = "qwen3"
    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str | None = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str | None = "ru"
    hotwords: list[str] = Field(default_factory=list)
    chunk_seconds: float = 60.0
    max_new_tokens: int = 2048
    batch_size: int = Field(default=4, ge=1)
    segment_target_seconds: float = Field(default=20.0, gt=0)
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True
    vad_min_silence_ms: int = Field(default=500, ge=0)
    condition_on_previous_text: bool = True
    hallucination_silence_threshold: float | None = Field(default=2.0, gt=0)


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    output_language: str = "ru"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: float = 300.0
    thinking: bool = False
    max_retries: int = Field(default=2, ge=0, le=5)
    math_audit: bool = True
    math_audit_min_equals: int = Field(default=4, ge=1, le=50)


class NotesConfig(BaseModel):
    architecture: Literal["knowledge", "legacy"] = "knowledge"
    chunk_target_seconds: float = Field(default=480.0, gt=0)
    chunk_overlap_seconds: float = Field(default=120.0, ge=0)
    boundary_context_seconds: float = Field(default=120.0, ge=0)
    knowledge_max_active_claims: int = Field(default=160, ge=20, le=1000)
    knowledge_recent_observations: int = Field(default=80, ge=10, le=1000)
    max_outline_sections: int = Field(default=40, ge=1, le=200)
    global_validation: bool = True
    global_validation_apply_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    visual_rule_selector: bool = True
    visual_llm_selector: bool = False
    visual_dedupe_seconds: float = 8.0
    max_low_confidence_visual_requests: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def validate_chunk_geometry(self) -> NotesConfig:
        if self.chunk_overlap_seconds >= self.chunk_target_seconds:
            raise ValueError("notes.chunk_overlap_seconds must be smaller than chunk_target_seconds")
        return self


class VisionConfig(BaseModel):
    frame_offsets_seconds: list[float] = Field(default_factory=lambda: [-3.0, 2.0, 7.0])
    youtube_video_format: str = "bestvideo[height<=1080]/best[height<=1080]"
    max_requests_per_chunk: int = 4
    max_workers: int = Field(default=3, ge=1, le=8)


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
