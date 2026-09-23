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
    backend: Literal["qwen3", "qwen3_hf", "faster_whisper", "gigaam"] = "qwen3"
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
    gigaam_fp16_encoder: bool = True
    gigaam_use_flash: bool = False
    gigaam_vad_enabled: bool = True
    gigaam_vad_max_speech_seconds: float = Field(default=22.0, gt=1.0, le=24.0)
    gigaam_vad_merge_gap_seconds: float = Field(default=0.35, ge=0.0, le=3.0)
    gigaam_vad_pad_seconds: float = Field(default=0.15, ge=0.0, le=1.0)


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
    architecture: Literal["linear", "state", "knowledge", "legacy"] = "knowledge"
    chunk_target_seconds: float = Field(default=480.0, gt=0)
    chunk_overlap_seconds: float = Field(default=120.0, ge=0)
    boundary_context_seconds: float = Field(default=120.0, ge=0)
    knowledge_max_active_claims: int = Field(default=160, ge=20, le=1000)
    knowledge_recent_observations: int = Field(default=80, ge=10, le=1000)
    max_outline_sections: int = Field(default=40, ge=1, le=200)
    global_validation: bool = True
    global_validation_apply_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    linear_global_editor_batch_chars: int = Field(default=16000, ge=4000, le=60000)
    linear_global_editor_catalog_excerpt_chars: int = Field(default=140, ge=40, le=800)
    linear_global_editor_conventions: list[str] = Field(default_factory=list, exclude=True)
    hierarchy_batch_episodes: int = Field(default=24, ge=2, le=100)
    episode_synthesis_max_evidence_chars: int = Field(default=24000, ge=4000, le=200000)
    state_section_max_evidence_chars: int = Field(default=28000, ge=8000, le=120000)
    episode_symbol_context_limit: int = Field(default=24, ge=0, le=200)
    episode_transcript_context_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=120.0,
        exclude=True,
    )
    visual_chunk_board_scan: bool = False
    visual_rule_selector: bool = True
    visual_llm_selector: bool = False
    visual_dedupe_seconds: float = 8.0
    max_low_confidence_visual_requests: int = Field(default=1, ge=0, le=10)
    linear_recent_blocks: int = Field(default=8, ge=0, le=50)
    linear_previous_transcript_segments: int = Field(default=8, ge=0, le=100)
    linear_correction_scan_enabled: bool = True
    linear_correction_catalog_chars: int = Field(default=320, ge=80, le=2000)
    linear_patch_apply_threshold: float = Field(default=0.90, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_geometry(self) -> NotesConfig:
        if self.chunk_overlap_seconds >= self.chunk_target_seconds:
            raise ValueError(
                "notes.chunk_overlap_seconds must be smaller than chunk_target_seconds"
            )
        return self


class FormulaDetectionConfig(BaseModel):
    enabled: bool = False
    backend: Literal["none", "yolov8"] = "none"
    model_path: Path | None = None
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=1280, ge=320, le=2048)
    device: Literal["cuda", "cpu"] = "cuda"
    max_crops_per_state: int = Field(default=6, ge=1, le=32)
    max_crops_per_chunk: int = Field(default=8, ge=1, le=64)
    min_width_px: int = Field(default=36, ge=4, le=2048)
    min_height_px: int = Field(default=18, ge=4, le=2048)
    padding_fraction: float = Field(default=0.08, ge=0.0, le=0.50)
    contact_sheet_enabled: bool = True
    contact_sheet_columns: int = Field(default=2, ge=1, le=4)
    normalize_dark_board: bool = True
    dark_board_threshold: int = Field(default=128, ge=0, le=255)


class MathOCRConfig(BaseModel):
    backend: Literal["none", "mathpix", "unimernet", "latexocr"] = "none"
    min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    board_scan_enabled: bool = False
    board_scan_max_images: int = Field(default=3, ge=1, le=8)
    device: Literal["cuda", "cpu"] = "cuda"
    mathpix_app_id_env: str = "MATHPIX_APP_ID"
    mathpix_app_key_env: str = "MATHPIX_APP_KEY"
    unimernet_config_path: Path | None = None
    unimernet_python_path: Path | None = None
    normalize_dark_formula: bool = True
    dark_formula_threshold: int = Field(default=128, ge=0, le=255)


class VisionConfig(BaseModel):
    frame_offsets_seconds: list[float] = Field(default_factory=lambda: [-3.0, 2.0, 7.0])
    youtube_video_format: str = "bestvideo[height<=1080]/best[height<=1080]"
    max_requests_per_chunk: int = 4
    max_workers: int = Field(default=3, ge=1, le=8)
    temporal_composite_enabled: bool = False
    temporal_window_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    temporal_sample_period_seconds: float = Field(default=2.0, ge=0.25, le=10.0)
    temporal_max_frames: int = Field(default=11, ge=3, le=61)

    # Host-side board ROI detection. The detector fails closed to ordinary raw frames. ``board_crop_roi``
    # is an optional normalized (left, top, right, bottom) override for whiteboards/unusual rooms.
    board_auto_crop_enabled: bool = True
    board_crop_roi: tuple[float, float, float, float] | None = None
    board_crop_tiles: int = Field(default=3, ge=1, le=4)
    board_crop_tile_overlap: float = Field(default=0.14, ge=0.0, le=0.45)
    board_crop_padding_fraction: float = Field(default=0.025, ge=0.0, le=0.20)
    board_crop_min_area_fraction: float = Field(default=0.16, ge=0.05, le=0.80)
    board_crop_axis_density: float = Field(default=0.25, ge=0.05, le=0.90)
    board_crop_color_distance: float = Field(default=0.22, ge=0.05, le=0.60)
    board_crop_max_luminance: float = Field(default=0.78, ge=0.20, le=0.98)
    board_crop_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    board_crop_max_vlm_images: int = Field(default=5, ge=2, le=8)
    board_sampling_mode: Literal["uniform", "change"] = "uniform"
    board_uniform_samples: int = Field(default=6, ge=2, le=12)
    board_change_probe_seconds: float = Field(default=8.0, ge=1.0, le=60.0)
    board_change_threshold: float = Field(default=0.20, ge=0.0, le=3.0)
    board_change_min_gap_seconds: float = Field(default=6.0, ge=0.0, le=60.0)
    board_change_max_probe_frames: int = Field(default=48, ge=3, le=240)

    # Fail-safe: if substantive mathematical content remains unresolved and readable visual evidence
    # exists, insert the best board crop directly into the notes instead of inventing a reconstruction.
    unresolved_board_snapshots_enabled: bool = True
    unresolved_board_min_visual_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    unresolved_board_max_per_chunk: int = Field(default=1, ge=0, le=4)
    unresolved_board_width_fraction: float = Field(default=0.88, ge=0.30, le=1.0)

    formula_detection: FormulaDetectionConfig = Field(default_factory=FormulaDetectionConfig)
    math_ocr: MathOCRConfig = Field(default_factory=MathOCRConfig)

    @model_validator(mode="after")
    def validate_board_roi(self) -> VisionConfig:
        if self.board_crop_roi is None:
            return self
        left, top, right, bottom = self.board_crop_roi
        if not all(0.0 <= value <= 1.0 for value in self.board_crop_roi):
            raise ValueError("vision.board_crop_roi values must lie in [0, 1]")
        if left >= right or top >= bottom:
            raise ValueError("vision.board_crop_roi must satisfy left<right and top<bottom")
        return self


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
    unimernet_config = cfg.vision.math_ocr.unimernet_config_path
    if unimernet_config is not None and not unimernet_config.is_absolute():
        cfg.vision.math_ocr.unimernet_config_path = (base / unimernet_config).resolve()
    formula_model = cfg.vision.formula_detection.model_path
    if formula_model is not None and not formula_model.is_absolute():
        cfg.vision.formula_detection.model_path = (base / formula_model).resolve()
    unimernet_python = cfg.vision.math_ocr.unimernet_python_path
    if unimernet_python is not None and not unimernet_python.is_absolute():
        # Do not call Path.resolve() here: venv/bin/python is normally a symlink to the base
        # interpreter, and resolving it silently discards the virtual environment entrypoint.
        cfg.vision.math_ocr.unimernet_python_path = (base / unimernet_python).absolute()
    return cfg
