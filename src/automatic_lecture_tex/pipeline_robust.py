"""Production pipeline with robust structured output and transcript reconstruction."""

import json

from . import pipeline as _base_pipeline
from .asr import make_asr_backend
from .correcting_asr import CorrectingASRBackend
from .episode_synthesis_resilient import EPISODE_SYNTHESIS_CACHE_VERSION
from .knowledge_pipeline_resilient import (
    KNOWLEDGE_CACHE_VERSION,
    run_knowledge_pipeline as resilient_knowledge_pipeline,
)
from .llm_robust import LectureModelClient
from .media import media_source_from_config
from .schemas import Transcript
from .transcript_correction import TRANSCRIPT_CORRECTION_CACHE_VERSION
from .util import atomic_json_dump, stable_hash

_BASE_ASR_CACHE_VERSION = _base_pipeline.ASR_CACHE_VERSION


class Pipeline(_base_pipeline.Pipeline):
    _expected_raw_transcript_fingerprint: str = ""
    _transcript_force: bool = False

    @property
    def llm(self) -> LectureModelClient:
        if self._llm is None:
            self._llm = LectureModelClient(self.config.llm)
        return self._llm

    @property
    def asr(self):
        if self._asr is None:
            if self.config.transcript_correction.enabled:
                self._asr = CorrectingASRBackend(
                    self.config.asr,
                    self.config.transcript_correction,
                    self.config.runtime,
                    llm=self.llm,
                    course_title=self.config.course.title,
                    language=self.config.course.language,
                    expected_raw_fingerprint=self._expected_raw_transcript_fingerprint,
                    reuse_raw=not self._transcript_force,
                )
            else:
                self._asr = make_asr_backend(self.config.asr, self.config.runtime)
        return self._asr

    def _ir_fingerprint(self, transcript, notation: dict[str, str]) -> str:
        return stable_hash(
            {
                "base": super()._ir_fingerprint(transcript, notation),
                "resilient_episode_synthesis_version": EPISODE_SYNTHESIS_CACHE_VERSION,
                "knowledge_integrity_cache_version": KNOWLEDGE_CACHE_VERSION,
                "transcript_correction_cache_version": TRANSCRIPT_CORRECTION_CACHE_VERSION,
                "transcript_correction": self.config.transcript_correction.model_dump(mode="json"),
            }
        )

    def _raw_transcript_fingerprint(self, lecture) -> str:
        source = media_source_from_config(
            lecture.source,
            self.config.runtime,
            self.config.vision,
        )
        return stable_hash(
            {
                "source": source.identity(),
                "asr": self.config.asr.model_dump(mode="json"),
                "asr_cache_version": _BASE_ASR_CACHE_VERSION,
            }
        )

    def _migrate_pre_reconstruction_transcript(self, lecture, raw_fingerprint: str) -> None:
        work = self._lecture_work_dir(lecture)
        raw_path = work / "raw_transcript.json"
        raw_meta_path = work / "raw_transcript_meta.json"
        transcript_path = work / "transcript.json"
        audit_path = work / "transcript_reconstruction.json"
        manifest_path = work / "manifest.json"
        if raw_path.exists() or audit_path.exists() or not transcript_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if manifest.get("transcript_fingerprint") != raw_fingerprint:
            return
        try:
            raw = Transcript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
        except ValueError:
            return
        atomic_json_dump(raw_path, raw.model_dump(mode="json"))
        atomic_json_dump(raw_meta_path, {"fingerprint": raw_fingerprint})

    def _corrected_asr_cache_token(self) -> str | int:
        if not self.config.transcript_correction.enabled:
            return _BASE_ASR_CACHE_VERSION
        return stable_hash(
            {
                "base_asr_cache_version": _BASE_ASR_CACHE_VERSION,
                "transcript_correction_cache_version": TRANSCRIPT_CORRECTION_CACHE_VERSION,
                "transcript_correction": self.config.transcript_correction.model_dump(mode="json"),
                "llm": self.config.llm.model_dump(mode="json"),
            }
        )

    def run_lecture(self, lecture, *, force: bool = False):
        original_knowledge = _base_pipeline.run_knowledge_pipeline
        original_asr_cache_version = _base_pipeline.ASR_CACHE_VERSION
        self._transcript_force = force

        if self.config.transcript_correction.enabled:
            raw_fingerprint = self._raw_transcript_fingerprint(lecture)
            self._expected_raw_transcript_fingerprint = raw_fingerprint
            if not force:
                self._migrate_pre_reconstruction_transcript(lecture, raw_fingerprint)
            # The wrapper is lecture-specific because raw provenance depends on source identity.
            self._asr = None

        _base_pipeline.run_knowledge_pipeline = resilient_knowledge_pipeline
        _base_pipeline.ASR_CACHE_VERSION = self._corrected_asr_cache_token()
        try:
            return super().run_lecture(lecture, force=force)
        finally:
            _base_pipeline.run_knowledge_pipeline = original_knowledge
            _base_pipeline.ASR_CACHE_VERSION = original_asr_cache_version
