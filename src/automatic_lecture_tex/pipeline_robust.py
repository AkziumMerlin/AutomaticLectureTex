"""Production pipeline with robust structured output and bounded semantic reconstruction."""

import json

from . import pipeline as _base_pipeline
from .asr import make_asr_backend
from .episode_synthesis_resilient import EPISODE_SYNTHESIS_CACHE_VERSION
from .gigaam_vad import VadGigaAMBackend
from .knowledge_pipeline_resilient import (
    KNOWLEDGE_CACHE_VERSION,
    run_knowledge_pipeline as resilient_knowledge_pipeline,
)
from .llm_robust import LectureModelClient
from .media import media_source_from_config
from .schemas import Transcript
from .util import atomic_json_dump, stable_hash


class Pipeline(_base_pipeline.Pipeline):
    @property
    def asr(self):
        if self._asr is None:
            if self.config.asr.backend == "gigaam" and self.config.asr.gigaam_vad_enabled:
                self._asr = VadGigaAMBackend(self.config.asr, self.config.runtime)
            else:
                self._asr = make_asr_backend(self.config.asr, self.config.runtime)
        return self._asr

    @property
    def llm(self) -> LectureModelClient:
        if self._llm is None:
            self._llm = LectureModelClient(self.config.llm)
        return self._llm

    def _ir_fingerprint(self, transcript, notation: dict[str, str]) -> str:
        return stable_hash(
            {
                "base": super()._ir_fingerprint(transcript, notation),
                "resilient_episode_synthesis_version": EPISODE_SYNTHESIS_CACHE_VERSION,
                "knowledge_integrity_cache_version": KNOWLEDGE_CACHE_VERSION,
            }
        )

    def _restore_raw_asr_cache(self, lecture) -> None:
        """Reuse the raw transcript saved by the removed transcript-correction layer."""

        work = self._lecture_work_dir(lecture)
        raw_path = work / "raw_transcript.json"
        raw_meta_path = work / "raw_transcript_meta.json"
        if not raw_path.exists() or not raw_meta_path.exists():
            return

        source = media_source_from_config(
            lecture.source,
            self.config.runtime,
            self.config.vision,
        )
        raw_fingerprint = stable_hash(
            {
                "source": source.identity(),
                "asr": self.config.asr.model_dump(mode="json"),
                "asr_cache_version": _base_pipeline.ASR_CACHE_VERSION,
            }
        )
        try:
            meta = json.loads(raw_meta_path.read_text(encoding="utf-8"))
            raw = Transcript.model_validate_json(raw_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return
        if meta.get("fingerprint") != raw_fingerprint:
            return

        transcript_path = work / "transcript.json"
        manifest_path = work / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}

        atomic_json_dump(transcript_path, raw.model_dump(mode="json"))
        manifest["transcript_fingerprint"] = raw_fingerprint
        manifest.pop("ir_fingerprint", None)
        atomic_json_dump(manifest_path, manifest)

        stale_audit = work / "transcript_reconstruction.json"
        if stale_audit.exists():
            stale_audit.unlink()

    def run_lecture(self, lecture, *, force: bool = False):
        original = _base_pipeline.run_knowledge_pipeline
        if not force:
            self._restore_raw_asr_cache(lecture)
        _base_pipeline.run_knowledge_pipeline = resilient_knowledge_pipeline
        try:
            return super().run_lecture(lecture, force=force)
        finally:
            _base_pipeline.run_knowledge_pipeline = original
