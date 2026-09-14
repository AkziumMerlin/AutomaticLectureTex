from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

from .asr import ASRBackend, make_asr_backend
from .config import ASRConfig, RuntimeConfig, TranscriptCorrectionConfig
from .schemas import Transcript
from .transcript_correction import reconstruct_transcript
from .util import atomic_json_dump

logger = logging.getLogger(__name__)


class CorrectingASRBackend(ASRBackend):
    """Preserve immutable raw ASR, then expose an LLM-reconstructed transcript downstream."""

    def __init__(
        self,
        primary_config: ASRConfig,
        correction_config: TranscriptCorrectionConfig,
        runtime: RuntimeConfig,
        *,
        llm,
        course_title: str,
        language: str,
        expected_raw_fingerprint: str,
        reuse_raw: bool,
    ) -> None:
        super().__init__(primary_config, runtime)
        self.correction_config = correction_config
        self.llm = llm
        self.course_title = course_title
        self.language = language
        self.expected_raw_fingerprint = expected_raw_fingerprint
        self.reuse_raw = reuse_raw
        self._primary: ASRBackend | None = None

    @property
    def primary(self) -> ASRBackend:
        if self._primary is None:
            self._primary = make_asr_backend(self.config, self.runtime)
        return self._primary

    def _work(self, lecture_id: str) -> Path:
        path = self.runtime.work_dir / lecture_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _raw_paths(self, lecture_id: str) -> tuple[Path, Path]:
        work = self._work(lecture_id)
        return work / "raw_transcript.json", work / "raw_transcript_meta.json"

    def _load_reusable_raw(self, lecture_id: str) -> Transcript | None:
        if not self.reuse_raw:
            return None
        raw_path, meta_path = self._raw_paths(lecture_id)
        if not raw_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != self.expected_raw_fingerprint:
                return None
            return Transcript.model_validate_json(raw_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return None

    def _save_raw(self, lecture_id: str, transcript: Transcript) -> None:
        raw_path, meta_path = self._raw_paths(lecture_id)
        atomic_json_dump(raw_path, transcript.model_dump(mode="json"))
        atomic_json_dump(meta_path, {"fingerprint": self.expected_raw_fingerprint})

    def _release_primary(self) -> None:
        self._primary = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(self, lecture_id: str, audio_path: Path) -> Transcript:
        raw = self._load_reusable_raw(lecture_id)
        if raw is None:
            logger.info("[%s] producing immutable raw ASR transcript", lecture_id)
            raw = self.primary.transcribe(lecture_id, audio_path)
            self._save_raw(lecture_id, raw)
        else:
            logger.info("[%s] reusing immutable raw ASR transcript", lecture_id)

        # Fallback ASR may use the same GPU. The primary model is no longer needed after raw ASR.
        self._release_primary()
        result = reconstruct_transcript(
            raw,
            llm=self.llm,
            config=self.correction_config,
            runtime=self.runtime,
            course_title=self.course_title,
            language=self.language,
            hotwords=self.config.hotwords,
            audio_path=audio_path,
        )
        work = self._work(lecture_id)
        audit = dict(result.audit)
        audit["metrics"] = result.metrics
        audit["raw_transcript"] = "raw_transcript.json"
        audit["corrected_transcript"] = "transcript.json"
        atomic_json_dump(work / "transcript_reconstruction.json", audit)
        logger.info(
            "[%s] transcript reconstruction: %d reconstructed, %d ambiguous, %d fallback spans",
            lecture_id,
            result.metrics["segments_reconstructed"],
            result.metrics["segments_ambiguous"],
            result.metrics["fallback_spans"],
        )
        return result.transcript
