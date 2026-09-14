"""Production pipeline with robust structured output and split-and-merge synthesis."""

from . import pipeline as _base_pipeline
from .episode_synthesis_resilient import EPISODE_SYNTHESIS_CACHE_VERSION
from .knowledge_pipeline_resilient import run_knowledge_pipeline as resilient_knowledge_pipeline
from .llm_robust import LectureModelClient
from .util import stable_hash


class Pipeline(_base_pipeline.Pipeline):
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
            }
        )

    def run_lecture(self, lecture, *, force: bool = False):
        original = _base_pipeline.run_knowledge_pipeline
        _base_pipeline.run_knowledge_pipeline = resilient_knowledge_pipeline
        try:
            return super().run_lecture(lecture, force=force)
        finally:
            _base_pipeline.run_knowledge_pipeline = original
