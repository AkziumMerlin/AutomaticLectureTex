"""Production pipeline with robust structured output and split-and-merge synthesis."""

from . import pipeline as _base_pipeline
from .knowledge_pipeline_resilient import run_knowledge_pipeline as resilient_knowledge_pipeline
from .llm_robust import LectureModelClient


class Pipeline(_base_pipeline.Pipeline):
    @property
    def llm(self) -> LectureModelClient:
        if self._llm is None:
            self._llm = LectureModelClient(self.config.llm)
        return self._llm

    def run_lecture(self, lecture, *, force: bool = False):
        original = _base_pipeline.run_knowledge_pipeline
        _base_pipeline.run_knowledge_pipeline = resilient_knowledge_pipeline
        try:
            return super().run_lecture(lecture, force=force)
        finally:
            _base_pipeline.run_knowledge_pipeline = original
