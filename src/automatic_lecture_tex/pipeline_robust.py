"""Production pipeline variant with backend-independent structured-output recovery."""

from .llm_robust import LectureModelClient
from .pipeline import Pipeline as BasePipeline


class Pipeline(BasePipeline):
    @property
    def llm(self) -> LectureModelClient:
        if self._llm is None:
            self._llm = LectureModelClient(self.config.llm)
        return self._llm
