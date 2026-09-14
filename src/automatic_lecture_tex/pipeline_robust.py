"""Compatibility layer that makes the production Pipeline use the robust LLM client."""

from . import pipeline as _pipeline
from .llm_robust import LectureModelClient

_pipeline.LectureModelClient = LectureModelClient
Pipeline = _pipeline.Pipeline

__all__ = ["Pipeline"]
