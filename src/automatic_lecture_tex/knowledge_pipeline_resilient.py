from __future__ import annotations

from . import knowledge_pipeline as _base
from .episode_synthesis_resilient import (
    EPISODE_SYNTHESIS_CACHE_VERSION,
    episode_evidence_batches,
    validate_episode_batch,
    write_episode_batch,
)


def run_knowledge_pipeline(*args, **kwargs):
    """Run the existing knowledge pipeline with split-and-merge synthesis hooks.

    The compatibility layer is scoped to one call and restores the base module afterwards, so the
    underlying pipeline remains import-compatible while production CLI gets resilient synthesis.
    """

    original_batches = _base.episode_evidence_batches
    original_write = _base.write_episode_batch
    original_validate = _base.validate_episode_batch
    original_version = _base.EPISODE_SYNTHESIS_CACHE_VERSION
    _base.episode_evidence_batches = episode_evidence_batches
    _base.write_episode_batch = write_episode_batch
    _base.validate_episode_batch = validate_episode_batch
    _base.EPISODE_SYNTHESIS_CACHE_VERSION = EPISODE_SYNTHESIS_CACHE_VERSION
    try:
        return _base.run_knowledge_pipeline(*args, **kwargs)
    finally:
        _base.episode_evidence_batches = original_batches
        _base.write_episode_batch = original_write
        _base.validate_episode_batch = original_validate
        _base.EPISODE_SYNTHESIS_CACHE_VERSION = original_version
