from __future__ import annotations

import json

from . import knowledge_pipeline as _base
from .episode_synthesis_resilient import (
    EPISODE_SYNTHESIS_CACHE_VERSION,
    episode_evidence_batches,
    merge_episode_batches,
    reset_synthesis_stats,
    synthesis_stats_snapshot,
    validate_episode_batch,
    write_episode_batch,
)
from .knowledge_integrity import IntegrityKnowledgeOrchestrator
from .util import atomic_json_dump

KNOWLEDGE_CACHE_VERSION = 3


def _patch_run_metrics(work) -> None:
    path = work / "run_metrics.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    stats = synthesis_stats_snapshot()
    payload["episode_synthesis_seconds"] = round(float(stats["synthesis_seconds"]), 3)
    payload["episode_validation_seconds"] = round(float(stats["validation_seconds"]), 3)
    payload["episode_write_calls"] = int(stats["write_calls"])
    payload["episode_validation_calls"] = int(stats["validation_calls"])
    payload["episode_boundary_validation_calls"] = int(stats["boundary_validation_calls"])
    payload["episode_proactive_splits"] = int(stats["proactive_splits"])
    payload["episode_structured_splits"] = int(stats["structured_splits"])
    payload["episode_validation_splits"] = int(stats["validation_splits"])
    payload["episode_coverage_splits"] = int(stats["coverage_splits"])
    payload["episode_coverage_unresolved"] = int(stats["coverage_unresolved"])
    payload["episode_boundary_validation_failures"] = int(
        stats["boundary_validation_failures"]
    )
    payload["episode_indivisible_failures"] = int(stats["indivisible_failures"])
    payload["episode_proof_merges"] = int(stats["proof_merges"])
    payload["episode_deduped_blocks"] = int(stats["deduped_blocks"])
    payload["knowledge_cache_version"] = KNOWLEDGE_CACHE_VERSION
    payload["episode_synthesis_cache_version"] = EPISODE_SYNTHESIS_CACHE_VERSION
    atomic_json_dump(path, payload)


def run_knowledge_pipeline(*args, **kwargs):
    """Run the knowledge pipeline with provenance, coverage, and split/merge integrity hooks."""

    transcript = kwargs["transcript"]
    work = kwargs["work"]

    original_orchestrator = _base.KnowledgeOrchestrator
    original_batches = _base.episode_evidence_batches
    original_write = _base.write_episode_batch
    original_validate = _base.validate_episode_batch
    original_merge = _base.merge_episode_batches
    original_synthesis_version = _base.EPISODE_SYNTHESIS_CACHE_VERSION
    original_knowledge_version = _base.KNOWLEDGE_CACHE_VERSION

    def orchestrator_factory(llm, config, output_language):
        return IntegrityKnowledgeOrchestrator(
            llm,
            config,
            output_language,
            transcript=transcript,
        )

    def bounded_batches(kb, episode, config, transcript_arg=None):
        del transcript_arg
        return episode_evidence_batches(kb, episode, config, transcript=transcript)

    reset_synthesis_stats()
    _base.KnowledgeOrchestrator = orchestrator_factory
    _base.episode_evidence_batches = bounded_batches
    _base.write_episode_batch = write_episode_batch
    _base.validate_episode_batch = validate_episode_batch
    _base.merge_episode_batches = merge_episode_batches
    _base.EPISODE_SYNTHESIS_CACHE_VERSION = EPISODE_SYNTHESIS_CACHE_VERSION
    _base.KNOWLEDGE_CACHE_VERSION = KNOWLEDGE_CACHE_VERSION
    try:
        result = _base.run_knowledge_pipeline(*args, **kwargs)
        _patch_run_metrics(work)
        return result
    finally:
        _base.KnowledgeOrchestrator = original_orchestrator
        _base.episode_evidence_batches = original_batches
        _base.write_episode_batch = original_write
        _base.validate_episode_batch = original_validate
        _base.merge_episode_batches = original_merge
        _base.EPISODE_SYNTHESIS_CACHE_VERSION = original_synthesis_version
        _base.KNOWLEDGE_CACHE_VERSION = original_knowledge_version
