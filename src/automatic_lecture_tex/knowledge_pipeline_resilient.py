from __future__ import annotations

import json

from . import episode_synthesis_resilient as _resilient
from . import knowledge_pipeline as _base
from .conservative_validation import validate_episode_batch_conservative
from .episode_synthesis_resilient import (
    EPISODE_SYNTHESIS_CACHE_VERSION,
    episode_evidence_batches,
    merge_episode_batches,
    reset_synthesis_stats,
    synthesis_stats_snapshot,
    validate_episode_batch,
    write_episode_batch,
)
from .knowledge_reconstruction_resilient import ResilientIntegrityKnowledgeOrchestrator
from .sensory_evidence import collect_visual_evidence
from .util import atomic_json_dump

KNOWLEDGE_CACHE_VERSION = 9


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
    payload["semantic_reconstruction"] = "raw_asr_to_canonical_events_resilient_split"
    payload["visual_evidence"] = "least_occluded_raw_plus_temporal_composite"
    atomic_json_dump(path, payload)


def run_knowledge_pipeline(*args, **kwargs):
    """Run bounded semantic reconstruction followed by provenance-safe synthesis."""

    transcript = kwargs["transcript"]
    work = kwargs["work"]

    original_orchestrator = _base.KnowledgeOrchestrator
    original_collect_visual = _base._collect_visual_evidence
    original_batches = _base.episode_evidence_batches
    original_write = _base.write_episode_batch
    original_validate = _base.validate_episode_batch
    original_merge = _base.merge_episode_batches
    original_assemble = _base.assemble_outline_sections
    original_synthesis_version = _base.EPISODE_SYNTHESIS_CACHE_VERSION
    original_knowledge_version = _base.KNOWLEDGE_CACHE_VERSION
    original_leaf_validator = _resilient._validate_once

    def orchestrator_factory(llm, config, output_language):
        return ResilientIntegrityKnowledgeOrchestrator(
            llm,
            config,
            output_language,
            transcript=transcript,
        )

    def canonical_batches(kb, episode, config, transcript_arg=None):
        del transcript_arg
        # Note synthesis consumes canonical events/claims/symbols only. Raw ASR is evidence for the
        # semantic reconstruction pass, not a second source that can re-introduce transcription noise.
        return episode_evidence_batches(kb, episode, config, transcript=None)

    def assemble_with_math_titles(sections, episode_notes, *, outline_unresolved=None):
        result = original_assemble(
            sections,
            episode_notes,
            outline_unresolved=outline_unresolved,
        )
        for section, notes in zip(sections, result, strict=True):
            notes.section_title = section.title
        return result

    reset_synthesis_stats()
    _base.KnowledgeOrchestrator = orchestrator_factory
    _base._collect_visual_evidence = collect_visual_evidence
    _base.episode_evidence_batches = canonical_batches
    _base.write_episode_batch = write_episode_batch
    _base.validate_episode_batch = validate_episode_batch
    _base.merge_episode_batches = merge_episode_batches
    _base.assemble_outline_sections = assemble_with_math_titles
    _base.EPISODE_SYNTHESIS_CACHE_VERSION = EPISODE_SYNTHESIS_CACHE_VERSION
    _base.KNOWLEDGE_CACHE_VERSION = KNOWLEDGE_CACHE_VERSION
    _resilient._validate_once = validate_episode_batch_conservative
    try:
        result = _base.run_knowledge_pipeline(*args, **kwargs)
        _patch_run_metrics(work)
        return result
    finally:
        _base.KnowledgeOrchestrator = original_orchestrator
        _base._collect_visual_evidence = original_collect_visual
        _base.episode_evidence_batches = original_batches
        _base.write_episode_batch = original_write
        _base.validate_episode_batch = original_validate
        _base.merge_episode_batches = original_merge
        _base.assemble_outline_sections = original_assemble
        _base.EPISODE_SYNTHESIS_CACHE_VERSION = original_synthesis_version
        _base.KNOWLEDGE_CACHE_VERSION = original_knowledge_version
        _resilient._validate_once = original_leaf_validator
