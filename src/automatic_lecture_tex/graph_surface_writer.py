from __future__ import annotations

from typing import Any

from .graph_revision import GraphState
from .reader_surface import (
    GeneratedReaderBlock,
    GraphSectionSpec,
    PlannedReaderBlock,
    ReaderDiscoursePlan,
    ReaderProjectionChoice,
    ReaderProjectionChoices,
    ReaderSectionProjection,
    ReaderSurfaceSegment,
    graph_section_specs,
    write_reader_surface,
)
from .schemas import LectureIR

GRAPH_SURFACE_WRITER_VERSION = 5


def write_graph_surface(
    orchestrator: Any,
    *,
    state: GraphState,
    lecture_id: str,
    lecture_title: str,
    fallback_ir: LectureIR,
    work,
    llm_config: dict[str, Any],
    force: bool,
) -> LectureIR:
    """Compatibility entry point for the typed reader-surface pipeline."""

    return write_reader_surface(
        orchestrator,
        state=state,
        lecture_id=lecture_id,
        lecture_title=lecture_title,
        fallback_ir=fallback_ir,
        work=work,
        llm_config=llm_config,
        force=force,
    )


__all__ = [
    "GRAPH_SURFACE_WRITER_VERSION",
    "GeneratedReaderBlock",
    "GraphSectionSpec",
    "PlannedReaderBlock",
    "ReaderDiscoursePlan",
    "ReaderProjectionChoice",
    "ReaderProjectionChoices",
    "ReaderSectionProjection",
    "ReaderSurfaceSegment",
    "graph_section_specs",
    "write_graph_surface",
]
