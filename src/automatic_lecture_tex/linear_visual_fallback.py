from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import VisionConfig
from .schemas import BlockType, ChunkNotes, NoteBlock, VisualEvidence, VisualKind

_BOARD_SNAPSHOT_PREFIX = "board-snapshot:"
_SUBSTANTIVE_UNRESOLVED = re.compile(
    r"(Audit block|формул|равен|выраж|знак|символ|индекс|обознач|доказ|переход|"
    r"теорем|определен|неоднознач|математ|formula|equation|symbol|index|proof|derivation|"
    r"\\[A-Za-z]+|\$)",
    re.IGNORECASE,
)
_STALE_FILES = (
    "episode_hierarchy.json",
    "global_validation.json",
    "lecture_kb.json",
    "lecture_outline.json",
)
_STALE_DIRS = (
    "knowledge_episode_batches",
    "knowledge_sections",
    "knowledge_windows",
)


def is_board_snapshot(block: NoteBlock) -> bool:
    return any(item.startswith(_BOARD_SNAPSHOT_PREFIX) for item in block.source_evidence_ids)


def _has_substantive_unresolved(notes: ChunkNotes) -> bool:
    return any(_SUBSTANTIVE_UNRESOLVED.search(item) is not None for item in notes.unresolved)


def inject_unresolved_board_snapshots(
    notes: ChunkNotes,
    evidence: list[VisualEvidence],
    config: VisionConfig,
    *,
    output_language: str,
) -> int:
    """Append source images only when unresolved mathematical content has readable board evidence."""

    if (
        not config.unresolved_board_snapshots_enabled
        or config.unresolved_board_max_per_chunk <= 0
        or not _has_substantive_unresolved(notes)
    ):
        return 0

    existing_assets = {block.asset_path for block in notes.blocks if block.asset_path}
    candidates = [
        item
        for item in evidence
        if item.asset_path
        and item.asset_path not in existing_assets
        and item.kind in {VisualKind.EQUATION, VisualKind.NOTATION, VisualKind.DIAGRAM}
        and item.confidence >= config.unresolved_board_min_visual_confidence
    ]
    candidates.sort(key=lambda item: item.confidence, reverse=True)

    inserted = 0
    for item in candidates[: config.unresolved_board_max_per_chunk]:
        caption = (
            "Фрагмент записи на доске: автоматическая текстовая реконструкция этого места "
            "осталась неоднозначной."
            if output_language.lower().startswith("ru")
            else "Board evidence for a mathematically relevant passage that remained unresolved."
        )
        notes.blocks.append(
            NoteBlock(
                type=BlockType.FIGURE,
                latex="",
                asset_path=item.asset_path,
                caption=caption,
                source_evidence_ids=[f"{_BOARD_SNAPSHOT_PREFIX}{item.request_id}"],
            )
        )
        existing_assets.add(item.asset_path)
        inserted += 1
    return inserted


def clean_stale_architecture_artifacts(work: Path) -> None:
    """Remove artifacts from the retired knowledge/episode path when running linear production."""

    for name in _STALE_FILES:
        path = work / name
        if path.exists():
            path.unlink()
    for name in _STALE_DIRS:
        path = work / name
        if path.exists():
            shutil.rmtree(path)

    frames = work / "frames"
    if frames.is_dir():
        for path in frames.glob("window_*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
