from __future__ import annotations

import shutil
from pathlib import Path

from .config import VisionConfig
from .schemas import BlockType, ChunkNotes, NoteBlock, VisualEvidence, VisualKind

_BOARD_SNAPSHOT_PREFIX = "board-snapshot:"
_OMITTED_MATH_PREFIX = "[omitted-math]"
_AUDIT_ISSUE_PREFIX = "audit-issue:"
_AUDIT_VISUAL_PREFIX = "audit-visual:"

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


def _is_high_confidence_audit_issue(block: NoteBlock) -> bool:
    for item in block.source_evidence_ids:
        if not item.startswith(_AUDIT_ISSUE_PREFIX):
            continue
        try:
            return float(item.removeprefix(_AUDIT_ISSUE_PREFIX)) >= 0.8
        except ValueError:
            continue
    return False


def _audit_visual_request_ids(block: NoteBlock) -> list[str]:
    return [
        item.removeprefix(_AUDIT_VISUAL_PREFIX)
        for item in block.source_evidence_ids
        if item.startswith(_AUDIT_VISUAL_PREFIX)
    ]


def _eligible_board_evidence(
    evidence: list[VisualEvidence],
    config: VisionConfig,
    *,
    existing_assets: set[str],
    request_ids: set[str] | None = None,
) -> list[VisualEvidence]:
    candidates = [
        item
        for item in evidence
        if item.asset_path
        and item.asset_path not in existing_assets
        and item.kind in {VisualKind.EQUATION, VisualKind.NOTATION, VisualKind.DIAGRAM}
        and item.confidence >= config.unresolved_board_min_visual_confidence
        and (request_ids is None or item.request_id in request_ids)
    ]
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


def _snapshot_block(
    item: VisualEvidence,
    *,
    output_language: str,
    source_claim_ids: list[str] | None = None,
) -> NoteBlock:
    caption = (
        "Фрагмент записи на доске: автоматическая текстовая реконструкция этого места "
        "осталась неоднозначной."
        if output_language.lower().startswith("ru")
        else "Board evidence for a mathematically relevant passage that remained unresolved."
    )
    return NoteBlock(
        type=BlockType.FIGURE,
        latex="",
        asset_path=item.asset_path,
        caption=caption,
        source_claim_ids=list(source_claim_ids or []),
        source_evidence_ids=[f"{_BOARD_SNAPSHOT_PREFIX}{item.request_id}"],
    )


def inject_unresolved_board_snapshots(
    notes: ChunkNotes,
    evidence: list[VisualEvidence],
    config: VisionConfig,
    *,
    output_language: str,
) -> int:
    """Replace unsafe audited blocks and cover explicit mathematical omissions with board evidence.

    Generic unresolved text never triggers a photo. High-confidence source-grounded audit issues
    suppress the exact retained block; a matching board crop is inserted at the same position when
    available. Separately, the writer can request a fallback for unique omitted mathematical content
    only via the explicit [omitted-math] marker.
    """

    snapshots_allowed = (
        config.unresolved_board_snapshots_enabled
        and config.unresolved_board_max_per_chunk > 0
    )
    existing_assets = {
        block.asset_path
        for block in notes.blocks
        if block.asset_path and not _is_high_confidence_audit_issue(block)
    }

    inserted = 0
    rewritten: list[NoteBlock] = []
    for block in notes.blocks:
        if not _is_high_confidence_audit_issue(block):
            rewritten.append(block)
            continue

        # A high-confidence source-grounded audit issue means the retained text is unsafe. Suppress
        # it even when snapshots are disabled. Only use visual evidence explicitly linked by the
        # audit; never substitute an unrelated board frame from the same technical chunk.
        replacement = None
        request_ids = set(_audit_visual_request_ids(block))
        if (
            snapshots_allowed
            and inserted < config.unresolved_board_max_per_chunk
            and request_ids
        ):
            candidates = _eligible_board_evidence(
                evidence,
                config,
                existing_assets=existing_assets,
                request_ids=request_ids,
            )
            if candidates:
                replacement = _snapshot_block(
                    candidates[0],
                    output_language=output_language,
                    source_claim_ids=list(block.source_claim_ids),
                )

        if replacement is not None:
            rewritten.append(replacement)
            assert replacement.asset_path is not None
            existing_assets.add(replacement.asset_path)
            inserted += 1

    notes.blocks = rewritten

    if not snapshots_allowed or inserted >= config.unresolved_board_max_per_chunk:
        return inserted

    omitted_count = sum(
        item.strip().casefold().startswith(_OMITTED_MATH_PREFIX)
        for item in notes.unresolved
    )
    remaining = min(
        omitted_count,
        config.unresolved_board_max_per_chunk - inserted,
    )
    if remaining <= 0:
        return inserted

    candidates = _eligible_board_evidence(
        evidence,
        config,
        existing_assets=existing_assets,
    )
    for item in candidates[:remaining]:
        notes.blocks.append(_snapshot_block(item, output_language=output_language))
        assert item.asset_path is not None
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
