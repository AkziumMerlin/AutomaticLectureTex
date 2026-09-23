from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .schemas import ExtractedFrame


def _normalized_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as raw:
        image = raw.convert("L").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = max(0.04, 1.4826 * mad)
    return np.clip((array - median) / scale, -6.0, 6.0)


def frame_occlusion_score(frame_path: Path, composite_path: Path) -> float:
    """Estimate transient foreground coverage relative to the temporal board composite.

    The score is deliberately cheap and photometric: global brightness is normalized away and the
    upper tail of absolute low-resolution differences dominates. A moving lecturer therefore costs
    much more than small chalk changes, while the selected image remains a real raw frame rather
    than a synthesized median image.
    """

    size = (256, 144)
    frame = _normalized_gray(frame_path, size)
    composite = _normalized_gray(composite_path, size)
    diff = np.abs(frame - composite)
    return float(0.7 * np.quantile(diff, 0.90) + 0.3 * diff.mean())


def select_least_occluded_frame(
    frames: list[ExtractedFrame],
    composite_path: Path,
    *,
    target_timestamp: float,
) -> ExtractedFrame:
    if not frames:
        raise ValueError("cannot select a board frame from an empty sequence")
    scored = [
        (
            frame_occlusion_score(frame.path, composite_path),
            abs(frame.timestamp - target_timestamp),
            index,
            frame,
        )
        for index, frame in enumerate(frames)
    ]
    return min(scored, key=lambda item: (item[0], item[1], item[2]))[-1]



def _edge_signature(path: Path, size: tuple[int, int] = (192, 108)) -> np.ndarray:
    """Cheap board-state signature that emphasizes writing over global brightness.

    We compare horizontal/vertical grayscale gradients instead of raw pixels. This makes the change
    score much less sensitive to exposure drift while remaining deterministic and host-side.
    """

    image = _normalized_gray(path, size)
    dx = np.diff(image, axis=1, prepend=image[:, :1])
    dy = np.diff(image, axis=0, prepend=image[:1, :])
    magnitude = np.sqrt(dx * dx + dy * dy)
    # Clip transient foreground edges (for example a lecturer crossing the board) so they cannot
    # dominate the whole score.
    cap = max(0.05, float(np.quantile(magnitude, 0.95)))
    return np.clip(magnitude, 0.0, cap) / cap


def _localized_writing_change(diff: np.ndarray) -> float:
    """Emphasize sparse local writing changes that global image averages would miss."""

    height, width = diff.shape
    block_h = max(6, height // 9)
    block_w = max(8, width // 12)
    scores: list[float] = []
    for y0 in range(0, height, block_h):
        for x0 in range(0, width, block_w):
            block = diff[y0 : min(height, y0 + block_h), x0 : min(width, x0 + block_w)]
            if block.size:
                scores.append(float(block.mean()))
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return float(sum(scores[: min(4, len(scores))]) / min(4, len(scores)))


def board_state_change_score(left: Path, right: Path) -> float:
    """Return a writing-sensitive visual-change score between board states.

    Small newly written formulas occupy little of a lecture frame and were previously diluted by
    the global mean. Combine global edge change with the strongest local writing regions so a new
    line or formula can survive selection without making the sampler depend on an LLM confidence.
    """

    first = _edge_signature(left)
    second = _edge_signature(right)
    diff = np.abs(first - second)
    global_score = float(0.55 * diff.mean() + 0.45 * np.quantile(diff, 0.90))
    local_score = _localized_writing_change(diff)
    sparse_score = float(np.quantile(diff, 0.97))
    return float(0.45 * global_score + 0.35 * local_score + 0.20 * sparse_score)


def select_board_state_frames(
    frames: list[ExtractedFrame],
    *,
    max_states: int,
    change_threshold: float,
    min_gap_seconds: float,
) -> list[ExtractedFrame]:
    """Select chronologically distinct board states without any LLM confidence signal.

    First discover all visually distinct states across the interval. Only afterwards compress that
    sequence to the multimodal image budget. This avoids the common failure where rapid writing near
    the beginning consumes every slot and the rest of the lecture interval is never represented.

    change_threshold is therefore only a photometric noise floor, not a lecturer-confidence
    parameter. When more distinct states exist than fit in the budget, retain states spread across
    the full chronological sequence, including its first and latest distinct state.
    """

    if not frames or max_states <= 0:
        return []
    ordered = sorted(frames, key=lambda item: item.timestamp)
    candidates = [ordered[0]]

    for frame in ordered[1:]:
        if frame.timestamp - candidates[-1].timestamp < min_gap_seconds:
            continue
        score = board_state_change_score(candidates[-1].path, frame.path)
        if score >= change_threshold:
            candidates.append(frame)

    # A semantically important final line can be too sparse for the ordinary threshold. Preserve
    # the latest probe when it carries at least half-threshold writing change relative to the last
    # selected state; unchanged tails still remain collapsed.
    latest = ordered[-1]
    if latest.path != candidates[-1].path and latest.timestamp - candidates[-1].timestamp >= min_gap_seconds:
        tail_score = board_state_change_score(candidates[-1].path, latest.path)
        if tail_score >= 0.5 * change_threshold:
            candidates.append(latest)

    if len(candidates) <= max_states:
        return candidates
    if max_states == 1:
        return [candidates[0]]

    # Uniformly cover the sequence of semantic visual changes, not wall-clock probe frames.
    indices = [
        round(index * (len(candidates) - 1) / (max_states - 1))
        for index in range(max_states)
    ]
    selected: list[ExtractedFrame] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        selected.append(candidates[index])
    return selected

