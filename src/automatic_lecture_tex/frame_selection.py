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


def board_state_change_score(left: Path, right: Path) -> float:
    """Return a deterministic visual-change score between two candidate board states."""

    first = _edge_signature(left)
    second = _edge_signature(right)
    diff = np.abs(first - second)
    return float(0.65 * diff.mean() + 0.35 * np.quantile(diff, 0.90))


def select_board_state_frames(
    frames: list[ExtractedFrame],
    *,
    max_states: int,
    change_threshold: float,
    min_gap_seconds: float,
) -> list[ExtractedFrame]:
    """Select chronologically distinct board states without any LLM confidence signal.

    The first state is always retained. A later probe is kept when the board differs sufficiently
    from the last retained state and is not temporally redundant. The final probe is retained when
    there is spare capacity and it is separated from the previous state. The result is bounded by
    the multimodal image budget and remains chronological.
    """

    if not frames or max_states <= 0:
        return []
    ordered = sorted(frames, key=lambda item: item.timestamp)
    selected = [ordered[0]]

    for frame in ordered[1:]:
        if len(selected) >= max_states:
            break
        if frame.timestamp - selected[-1].timestamp < min_gap_seconds:
            continue
        score = board_state_change_score(selected[-1].path, frame.path)
        if score >= change_threshold:
            selected.append(frame)

    last = ordered[-1]
    if (
        len(selected) < max_states
        and last.path != selected[-1].path
        and last.timestamp - selected[-1].timestamp >= min_gap_seconds
    ):
        selected.append(last)

    return selected[:max_states]
