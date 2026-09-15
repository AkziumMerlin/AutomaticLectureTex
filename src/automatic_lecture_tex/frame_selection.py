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
