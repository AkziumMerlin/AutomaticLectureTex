from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import VisionConfig
from .schemas import ExtractedFrame


def temporal_sample_offsets(config: VisionConfig) -> list[float]:
    """Return symmetric offsets for a bounded temporal board window.

    The number of sampled frames is capped deterministically. A temporal median removes objects that
    occupy a pixel for fewer than half of these samples, which is a useful cheap approximation for
    removing a moving lecturer while retaining persistent writing on a static-camera lecture video.
    """

    window = float(config.temporal_window_seconds)
    period = float(config.temporal_sample_period_seconds)
    count = int((2.0 * window) // period) + 1
    count = max(3, min(count, config.temporal_max_frames))
    if count % 2 == 0:
        count = max(3, count - 1)
    if count == 1:
        return [0.0]
    step = (2.0 * window) / (count - 1)
    return [-window + index * step for index in range(count)]


def build_temporal_board_composite(
    frames: list[ExtractedFrame],
    output_path: Path,
    *,
    autocontrast: bool = True,
) -> Path:
    """Build a robust board image from time-adjacent frames using a per-pixel median."""

    if len(frames) < 3:
        raise ValueError("temporal board composite requires at least three frames")

    images: list[Image.Image] = []
    target_size: tuple[int, int] | None = None
    for frame in frames:
        with Image.open(frame.path) as raw:
            image = raw.convert("RGB")
            if target_size is None:
                target_size = image.size
            elif image.size != target_size:
                image = image.resize(target_size, Image.Resampling.BILINEAR)
            images.append(image.copy())

    stack = np.stack([np.asarray(image, dtype=np.uint8) for image in images], axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    composite = Image.fromarray(median, mode="RGB")
    if autocontrast:
        composite = ImageOps.autocontrast(composite, cutoff=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, format="JPEG", quality=95, subsampling=0)
    return output_path
