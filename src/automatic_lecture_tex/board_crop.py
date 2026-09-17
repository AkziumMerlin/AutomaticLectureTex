from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import VisionConfig
from .schemas import ExtractedFrame


@dataclass(frozen=True)
class BoardCropResult:
    roi: tuple[int, int, int, int]
    score: float
    frames: list[ExtractedFrame]


def _runs(values: np.ndarray, threshold: float, *, max_gap: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values >= threshold)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current - previous <= max_gap + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = previous = current
    runs.append((start, previous))
    return runs


def _best_run(runs: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not runs:
        return None
    return max(runs, key=lambda item: item[1] - item[0] + 1)


def _dominant_board_roi(image: Image.Image, config: VisionConfig) -> tuple[tuple[int, int, int, int], float] | None:
    width, height = image.size
    if width < 64 or height < 64:
        return None

    sample_width = min(240, width)
    sample_height = max(48, round(height * sample_width / width))
    sample = image.resize((sample_width, sample_height), Image.Resampling.BILINEAR)
    rgb = np.asarray(sample, dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # Quantized dominant-colour detection works well for chalk/black boards while remaining
    # independent of a particular green hue. Very bright wall/projector pixels are excluded.
    bins = 16
    quantized = np.clip((rgb * (bins - 1)).astype(np.int16), 0, bins - 1)
    keys = quantized[..., 0] * (bins * bins) + quantized[..., 1] * bins + quantized[..., 2]
    candidate_mask = (luminance >= 0.04) & (luminance <= config.board_crop_max_luminance)
    if int(candidate_mask.sum()) < 32:
        return None
    values, counts = np.unique(keys[candidate_mask], return_counts=True)
    target_key = int(values[int(np.argmax(counts))])
    dominant = keys == target_key
    target = rgb[dominant].mean(axis=0)

    distance = np.linalg.norm(rgb - target, axis=2)
    mask = distance <= config.board_crop_color_distance
    row_density = mask.mean(axis=1)
    col_density = mask.mean(axis=0)

    row = _best_run(
        _runs(
            row_density,
            config.board_crop_axis_density,
            max_gap=max(1, round(sample_height * 0.03)),
        )
    )
    col = _best_run(
        _runs(
            col_density,
            config.board_crop_axis_density,
            max_gap=max(1, round(sample_width * 0.035)),
        )
    )
    if row is None or col is None:
        return None

    x0s, x1s = col
    y0s, y1s = row
    sample_area = max(1, sample_width * sample_height)
    box_area = (x1s - x0s + 1) * (y1s - y0s + 1)
    area_fraction = box_area / sample_area
    if area_fraction < config.board_crop_min_area_fraction:
        return None

    inside = mask[y0s : y1s + 1, x0s : x1s + 1]
    fill = float(inside.mean()) if inside.size else 0.0
    score = min(1.0, 0.55 * area_fraction / max(config.board_crop_min_area_fraction, 1e-6) + 0.45 * fill)

    x0 = round(x0s * width / sample_width)
    x1 = round((x1s + 1) * width / sample_width)
    y0 = round(y0s * height / sample_height)
    y1 = round((y1s + 1) * height / sample_height)
    pad_x = round((x1 - x0) * config.board_crop_padding_fraction)
    pad_y = round((y1 - y0) * config.board_crop_padding_fraction)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    if (x1 - x0) < width * 0.35 or (y1 - y0) < height * 0.22:
        return None
    return (x0, y0, x1, y1), score


def detect_board_roi(image_path: Path, config: VisionConfig) -> tuple[tuple[int, int, int, int], float] | None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if config.board_crop_roi is not None:
            left, top, right, bottom = config.board_crop_roi
            width, height = image.size
            roi = (
                max(0, min(width - 1, round(left * width))),
                max(0, min(height - 1, round(top * height))),
                max(1, min(width, round(right * width))),
                max(1, min(height, round(bottom * height))),
            )
            if roi[2] > roi[0] and roi[3] > roi[1]:
                return roi, 1.0
            return None
        return _dominant_board_roi(image, config)


def _tile_boxes(
    roi: tuple[int, int, int, int], *, count: int, overlap_fraction: float
) -> list[tuple[int, int, int, int]]:
    if count <= 1:
        return [roi]
    x0, y0, x1, y1 = roi
    width = x1 - x0
    if width <= 0:
        return []
    overlap = max(0.0, min(0.45, overlap_fraction))
    tile_width = width / (count - (count - 1) * overlap)
    step = tile_width * (1.0 - overlap)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(count):
        left = round(x0 + index * step)
        right = round(left + tile_width)
        if index == count - 1:
            right = x1
        boxes.append((max(x0, left), y0, min(x1, right), y1))
    return boxes


def generate_board_crops(
    frame: ExtractedFrame,
    output_dir: Path,
    config: VisionConfig,
) -> BoardCropResult | None:
    if not config.board_auto_crop_enabled:
        return None
    detected = detect_board_roi(frame.path, config)
    if detected is None:
        return None
    roi, score = detected
    if score < config.board_crop_min_score:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[ExtractedFrame] = []
    with Image.open(frame.path) as image:
        image = image.convert("RGB")
        full_path = output_dir / "board_full.jpg"
        image.crop(roi).save(full_path, quality=95)
        result.append(ExtractedFrame(timestamp=frame.timestamp, path=full_path))

        count = config.board_crop_tiles
        if count > 0:
            for index, box in enumerate(
                _tile_boxes(roi, count=count, overlap_fraction=config.board_crop_tile_overlap)
            ):
                tile_path = output_dir / f"board_tile_{index:02d}.jpg"
                image.crop(box).save(tile_path, quality=95)
                result.append(ExtractedFrame(timestamp=frame.timestamp, path=tile_path))

    return BoardCropResult(roi=roi, score=score, frames=result)
