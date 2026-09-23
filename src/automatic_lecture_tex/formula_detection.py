from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import FormulaDetectionConfig
from .schemas import ExtractedFrame


@dataclass(frozen=True)
class DetectedFormulaCrop:
    id: str
    frame: ExtractedFrame
    bbox: tuple[int, int, int, int]
    confidence: float


class FormulaDetector:
    """Detect mathematical-expression regions and materialize high-resolution crops."""

    def __init__(self, config: FormulaDetectionConfig) -> None:
        self.config = config
        if config.model_path is None:
            raise RuntimeError(
                "Formula detection is enabled but vision.formula_detection.model_path is unset."
            )
        if not config.model_path.is_file():
            raise RuntimeError(
                f"Formula detector weights do not exist: {config.model_path}. "
                "Run scripts/download_formula_models.py first."
            )

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Formula detection requires ultralytics. Install with "
                "pip install -e '.[formula-vision]'"
            ) from exc

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Formula detection is configured for CUDA, but torch.cuda.is_available() is False."
            )

        self._device = 0 if config.device == "cuda" else "cpu"
        self._model = YOLO(str(config.model_path))

    def detect(
        self,
        frame: ExtractedFrame,
        output_dir: Path,
        *,
        id_prefix: str,
    ) -> list[DetectedFormulaCrop]:
        detector_source = frame.path
        if self.config.normalize_dark_board:
            with Image.open(frame.path) as raw:
                gray = raw.convert("L")
                histogram = gray.histogram()
                midpoint = sum(histogram) / 2
                running = 0
                median = 255
                for value, count in enumerate(histogram):
                    running += count
                    if running >= midpoint:
                        median = value
                        break
                if median < self.config.dark_board_threshold:
                    normalized = ImageOps.autocontrast(ImageOps.invert(gray)).convert("RGB")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    detector_source = output_dir / "_mfd_normalized.jpg"
                    normalized.save(detector_source, quality=95)

        result = self._model.predict(
            source=str(detector_source),
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self._device,
            verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[tuple[float, tuple[float, float, float, float]]] = []
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        for confidence, coords in zip(confidences, xyxy, strict=True):
            left, top, right, bottom = [float(value) for value in coords]
            if right <= left or bottom <= top:
                continue
            detections.append((float(confidence), (left, top, right, bottom)))

        # First retain the strongest detections, then restore approximate reading order so crop ids
        # are stable and useful in multimodal prompts.
        detections.sort(key=lambda item: item[0], reverse=True)
        detections = detections[: self.config.max_crops_per_state]
        detections.sort(key=lambda item: (item[1][1], item[1][0]))

        output_dir.mkdir(parents=True, exist_ok=True)
        crops: list[DetectedFormulaCrop] = []
        with Image.open(frame.path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            for index, (confidence, coords) in enumerate(detections):
                left, top, right, bottom = coords
                box_width = right - left
                box_height = bottom - top
                if (
                    box_width < self.config.min_width_px
                    or box_height < self.config.min_height_px
                ):
                    continue

                pad_x = box_width * self.config.padding_fraction
                pad_y = box_height * self.config.padding_fraction
                x0 = max(0, int(round(left - pad_x)))
                y0 = max(0, int(round(top - pad_y)))
                x1 = min(width, int(round(right + pad_x)))
                y1 = min(height, int(round(bottom + pad_y)))
                if x1 <= x0 or y1 <= y0:
                    continue

                crop_id = f"{id_prefix}_f{index:02d}"
                crop_path = output_dir / f"{crop_id}.jpg"
                image.crop((x0, y0, x1, y1)).save(crop_path, quality=97)
                crops.append(
                    DetectedFormulaCrop(
                        id=crop_id,
                        frame=ExtractedFrame(timestamp=frame.timestamp, path=crop_path),
                        bbox=(x0, y0, x1, y1),
                        confidence=confidence,
                    )
                )
        return crops





def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Temporal formula proposals require OpenCV. Install with "
            "pip install -e '.[formula-vision]'"
        ) from exc
    return cv2


def _dominant_surface_color(rgb: np.ndarray) -> np.ndarray:
    """Estimate the dominant board colour while ignoring very dark/bright outliers."""

    flat = rgb.reshape(-1, 3).astype(np.float32)
    luminance = 0.2126 * flat[:, 0] + 0.7152 * flat[:, 1] + 0.0722 * flat[:, 2]
    candidate = flat[(luminance >= 18.0) & (luminance <= 210.0)]
    if candidate.size == 0:
        candidate = flat
    quantized = np.clip((candidate / 16).astype(np.int16), 0, 15)
    keys = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    key = int(values[int(np.argmax(counts))])
    selected = candidate[keys == key]
    return selected.mean(axis=0) if len(selected) else candidate.mean(axis=0)


def _stroke_masks(
    rgb: np.ndarray,
    config: FormulaDetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return thin chalk strokes and a conservative thick-foreground mask.

    Large connected components are a bad lecturer proxy on dense mathematics: after closing, an
    entire formula can become one component. Instead, use distance-to-background thickness. Chalk
    strokes are thin; faces, hands, clothes, board frames and other foreground contain thick cores.
    """

    cv2 = _cv2()
    image = rgb.astype(np.float32)
    background = _dominant_surface_color(rgb)
    distance = np.linalg.norm(image - background[None, None, :], axis=2)
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    board_luminance = float(
        0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    )

    candidate = (
        (luminance >= max(82.0, board_luminance + 18.0))
        & (distance >= 28.0)
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )

    thickness = cv2.distanceTransform(
        (candidate > 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    thick_core = (
        thickness >= config.stroke_foreground_core_radius_px
    ).astype(np.uint8) * 255
    if np.any(thick_core):
        radius = config.stroke_foreground_core_dilate_px
        if radius > 0:
            thick_core = cv2.dilate(
                thick_core,
                np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8),
                iterations=1,
            )

    chalk = candidate.copy()
    chalk[thick_core > 0] = 0

    # Auto-crops often include a few pixels of the metal board frame. Those long edge strokes are
    # not mathematical content and otherwise dominate row/column projections.
    height, width = chalk.shape
    border_x = max(2, int(round(width * config.stroke_border_fraction)))
    border_y = max(2, int(round(height * config.stroke_border_fraction)))
    chalk[:border_y, :] = 0
    chalk[-border_y:, :] = 0
    chalk[:, :border_x] = 0
    chalk[:, -border_x:] = 0
    return chalk, thick_core


def _merge_active_runs(
    active: np.ndarray,
    *,
    max_gap: int,
    min_height: int,
) -> list[tuple[int, int]]:
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current - previous <= max_gap + 1:
            previous = current
            continue
        if previous - start + 1 >= min_height:
            result.append((start, previous + 1))
        start = previous = current
    if previous - start + 1 >= min_height:
        result.append((start, previous + 1))
    return result


def _index_runs(indices: np.ndarray, *, max_gap: int) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current - previous <= max_gap + 1:
            previous = current
            continue
        result.append((start, previous + 1))
        start = previous = current
    result.append((start, previous + 1))
    return result


def _line_boxes(
    rgb: np.ndarray,
    config: FormulaDetectionConfig,
) -> list[tuple[int, int, int, int]]:
    """Split a coarse board region into OCR-sized horizontal writing bands."""

    height, width = rgb.shape[:2]
    chalk, _foreground = _stroke_masks(rgb, config)
    row_counts = (chalk > 0).sum(axis=1)
    row_threshold = max(4, int(round(width * config.line_split_row_density)))
    bands = _merge_active_runs(
        row_counts >= row_threshold,
        max_gap=max(3, int(round(height * config.line_split_max_gap_fraction))),
        min_height=config.line_split_min_band_height_px,
    )
    if not bands:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in bands:
        band = chalk[y0:y1]
        column_counts = (band > 0).sum(axis=0)
        column_threshold = max(
            2,
            int(round((y1 - y0) * config.line_split_column_density)),
        )
        active_columns = np.flatnonzero(column_counts >= column_threshold)
        if not len(active_columns):
            continue

        # Ignore narrow border remnants while retaining separated pieces of one expression. This is
        # deliberately not horizontal connected-component segmentation: large formulas can contain
        # substantial whitespace, fractions and side conditions.
        runs = _index_runs(
            active_columns,
            max_gap=max(4, int(round((y1 - y0) * 0.04))),
        )
        filtered: list[tuple[int, int]] = []
        edge_margin = width * 0.04
        narrow_edge = max(20, int(round(width * 0.08)))
        for left, right in runs:
            near_edge = left <= edge_margin or right >= width - edge_margin
            if near_edge and (right - left) < narrow_edge:
                continue
            filtered.append((left, right))
        if not filtered:
            filtered = runs
        if not filtered:
            continue

        x0 = min(left for left, _right in filtered)
        x1 = max(right for _left, right in filtered)
        box_width = x1 - x0
        box_height = y1 - y0
        pad_x = max(6, int(round(box_width * config.line_split_padding_fraction)))
        pad_y = max(4, int(round(box_height * config.line_split_padding_fraction)))
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(height, y1 + pad_y)
        if (
            x1 - x0 >= config.min_width_px
            and y1 - y0 >= config.min_height_px
        ):
            boxes.append((x0, y0, x1, y1))
    return boxes


def split_oversized_formula_crops(
    crops: list[DetectedFormulaCrop],
    output_dir: Path,
    config: FormulaDetectionConfig,
) -> list[DetectedFormulaCrop]:
    """Turn board-sized MFD detections into line-sized crops without deleting dense mathematics."""

    if not config.line_split_enabled:
        return crops
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[DetectedFormulaCrop] = []
    for crop in crops:
        with Image.open(crop.frame.path) as raw:
            image = raw.convert("RGB")
            if image.height < config.line_split_min_height_px:
                result.append(crop)
                continue
            boxes = _line_boxes(np.asarray(image), config)
            if not boxes:
                # Fail closed. OCR'ing an unsplittable board-sized MFD region was the source of
                # hundreds of long false LaTeX candidates.
                continue

            parent_x0, parent_y0, _parent_x1, _parent_y1 = crop.bbox
            for index, (x0, y0, x1, y1) in enumerate(boxes):
                crop_id = f"{crop.id}_l{index:02d}"
                path = output_dir / f"{crop_id}.jpg"
                image.crop((x0, y0, x1, y1)).save(path, quality=97)
                result.append(
                    DetectedFormulaCrop(
                        id=crop_id,
                        frame=ExtractedFrame(timestamp=crop.frame.timestamp, path=path),
                        bbox=(
                            parent_x0 + x0,
                            parent_y0 + y0,
                            parent_x0 + x1,
                            parent_y0 + y1,
                        ),
                        confidence=crop.confidence,
                    )
                )
    return result


def _registered_homography(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    config: FormulaDetectionConfig,
):
    """Estimate source->target homography and reject real shot changes."""

    cv2 = _cv2()
    source = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    target = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=2500)
    keypoints_a, descriptors_a = sift.detectAndCompute(source, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(target, None)
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(keypoints_a) < 8
        or len(keypoints_b) < 8
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(good) < config.temporal_min_good_matches:
        return None

    source_points = np.float32(
        [keypoints_a[item.queryIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [keypoints_b[item.trainIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        4.0,
    )
    if homography is None or inliers is None:
        return None
    inlier_ratio = float(inliers.ravel().mean())
    if inlier_ratio < config.temporal_min_inlier_ratio:
        return None
    return homography, inlier_ratio


def _seed_components(
    seed: np.ndarray,
    config: FormulaDetectionConfig,
) -> list[np.ndarray]:
    """Return coherent local change seeds instead of treating every changed pixel as one proposal."""

    cv2 = _cv2()
    joined = cv2.dilate(
        seed,
        np.ones((3, 11), dtype=np.uint8),
        iterations=1,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(joined, 8)
    result: list[np.ndarray] = []
    frame_area = seed.shape[0] * seed.shape[1]
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if width * height > frame_area * 0.08:
            continue
        component = np.logical_and(labels == index, seed > 0).astype(np.uint8) * 255
        if int((component > 0).sum()) < config.temporal_min_new_pixels:
            continue
        result.append(component)
    result.sort(key=lambda item: int((item > 0).sum()), reverse=True)
    return result[: max(1, 2 * config.temporal_max_crops_per_state)]


def _local_formula_box(
    rgb: np.ndarray,
    seed: np.ndarray,
    config: FormulaDetectionConfig,
) -> tuple[int, int, int, int] | None:
    """Expand one temporal attention seed to a complete local writing band."""

    ys, xs = np.nonzero(seed)
    if not len(xs):
        return None
    height, width = seed.shape
    seed_width = int(xs.max() - xs.min() + 1)
    seed_height = int(ys.max() - ys.min() + 1)
    margin_x = max(
        config.temporal_context_min_width_px,
        int(round(seed_width * config.temporal_context_width_factor)),
    )
    margin_y = max(
        config.temporal_context_min_height_px,
        int(round(seed_height * config.temporal_context_height_factor)),
    )
    roi = (
        max(0, int(xs.min()) - margin_x),
        max(0, int(ys.min()) - margin_y),
        min(width, int(xs.max()) + 1 + margin_x),
        min(height, int(ys.max()) + 1 + margin_y),
    )
    x0, y0, x1, y1 = roi
    local_rgb = rgb[y0:y1, x0:x1]
    local_seed = seed[y0:y1, x0:x1]
    boxes = _line_boxes(local_rgb, config)
    if not boxes:
        return None

    best: tuple[int, tuple[int, int, int, int]] | None = None
    for box in boxes:
        bx0, by0, bx1, by1 = box
        overlap = int((local_seed[by0:by1, bx0:bx1] > 0).sum())
        if overlap <= 0:
            continue
        candidate = (overlap, box)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None

    bx0, by0, bx1, by1 = best[1]
    return x0 + bx0, y0 + by0, x0 + bx1, y0 + by1


def detect_temporal_formula_crops(
    frames: list[ExtractedFrame],
    output_dir: Path,
    config: FormulaDetectionConfig,
    *,
    id_prefix: str,
) -> list[DetectedFormulaCrop]:
    """Use temporal change only as attention, then OCR a clean future full-line view.

    The previous implementation sent the changed connected component itself to OCR, producing hands,
    faces and isolated glyphs. Here a coherent new-stroke seed must persist into a later registered
    state. The crop is materialized from the cleanest future state and expanded to the containing
    writing band before UniMERNet sees it.
    """

    if not config.temporal_proposals_enabled or len(frames) < 3:
        return []

    cv2 = _cv2()
    ordered = sorted(frames, key=lambda item: item.timestamp)
    images: list[np.ndarray] = []
    stroke_masks: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in ordered:
        with Image.open(frame.path) as raw:
            rgb = np.asarray(raw.convert("RGB"))
        images.append(rgb)
        stroke_masks.append(_stroke_masks(rgb, config))

    output_dir.mkdir(parents=True, exist_ok=True)
    proposals: list[DetectedFormulaCrop] = []
    for index in range(1, len(ordered) - 1):
        previous_rgb = images[index - 1]
        current_rgb = images[index]
        previous_registration = _registered_homography(previous_rgb, current_rgb, config)
        if previous_registration is None:
            continue
        previous_h, previous_ratio = previous_registration

        current_chalk, _current_foreground = stroke_masks[index]
        current_height, current_width = current_chalk.shape
        previous_warped = cv2.warpPerspective(
            stroke_masks[index - 1][0],
            previous_h,
            (current_width, current_height),
            flags=cv2.INTER_NEAREST,
        )
        tolerance = max(1, config.temporal_registration_tolerance_px)
        previous_warped = cv2.dilate(
            previous_warped,
            np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8),
            iterations=1,
        )
        raw_seed = np.logical_and(
            current_chalk > 0,
            previous_warped == 0,
        ).astype(np.uint8) * 255
        raw_seed = cv2.morphologyEx(
            raw_seed,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )

        for component_index, component in enumerate(_seed_components(raw_seed, config)):
            best: tuple[
                float,
                int,
                np.ndarray,
                tuple[int, int, int, int],
                float,
            ] | None = None
            max_future = min(
                len(ordered) - 1,
                index + config.temporal_lookahead_states,
            )
            # Require at least one later frame. This is what turns "motion now" into persistent
            # writing rather than a hand/face crop.
            for future_index in range(index + 1, max_future + 1):
                registration = _registered_homography(
                    current_rgb,
                    images[future_index],
                    config,
                )
                if registration is None:
                    # A real shot cut ends the temporal track.
                    break
                homography, future_ratio = registration
                future_height, future_width = stroke_masks[future_index][0].shape
                future_seed = cv2.warpPerspective(
                    component,
                    homography,
                    (future_width, future_height),
                    flags=cv2.INTER_NEAREST,
                )
                seed_pixels = int((future_seed > 0).sum())
                if seed_pixels <= 0:
                    continue

                future_chalk, future_foreground = stroke_masks[future_index]
                chalk_nearby = cv2.dilate(
                    future_chalk,
                    np.ones((7, 7), dtype=np.uint8),
                    iterations=1,
                )
                persistent_pixels = int(
                    np.logical_and(future_seed > 0, chalk_nearby > 0).sum()
                )
                persistence = persistent_pixels / max(1, seed_pixels)
                if persistence < config.temporal_min_persistence_ratio:
                    continue

                box = _local_formula_box(
                    images[future_index],
                    future_seed,
                    config,
                )
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                foreground_fraction = float(
                    (future_foreground[y0:y1, x0:x1] > 0).mean()
                )
                # Prefer a clean later view; persistence and registration quality break ties.
                score = (
                    1.0
                    - min(1.0, foreground_fraction)
                    + 0.20 * persistence
                    + 0.10 * future_ratio
                    + 0.01 * (future_index - index)
                )
                candidate = (
                    score,
                    future_index,
                    future_seed,
                    box,
                    min(previous_ratio, future_ratio),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate

            if best is None:
                continue
            _score, future_index, _future_seed, box, registration_confidence = best
            crop_id = f"{id_prefix}_s{index:02d}_t{component_index:02d}"
            path = output_dir / f"{crop_id}.jpg"
            with Image.open(ordered[future_index].path) as raw:
                raw.convert("RGB").crop(box).save(path, quality=97)
            proposals.append(
                DetectedFormulaCrop(
                    id=crop_id,
                    frame=ExtractedFrame(
                        timestamp=ordered[future_index].timestamp,
                        path=path,
                    ),
                    bbox=box,
                    # Proposal confidence is intentionally below a good MFD detection. It measures
                    # geometric support, not OCR correctness.
                    confidence=min(0.60, 0.55 * registration_confidence),
                )
            )

    # Deduplicate temporal attention seeds that expanded to the same clean formula line.
    distinct: list[DetectedFormulaCrop] = []
    for crop in sorted(
        proposals,
        key=lambda item: (-item.confidence, item.frame.timestamp, item.bbox[1], item.bbox[0]),
    ):
        if any(
            abs(existing.frame.timestamp - crop.frame.timestamp) < 1e-3
            and _bbox_iou(existing.bbox, crop.bbox) >= 0.55
            for existing in distinct
        ):
            continue
        distinct.append(crop)
    return sorted(
        distinct,
        key=lambda item: (item.frame.timestamp, item.bbox[1], item.bbox[0]),
    )[: config.temporal_max_crops_per_state * max(1, len(ordered) - 2)]


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def merge_formula_crops(
    crops: list[DetectedFormulaCrop],
) -> list[DetectedFormulaCrop]:
    """Merge MFD and temporal proposals, de-duplicating only within the same board state."""

    ordered = sorted(
        crops,
        key=lambda item: (
            item.frame.timestamp,
            -item.confidence,
            item.bbox[1],
            item.bbox[0],
        ),
    )
    kept: list[DetectedFormulaCrop] = []
    for crop in ordered:
        duplicate = next(
            (
                existing
                for existing in kept
                if abs(existing.frame.timestamp - crop.frame.timestamp) < 1e-3
                and _bbox_iou(existing.bbox, crop.bbox) >= 0.70
            ),
            None,
        )
        if duplicate is None:
            kept.append(crop)
    return sorted(
        kept,
        key=lambda item: (item.frame.timestamp, item.bbox[1], item.bbox[0]),
    )


def make_formula_detector(config: FormulaDetectionConfig) -> FormulaDetector | None:
    if not config.enabled or config.backend == "none":
        return None
    if config.backend == "yolov8":
        return FormulaDetector(config)
    raise ValueError(f"unsupported formula detector backend: {config.backend}")


def build_formula_contact_sheet(
    crops: list[DetectedFormulaCrop],
    output_path: Path,
    *,
    columns: int = 2,
    max_items: int = 8,
) -> Path | None:
    """Build one VLM-friendly numbered sheet while preserving individual crop pixels."""

    selected = crops[:max_items]
    if not selected:
        return None

    columns = max(1, min(columns, len(selected)))
    rows = (len(selected) + columns - 1) // columns
    cell_width = 760
    image_height = 260
    label_height = 34
    margin = 12
    cell_height = image_height + label_height + 2 * margin

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, crop in enumerate(selected):
        row = index // columns
        column = index % columns
        x0 = column * cell_width
        y0 = row * cell_height

        with Image.open(crop.frame.path) as raw:
            image = raw.convert("RGB")
            available_width = cell_width - 2 * margin
            available_height = image_height - 2 * margin
            scale = min(
                available_width / max(1, image.width),
                available_height / max(1, image.height),
            )
            scale = max(scale, 1e-6)
            resized = image.resize(
                (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                ),
                Image.Resampling.LANCZOS,
            )

        image_x = x0 + margin + (available_width - resized.width) // 2
        image_y = y0 + margin + (available_height - resized.height) // 2
        sheet.paste(resized, (image_x, image_y))

        label = (
            f"{crop.id}  t={crop.frame.timestamp:.1f}s  "
            f"det={crop.confidence:.2f}"
        )
        draw.text((x0 + margin, y0 + image_height + 4), label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path
