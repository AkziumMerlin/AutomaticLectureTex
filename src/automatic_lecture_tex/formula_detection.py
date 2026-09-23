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


def _chalk_mask(rgb: np.ndarray, *, large_component_fraction: float) -> np.ndarray:
    """Extract thin bright writing and suppress large foreground objects such as the lecturer."""

    cv2 = _cv2()
    image = rgb.astype(np.float32)
    background = _dominant_surface_color(rgb)
    distance = np.linalg.norm(image - background[None, None, :], axis=2)
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    board_luminance = float(
        0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    )

    # Chalk is both substantially brighter than the board and chromatically distinct. This also
    # admits some foreground pixels, which are removed below by connected-component geometry.
    mask = (
        (luminance >= max(82.0, board_luminance + 18.0))
        & (distance >= 28.0)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )

    # Join filled foreground regions before measuring their area. Thin chalk strokes remain too
    # sparse to form a large filled component at this scale.
    joined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(joined, 8)
    max_area = max(1, int(mask.shape[0] * mask.shape[1] * large_component_fraction))
    foreground = np.zeros_like(mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= max_area:
            foreground[labels == index] = 255
    if np.any(foreground):
        foreground = cv2.dilate(
            foreground,
            np.ones((11, 11), dtype=np.uint8),
            iterations=1,
        )
        mask[foreground > 0] = 0
    return mask


def _merge_active_runs(
    active: np.ndarray,
    *,
    max_gap: int,
    min_height: int,
) -> list[tuple[int, int]]:
    rows = np.flatnonzero(active)
    if not len(rows):
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(rows[0])
    for raw in rows[1:]:
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


def _line_boxes(
    rgb: np.ndarray,
    config: FormulaDetectionConfig,
) -> list[tuple[int, int, int, int]]:
    """Split a coarse board region into tight horizontal writing bands."""

    height, width = rgb.shape[:2]
    if height < config.line_split_min_height_px:
        return [(0, 0, width, height)]

    mask = _chalk_mask(
        rgb,
        large_component_fraction=config.foreground_component_area_fraction,
    )
    row_counts = (mask > 0).sum(axis=1)
    row_threshold = max(3, int(round(width * 0.0045)))
    active = row_counts >= row_threshold
    max_gap = max(2, int(round(height * config.line_split_max_gap_fraction)))
    bands = _merge_active_runs(
        active,
        max_gap=max_gap,
        min_height=config.line_split_min_band_height_px,
    )
    if len(bands) <= 1:
        return [(0, 0, width, height)]

    boxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in bands:
        band_mask = mask[y0:y1]
        columns = np.flatnonzero((band_mask > 0).sum(axis=0) >= 1)
        if not len(columns):
            continue
        x0 = int(columns[0])
        x1 = int(columns[-1]) + 1
        band_width = x1 - x0
        band_height = y1 - y0
        pad_x = int(round(band_width * config.line_split_padding_fraction))
        pad_y = int(round(band_height * config.line_split_padding_fraction))
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(height, y1 + pad_y)
        if (
            x1 - x0 >= config.min_width_px
            and y1 - y0 >= config.min_height_px
        ):
            boxes.append((x0, y0, x1, y1))
    return boxes or [(0, 0, width, height)]


def split_oversized_formula_crops(
    crops: list[DetectedFormulaCrop],
    output_dir: Path,
    config: FormulaDetectionConfig,
) -> list[DetectedFormulaCrop]:
    """Turn board-sized MFD detections into OCR-sized chalk-line crops."""

    if not config.line_split_enabled:
        return crops
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[DetectedFormulaCrop] = []
    for crop in crops:
        with Image.open(crop.frame.path) as raw:
            image = raw.convert("RGB")
            rgb = np.asarray(image)
            boxes = _line_boxes(rgb, config)
            if len(boxes) == 1 and boxes[0] == (0, 0, image.width, image.height):
                result.append(crop)
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


def _boxes_from_temporal_seed(
    current_rgb: np.ndarray,
    seed: np.ndarray,
    config: FormulaDetectionConfig,
) -> list[tuple[int, int, int, int]]:
    """Grow newly written seed pixels to their containing chalk line."""

    cv2 = _cv2()
    chalk = _chalk_mask(
        current_rgb,
        large_component_fraction=config.foreground_component_area_fraction,
    )
    height, width = chalk.shape
    horizontal = max(15, width // 55)
    vertical = max(3, height // 240)
    lines = cv2.dilate(
        chalk,
        np.ones((vertical, horizontal), dtype=np.uint8),
        iterations=1,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(lines, 8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        component = labels == index
        if int(np.logical_and(component, seed > 0).sum()) < config.temporal_min_new_pixels:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        local = chalk[y : y + h, x : x + w]
        ys, xs = np.nonzero(local)
        if not len(xs):
            continue
        x0 = x + int(xs.min())
        x1 = x + int(xs.max()) + 1
        y0 = y + int(ys.min())
        y1 = y + int(ys.max()) + 1
        pad_x = max(6, int(round((x1 - x0) * config.padding_fraction)))
        pad_y = max(4, int(round((y1 - y0) * config.padding_fraction)))
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(height, y1 + pad_y)
        if (
            x1 - x0 >= config.min_width_px
            and y1 - y0 >= config.min_height_px
        ):
            boxes.append((x0, y0, x1, y1))

    # Keep geometrically distinct regions in reading order.
    boxes.sort(key=lambda box: (box[1], box[0]))
    distinct: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(_bbox_iou(box, existing) < 0.65 for existing in distinct):
            distinct.append(box)
    return distinct[: config.temporal_max_crops_per_state]


def detect_temporal_formula_crops(
    frames: list[ExtractedFrame],
    output_dir: Path,
    config: FormulaDetectionConfig,
    *,
    id_prefix: str,
) -> list[DetectedFormulaCrop]:
    """Localize newly written mathematics from registered neighboring board states.

    A proposal is emitted only when the previous and next board state both belong to the same shot
    as the current frame. New chalk must be absent in the registered previous state and persist in
    the registered next state, which rejects most lecturer motion without interpreting OCR output.
    """

    if not config.temporal_proposals_enabled or len(frames) < 3:
        return []

    cv2 = _cv2()
    ordered = sorted(frames, key=lambda item: item.timestamp)
    images: list[np.ndarray] = []
    for frame in ordered:
        with Image.open(frame.path) as raw:
            images.append(np.asarray(raw.convert("RGB")))

    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[DetectedFormulaCrop] = []
    for index in range(1, len(ordered) - 1):
        previous_rgb = images[index - 1]
        current_rgb = images[index]
        next_rgb = images[index + 1]
        current_height, current_width = current_rgb.shape[:2]

        previous_registration = _registered_homography(previous_rgb, current_rgb, config)
        next_registration = _registered_homography(next_rgb, current_rgb, config)
        if previous_registration is None or next_registration is None:
            continue
        previous_h, previous_ratio = previous_registration
        next_h, next_ratio = next_registration

        previous_mask = _chalk_mask(
            previous_rgb,
            large_component_fraction=config.foreground_component_area_fraction,
        )
        current_mask = _chalk_mask(
            current_rgb,
            large_component_fraction=config.foreground_component_area_fraction,
        )
        next_mask = _chalk_mask(
            next_rgb,
            large_component_fraction=config.foreground_component_area_fraction,
        )
        previous_warped = cv2.warpPerspective(
            previous_mask,
            previous_h,
            (current_width, current_height),
            flags=cv2.INTER_NEAREST,
        )
        next_warped = cv2.warpPerspective(
            next_mask,
            next_h,
            (current_width, current_height),
            flags=cv2.INTER_NEAREST,
        )
        tolerance = np.ones((5, 5), dtype=np.uint8)
        previous_warped = cv2.dilate(previous_warped, tolerance, iterations=1)
        next_warped = cv2.dilate(next_warped, tolerance, iterations=1)

        new_seed = np.logical_and(current_mask > 0, previous_warped == 0)
        persistent = np.logical_and(new_seed, next_warped > 0).astype(np.uint8) * 255
        persistent = cv2.morphologyEx(
            persistent,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )
        if int((persistent > 0).sum()) < config.temporal_min_new_pixels:
            continue

        boxes = _boxes_from_temporal_seed(current_rgb, persistent, config)
        if not boxes:
            continue

        confidence = min(previous_ratio, next_ratio)
        with Image.open(ordered[index].path) as raw:
            image = raw.convert("RGB")
            for local_index, box in enumerate(boxes):
                crop_id = f"{id_prefix}_s{index:02d}_t{local_index:02d}"
                path = output_dir / f"{crop_id}.jpg"
                image.crop(box).save(path, quality=97)
                result.append(
                    DetectedFormulaCrop(
                        id=crop_id,
                        frame=ExtractedFrame(
                            timestamp=ordered[index].timestamp,
                            path=path,
                        ),
                        bbox=box,
                        confidence=confidence,
                    )
                )
    return result


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
