from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
