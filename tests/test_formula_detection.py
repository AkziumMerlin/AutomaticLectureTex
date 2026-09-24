from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from automatic_lecture_tex.config import FormulaDetectionConfig
from automatic_lecture_tex.formula_detection import (
    DetectedFormulaCrop,
    FormulaDetector,
    build_formula_contact_sheet,
    detect_temporal_formula_crops,
    split_oversized_formula_crops,
)
from automatic_lecture_tex.schemas import ExtractedFrame


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor(
            [
                [20.0, 10.0, 180.0, 55.0],
                [30.0, 80.0, 220.0, 120.0],
            ]
        )
        self.conf = _FakeTensor([0.91, 0.74])

    def __len__(self):
        return 2


def _install_fake_yolo(monkeypatch):
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: True)

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, **kwargs):
            assert kwargs["device"] == 0
            return [SimpleNamespace(boxes=_FakeBoxes())]

    ultralytics_module = ModuleType("ultralytics")
    ultralytics_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)


def test_formula_detector_materializes_ordered_crops(tmp_path, monkeypatch):
    _install_fake_yolo(monkeypatch)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    image_path = tmp_path / "board.jpg"
    Image.new("RGB", (320, 180), "white").save(image_path)

    detector = FormulaDetector(
        FormulaDetectionConfig(
            enabled=True,
            backend="yolov8",
            model_path=weights,
            device="cuda",
            padding_fraction=0.0,
        )
    )
    crops = detector.detect(
        ExtractedFrame(timestamp=12.5, path=image_path),
        tmp_path / "crops",
        id_prefix="state_00",
    )

    assert [crop.id for crop in crops] == ["state_00_f00", "state_00_f01"]
    assert [crop.bbox for crop in crops] == [
        (20, 10, 180, 55),
        (30, 80, 220, 120),
    ]
    assert all(crop.frame.path.is_file() for crop in crops)
    assert all(crop.frame.timestamp == 12.5 for crop in crops)


def test_formula_contact_sheet_preserves_crop_ids(tmp_path):
    crops = []
    for index in range(3):
        image_path = tmp_path / f"formula_{index}.jpg"
        Image.new("RGB", (180 + 20 * index, 60), "white").save(image_path)
        crops.append(
            DetectedFormulaCrop(
                id=f"f{index}",
                frame=ExtractedFrame(timestamp=float(index), path=image_path),
                bbox=(0, 0, 100, 40),
                confidence=0.9 - 0.1 * index,
            )
        )

    output = build_formula_contact_sheet(
        crops,
        tmp_path / "contact.jpg",
        columns=2,
        max_items=3,
    )

    assert output is not None
    assert output.is_file()
    with Image.open(output) as sheet:
        assert sheet.width > 1000
        assert sheet.height > 500



def test_oversized_mfd_crop_is_split_into_chalk_lines(tmp_path):
    image_path = tmp_path / "coarse.jpg"
    image = Image.new("RGB", (640, 360), (35, 88, 58))
    draw = ImageDraw.Draw(image)
    # Two mathematical lines.
    for x in range(180, 560, 28):
        draw.rectangle((x, 75, x + 14, 84), fill=(235, 235, 220))
    for x in range(205, 600, 24):
        draw.rectangle((x, 238, x + 12, 248), fill=(238, 238, 225))
    # Large bright foreground object, representing the lecturer, should be removed before row split.
    draw.rectangle((10, 35, 140, 335), fill=(190, 160, 135))
    image.save(image_path)

    crop = DetectedFormulaCrop(
        id="coarse",
        frame=ExtractedFrame(timestamp=1.0, path=image_path),
        bbox=(100, 50, 740, 410),
        confidence=0.8,
    )
    config = FormulaDetectionConfig(
        line_split_enabled=True,
        line_split_min_height_px=120,
        line_split_min_band_height_px=5,
        stroke_foreground_core_radius_px=8.0,
        stroke_foreground_core_dilate_px=7,
        min_width_px=20,
        min_height_px=5,
    )

    refined = split_oversized_formula_crops(
        [crop],
        tmp_path / "refined",
        config,
    )

    assert len(refined) == 2
    assert all(item.frame.path.is_file() for item in refined)
    assert all(item.id.startswith("coarse_l") for item in refined)
    assert refined[0].bbox[1] < refined[1].bbox[1]


def test_temporal_proposal_requires_persistent_new_writing(tmp_path, monkeypatch):
    import automatic_lecture_tex.formula_detection as module

    paths = []
    for index in range(3):
        path = tmp_path / f"state_{index}.png"
        image = Image.new("RGB", (640, 320), (35, 88, 58))
        draw = ImageDraw.Draw(image)
        # Fixed board landmarks/writing.
        for x in range(80, 560, 45):
            draw.rectangle((x, 55, x + 10, 64), fill=(235, 235, 220))
        # New line appears at current state and persists into the next state.
        if index >= 1:
            for x in range(220, 520, 25):
                draw.rectangle((x, 205, x + 12, 214), fill=(240, 240, 225))
        image.save(path)
        paths.append(path)

    monkeypatch.setattr(
        module,
        "_registered_homography",
        lambda _source, _target, _config: (np.eye(3, dtype=np.float32), 0.9),
    )
    config = FormulaDetectionConfig(
        temporal_proposals_enabled=True,
        temporal_min_new_pixels=8,
        temporal_max_crops_per_state=3,
        temporal_lookahead_states=2,
        temporal_min_persistence_ratio=0.6,
        line_split_min_band_height_px=5,
        stroke_foreground_core_radius_px=8.0,
        stroke_foreground_core_dilate_px=7,
        min_width_px=20,
        min_height_px=5,
    )
    frames = [
        ExtractedFrame(timestamp=float(index), path=path)
        for index, path in enumerate(paths)
    ]

    crops = detect_temporal_formula_crops(
        frames,
        tmp_path / "temporal",
        config,
        id_prefix="window",
    )

    assert crops
    # Temporal change is only an attention seed: OCR uses the later clean state where the writing
    # has persisted, never the just-changing frame itself.
    assert all(crop.frame.timestamp == 2.0 for crop in crops)
    assert any(crop.bbox[1] < 205 < crop.bbox[3] for crop in crops)
    assert any(crop.bbox[0] < 300 < crop.bbox[2] for crop in crops)
    assert all(crop.frame.path.is_file() for crop in crops)



def test_temporal_proposal_rejects_transient_motion(tmp_path, monkeypatch):
    import automatic_lecture_tex.formula_detection as module

    paths = []
    for index in range(3):
        path = tmp_path / f"motion_{index}.png"
        image = Image.new("RGB", (640, 320), (35, 88, 58))
        draw = ImageDraw.Draw(image)
        for x in range(80, 560, 45):
            draw.rectangle((x, 55, x + 10, 64), fill=(235, 235, 220))
        if index == 1:
            # A transient thin object is visible only in the changing frame.
            draw.line((220, 205, 520, 205), fill=(240, 240, 225), width=4)
        image.save(path)
        paths.append(path)

    monkeypatch.setattr(
        module,
        "_registered_homography",
        lambda _source, _target, _config: (np.eye(3, dtype=np.float32), 0.9),
    )
    config = FormulaDetectionConfig(
        temporal_proposals_enabled=True,
        temporal_min_new_pixels=8,
        temporal_lookahead_states=2,
        temporal_min_persistence_ratio=0.6,
        min_width_px=20,
        min_height_px=5,
    )
    frames = [
        ExtractedFrame(timestamp=float(index), path=path)
        for index, path in enumerate(paths)
    ]

    crops = detect_temporal_formula_crops(
        frames,
        tmp_path / "transient",
        config,
        id_prefix="window",
    )

    assert crops == []



def test_layout_split_respects_board_divider_and_oval_bridges(tmp_path):
    image_path = tmp_path / "board_scene.jpg"
    image = Image.new("RGB", (900, 500), (35, 88, 58))
    draw = ImageDraw.Draw(image)

    # Two separate board panels.
    draw.rectangle((435, 0, 465, 499), fill=(240, 240, 235))

    # Left panel: two writing bands.
    for x in range(60, 380, 26):
        draw.rectangle((x, 95, x + 12, 106), fill=(238, 238, 225))
    for x in range(80, 410, 24):
        draw.rectangle((x, 315, x + 12, 326), fill=(238, 238, 225))

    # Right panel: two bands plus a large oval/brace-like stroke spanning them. The oval should
    # not glue the two rows into one board-sized OCR crop.
    for x in range(520, 840, 24):
        draw.rectangle((x, 105, x + 12, 116), fill=(238, 238, 225))
    for x in range(500, 825, 26):
        draw.rectangle((x, 330, x + 12, 341), fill=(238, 238, 225))
    draw.ellipse((500, 70, 850, 385), outline=(235, 235, 220), width=4)
    image.save(image_path)

    crop = DetectedFormulaCrop(
        id="scene",
        frame=ExtractedFrame(timestamp=1.0, path=image_path),
        bbox=(100, 50, 1000, 550),
        confidence=0.8,
    )
    config = FormulaDetectionConfig(
        line_split_enabled=True,
        line_split_min_height_px=100,
        line_split_min_band_height_px=5,
        line_split_min_panel_width_px=120,
        line_split_max_band_height_px=160,
        stroke_foreground_core_radius_px=8.0,
        stroke_foreground_core_dilate_px=5,
        min_width_px=30,
        min_height_px=5,
    )

    refined = split_oversized_formula_crops(
        [crop],
        tmp_path / "layout_refined",
        config,
    )

    assert len(refined) >= 4
    divider = 100 + 450
    assert all(
        item.bbox[2] <= divider or item.bbox[0] >= divider
        for item in refined
    )
    assert max(item.bbox[3] - item.bbox[1] for item in refined) < 220
