from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

from automatic_lecture_tex.config import FormulaDetectionConfig
from automatic_lecture_tex.formula_detection import (
    DetectedFormulaCrop,
    FormulaDetector,
    build_formula_contact_sheet,
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
