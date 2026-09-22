from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .config import MathOCRConfig
from .schemas import MathOCRCandidate

_MAX_OCR_TEXT_CHARS = 8000


class MathOCRBackend(ABC):
    @abstractmethod
    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        raise NotImplementedError


class LatexOCRBackend(MathOCRBackend):
    """Local pix2tex/LaTeX-OCR backend.

    This backend is intentionally formula-only. It is used as a literal glyph/transcription
    sensor; semantic interpretation remains the responsibility of the multimodal reconstruction
    stage.
    """

    def __init__(self, config: MathOCRConfig) -> None:
        self.config = config
        try:
            import torch
            from pix2tex.cli import LatexOCR
        except ImportError as exc:
            raise RuntimeError(
                "LaTeX-OCR backend requires pix2tex. Install with "
                "pip install 'automatic-lecture-tex[latexocr]'"
            ) from exc

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LaTeX-OCR is configured for CUDA, but torch.cuda.is_available() is False. "
                "Fix the PyTorch/CUDA installation or set vision.math_ocr.device=cpu explicitly."
            )

        # pix2tex's Python API defaults to no_cuda=True when LatexOCR() is called without
        # arguments, so always pass the device choice explicitly.
        arguments = argparse.Namespace(
            config="settings/config.yaml",
            checkpoint="checkpoints/weights.pth",
            no_cuda=config.device == "cpu",
            no_resize=False,
        )
        self.model = LatexOCR(arguments)

    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("LaTeX-OCR backend requires Pillow") from exc

        with Image.open(image_path) as image:
            text = str(self.model(image.convert("RGB")) or "").strip()[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        return MathOCRCandidate(backend="latexocr", text=text)


class MathpixBackend(MathOCRBackend):
    def __init__(self, config: MathOCRConfig) -> None:
        self.config = config
        self.app_id = os.getenv(config.mathpix_app_id_env, "").strip()
        self.app_key = os.getenv(config.mathpix_app_key_env, "").strip()
        if not self.app_id or not self.app_key:
            raise RuntimeError(
                "Mathpix OCR is enabled but credentials are missing. Set "
                f"{config.mathpix_app_id_env} and {config.mathpix_app_key_env}."
            )

    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "src": f"data:{mime};base64,{encoded}",
            "formats": ["text"],
            "rm_spaces": True,
            "enable_document_layout": True,
        }
        request = urllib.request.Request(
            "https://api.mathpix.com/v3/text",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
        text = str(result.get("text") or "").strip()[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        confidence = result.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return MathOCRCandidate(backend="mathpix", text=text, confidence=confidence)


class UniMERNetBackend(MathOCRBackend):
    """Local UniMERNet wrapper following the upstream inference API.

    UniMERNet is most useful when the temporal composite contains a dominant formula or a cropped
    board region. The backend is optional because its model environment is substantially heavier
    than the core lecture pipeline.
    """

    def __init__(self, config: MathOCRConfig) -> None:
        if config.unimernet_config_path is None:
            raise RuntimeError("UniMERNet OCR requires vision.math_ocr.unimernet_config_path")
        try:
            import torch
            import unimernet.tasks as tasks
            from PIL import Image
            from unimernet.common.config import Config
            from unimernet.processors import load_processor
        except ImportError as exc:
            raise RuntimeError(
                "UniMERNet OCR requires the upstream package, for example: "
                "pip install 'unimernet[full]'"
            ) from exc

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "UniMERNet OCR is configured for CUDA, but torch.cuda.is_available() is False."
            )

        self.torch = torch
        self.Image = Image
        args = argparse.Namespace(cfg_path=str(config.unimernet_config_path), options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        self.device = torch.device(config.device)
        self.model = task.build_model(cfg).to(self.device)
        self.model.eval()
        self.processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )

    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        raw_image = self.Image.open(image_path).convert("RGB")
        image = self.processor(raw_image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate({"image": image})
        text = str(output["pred_str"][0]).strip()[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        return MathOCRCandidate(backend="unimernet", text=text)


def make_math_ocr_backend(config: MathOCRConfig) -> MathOCRBackend | None:
    if config.backend == "none":
        return None
    if config.backend == "mathpix":
        return MathpixBackend(config)
    if config.backend == "unimernet":
        return UniMERNetBackend(config)
    if config.backend == "latexocr":
        return LatexOCRBackend(config)
    raise ValueError(f"unsupported math OCR backend: {config.backend}")
