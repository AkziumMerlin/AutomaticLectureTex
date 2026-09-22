from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from automatic_lecture_tex.config import MathOCRConfig
from automatic_lecture_tex.math_ocr import LatexOCRBackend


def _install_fake_pix2tex(monkeypatch, *, cuda_available: bool):
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: cuda_available)

    captured = {}

    class FakeLatexOCR:
        def __init__(self, arguments):
            captured["arguments"] = arguments

        def __call__(self, image):
            del image
            return r"x=1"

    pix2tex_module = ModuleType("pix2tex")
    cli_module = ModuleType("pix2tex.cli")
    cli_module.LatexOCR = FakeLatexOCR

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "pix2tex", pix2tex_module)
    monkeypatch.setitem(sys.modules, "pix2tex.cli", cli_module)
    return captured


def test_latexocr_cuda_is_explicitly_enabled(monkeypatch):
    captured = _install_fake_pix2tex(monkeypatch, cuda_available=True)

    LatexOCRBackend(MathOCRConfig(backend="latexocr", device="cuda"))

    assert captured["arguments"].no_cuda is False


def test_latexocr_cuda_fails_fast_when_unavailable(monkeypatch):
    _install_fake_pix2tex(monkeypatch, cuda_available=False)

    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\) is False"):
        LatexOCRBackend(MathOCRConfig(backend="latexocr", device="cuda"))


def test_latexocr_cpu_must_be_requested_explicitly(monkeypatch):
    captured = _install_fake_pix2tex(monkeypatch, cuda_available=False)

    LatexOCRBackend(MathOCRConfig(backend="latexocr", device="cpu"))

    assert captured["arguments"].no_cuda is True
