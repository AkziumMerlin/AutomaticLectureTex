from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from automatic_lecture_tex import math_ocr as math_ocr_module
from automatic_lecture_tex.config import MathOCRConfig
from automatic_lecture_tex.math_ocr import LatexOCRBackend, UniMERNetBackend


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


class _FakeWorkerStream:
    def __init__(self, lines=None):
        self.lines = list(lines or [])
        self.writes = []

    def readline(self):
        if not self.lines:
            return ""
        return self.lines.pop(0)

    def write(self, value):
        self.writes.append(value)
        return len(value)

    def flush(self):
        return None


class _FakeWorkerProcess:
    def __init__(self):
        self.stdin = _FakeWorkerStream()
        self.stdout = _FakeWorkerStream(
            [
                '{"ready": true}\n',
                '{"text": "\\\\frac{x}{y}"}\n',
            ]
        )
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def test_unimernet_uses_persistent_isolated_worker(tmp_path, monkeypatch):
    python_path = tmp_path / "unimernet-python"
    config_path = tmp_path / "unimernet.yaml"
    image_path = tmp_path / "formula.png"
    python_path.write_text("", encoding="utf-8")
    config_path.write_text("", encoding="utf-8")
    image_path.write_bytes(b"image")

    process = _FakeWorkerProcess()
    captured = {}

    def fake_run(args, **kwargs):
        captured["preflight_args"] = args
        captured["preflight_kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="/tmp/unimernet/__init__.py\n", stderr="")

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(math_ocr_module.subprocess, "run", fake_run)
    monkeypatch.setattr(math_ocr_module.subprocess, "Popen", fake_popen)

    backend = UniMERNetBackend(
        MathOCRConfig(
            backend="unimernet",
            device="cuda",
            unimernet_config_path=config_path,
            unimernet_python_path=python_path,
        )
    )
    candidate = backend.recognize(image_path)

    assert candidate is not None
    assert candidate.backend == "unimernet"
    assert candidate.text == r"\frac{x}{y}"
    assert str(python_path) == captured["preflight_args"][0]
    assert "import pathlib, sys, unimernet, unimernet.tasks" in captured["preflight_args"][2]
    assert str(python_path) == captured["args"][0]
    assert "--config" in captured["args"]
    assert any('"image"' in item for item in process.stdin.writes)

    backend.close()
    assert any('"shutdown"' in item for item in process.stdin.writes)
