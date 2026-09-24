from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from automatic_lecture_tex import math_ocr as math_ocr_module
from automatic_lecture_tex.config import LLMConfig, MathOCRConfig
from automatic_lecture_tex.math_ocr import (
    LatexOCRBackend,
    QwenVLMOCRBackend,
    UniMERNetBackend,
    UniMuMERBackend,
)


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
    formula_root = tmp_path / "models" / "formula"
    python_path = formula_root / "unimernet-env" / "bin" / "python"
    config_path = formula_root / "unimernet_small" / "unimernet.yaml"
    source_root = formula_root / "UniMERNet-src"
    image_path = tmp_path / "formula.png"
    python_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    source_root.mkdir(parents=True)
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
    assert str(source_root) in captured["preflight_kwargs"]["env"]["PYTHONPATH"]
    assert str(python_path) == captured["args"][0]
    assert "--config" in captured["args"]
    assert any('"image"' in item for item in process.stdin.writes)

    backend.close()
    assert any('"shutdown"' in item for item in process.stdin.writes)



def test_unimumer_uses_persistent_isolated_worker(tmp_path, monkeypatch):
    python_path = tmp_path / "unimumer-env" / "bin" / "python"
    image_path = tmp_path / "formula.png"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    image_path.write_bytes(b"image")

    process = _FakeWorkerProcess()
    captured = {}

    def fake_run(args, **kwargs):
        captured["preflight_args"] = args
        return SimpleNamespace(returncode=0, stdout="0.27.1\n", stderr="")

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(math_ocr_module.subprocess, "run", fake_run)
    monkeypatch.setattr(math_ocr_module.subprocess, "Popen", fake_popen)

    backend = UniMuMERBackend(
        MathOCRConfig(
            backend="unimumer",
            device="cuda",
            unimumer_model="phxember/Uni-MuMER-Qwen3.5-4B",
            unimumer_python_path=python_path,
        )
    )
    candidate = backend.recognize(image_path)

    assert candidate is not None
    assert candidate.backend == "unimumer"
    assert candidate.text == r"\frac{x}{y}"
    assert str(python_path) == captured["preflight_args"][0]
    assert "vllm" in captured["preflight_args"][2]
    assert str(python_path) == captured["args"][0]
    assert "--model" in captured["args"]
    assert "phxember/Uni-MuMER-Qwen3.5-4B" in captured["args"]
    assert any('"image"' in item for item in process.stdin.writes)

    backend.close()
    assert any('"shutdown"' in item for item in process.stdin.writes)


def test_qwen_vlm_ocr_reuses_configured_multimodal_server(tmp_path, monkeypatch):
    image_path = tmp_path / "formula.png"
    Image.new("RGB", (120, 60), "white").save(image_path)
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=r"\frac{x}{y}")
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    openai_module = ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    backend = QwenVLMOCRBackend(
        MathOCRConfig(
            backend="qwen_vlm",
            normalize_dark_formula=False,
        ),
        LLMConfig(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            model="Qwen/test-vlm",
        ),
    )
    candidate = backend.recognize(image_path)

    assert candidate is not None
    assert candidate.backend == "qwen_vlm"
    assert candidate.text == r"\frac{x}{y}"
    assert captured["model"] == "Qwen/test-vlm"
    assert captured["temperature"] == 0.0
    assert captured["client"]["base_url"] == "http://127.0.0.1:8000/v1"
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "handwritten mathematical expression" in content[1]["text"]
