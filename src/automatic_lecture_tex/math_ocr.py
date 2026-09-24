from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .config import MathOCRConfig
from .schemas import MathOCRCandidate

_MAX_OCR_TEXT_CHARS = 8000


def _normalize_formula_image(image, config: MathOCRConfig):
    if not config.normalize_dark_formula:
        return image.convert("RGB")

    from PIL import ImageOps

    gray = image.convert("L")
    histogram = gray.histogram()
    midpoint = sum(histogram) / 2
    running = 0
    median = 255
    for value, count in enumerate(histogram):
        running += count
        if running >= midpoint:
            median = value
            break
    if median < config.dark_formula_threshold:
        gray = ImageOps.invert(gray)
    return ImageOps.autocontrast(gray).convert("RGB")


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
            prepared = _normalize_formula_image(image, self.config)
            text = str(self.model(prepared) or "").strip()[:_MAX_OCR_TEXT_CHARS]
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


class UniMuMERBackend(MathOCRBackend):
    """Specialized handwritten-math OCR via an isolated Uni-MuMER/Qwen worker."""

    def __init__(self, config: MathOCRConfig) -> None:
        self.config = config
        self._worker: subprocess.Popen[str] | None = None
        self._start_worker()

    def _start_worker(self) -> None:
        python_path = self.config.unimumer_python_path
        if python_path is None:
            raise RuntimeError(
                "Uni-MuMER OCR requires vision.math_ocr.unimumer_python_path. "
                "Run scripts/create_unimumer_worker_env.sh first."
            )
        if not python_path.is_file():
            raise RuntimeError(
                f"Uni-MuMER worker Python does not exist: {python_path}. "
                "Run scripts/create_unimumer_worker_env.sh first."
            )

        preflight = subprocess.run(  # noqa: S603
            [
                str(python_path),
                "-c",
                "import vllm, transformers, qwen_vl_utils; print(vllm.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if preflight.returncode != 0:
            raise RuntimeError(
                "Uni-MuMER worker preflight failed. "
                f"python={python_path}, stderr={preflight.stderr.strip()!r}"
            )

        worker_script = Path(__file__).with_name("unimumer_worker.py")
        process = subprocess.Popen(  # noqa: S603
            [
                str(python_path),
                str(worker_script),
                "--model",
                self.config.unimumer_model,
                "--device",
                self.config.device,
                "--max-tokens",
                str(self.config.unimumer_max_tokens),
                "--temperature",
                str(self.config.unimumer_temperature),
                "--top-p",
                str(self.config.unimumer_top_p),
                "--gpu-memory-utilization",
                str(self.config.unimumer_gpu_memory_utilization),
                "--dark-formula-threshold",
                str(self.config.dark_formula_threshold),
                *(
                    ["--normalize-dark-formula"]
                    if self.config.normalize_dark_formula
                    else []
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if not ready_line:
            return_code = process.poll()
            process.terminate()
            raise RuntimeError(
                "Uni-MuMER worker exited before reporting readiness"
                + (f" (return code {return_code})" if return_code is not None else "")
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            process.terminate()
            raise RuntimeError(
                f"Invalid Uni-MuMER worker startup response: {ready_line!r}"
            ) from exc
        if not ready.get("ready"):
            process.terminate()
            raise RuntimeError(f"Uni-MuMER worker failed to initialize: {ready!r}")
        self._worker = process

    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        process = self._worker
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Uni-MuMER worker is not available")
        if process.poll() is not None:
            raise RuntimeError(
                f"Uni-MuMER worker exited unexpectedly with return code {process.returncode}"
            )

        process.stdin.write(
            json.dumps({"image": str(image_path)}, ensure_ascii=False) + "\n"
        )
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError("Uni-MuMER worker closed stdout during inference")
        response = json.loads(response_line)
        if response.get("error"):
            raise RuntimeError(f"Uni-MuMER worker error: {response['error']}")
        text = str(response.get("text") or "").strip()[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        return MathOCRCandidate(backend="unimumer", text=text)

    def close(self) -> None:
        process = self._worker
        self._worker = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=5)
        except Exception:
            process.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class QwenVLMOCRBackend(MathOCRBackend):
    """Reuse an OpenAI-compatible multimodal Qwen server as a literal formula transcriber."""

    _PROMPT = (
        "Transcribe this cropped handwritten mathematical expression from a chalkboard into LaTeX. "
        "Return only the LaTeX expression, without markdown fences or explanation. "
        "Read the visible symbols literally; do not complete or infer missing mathematics."
    )

    def __init__(self, config: MathOCRConfig, llm_config) -> None:
        if llm_config is None and (
            not config.qwen_vlm_base_url or not config.qwen_vlm_model
        ):
            raise RuntimeError(
                "qwen_vlm OCR requires the application llm config or explicit "
                "qwen_vlm_base_url/qwen_vlm_model values"
            )

        from openai import OpenAI

        self.config = config
        base_url = config.qwen_vlm_base_url or llm_config.base_url
        api_key = config.qwen_vlm_api_key or llm_config.api_key
        self.model = config.qwen_vlm_model or llm_config.model
        timeout = getattr(llm_config, "timeout_seconds", 300.0)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        from PIL import Image

        with Image.open(image_path) as raw:
            prepared = _normalize_formula_image(raw, self.config)
            buffer = io.BytesIO()
            prepared.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=self.config.qwen_vlm_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + encoded},
                        },
                        {"type": "text", "text": self._PROMPT},
                    ],
                }
            ],
        )
        text = str(response.choices[0].message.content or "").strip()
        if text.startswith("~~~") and text.endswith("~~~"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        text = text[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        return MathOCRCandidate(backend="qwen_vlm", text=text)


class UniMERNetBackend(MathOCRBackend):
    """UniMERNet wrapper with an optional persistent isolated worker process.

    UniMERNet 0.2.3 pins an older Transformers release than Qwen-ASR. When an isolated Python path
    is configured, inference runs in that interpreter and the model stays resident across all crop
    requests. The legacy in-process path remains available for dedicated environments.
    """

    def __init__(self, config: MathOCRConfig) -> None:
        if config.unimernet_config_path is None:
            raise RuntimeError("UniMERNet OCR requires vision.math_ocr.unimernet_config_path")

        self.config = config
        self._worker: subprocess.Popen[str] | None = None
        self.torch = None
        self.Image = None
        self.device = None
        self.model = None
        self.processor = None

        if config.unimernet_python_path is not None:
            self._start_worker()
        else:
            self._init_in_process()

    def _start_worker(self) -> None:
        python_path = self.config.unimernet_python_path
        assert python_path is not None
        if not python_path.is_file():
            raise RuntimeError(
                f"UniMERNet worker Python does not exist: {python_path}. "
                "Run scripts/create_unimernet_worker_env.sh first."
            )
        worker_script = Path(__file__).with_name("unimernet_worker.py")

        # Do not rely only on editable-install import hooks in the isolated environment. The
        # official UniMERNet source tree is kept next to the worker venv, so make that source root
        # explicit for both the preflight and the long-lived worker process.
        worker_env = os.environ.copy()
        # The Python entrypoint may itself be a symlink, so never infer the model/source root from
        # its target. The UniMERNet config lives at <formula_root>/unimernet_small/*.yaml.
        formula_root = self.config.unimernet_config_path.parent.parent
        source_root = formula_root / "UniMERNet-src"
        if source_root.is_dir():
            existing_pythonpath = worker_env.get("PYTHONPATH", "")
            worker_env["PYTHONPATH"] = (
                str(source_root)
                if not existing_pythonpath
                else str(source_root) + os.pathsep + existing_pythonpath
            )

        preflight = subprocess.run(  # noqa: S603
            [
                str(python_path),
                "-c",
                (
                    "import pathlib, sys, unimernet, unimernet.tasks; "
                    "print(pathlib.Path(unimernet.__file__).resolve()); "
                    "print(sys.executable)"
                ),
            ],
            env=worker_env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if preflight.returncode != 0:
            raise RuntimeError(
                "UniMERNet worker preflight import failed. "
                f"python={python_path}, source_root={source_root}, "
                f"stderr={preflight.stderr.strip()!r}"
            )

        process = subprocess.Popen(  # noqa: S603
            [
                str(python_path),
                str(worker_script),
                "--config",
                str(self.config.unimernet_config_path),
                "--device",
                self.config.device,
                "--dark-formula-threshold",
                str(self.config.dark_formula_threshold),
                *(
                    ["--normalize-dark-formula"]
                    if self.config.normalize_dark_formula
                    else []
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if not ready_line:
            return_code = process.poll()
            process.terminate()
            raise RuntimeError(
                "UniMERNet worker exited before reporting readiness"
                + (f" (return code {return_code})" if return_code is not None else "")
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            process.terminate()
            raise RuntimeError(
                f"Invalid UniMERNet worker startup response: {ready_line!r}"
            ) from exc
        if not ready.get("ready"):
            process.terminate()
            raise RuntimeError(f"UniMERNet worker failed to initialize: {ready!r}")
        self._worker = process

    def _init_in_process(self) -> None:
        try:
            import torch
            import unimernet.tasks as tasks
            from PIL import Image
            from unimernet.common.config import Config
            from unimernet.processors import load_processor
        except ImportError as exc:
            raise RuntimeError(
                "In-process UniMERNet OCR requires the unimernet package. Prefer the isolated "
                "worker via vision.math_ocr.unimernet_python_path."
            ) from exc

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "UniMERNet OCR is configured for CUDA, but torch.cuda.is_available() is False."
            )

        self.torch = torch
        self.Image = Image
        args = argparse.Namespace(
            cfg_path=str(self.config.unimernet_config_path),
            options=None,
        )
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        self.device = torch.device(self.config.device)
        self.model = task.build_model(cfg).to(self.device)
        self.model.eval()
        self.processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )

    def _recognize_worker(self, image_path: Path) -> MathOCRCandidate | None:
        process = self._worker
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("UniMERNet worker is not available")
        if process.poll() is not None:
            raise RuntimeError(
                f"UniMERNet worker exited unexpectedly with return code {process.returncode}"
            )

        process.stdin.write(
            json.dumps({"image": str(image_path)}, ensure_ascii=False) + "\n"
        )
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError("UniMERNet worker closed stdout during inference")
        response = json.loads(response_line)
        if response.get("error"):
            raise RuntimeError(f"UniMERNet worker error: {response['error']}")
        text = str(response.get("text") or "").strip()[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        return MathOCRCandidate(backend="unimernet", text=text)

    def recognize(self, image_path: Path) -> MathOCRCandidate | None:
        if self._worker is not None:
            return self._recognize_worker(image_path)

        assert self.Image is not None
        assert self.processor is not None
        assert self.device is not None
        assert self.torch is not None
        assert self.model is not None

        with self.Image.open(image_path) as raw:
            raw_image = _normalize_formula_image(raw, self.config)
        image = self.processor(raw_image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate({"image": image})
        text = str(output["pred_str"][0]).strip()[:_MAX_OCR_TEXT_CHARS]
        if not text:
            return None
        return MathOCRCandidate(backend="unimernet", text=text)

    def close(self) -> None:
        process = self._worker
        self._worker = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=5)
        except Exception:
            process.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_math_ocr_backend(
    config: MathOCRConfig,
    llm_config=None,
) -> MathOCRBackend | None:
    if config.backend == "none":
        return None
    if config.backend == "mathpix":
        return MathpixBackend(config)
    if config.backend == "unimernet":
        return UniMERNetBackend(config)
    if config.backend == "unimumer":
        return UniMuMERBackend(config)
    if config.backend == "qwen_vlm":
        return QwenVLMOCRBackend(config, llm_config)
    if config.backend == "latexocr":
        return LatexOCRBackend(config)
    raise ValueError(f"unsupported math OCR backend: {config.backend}")
