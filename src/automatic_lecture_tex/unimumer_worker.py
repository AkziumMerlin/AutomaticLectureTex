from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path


# This module is itself a dedicated long-lived subprocess. Running vLLM's V1 EngineCore in yet
# another process adds a ZMQ startup handshake that is unnecessary here and can hang when vLLM is
# embedded as a library. Keep the engine core in this worker process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


HMER_PROMPT = (
    "I have an image of a handwritten mathematical expression. "
    "Please write out the expression of the formula in the image using LaTeX format."
)


def _normalize_image(
    image_path: Path,
    *,
    normalize_dark_formula: bool,
    dark_formula_threshold: int,
) -> tuple[Path, Path | None]:
    from PIL import Image, ImageOps

    if not normalize_dark_formula:
        return image_path, None

    with Image.open(image_path) as raw:
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
        if median < dark_formula_threshold:
            gray = ImageOps.invert(gray)
        prepared = ImageOps.autocontrast(gray).convert("RGB")

    handle = tempfile.NamedTemporaryFile(
        prefix="automatic-lecture-tex-unimumer-",
        suffix=".png",
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    prepared.save(temp_path)
    return temp_path, temp_path


def _load_engine(args):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Uni-MuMER worker requested CUDA, but torch.cuda.is_available() is False."
            )

        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max(4096, args.max_tokens + 1024),
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
        )
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
        )
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=50,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
    return llm, processor, sampling


def _infer(llm, processor, sampling, image_path: Path) -> str:
    from qwen_vl_utils import process_vision_info

    image_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": HMER_PROMPT},
            ],
        },
    ]
    final_prompt = processor.apply_chat_template(
        image_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _, _ = process_vision_info(
        image_messages,
        return_video_kwargs=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    with contextlib.redirect_stdout(sys.stderr):
        outputs = llm.generate(
            [{"prompt": final_prompt, "multi_modal_data": mm_data}],
            sampling,
            use_tqdm=False,
        )
    return str(outputs[0].outputs[0].text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent Uni-MuMER JSONL inference worker.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--normalize-dark-formula", action="store_true")
    parser.add_argument("--dark-formula-threshold", type=int, default=128)
    args = parser.parse_args()

    # vLLM respects CUDA_VISIBLE_DEVICES inherited from the parent lecture process.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    llm, processor, sampling = _load_engine(args)
    print(json.dumps({"ready": True, "model": args.model}), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        temp_path: Path | None = None
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                return

            source_path = Path(request["image"])
            prepared_path, temp_path = _normalize_image(
                source_path,
                normalize_dark_formula=args.normalize_dark_formula,
                dark_formula_threshold=args.dark_formula_threshold,
            )
            text = _infer(llm, processor, sampling, prepared_path)
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
