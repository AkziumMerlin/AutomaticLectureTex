from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path


# Keep transient OCR allocations from fragmenting the few GiB left beside the resident lecture LLM.
# This must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def _load_model(args):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Uni-MuMER worker requested CUDA, but torch.cuda.is_available() is False."
                )
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            requested_bytes = int(args.max_gpu_memory_gib * 1024**3)
            fraction = min(1.0, requested_bytes / total_bytes)
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)

            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            if free_bytes < 2 * 1024**3:
                raise RuntimeError(
                    "Uni-MuMER worker has less than 2 GiB free GPU memory before model load: "
                    f"{free_bytes / 1024**3:.2f} GiB"
                )

        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if args.device == "cuda":
            model_kwargs["device_map"] = {"": 0}
            model_kwargs["dtype"] = torch.bfloat16
            if args.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        else:
            model_kwargs["device_map"] = {"": "cpu"}
            model_kwargs["dtype"] = torch.float32

        model = AutoModelForMultimodalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )
        model.eval()

    return torch, model, processor


def _infer(torch, model, processor, args, image_path: Path) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path.resolve())},
                {"type": "text", "text": HMER_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    try:
        model_device = next(model.parameters()).device
        inputs = inputs.to(model_device)
        generation_kwargs = {
            "max_new_tokens": args.max_tokens,
            "do_sample": args.temperature > 0,
            "use_cache": True,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated = outputs[0, prompt_tokens:]
        return processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
    finally:
        del inputs
        if args.device == "cuda":
            torch.cuda.empty_cache()


def _gpu_snapshot(torch) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
        "free_gib": round(free_bytes / 1024**3, 3),
        "total_gib": round(total_bytes / 1024**3, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent low-memory Uni-MuMER Transformers JSONL inference worker."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=5.5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--normalize-dark-formula", action="store_true")
    parser.add_argument("--dark-formula-threshold", type=int, default=128)
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    torch, model, processor = _load_model(args)
    print(
        json.dumps(
            {
                "ready": True,
                "model": args.model,
                "load_in_4bit": args.load_in_4bit,
                "max_gpu_memory_gib": args.max_gpu_memory_gib,
                "gpu": _gpu_snapshot(torch),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

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
            text = _infer(torch, model, processor, args, prepared_path)
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
