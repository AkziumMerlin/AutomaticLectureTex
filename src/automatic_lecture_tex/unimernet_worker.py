from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def _load_processor(config_path: Path, device_name: str):
    # UniMERNet prints model-construction diagnostics to stdout. Keep stdout reserved for the JSONL
    # protocol so the parent process can parse replies reliably.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import unimernet.tasks as tasks
        from PIL import Image
        from unimernet.common.config import Config
        from unimernet.processors import load_processor

        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "UniMERNet worker requested CUDA, but torch.cuda.is_available() is False."
            )

        args = argparse.Namespace(cfg_path=str(config_path), options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        device = torch.device(device_name)
        model = task.build_model(cfg).to(device)
        model.eval()
        processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )

    return torch, Image, device, model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent UniMERNet JSONL inference worker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--normalize-dark-formula", action="store_true")
    parser.add_argument("--dark-formula-threshold", type=int, default=128)
    args = parser.parse_args()

    torch, Image, device, model, processor = _load_processor(args.config, args.device)
    print(json.dumps({"ready": True}), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                print(json.dumps({"ok": True}), flush=True)
                return

            image_path = Path(request["image"])
            with contextlib.redirect_stdout(sys.stderr):
                from PIL import ImageOps

                with Image.open(image_path) as raw:
                    gray = raw.convert("L")
                    if args.normalize_dark_formula:
                        histogram = gray.histogram()
                        midpoint = sum(histogram) / 2
                        running = 0
                        median = 255
                        for value, count in enumerate(histogram):
                            running += count
                            if running >= midpoint:
                                median = value
                                break
                        if median < args.dark_formula_threshold:
                            gray = ImageOps.invert(gray)
                        raw_image = ImageOps.autocontrast(gray).convert("RGB")
                    else:
                        raw_image = raw.convert("RGB")

                image = processor(raw_image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    output = model.generate({"image": image})
            text = str(output["pred_str"][0]).strip()
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
