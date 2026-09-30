from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import shutil
import sys

from .config import load_config
from .graph_reconstruction import run_graph_reconstruction
from .pipeline_robust import Pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automatic-lecture-tex")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process one lecture or the whole configured course")
    run.add_argument("--config", required=True)
    run.add_argument("--lecture")
    run.add_argument("--force", action="store_true")

    build = sub.add_parser("build", help="render existing LectureIR artifacts to TeX")
    build.add_argument("--config", required=True)

    review = sub.add_parser("review", help="review reconstructed notes against local literature")
    review.add_argument("--config", required=True)
    review.add_argument("--lecture")

    doctor = sub.add_parser("doctor", help="check external executables")
    doctor.add_argument("--config", required=True)

    reconstruct = sub.add_parser(
        "reconstruct-graph",
        help="experimental global reconstruction from noisy lecture observations",
    )
    reconstruct.add_argument("--config", required=True)
    reconstruct.add_argument("--lecture", required=True)
    reconstruct.add_argument("--start-seconds", type=float)
    reconstruct.add_argument("--end-seconds", type=float)
    reconstruct.add_argument("--max-observations", type=int)
    reconstruct.add_argument("--candidate-count", type=int, default=4)
    reconstruct.add_argument("--candidate-batch-size", type=int, default=6)
    reconstruct.add_argument("--edge-batch-size", type=int, default=5)
    reconstruct.add_argument("--neighbor-span", type=int, default=2)
    reconstruct.add_argument("--max-gap-seconds", type=float, default=90.0)
    reconstruct.add_argument("--symbol-gap-seconds", type=float, default=240.0)
    reconstruct.add_argument("--beam-width", type=int, default=256)
    reconstruct.add_argument("--top-k", type=int, default=8)
    reconstruct.add_argument("--pairwise-weight", type=float, default=1.0)
    reconstruct.add_argument("--force", action="store_true")

    return parser


def _doctor(cfg) -> int:
    required = [cfg.runtime.ffmpeg, cfg.runtime.ffprobe]
    if any(lecture.source.type == "youtube" for lecture in cfg.course.lectures):
        required.append(cfg.runtime.yt_dlp)
    missing = [binary for binary in required if shutil.which(binary) is None]
    problems = []
    if missing:
        problems.append("Missing executables: " + ", ".join(missing))

    package_by_asr = {
        "faster_whisper": ("faster_whisper", "pip install -e '.[whisper]'"),
        "qwen3": ("qwen_asr", "pip install -e '.[qwen-asr]'"),
        "gigaam": (
            "gigaam",
            "pip install -e '.[gigaam]'",
        ),
    }
    package = package_by_asr.get(cfg.asr.backend)
    if package is not None and importlib.util.find_spec(package[0]) is None:
        problems.append(f"Missing Python package {package[0]!r}; install with: {package[1]}")

    if cfg.asr.backend == "gigaam":
        if importlib.util.find_spec("torchaudio") is None:
            problems.append(
                "GigaAM requires torchaudio matching the installed torch/CUDA build. "
                "Install torchaudio from the same PyTorch CUDA index as torch."
            )
        elif importlib.util.find_spec("gigaam") is not None:
            try:
                importlib.import_module("gigaam")
            except (ImportError, OSError) as exc:
                problems.append(f"GigaAM is installed but cannot be imported: {exc}")
        if cfg.asr.gigaam_vad_enabled and importlib.util.find_spec("faster_whisper") is None:
            problems.append(
                "VAD-aware GigaAM uses faster-whisper only for standalone Silero VAD. "
                "Install with: pip install -e '.[whisper]'"
            )

    math_ocr = cfg.vision.math_ocr
    if math_ocr.backend == "mathpix":
        if not os.getenv(math_ocr.mathpix_app_id_env) or not os.getenv(math_ocr.mathpix_app_key_env):
            problems.append(
                "Mathpix OCR requires environment variables "
                f"{math_ocr.mathpix_app_id_env} and {math_ocr.mathpix_app_key_env}"
            )
    elif math_ocr.backend == "unimernet":
        if importlib.util.find_spec("unimernet") is None:
            problems.append("UniMERNet OCR requires: pip install 'unimernet[full]'")
        if math_ocr.unimernet_config_path is None or not math_ocr.unimernet_config_path.is_file():
            problems.append("UniMERNet OCR requires an existing vision.math_ocr.unimernet_config_path")

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print("Environment: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        return _doctor(cfg)

    pipeline = Pipeline(cfg)
    if args.command == "reconstruct-graph":
        lecture = next(
            (item for item in cfg.course.lectures if item.id == args.lecture),
            None,
        )
        if lecture is None:
            available = ", ".join(item.id for item in cfg.course.lectures)
            parser.error(f"unknown lecture {args.lecture!r}; available: {available}")
        artifact = run_graph_reconstruction(
            config=cfg,
            lecture=lecture,
            llm=pipeline.llm,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            max_observations=args.max_observations,
            candidate_count=args.candidate_count,
            candidate_batch_size=args.candidate_batch_size,
            edge_batch_size=args.edge_batch_size,
            neighbor_span=args.neighbor_span,
            max_gap_seconds=args.max_gap_seconds,
            symbol_gap_seconds=args.symbol_gap_seconds,
            beam_width=args.beam_width,
            top_k=args.top_k,
            pairwise_weight=args.pairwise_weight,
            force=args.force,
        )
        print(artifact)
        return 0

    if args.command == "review":
        reports = pipeline.review(args.lecture)
        for report in reports:
            print(cfg.runtime.work_dir / report.lecture_id / "review.json")
        return 0
    if args.command == "build":
        print(pipeline.build())
        return 0
    pipeline.run(args.lecture, force=args.force)
    print(cfg.latex.output_dir / "main.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
