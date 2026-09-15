from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import sys

from .config import load_config
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
            "pip install 'git+https://github.com/salute-developers/GigaAM.git'",
        ),
    }
    package = package_by_asr.get(cfg.asr.backend)
    if package is not None and importlib.util.find_spec(package[0]) is None:
        problems.append(f"Missing Python package {package[0]!r}; install with: {package[1]}")

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
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        return _doctor(cfg)

    pipeline = Pipeline(cfg)
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
