from __future__ import annotations

import argparse
import shutil
import sys

from .config import load_config
from .pipeline import Pipeline


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        required = [cfg.runtime.ffmpeg, cfg.runtime.ffprobe]
        if any(lecture.source.type == "youtube" for lecture in cfg.course.lectures):
            required.append(cfg.runtime.yt_dlp)
        missing = [binary for binary in required if shutil.which(binary) is None]
        if missing:
            print("Missing executables: " + ", ".join(missing), file=sys.stderr)
            return 1
        print("External executables: OK")
        return 0

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
