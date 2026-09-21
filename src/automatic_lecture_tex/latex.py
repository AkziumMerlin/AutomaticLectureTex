from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import BlockType, LectureIR, NoteBlock
from .tex_safety import (
    canonicalize_math_fragment,
    looks_like_math_fragment,
    normalize_heading_math,
    normalize_math_spans,
    strip_control_chars,
    validate_tex_source,
)
from .util import run_checked

_TEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_INLINE_DOLLAR_MATH = re.compile(r"(\$[^$\n]*\$)")
_BOARD_SNAPSHOT_PREFIX = "board-snapshot:"


def escape_tex(text: str) -> str:
    clean = strip_control_chars(text)
    return "".join(_TEX_ESCAPES.get(char, char) for char in clean)


def escape_tex_mixed(text: str) -> str:
    """Escape prose while preserving or inserting safe `$...$` math in headings/captions."""

    clean = normalize_heading_math(strip_control_chars(text))
    if clean.count("$") % 2:
        return escape_tex(clean)
    parts = _INLINE_DOLLAR_MATH.split(clean)
    return "".join(
        canonicalize_math_fragment(part)
        if _INLINE_DOLLAR_MATH.fullmatch(part or "")
        else escape_tex(part)
        for part in parts
    )


def _body(block: NoteBlock) -> str:
    return normalize_math_spans(strip_control_chars(block.latex)).strip()


def _environment(block: NoteBlock, environment: str) -> str:
    title = f"[{escape_tex_mixed(block.title)}]" if block.title else ""
    return f"\\begin{{{environment}}}{title}\n{_body(block)}\n\\end{{{environment}}}\n"


def _is_board_snapshot(block: NoteBlock) -> bool:
    return any(item.startswith(_BOARD_SNAPSHOT_PREFIX) for item in block.source_evidence_ids)


def render_block(block: NoteBlock) -> str:
    body = _body(block)
    if block.type == BlockType.PARAGRAPH:
        return body + "\n"
    if block.type == BlockType.EQUATION:
        math = canonicalize_math_fragment(strip_control_chars(block.latex)).strip()
        if looks_like_math_fragment(math):
            return "\\[\n" + math + "\n\\]\n"
        # Defensive compatibility path for stale/bad IR. Never wrap prose or already-delimited math
        # in another display environment; normalize delimiters and render it as ordinary TeX.
        return normalize_math_spans(strip_control_chars(block.latex)).strip() + "\n"
    if block.type == BlockType.FIGURE:
        if not block.asset_path:
            return body + "\n"
        caption = escape_tex_mixed(block.caption or block.title or "")
        if _is_board_snapshot(block):
            return (
                "\\par\\medskip\n"
                "\\begin{center}\n"
                f"\\includegraphics[width=0.88\\textwidth]{{{block.asset_path}}}\n"
                "\\end{center}\n"
                + (f"\\noindent\\textit{{{caption}}}\\par\n" if caption else "")
                + "\\medskip\n"
            )
        return (
            "\\begin{figure}[ht]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.9\\textwidth]{{{block.asset_path}}}\n"
            + (f"\\caption{{{caption}}}\n" if caption else "")
            + "\\end{figure}\n"
        )
    environment_map = {
        BlockType.DEFINITION: "definition",
        BlockType.THEOREM: "theorem",
        BlockType.LEMMA: "lemma",
        BlockType.PROPOSITION: "proposition",
        BlockType.COROLLARY: "corollary",
        BlockType.PROOF: "proof",
        BlockType.EXAMPLE: "example",
        BlockType.REMARK: "remark",
        BlockType.EXERCISE: "exercise",
    }
    return _environment(block, environment_map[block.type])


def render_lecture(ir: LectureIR) -> str:
    lines = [f"\\chapter{{{escape_tex_mixed(ir.title)}}}", ""]
    previous_section = None
    for chunk in ir.chunks:
        section = chunk.section_title.strip() or "Без названия"
        if section != previous_section:
            lines.extend([f"\\section{{{escape_tex_mixed(section)}}}", ""])
            previous_section = section
        for block in chunk.blocks:
            lines.append(render_block(block))
    rendered = "\n".join(lines).rstrip() + "\n"
    validate_tex_source(rendered)
    return rendered


def _prune_unreferenced_figure_assets(ir: LectureIR, output_dir: Path) -> None:
    """Keep only generated figure assets referenced by the current lecture IR."""

    figures_root = output_dir / "figures" / ir.lecture_id
    if not figures_root.is_dir():
        return
    referenced = {
        (output_dir / block.asset_path).resolve()
        for chunk in ir.chunks
        for block in chunk.blocks
        if block.asset_path
    }
    for path in figures_root.rglob("*"):
        if path.is_file() and path.resolve() not in referenced:
            path.unlink()
    directories = [path for path in figures_root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _audit_payload(ir: LectureIR) -> dict:
    return {
        "lecture_id": ir.lecture_id,
        "title": ir.title,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "start": chunk.start,
                "end": chunk.end,
                "section_title": chunk.section_title,
                "corrections": [item.model_dump(mode="json") for item in chunk.corrections],
                "unresolved": list(chunk.unresolved),
            }
            for chunk in ir.chunks
            if chunk.corrections or chunk.unresolved
        ],
    }


PREAMBLE = r"""\documentclass[12pt,a4paper]{book}
\usepackage{fontspec}
\usepackage{polyglossia}
\setdefaultlanguage{russian}
\setotherlanguage{english}
\IfFontExistsTF{CMU Serif}{\setmainfont{CMU Serif}}{\setmainfont{Latin Modern Roman}}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{microtype}
\newtheorem{theorem}{Теорема}[chapter]
\newtheorem{lemma}[theorem]{Лемма}
\newtheorem{proposition}[theorem]{Предложение}
\newtheorem{corollary}[theorem]{Следствие}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Определение}
\newtheorem{example}[theorem]{Пример}
\newtheorem{exercise}[theorem]{Упражнение}
\theoremstyle{remark}
\newtheorem*{remark}{Замечание}
"""


def write_course_tex(course_title: str, lectures: list[LectureIR], output_dir: Path) -> Path:
    lectures_dir = output_dir / "lectures"
    audit_dir = output_dir / "audit"
    lectures_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    includes: list[str] = []
    for ir in lectures:
        _prune_unreferenced_figure_assets(ir, output_dir)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ir.lecture_id)
        path = lectures_dir / f"{safe}.tex"
        path.write_text(render_lecture(ir), encoding="utf-8")
        audit_path = audit_dir / f"{safe}.json"
        audit_path.write_text(
            json.dumps(_audit_payload(ir), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        includes.append(f"\\input{{lectures/{safe}.tex}}")

    main = output_dir / "main.tex"
    main_text = (
        PREAMBLE
        + "\n\\begin{document}\n"
        + f"\\title{{{escape_tex_mixed(course_title)}}}\n\\maketitle\n\\tableofcontents\n"
        + "\n".join(includes)
        + "\n\\end{document}\n"
    )
    validate_tex_source(main_text)
    main.write_text(main_text, encoding="utf-8")
    return main


def compile_tex(main_tex: Path, compiler: str) -> None:
    if compiler == "latexmk":
        run_checked(
            [compiler, "-xelatex", "-interaction=nonstopmode", main_tex.name], cwd=main_tex.parent
        )
    else:
        run_checked([compiler, "-interaction=nonstopmode", main_tex.name], cwd=main_tex.parent)
