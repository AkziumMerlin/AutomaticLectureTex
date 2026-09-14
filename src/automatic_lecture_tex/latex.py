from __future__ import annotations

import re
from pathlib import Path

from .schemas import BlockType, LectureIR, NoteBlock
from .tex_safety import (
    looks_like_math_fragment,
    normalize_math_spans,
    normalize_math_unicode,
    strip_control_chars,
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


def escape_tex(text: str) -> str:
    clean = strip_control_chars(text)
    return "".join(_TEX_ESCAPES.get(char, char) for char in clean)


def escape_tex_mixed(text: str) -> str:
    """Escape prose while preserving balanced `$...$` math in headings/captions."""

    clean = strip_control_chars(text)
    if clean.count("$") % 2:
        return escape_tex(clean)
    parts = _INLINE_DOLLAR_MATH.split(clean)
    return "".join(
        normalize_math_unicode(part) if _INLINE_DOLLAR_MATH.fullmatch(part or "") else escape_tex(part)
        for part in parts
    )


def _body(block: NoteBlock) -> str:
    return normalize_math_spans(strip_control_chars(block.latex)).strip()


def _environment(block: NoteBlock, environment: str) -> str:
    title = f"[{escape_tex_mixed(block.title)}]" if block.title else ""
    return f"\\begin{{{environment}}}{title}\n{_body(block)}\n\\end{{{environment}}}\n"


def render_block(block: NoteBlock) -> str:
    body = _body(block)
    if block.type == BlockType.PARAGRAPH:
        return body + "\n"
    if block.type == BlockType.EQUATION:
        math = normalize_math_unicode(strip_control_chars(block.latex)).strip()
        if looks_like_math_fragment(math):
            return "\\[\n" + math + "\n\\]\n"
        # Defensive compatibility path for stale/bad IR. Never wrap prose or already-delimited math
        # in another display environment; render it as ordinary TeX instead.
        return normalize_math_spans(strip_control_chars(block.latex)).strip() + "\n"
    if block.type == BlockType.FIGURE:
        if not block.asset_path:
            return body + "\n"
        caption = escape_tex_mixed(block.caption or block.title or "")
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


def _comment_text(value: str) -> str:
    return " ".join(strip_control_chars(value).replace("\r", " ").replace("\n", " ").split())


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
        if chunk.corrections:
            lines.append("% Reconstruction corrections:")
            lines.extend(
                "% - "
                + _comment_text(
                    f"[{item.basis}, confidence={item.confidence:.2f}] "
                    f"{item.original!r} -> {item.corrected!r}: {item.reason}"
                )
                for item in chunk.corrections
            )
            lines.append("")
        if chunk.unresolved:
            lines.append("% Unresolved reconstruction issues:")
            lines.extend(f"% - {_comment_text(item)}" for item in chunk.unresolved)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


PREAMBLE = r"""\documentclass[12pt,a4paper]{book}
\usepackage{fontspec}
\usepackage{polyglossia}
\setdefaultlanguage{russian}
\setotherlanguage{english}
\setmainfont{CMU Serif}
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
    lectures_dir.mkdir(parents=True, exist_ok=True)
    includes: list[str] = []
    for ir in lectures:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ir.lecture_id)
        path = lectures_dir / f"{safe}.tex"
        path.write_text(render_lecture(ir), encoding="utf-8")
        includes.append(f"\\input{{lectures/{safe}.tex}}")

    main = output_dir / "main.tex"
    main.write_text(
        PREAMBLE
        + "\n\\begin{document}\n"
        + f"\\title{{{escape_tex_mixed(course_title)}}}\n\\maketitle\n\\tableofcontents\n"
        + "\n".join(includes)
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
    return main


def compile_tex(main_tex: Path, compiler: str) -> None:
    if compiler == "latexmk":
        run_checked(
            [compiler, "-xelatex", "-interaction=nonstopmode", main_tex.name], cwd=main_tex.parent
        )
    else:
        run_checked([compiler, "-interaction=nonstopmode", main_tex.name], cwd=main_tex.parent)
