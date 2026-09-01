from __future__ import annotations

import re
from pathlib import Path

from .schemas import BlockType, LectureIR, NoteBlock
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


def escape_tex(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(char, char) for char in text)


def _environment(block: NoteBlock, environment: str) -> str:
    title = f"[{escape_tex(block.title)}]" if block.title else ""
    return f"\\begin{{{environment}}}{title}\n{block.latex.strip()}\n\\end{{{environment}}}\n"


def render_block(block: NoteBlock) -> str:
    if block.type == BlockType.PARAGRAPH:
        return block.latex.strip() + "\n"
    if block.type == BlockType.EQUATION:
        return "\\[\n" + block.latex.strip() + "\n\\]\n"
    if block.type == BlockType.FIGURE:
        if not block.asset_path:
            return block.latex.strip() + "\n"
        caption = escape_tex(block.caption or block.title or "")
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
    lines = [f"\\chapter{{{escape_tex(ir.title)}}}", ""]
    previous_section = None
    for chunk in ir.chunks:
        section = chunk.section_title.strip() or "Без названия"
        if section != previous_section:
            lines.extend([f"\\section{{{escape_tex(section)}}}", ""])
            previous_section = section
        for block in chunk.blocks:
            lines.append(render_block(block))
        if chunk.unresolved:
            lines.append("% Unresolved reconstruction issues:")
            lines.extend(f"% - {item}" for item in chunk.unresolved)
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
        + f"\\title{{{escape_tex(course_title)}}}\n\\maketitle\n\\tableofcontents\n"
        + "\n".join(includes)
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
    return main


def compile_tex(main_tex: Path, compiler: str) -> None:
    if compiler == "latexmk":
        run_checked([compiler, "-xelatex", "-interaction=nonstopmode", main_tex.name], cwd=main_tex.parent)
    else:
        run_checked([compiler, "-interaction=nonstopmode", main_tex.name], cwd=main_tex.parent)
