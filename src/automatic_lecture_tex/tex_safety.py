from __future__ import annotations

import re

_CONTROL_TRANSLATION = {
    codepoint: None
    for codepoint in [*range(0x00, 0x09), *range(0x0B, 0x20), 0x7F]
}

_MATH_UNICODE = {
    "∈": r"\in ",
    "∉": r"\notin ",
    "≠": r"\neq ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "→": r"\to ",
    "⇒": r"\Rightarrow ",
    "⇔": r"\Leftrightarrow ",
    "∞": r"\infty ",
    "∑": r"\sum ",
    "∏": r"\prod ",
    "∫": r"\int ",
    "∥": r"\|",
    "·": r"\cdot ",
    "×": r"\times ",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\varepsilon ",
    "ϵ": r"\epsilon ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "ν": r"\nu ",
    "π": r"\pi ",
    "φ": r"\varphi ",
    "ϕ": r"\phi ",
    "τ": r"\tau ",
    "ℂ": r"\mathbb{C}",
    "ℝ": r"\mathbb{R}",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
}

_INLINE_MATH = re.compile(r"(\$[^$\n]*\$|\\\([^\n]*?\\\)|\\\[[\s\S]*?\\\])")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_TEXT_COMMAND = re.compile(r"\\text\{[^{}]*\}")


def strip_control_chars(value: str) -> str:
    """Drop C0/DEL characters that cannot appear safely in a TeX source."""

    return value.translate(_CONTROL_TRANSLATION)


def normalize_math_unicode(value: str) -> str:
    result = strip_control_chars(value)
    for source, replacement in _MATH_UNICODE.items():
        result = result.replace(source, replacement)
    return result


def normalize_math_spans(value: str) -> str:
    """Normalize Unicode math only inside explicit inline/display math spans."""

    clean = strip_control_chars(value)
    parts = _INLINE_MATH.split(clean)
    return "".join(
        normalize_math_unicode(part) if _INLINE_MATH.fullmatch(part or "") else part
        for part in parts
    )


def looks_like_math_fragment(value: str) -> bool:
    """Conservative check for a renderer-owned display-math fragment."""

    clean = strip_control_chars(value).strip()
    if not clean:
        return False
    if "$" in clean or r"\[" in clean or r"\]" in clean:
        return False
    without_text = _TEXT_COMMAND.sub("", clean)
    return _CYRILLIC.search(without_text) is None
