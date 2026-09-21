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
    "⊂": r"\subset ",
    "⊆": r"\subseteq ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "→": r"\to ",
    "↦": r"\mapsto ",
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
    "Γ": r"\Gamma ",
    "Δ": r"\Delta ",
    "Φ": r"\Phi ",
    "Ψ": r"\Psi ",
    "Ω": r"\Omega ",
    "ℂ": r"\mathbb{C}",
    "ℝ": r"\mathbb{R}",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
}
_MATH_ONLY_UNICODE = {"Ф": r"\Phi "}

_DOUBLE_DOLLAR_MATH = re.compile(r"\$\$([\s\S]*?)\$\$")
_INLINE_MATH = re.compile(r"(\$[^$\n]*\$|\\\([^\n]*?\\\)|\\\[[\s\S]*?\\\])")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_TEXT_COMMAND = re.compile(r"\\text\{[^{}]*\}")

_COMMAND_NAMES = (
    "alpha|beta|gamma|delta|epsilon|varepsilon|lambda|mu|nu|pi|phi|varphi|tau|"
    "Gamma|Delta|Phi|Psi|Omega|in|notin|neq|leq|geq|leqslant|geqslant|ell|"
    "subset|subseteq|cup|cap|bigcup|bigcap|to|mapsto|Rightarrow|Leftrightarrow|infty|sqrt"
)
_BAD_TAB_COMMAND = re.compile(rf"\\t({_COMMAND_NAMES})\b")
_BARE_COMMAND_WORD = re.compile(rf"(?<![\\A-Za-z])({_COMMAND_NAMES})\b")
_BARE_MATH_COMMAND = re.compile(
    rf"(\\(?:{_COMMAND_NAMES})(?:(?:\\?_\{{?[^\s,.;:)]+\}}?)|(?:\^\{{?[^\s,.;:)]+\}}?))*)"
)
_BARE_INDEXED_SET_OPERATOR = re.compile(r"(?<![\\A-Za-z])(?:(?:big)?(cap|cup))(?=_)")
_SIZE_COMMAND = re.compile(r"\\(?:big|Big|bigg|Bigg)[lr]?")
_VALID_SIZED_DELIMITER = re.compile(
    r"(?:[()\[\]|.]|\\[{}]|\\(?:langle|rangle|lvert|rvert|lVert|rVert|vert|Vert|"
    r"lfloor|rfloor|lceil|rceil)\b)"
)


def strip_control_chars(value: str) -> str:
    """Drop C0/DEL characters that cannot appear safely in a TeX source."""

    return value.translate(_CONTROL_TRANSLATION)


def normalize_math_unicode(value: str) -> str:
    result = strip_control_chars(value)
    for source, replacement in _MATH_UNICODE.items():
        result = result.replace(source, replacement)
    return result


def _drop_orphan_sizing_commands(value: str) -> str:
    """Drop \bigl/\bigr-style commands when no TeX delimiter follows them."""

    def replace(match: re.Match[str]) -> str:
        tail = match.string[match.end() :]
        return match.group(0) if _VALID_SIZED_DELIMITER.match(tail) else ""

    return _SIZE_COMMAND.sub(replace, value)


def canonicalize_math_fragment(value: str) -> str:
    """Repair deterministic serialization/typography damage inside mathematical material."""

    result = _drop_orphan_sizing_commands(normalize_math_unicode(value))
    # Nested math delimiters are a common model serialization error: once a fragment is already
    # known to be mathematical, inner \(...\)/\[...\] wrappers are invalid and redundant.
    result = result.replace(r"\(", "").replace(r"\)", "")
    result = result.replace(r"\[", "").replace(r"\]", "")
    result = _BARE_INDEXED_SET_OPERATOR.sub(
        lambda match: r"\big" + match.group(1),
        result,
    )
    for source, replacement in _MATH_ONLY_UNICODE.items():
        result = result.replace(source, replacement)
    result = _BAD_TAB_COMMAND.sub(lambda match: "\\" + match.group(1), result)
    result = result.replace(r"\_", "_")
    result = _BARE_COMMAND_WORD.sub(lambda match: "\\" + match.group(1), result)
    return result


def _normalize_delimited_math(value: str) -> str:
    if value.startswith("$") and value.endswith("$"):
        return "$" + canonicalize_math_fragment(value[1:-1]).strip() + "$"
    if value.startswith(r"\(") and value.endswith(r"\)"):
        return r"\(" + canonicalize_math_fragment(value[2:-2]).strip() + r"\)"
    if value.startswith(r"\[") and value.endswith(r"\]"):
        return r"\[" + canonicalize_math_fragment(value[2:-2]).strip() + r"\]"
    return canonicalize_math_fragment(value)


def _wrap_unicode_math_in_prose(value: str, *, dollars: bool) -> str:
    pieces: list[str] = []
    for char in value:
        replacement = _MATH_UNICODE.get(char)
        if replacement is None:
            pieces.append(char)
            continue
        math = replacement.strip()
        pieces.append(f"${math}$" if dollars else rf"\({math}\)")
    return "".join(pieces)


def _wrap_bare_commands(value: str, *, dollars: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        math = canonicalize_math_fragment(match.group(1)).strip()
        return f"${math}$" if dollars else rf"\({math}\)"

    return _BARE_MATH_COMMAND.sub(replace, value)


_SPLIT_SLANT_COMMAND = re.compile(r"\\\((\\(?:leq|geq))\\\)slant")


def _repair_unmatched_display_lines(value: str) -> str:
    """Repair multiple independent one-line \`$$formula\` serialization failures."""

    repaired: list[str] = []
    for line in value.splitlines():
        if line.count("$$") != 1:
            repaired.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("$$"):
            content = stripped[2:].strip()
        elif stripped.endswith("$$"):
            content = stripped[:-2].strip()
        else:
            repaired.append(line)
            continue
        if not content:
            repaired.append(line)
            continue
        repaired.append(r"\[" + canonicalize_math_fragment(content) + r"\]")
    return "\n".join(repaired)


def _repair_single_unmatched_display(value: str) -> str:
    """Repair the common model failure `$$formula` (or `formula$$`) without guessing prose spans."""

    if value.count("$$") != 1:
        return value
    marker = value.find("$$")
    before = value[:marker].strip()
    after = value[marker + 2 :].strip()
    if not before and after:
        return r"\[" + canonicalize_math_fragment(after).strip() + r"\]"
    if before and not after:
        return r"\[" + canonicalize_math_fragment(before).strip() + r"\]"
    return value


def normalize_math_spans(value: str) -> str:
    """Normalize math syntax in delimited spans and safely wrap raw math glyphs in prose."""

    clean = _repair_unmatched_display_lines(strip_control_chars(value))
    clean = _repair_single_unmatched_display(clean)
    clean = _SPLIT_SLANT_COMMAND.sub(
        lambda match: r"\(" + match.group(1) + "slant" + r"\)",
        clean,
    )
    clean = _DOUBLE_DOLLAR_MATH.sub(
        lambda match: r"\[" + canonicalize_math_fragment(match.group(1)).strip() + r"\]",
        clean,
    )
    parts = _INLINE_MATH.split(clean)
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        if _INLINE_MATH.fullmatch(part):
            result.append(_normalize_delimited_math(part))
        else:
            prose = _wrap_unicode_math_in_prose(part, dollars=False)
            prose = _wrap_bare_commands(prose, dollars=False)
            result.append(prose)
    return "".join(result)


def normalize_heading_math(value: str) -> str:
    """Make math commands in theorem/section titles safe for text-mode rendering."""

    clean = strip_control_chars(value)
    clean = _DOUBLE_DOLLAR_MATH.sub(
        lambda match: "$" + canonicalize_math_fragment(match.group(1)).strip() + "$",
        clean,
    )
    clean = clean.replace(r"\[", "$").replace(r"\]", "$")
    clean = clean.replace(r"\(", "$").replace(r"\)", "$")
    parts = _INLINE_MATH.split(clean)
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        if _INLINE_MATH.fullmatch(part):
            result.append(_normalize_delimited_math(part))
        else:
            prose = _wrap_unicode_math_in_prose(part, dollars=True)
            prose = _wrap_bare_commands(prose, dollars=True)
            result.append(prose)
    return "".join(result)


def looks_like_math_fragment(value: str) -> bool:
    """Conservative check for a renderer-owned display-math fragment."""

    clean = strip_control_chars(value).strip()
    if not clean:
        return False
    if "$" in clean or r"\[" in clean or r"\]" in clean:
        return False
    without_text = _TEXT_COMMAND.sub("", clean)
    return _CYRILLIC.search(without_text) is None


_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def assert_balanced_math_delimiters(value: str) -> None:
    """Reject residual malformed math delimiters after deterministic normalization."""

    clean = strip_control_chars(value)
    if "$$" in clean:
        raise ValueError("raw $$ display delimiter survived TeX normalization")
    if len(_UNESCAPED_DOLLAR.findall(clean)) % 2:
        raise ValueError("unbalanced $ math delimiter")
    if clean.count(r"\(") != clean.count(r"\)"):
        raise ValueError("unbalanced \\( ... \\) math delimiter")
    if clean.count(r"\[") != clean.count(r"\]"):
        raise ValueError("unbalanced \\[ ... \\] math delimiter")
