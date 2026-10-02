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
_SIZE_COMMAND = re.compile(r"\\(?:big|Big|bigg|Bigg)[lr]?(?![A-Za-z])")
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



_STRUCTURED_DISPLAY_ENVIRONMENTS = re.compile(
    r"\\begin\{(?:aligned|alignedat|gathered|multlined|cases|array|matrix|pmatrix|bmatrix|"
    r"Bmatrix|vmatrix|Vmatrix|split)\}"
)


def _split_tex_top_level(value: str, token: str) -> list[str]:
    """Split on a token only outside TeX brace groups and nested environments."""

    parts: list[str] = []
    start = 0
    brace_depth = 0
    env_depth = 0
    delimiter_depth = 0
    index = 0
    while index < len(value):
        if value.startswith(r"\begin{", index):
            env_depth += 1
        elif value.startswith(r"\end{", index):
            env_depth = max(0, env_depth - 1)
        elif value.startswith(r"\left", index):
            delimiter_depth += 1
        elif value.startswith(r"\right", index):
            delimiter_depth = max(0, delimiter_depth - 1)

        char = value[index]
        escaped = index > 0 and value[index - 1] == "\\"
        if char == "{" and not escaped:
            brace_depth += 1
        elif char == "}" and not escaped:
            brace_depth = max(0, brace_depth - 1)

        if (
            brace_depth == 0
            and env_depth == 0
            and delimiter_depth == 0
            and not escaped
            and value.startswith(token, index)
        ):
            parts.append(value[start:index])
            index += len(token)
            start = index
            continue
        index += 1

    parts.append(value[start:])
    return parts


def _aligned_lines(parts: list[str], *, delimiter: str = "") -> str:
    lines: list[str] = []
    last = len(parts) - 1
    for index, part in enumerate(parts):
        suffix = delimiter if delimiter and index < last else ""
        lines.append("&" + part.strip() + suffix)
    return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"


def layout_display_math(value: str, *, target_chars: int = 78) -> str:
    """Lay out long display math without shrinking it.

    Only top-level separators are used, so commands nested in braces/environments are preserved.
    Short or already-structured displays are left unchanged.
    """

    math = value.strip()
    if not math or _STRUCTURED_DISPLAY_ENVIRONMENTS.search(math):
        return math
    if len(math) <= target_chars:
        return math

    # Several mathematical clauses on one display are best rendered as left-aligned lines.
    semicolon_parts = [part.strip() for part in _split_tex_top_level(math, ";")]
    if len(semicolon_parts) >= 2 and all(semicolon_parts):
        return _aligned_lines(semicolon_parts, delimiter=";")

    for separator in (r"\qquad", r"\quad"):
        parts = [part.strip() for part in _split_tex_top_level(math, separator)]
        if len(parts) >= 2 and all(parts):
            return _aligned_lines(parts)

    # Equality chains should visually align on the equality sign. Do not mistake a
    # comma-separated system/list containing independent equalities for one chain.
    comma_parts = [part.strip() for part in _split_tex_top_level(math, ",")]
    equality_parts = [part.strip() for part in _split_tex_top_level(math, "=")]
    if len(equality_parts) >= 3 and all(equality_parts) and len(comma_parts) == 1:
        lines = [f"{equality_parts[0]} &={equality_parts[1]}"]
        lines.extend(f"&={part}" for part in equality_parts[2:])
        return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"

    for operator in (
        r"\Longleftrightarrow",
        r"\Leftrightarrow",
        r"\Longrightarrow",
        r"\Rightarrow",
    ):
        parts = [part.strip() for part in _split_tex_top_level(math, operator)]
        if len(parts) >= 2 and all(parts):
            lines = [parts[0]]
            lines.extend(operator + r"\ " + part for part in parts[1:])
            return _aligned_lines(lines)

    # A long comma-separated list has no natural alignment column; multlined gives it room to wrap
    # while preserving the mathematical order and punctuation.
    if len(comma_parts) >= 3 and all(comma_parts):
        lines: list[str] = []
        current = ""
        for index, part in enumerate(comma_parts):
            piece = part + ("," if index < len(comma_parts) - 1 else "")
            if current and len(current) + 1 + len(piece) > target_chars:
                lines.append(current)
                current = piece
            else:
                current = (current + " " + piece).strip()
        if current:
            lines.append(current)
        if len(lines) >= 2:
            return "\\begin{multlined}\n" + " \\\\\n".join(lines) + "\n\\end{multlined}"

    return math


def _normalize_delimited_math(value: str) -> str:
    if value.startswith("$") and value.endswith("$"):
        return "$" + canonicalize_math_fragment(value[1:-1]).strip() + "$"
    if value.startswith(r"\(") and value.endswith(r"\)"):
        return r"\(" + canonicalize_math_fragment(value[2:-2]).strip() + r"\)"
    if value.startswith(r"\[") and value.endswith(r"\]"):
        math = canonicalize_math_fragment(value[2:-2]).strip()
        return r"\[" + layout_display_math(math) + r"\]"
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



def _wrap_prose_math_atoms(value: str, *, dollars: bool) -> str:
    """Wrap raw math atoms without re-wrapping commands introduced by the same pass."""

    with_unicode = _wrap_unicode_math_in_prose(value, dollars=dollars)
    parts = _INLINE_MATH.split(with_unicode)
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        if _INLINE_MATH.fullmatch(part):
            result.append(_normalize_delimited_math(part))
        else:
            result.append(_wrap_bare_commands(part, dollars=dollars))
    return "".join(result)


_SPLIT_SLANT_COMMAND = re.compile(r"\\\((\\(?:leq|geq))\\\)slant")


def _repair_unmatched_display_lines(value: str) -> str:
    """Repair multiple independent one-line `$$formula` serialization failures."""

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
        lambda match: (
            r"\["
            + layout_display_math(
                canonicalize_math_fragment(match.group(1)).strip()
            )
            + r"\]"
        ),
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
            result.append(_wrap_prose_math_atoms(part, dollars=False))
    return "".join(result)


_HEADING_GREEK_CHARS = "αβγδεϵζηθικλμνξοπρστυφϕχψωΓΔΘΛΞΠΣΦΨΩ"
_HEADING_BARE_LATIN_ATOM = re.compile(
    r"(?<![A-Za-z0-9\\$_^])"
    r"([A-Za-z]"
    r"(?:_(?:\{[^{}\n]+\}|[A-Za-z0-9]+))?"
    r"(?:\^(?:\{[^{}\n]+\}|\*+|[A-Za-z0-9]+))?"
    r"\*{0,2}"
    r"(?:\([^()\n]*\))?"
    r")"
    r"(?![A-Za-z0-9])"
)
_HEADING_LATIN_SUB_GREEK = re.compile(
    rf"(?<![A-Za-z0-9\\])([A-Za-z][A-Za-z0-9]*)_([{_HEADING_GREEK_CHARS}])"
)
_HEADING_GREEK_SUB_LATIN = re.compile(
    rf"([{_HEADING_GREEK_CHARS}])_([A-Za-z][A-Za-z0-9]*)"
)


def _wrap_heading_bare_latin_atoms(value: str) -> str:
    """Wrap standalone Latin mathematical identifiers in heading prose.

    This is intentionally limited to one-letter identifiers (possibly decorated/indexed or used as
    a simple function call), so ordinary Latin words and theorem names remain text. A trailing
    asterisk is normalized as a dual-space superscript.
    """

    has_cyrillic = _CYRILLIC.search(value) is not None

    def replace(match: re.Match[str]) -> str:
        atom = match.group(1)
        # Avoid turning the English article/pronoun into mathematics in otherwise English prose.
        # In Russian mathematical headings the same letters are overwhelmingly identifiers.
        if not has_cyrillic and atom in {"A", "a", "I"}:
            return atom
        if atom.endswith("**"):
            atom = atom[:-2] + "^{**}"
        elif atom.endswith("*"):
            atom = atom[:-1] + "^*"
        return "$" + atom + "$"

    return _HEADING_BARE_LATIN_ATOM.sub(replace, value)


def _normalize_heading_compound_math(value: str) -> str:
    """Preserve simple subscripted symbols as one math atom before per-glyph wrapping."""

    def latin_sub_greek(match: re.Match[str]) -> str:
        greek = _MATH_UNICODE[match.group(2)].strip()
        return "$" + match.group(1) + "_{" + greek + "}$"

    def greek_sub_latin(match: re.Match[str]) -> str:
        greek = _MATH_UNICODE[match.group(1)].strip()
        return "$" + greek + "_{" + match.group(2) + "}$"

    value = _HEADING_LATIN_SUB_GREEK.sub(latin_sub_greek, value)
    return _HEADING_GREEK_SUB_LATIN.sub(greek_sub_latin, value)

def normalize_heading_math(value: str) -> str:
    """Make math commands in theorem/section titles safe for text-mode rendering."""

    clean = _normalize_heading_compound_math(strip_control_chars(value))
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
            result.append(
                _wrap_prose_math_atoms(
                    _wrap_heading_bare_latin_atoms(part),
                    dollars=True,
                )
            )
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
