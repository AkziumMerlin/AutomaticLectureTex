from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_checked(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        command = " ".join(args)
        raise RuntimeError(f"command failed ({proc.returncode}): {command}\n{proc.stderr.strip()}")
    return proc


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required executable not found in PATH: {name}")
    return resolved


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


_JSON_HEX = frozenset("0123456789abcdefABCDEF")
_JSON_CONTROL_ESCAPES = {
    "\b": r"\b",
    "\f": r"\f",
    "\n": r"\n",
    "\r": r"\r",
    "\t": r"\t",
}


def repair_json_string_escapes(text: str) -> str:
    """Repair narrow JSON-string violations commonly produced around LaTeX.

    Some OpenAI-compatible guided-decoding backends still emit raw control characters for LaTeX
    commands such as ``\\theta``/``\\beta`` or a single unescaped backslash for commands such as
    ``\\alpha``. Repair only characters *inside JSON strings*. Structural JSON remains untouched,
    so truncation, missing delimiters and other malformed responses still fail normally and enter
    the caller's retry path.
    """

    out: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            out.append(char)
            in_string = False
            index += 1
            continue

        if ord(char) < 0x20:
            out.append(_JSON_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
            index += 1
            continue

        if char != "\\":
            out.append(char)
            index += 1
            continue

        if index + 1 >= len(text):
            out.append("\\\\")
            index += 1
            continue

        next_char = text[index + 1]
        if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            out.append(text[index : index + 2])
            index += 2
            continue

        if next_char == "u":
            codepoint = text[index + 2 : index + 6]
            if len(codepoint) == 4 and all(char in _JSON_HEX for char in codepoint):
                out.append(text[index : index + 6])
                index += 6
                continue

        # Invalid JSON escape: treat the slash as a literal LaTeX backslash.
        out.append("\\\\")
        index += 1

    return "".join(out)


def strip_thinking_and_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return repair_json_string_escapes(text.strip())
