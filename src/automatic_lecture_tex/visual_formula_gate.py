from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .schemas import VisualEvidence

_TOKEN = re.compile(
    r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|:=|<=|>=|!=|[=+\-*/^_<>|(),{}\[\]]"
)
_MATH_SPAN = re.compile(r"\$([^$\n]+)\$|\\\((.*?)\\\)|\\\[([\s\S]*?)\\\]")
_FORMATTING_COMMANDS = {
    r"\left",
    r"\right",
    r"\big",
    r"\Big",
    r"\bigl",
    r"\bigr",
    r"\Bigl",
    r"\Bigr",
}
_UNPROTECTED = {"(", ")", "[", "]", "{", "}", ",", "|"}


@dataclass(frozen=True)
class FormulaGateViolation:
    observation_index: int
    evidence_id: str
    visual_formula: str
    generated_formula: str
    reason: str


def _tokens(value: str) -> list[str]:
    value = value.replace(r"\,", " ").replace(r"\!", " ").replace(r"\;", " ")
    return [token for token in _TOKEN.findall(value) if token not in _FORMATTING_COMMANDS]


def _protected(token: str) -> bool:
    if token in _UNPROTECTED:
        return False
    if token.startswith("\\"):
        return token not in _FORMATTING_COMMANDS
    if token.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?", token):
        return True
    if token in {"=", "+", "-", "*", "/", "^", "_", "<", ">", ":="}:
        return True
    return token.isalpha() and len(token) <= 5


def _fragments(value: str) -> list[str]:
    clean = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", "", value)
    parts = re.split(r"(?:\\\\|\n|;)+", clean)
    return [part.strip() for part in parts if part.strip()]


def _candidate_formulas(latex: str | None, text: str) -> list[str]:
    result: list[str] = []
    if latex and latex.strip():
        result.extend(_fragments(latex))
    for match in _MATH_SPAN.finditer(text):
        value = next((group for group in match.groups() if group is not None), "")
        if value.strip():
            result.extend(_fragments(value))
    return list(dict.fromkeys(result))


def _visual_formulas(evidence: VisualEvidence, min_confidence: float) -> list[str]:
    if evidence.confidence < min_confidence:
        return []
    values = [evidence.latex, evidence.raw_latex]
    for candidate in evidence.math_ocr_candidates:
        # Uncalibrated local OCR is useful reconstruction evidence but must not become a hard
        # rejection gate. Only backends that report an actual confidence can veto generated math.
        if candidate.confidence is not None and candidate.confidence >= min_confidence:
            values.append(candidate.text)
    result: list[str] = []
    for value in values:
        if value and value.strip():
            result.extend(_fragments(value))
    return list(dict.fromkeys(result))


def _comparison(visual: str, generated: str) -> tuple[float, str | None]:
    source = _tokens(visual)
    target = _tokens(generated)
    if len(source) < 2 or len(target) < 2:
        return 0.0, None

    matcher = SequenceMatcher(a=source, b=target, autojunk=False)
    shared = sum(block.size for block in matcher.get_matching_blocks())
    coverage = shared / max(1, min(len(source), len(target)))
    if coverage < 0.55:
        return coverage, None

    changed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in {"replace", "delete"}:
            continue
        source_changed = [token for token in source[i1:i2] if _protected(token)]
        if not source_changed:
            continue
        target_changed = [token for token in target[j1:j2] if _protected(token)]
        if tag == "delete":
            changed.append(f"deleted {source_changed!r}")
        else:
            changed.append(f"replaced {source_changed!r} with {target_changed!r}")
    return coverage, "; ".join(changed) if changed else None


def find_formula_gate_violations(
    observations,
    evidence: list[VisualEvidence],
    *,
    min_visual_confidence: float = 0.80,
) -> list[FormulaGateViolation]:
    visual_by_id = {item.request_id: item for item in evidence if item.request_id}
    violations: list[FormulaGateViolation] = []

    for index, observation in enumerate(observations):
        generated = _candidate_formulas(observation.latex, observation.text)
        if not generated:
            continue
        for evidence_id in observation.visual_evidence_ids:
            visual = visual_by_id.get(evidence_id)
            if visual is None:
                continue
            formulas = _visual_formulas(visual, min_visual_confidence)
            best: tuple[float, str, str, str] | None = None
            for visual_formula in formulas:
                for generated_formula in generated:
                    coverage, reason = _comparison(visual_formula, generated_formula)
                    if reason is None:
                        continue
                    candidate = (coverage, visual_formula, generated_formula, reason)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None:
                coverage, visual_formula, generated_formula, reason = best
                violations.append(
                    FormulaGateViolation(
                        observation_index=index,
                        evidence_id=evidence_id,
                        visual_formula=visual_formula,
                        generated_formula=generated_formula,
                        reason=f"coverage={coverage:.2f}; {reason}",
                    )
                )
    return violations
