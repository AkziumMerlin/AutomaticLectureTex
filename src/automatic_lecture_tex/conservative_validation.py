from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from .schemas import MathAudit, MathAuditCorrection


class ConservativeCorrection(BaseModel):
    block_index: int = Field(ge=0)
    corrected_latex: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: Literal[
        "mathematical_contradiction",
        "latex_damage",
        "lecturer_correction",
    ]


class ConservativeAudit(BaseModel):
    corrections: list[ConservativeCorrection] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


def validate_episode_batch_conservative(orchestrator, evidence, notes) -> MathAudit:
    draft = [
        {"block_index": index, **block.model_dump(mode="json")}
        for index, block in enumerate(notes.blocks)
    ]
    prompt = f"""Audit ONE bounded generated lecture-note batch against exactly its source
evidence. You are a conservative mathematical guardrail, NOT a second writer and NOT an ASR editor.

Evidence:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}

Generated blocks:
{json.dumps(draft, ensure_ascii=False, separators=(",", ":"))}

A correction is allowed ONLY for one of these categories:
1. mathematical_contradiction: the generated block changes a mathematical assertion supported by
   the evidence (wrong sign, variable, quantifier, domain/codomain, implication, formula, or proof
   dependency);
2. latex_damage: the intended mathematical content is preserved in evidence but the generated LaTeX
   is syntactically or semantically damaged;
3. lecturer_correction: the generated block kept content that the lecturer explicitly corrected or
   retracted in this same bounded evidence.

Do NOT issue a correction merely because generated prose is not verbatim transcript text. In
particular, NEVER revert grammatical cleanup, punctuation repair, removal of filler words, expansion
of a locally reconstructed technical term, or an equivalent mathematical paraphrase back toward
noisy ASR wording. Do not prefer a low-confidence/garbled transcript phrase over coherent generated
prose unless the garbled phrase still supplies unambiguous mathematical information that the draft
actually contradicts. Do not add textbook facts or missing proof steps.

If the evidence is ambiguous, report `unresolved` instead of rewriting. `corrected_latex` must be the
complete corrected body of the affected block and must preserve all content not involved in the
specific error. Write reasons in language code `{orchestrator.output_language}`.
"""
    result = orchestrator._structured(  # noqa: SLF001
        prompt,
        ConservativeAudit,
        operation="episode_validation",
        max_tokens=2048,
    )
    return MathAudit(
        corrections=[
            MathAuditCorrection(
                block_index=item.block_index,
                corrected_latex=item.corrected_latex,
                reason=f"[{item.category}] {item.reason}",
                confidence=item.confidence,
            )
            for item in result.corrections
        ],
        unresolved=list(result.unresolved),
    )
