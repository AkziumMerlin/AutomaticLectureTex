import json

from pydantic import BaseModel

from automatic_lecture_tex.config import LLMConfig
from automatic_lecture_tex.linear_llm_policy import (
    LINEAR_SOURCE_POLICY_VERSION,
    AuditEvidence,
    LectureModelClient,
    SourceGroundedMathAudit,
    _current_correction_transcript,
    audit_evidence_supported,
    has_explicit_correction_signal,
)
from automatic_lecture_tex.linear_notes import LinearCorrectionScan
from automatic_lecture_tex.llm_robust import LectureModelClient as RobustLectureModelClient
from automatic_lecture_tex.pipeline_robust import _run_linear_pipeline_with_policy
from automatic_lecture_tex.schemas import BlockType, ChunkNotes, LectureChunk, NoteBlock


def _chunk(text: str = "Продолжаем доказательство.") -> LectureChunk:
    return LectureChunk(
        id="chunk_test",
        start=0.0,
        end=10.0,
        segment_ids=["seg_0"],
        text=text,
        timestamped_text=f"[00:00-00:10] {text}",
    )


def test_correction_trigger_looks_only_at_current_transcript():
    prompt = """Current timestamped transcript:
[00:10-00:20] Продолжаем доказательство без изменений.

Current visual evidence:
[]

Compact catalog of earlier note blocks:
[{"latex_excerpt":"Здесь была ошибка."}]
"""

    transcript = _current_correction_transcript(prompt)

    assert "Продолжаем" in transcript
    assert "ошибка" not in transcript
    assert has_explicit_correction_signal(transcript) is False


def test_explicit_lecturer_correction_triggers_scan():
    assert has_explicit_correction_signal("Я оговорился: здесь должен быть минус.") is True
    assert has_explicit_correction_signal("Точнее, индекс здесь n+1.") is True
    assert has_explicit_correction_signal("Продолжаем доказательство теоремы.") is False


def test_correction_scan_short_circuits_without_llm_call():
    client = object.__new__(LectureModelClient)
    prompt = """Current timestamped transcript:
[00:10-00:20] Продолжаем доказательство.

Current visual evidence:
[]
"""

    result = client._structured(
        prompt,
        LinearCorrectionScan,
        operation="linear_correction_scan",
    )

    assert result.patches == []
    assert result.unresolved == []


def test_finalize_prompt_removes_old_textbook_completion_permissions(monkeypatch):
    captured = {}

    def fake_structured(
        self,
        prompt,
        schema,
        images=None,
        max_tokens=None,
        *,
        guided_json=True,
        operation="structured",
    ):
        captured[operation] = prompt
        return schema.model_validate({})

    monkeypatch.setattr(RobustLectureModelClient, "_structured", fake_structured)
    client = object.__new__(LectureModelClient)

    class EmptyModel(BaseModel):
        pass

    old_writer = """Create notes.
You may actively correct ASR/OCR errors, normalize terminology, reconstruct formulas from combined
audio and video evidence, and complete a short derivation when its mathematical conclusion is
reliable. Do not add unrelated textbook exposition.
For figure blocks, asset_path must be copied exactly from visual evidence. Record unresolved
ambiguities in `unresolved`.
Use the speech, known
notation, and mathematical consistency here to make any further correction or inference, and record
every such content-changing step in `corrections`.
"""
    client._structured(old_writer, EmptyModel, operation="finalize_chunk")
    client._structured("audit", EmptyModel, operation="math_audit")

    final_prompt = captured["finalize_chunk"]
    assert "complete a short derivation" not in final_prompt
    assert "make any further correction or inference" not in final_prompt
    assert "Do NOT invent or complete a multi-step derivation" in final_prompt
    assert "garbled ASR fragments" in final_prompt
    assert "z_f versus y_f" in captured["math_audit"]
    assert "Never change a lecturer statement solely" in captured["math_audit"]


def test_audit_evidence_must_exist_in_declared_current_source():
    chunk = _chunk("Лектор говорит: z_f определяется с точностью до скаляра.")
    visual_json = json.dumps(
        [{"raw_latex": r"f(ix)=u(ix)+iv(ix)\quad if(x)=iu(x)-v(x)"}],
        ensure_ascii=False,
    )

    assert audit_evidence_supported(
        [AuditEvidence(source="transcript", quote="z_f определяется с точностью до скаляра")],
        chunk=chunk,
        evidence_json=visual_json,
    )
    assert audit_evidence_supported(
        [AuditEvidence(source="visual", quote=r"if(x)=iu(x)-v(x)")],
        chunk=chunk,
        evidence_json=visual_json,
    )
    assert not audit_evidence_supported(
        [AuditEvidence(source="transcript", quote="по теореме Рисса вектор единственен")],
        chunk=chunk,
        evidence_json=visual_json,
    )


def test_source_backed_visual_audit_correction_is_applied(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    notes = ChunkNotes(
        section_title="Комплексный функционал",
        blocks=[
            NoteBlock(
                type=BlockType.EQUATION,
                latex=r"v(ix)=-u(x),\qquad u(ix)=-v(x)",
            )
        ],
    )
    evidence_json = json.dumps(
        [
            {
                "raw_latex": (
                    r"f(ix)=u(ix)+iv(ix)\qquad if(x)=iu(x)-v(x)"
                )
            }
        ]
    )

    def fake_structured(self, prompt, schema, **kwargs):
        assert schema is SourceGroundedMathAudit
        return schema.model_validate(
            {
                "corrections": [
                    {
                        "block_index": 0,
                        "corrected_latex": r"u(ix)=-v(x),\qquad v(ix)=u(x)",
                        "reason": "Знаки и роли u,v должны следовать двум формулам на доске.",
                        "confidence": 0.97,
                        "evidence": [
                            {"source": "visual", "quote": r"f(ix)=u(ix)+iv(ix)"},
                            {"source": "visual", "quote": r"if(x)=iu(x)-v(x)"},
                        ],
                    }
                ],
                "issues": [],
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=_chunk(),
        evidence_json=evidence_json,
        previous_context=None,
    )

    assert result.blocks[0].latex == r"u(ix)=-v(x),\qquad v(ix)=u(x)"
    assert len(result.corrections) == 1
    assert str(result.corrections[0].basis) == "visual"


def test_textbook_only_audit_correction_is_not_applied(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    original = "Представляющий вектор y_f определён с точностью до скаляра."
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[NoteBlock(type=BlockType.PARAGRAPH, latex=original)],
    )
    chunk = _chunk("z_f определяется с точностью до скаляра, а y_f определён однозначно.")

    def fake_structured(self, prompt, schema, **kwargs):
        return schema.model_validate(
            {
                "corrections": [
                    {
                        "block_index": 0,
                        "corrected_latex": "Представляющий вектор y_f единственен.",
                        "reason": "Так утверждает стандартная теорема Рисса.",
                        "confidence": 0.99,
                        "evidence": [
                            {"source": "transcript", "quote": "стандартная теорема Рисса"}
                        ],
                    }
                ],
                "issues": [],
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert result.blocks[0].latex == original
    assert result.corrections == []
    assert any("source support not verified" in item for item in result.unresolved)


def test_linear_audit_runs_on_prose_only_chunk(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True, math_audit_min_equals=50)
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Представляющий вектор определяется однозначно.",
            )
        ],
    )
    seen = {}

    def fake_structured(self, prompt, schema, **kwargs):
        seen["called"] = True
        return schema.model_validate({"corrections": [], "issues": []})

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=_chunk(),
        evidence_json="[]",
        previous_context=None,
    )

    assert result is notes
    assert seen["called"] is True


def test_policy_version_is_in_linear_chunk_cache_identity(monkeypatch):
    captured = {}

    def fake_runner(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("automatic_lecture_tex.pipeline_robust.run_linear_pipeline", fake_runner)

    result = _run_linear_pipeline_with_policy(source_identity={"url": "lecture"})

    assert result == "ok"
    assert captured["source_identity"] == {
        "media_source": {"url": "lecture"},
        "linear_source_policy_version": LINEAR_SOURCE_POLICY_VERSION,
    }
