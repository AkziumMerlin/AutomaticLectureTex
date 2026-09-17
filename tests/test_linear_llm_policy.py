from pydantic import BaseModel

from automatic_lecture_tex.config import LLMConfig
from automatic_lecture_tex.linear_llm_policy import (
    LINEAR_SOURCE_POLICY_VERSION,
    LectureModelClient,
    _current_correction_transcript,
    has_explicit_correction_signal,
)
from automatic_lecture_tex.linear_notes import LinearCorrectionScan
from automatic_lecture_tex.llm_robust import LectureModelClient as RobustLectureModelClient
from automatic_lecture_tex.pipeline_robust import _run_linear_pipeline_with_policy
from automatic_lecture_tex.schemas import BlockType, ChunkNotes, NoteBlock


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


def test_finalize_and_audit_prompts_receive_source_policy(monkeypatch):
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

    client._structured("writer", EmptyModel, operation="finalize_chunk")
    client._structured("audit", EmptyModel, operation="math_audit")

    assert "Do NOT introduce a new sequence" in captured["finalize_chunk"]
    assert "Mathematical plausibility is not evidence" in captured["finalize_chunk"]
    assert "z_f versus y_f" in captured["math_audit"]
    assert "Never change a lecturer statement solely" in captured["math_audit"]


def test_linear_audit_runs_even_without_equals(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True, math_audit_min_equals=4)
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

    def fake_audit(self, current_notes, **kwargs):
        seen["threshold"] = self.config.math_audit_min_equals
        return current_notes

    monkeypatch.setattr(RobustLectureModelClient, "_audit_math", fake_audit)

    result = client._audit_math(
        notes,
        chunk=None,
        evidence_json="[]",
        previous_context=None,
    )

    assert result is notes
    assert seen["threshold"] == 0
    assert client.config.math_audit_min_equals == 4


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
