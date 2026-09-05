import json
import threading
from types import SimpleNamespace

from pydantic import BaseModel

from automatic_lecture_tex.llm import LectureModelClient
from automatic_lecture_tex.schemas import (
    ChunkAnalysis,
    ChunkNotes,
    CorrectionRecord,
    LectureChunk,
    MathAudit,
    MathAuditCorrection,
    NoteBlock,
    VisualEvidence,
    VisualRequest,
)


class LatexPayload(BaseModel):
    latex: str


def test_usage_accounting_tracks_operations_and_cache_tokens() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    client._usage_lock = threading.Lock()
    client.reset_usage()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
    )

    client._record_usage("visual_ocr", response)
    usage = client.usage_snapshot()

    assert usage["requests"] == 1
    assert usage["total_tokens"] == 150
    assert usage["cached_prompt_tokens"] == 40
    assert usage["reasoning_tokens"] == 5
    assert usage["by_operation"]["visual_ocr"]["prompt_tokens"] == 120

    client._record_usage("math_audit", response)
    after = client.usage_snapshot()
    delta = LectureModelClient.usage_delta(after, usage)
    combined = LectureModelClient.combine_usage([usage, delta])

    assert delta["requests"] == 1
    assert delta["by_operation"]["math_audit"]["total_tokens"] == 150
    assert combined == after


def test_response_format_uses_json_schema() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    response_format = client._response_format(LatexPayload)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "LatexPayload"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == LatexPayload.model_json_schema()


def test_parse_json_preserves_latex_backslashes() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    latex = r"\theta + \frac{1}{2} + \beta"
    raw = json.dumps({"latex": latex})

    parsed = client._parse_json(raw, LatexPayload)

    assert parsed.latex == latex


def test_parse_json_restores_single_escaped_latex_command() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    parsed = client._parse_json(r'{"latex":"\boldsymbol{C}"}', LatexPayload)

    assert parsed.latex == r"\boldsymbol{C}"


def test_parse_json_restores_latex_command_misread_as_newline() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    parsed = client._parse_json(r'{"latex":"$x \neq 0$, $x \notin A$"}', LatexPayload)

    assert parsed.latex == r"$x \neq 0$, $x \notin A$"


def test_parse_json_restores_variable_misread_as_vertical_tab() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    parsed = client._parse_json('{"latex":"$\\u000b(x)$"}', LatexPayload)

    assert parsed.latex == "$v(x)$"


def test_speculative_note_block_is_demoted_to_unresolved() -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    notes = ChunkNotes(
        section_title="Section",
        blocks=[
            NoteBlock(type="paragraph", latex="Подтвержденный факт."),
            NoteBlock(type="equation", latex="Вероятно, $x=2$, или аналогичное выражение."),
        ],
    )

    result = client._demote_speculative_blocks(notes)

    assert [block.latex for block in result.blocks] == ["Подтвержденный факт."]
    assert "Вероятно" in result.unresolved[0]


def test_analyze_chunk_discards_ungrounded_timestamps(monkeypatch) -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    analysis = ChunkAnalysis(
        visual_requests=[
            VisualRequest(id="valid", timestamp=15, reason="x", question="x"),
            VisualRequest(id="invalid", timestamp=150, reason="y", question="y"),
        ]
    )
    monkeypatch.setattr(client, "_structured", lambda *args, **kwargs: analysis)
    chunk = LectureChunk(
        id="c", start=10, end=20, segment_ids=["s"], text="text", timestamped_text="timed"
    )

    result = client.analyze_chunk(chunk, {})

    assert [request.id for request in result.visual_requests] == ["valid"]


def test_finalize_chunk_keeps_visual_correction_audit_trail(monkeypatch) -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    client.config = type(
        "Config",
        (),
        {"output_language": "ru", "math_audit": False, "math_audit_min_equals": 4},
    )()
    generated = ChunkNotes(section_title="Section", blocks=[])
    monkeypatch.setattr(client, "_structured", lambda *args, **kwargs: generated)
    correction = CorrectionRecord(
        original="w(ix)",
        corrected="u(ix)",
        reason="The board and established notation agree on u.",
        basis="visual",
        confidence=0.95,
    )
    evidence = VisualEvidence(corrections=[correction])
    chunk = LectureChunk(id="c", start=0, end=10, segment_ids=["s"], text="text")

    result = client.finalize_chunk(chunk, [evidence], {})

    assert result.corrections == [correction]


def test_math_audit_applies_high_confidence_correction(monkeypatch) -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    client.config = type(
        "Config",
        (),
        {"output_language": "ru", "math_audit": True, "math_audit_min_equals": 1},
    )()
    audit = MathAudit(
        corrections=[
            MathAuditCorrection(
                block_index=0,
                corrected_latex=r"$(\nu-i\mu)w(ix)=-i\alpha w(ix)$",
                reason="Коэффициент равен -i alpha.",
                confidence=0.99,
            )
        ]
    )
    monkeypatch.setattr(client, "_structured", lambda *args, **kwargs: audit)
    notes = ChunkNotes(
        section_title="Section",
        blocks=[NoteBlock(type="equation", latex=r"$(\nu-i\mu)w(ix)=i\alpha w(ix)$")],
    )
    chunk = LectureChunk(id="c", start=0, end=10, segment_ids=["s"], text="proof")

    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert result.blocks[0].latex == r"$(\nu-i\mu)w(ix)=-i\alpha w(ix)$"
    assert result.corrections[0].basis == "mathematical_consistency"


def test_math_audit_failure_preserves_primary_notes(monkeypatch) -> None:
    client = LectureModelClient.__new__(LectureModelClient)
    client.config = type(
        "Config",
        (),
        {"output_language": "ru", "math_audit": True, "math_audit_min_equals": 1},
    )()
    monkeypatch.setattr(
        client,
        "_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(json.JSONDecodeError("cut", "", 0)),
    )
    notes = ChunkNotes(
        section_title="Section",
        blocks=[NoteBlock(type="equation", latex=r"x=x")],
    )
    chunk = LectureChunk(id="c", start=0, end=10, segment_ids=["s"], text="proof")

    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert result.blocks[0].latex == r"x=x"
    assert "audit" in result.unresolved[0]
