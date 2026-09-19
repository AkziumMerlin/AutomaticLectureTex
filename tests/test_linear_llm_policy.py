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
from automatic_lecture_tex.schemas import (
    BlockType,
    ChunkNotes,
    LectureChunk,
    NoteBlock,
    VisualEvidence,
)


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
    assert "[omitted-math]" in final_prompt
    assert "z_f versus y_f" in captured["math_audit"]
    assert "Produce exactly one keep/replace/suppress verdict" in captured["math_audit"]
    assert "transcript is noisy ASR" in captured["math_audit"]
    assert "Never guess a theorem/person/name" in captured["math_audit"]


def test_finalize_chunk_sends_board_scan_frames_directly(tmp_path, monkeypatch):
    image_paths = []
    for index in range(5):
        path = tmp_path / f"board_{index}.jpg"
        path.write_bytes(b"image")
        image_paths.append(path)

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
        captured[operation] = {
            "prompt": prompt,
            "images": list(images or []),
            "guided_json": guided_json,
        }
        return schema.model_validate(
            {
                "section_title": "Комплексные функционалы",
                "blocks": [{"type": "paragraph", "latex": "Продолжаем доказательство."}],
            }
        )

    monkeypatch.setattr(RobustLectureModelClient, "_structured", fake_structured)
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=False)

    evidence = [
        VisualEvidence(
            request_id="chunk_test_board_scan_00",
            kind="board_scan",
            confidence=1.0,
            frame_paths=[str(path) for path in image_paths],
            frame_timestamps=[1.0, 3.0, 5.0, 7.0, 9.0],
        )
    ]

    notes = client.finalize_chunk(_chunk(), evidence, {}, None)

    assert notes.blocks
    assert captured["finalize_chunk"]["images"] == image_paths
    assert captured["finalize_chunk"]["guided_json"] is False
    assert "transcript is a noisy observation" in captured["finalize_chunk"]["prompt"].lower()
    assert "never identify a named theorem/person" in captured["finalize_chunk"]["prompt"].lower()


def test_multimodal_verifier_can_apply_visual_only_replacement(tmp_path, monkeypatch):
    image = tmp_path / "board.jpg"
    image.write_bytes(b"image")
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    notes = ChunkNotes(
        section_title="Продолжение функционала",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="По теореме Гельфанда-Неймана функционал продолжается.",
            )
        ],
    )
    evidence_json = json.dumps(
        [
            {
                "request_id": "chunk_test_board_scan_00",
                "kind": "board_scan",
                "frame_paths": [str(image)],
                "frame_timestamps": [5.0],
            }
        ],
        ensure_ascii=False,
    )
    seen = {}

    def fake_structured(self, prompt, schema, **kwargs):
        seen.update(kwargs)
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "replace",
                        "target_excerpt": "теореме Гельфанда-Неймана",
                        "replacement_latex": (
                            "По теореме о продолжении линейного функционала с сохранением нормы "
                            "функционал продолжается."
                        ),
                        "reason": "Имя не подтверждается; содержание продолжения видно из доски.",
                        "confidence": 0.96,
                        "support": "visual",
                    }
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=_chunk("По теореме Анванкова продолжаем функционал."),
        evidence_json=evidence_json,
        previous_context=None,
        images=[image],
    )

    assert "Гельфанда-Неймана" not in result.blocks[0].latex
    assert "продолжении линейного функционала" in result.blocks[0].latex
    assert str(result.corrections[0].basis) == "visual"
    assert seen["images"] == [image]
    assert seen["guided_json"] is False


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
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "replace",
                        "target_excerpt": r"v(ix)=-u(x)",
                        "replacement_latex": r"u(ix)=-v(x),\qquad v(ix)=u(x)",
                        "reason": "Знаки и роли u,v должны следовать двум формулам на доске.",
                        "confidence": 0.97,
                        "evidence": [
                            {"source": "visual", "quote": r"f(ix)=u(ix)+iv(ix)"},
                            {"source": "visual", "quote": r"if(x)=iu(x)-v(x)"},
                        ],
                    }
                ]
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
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "replace",
                        "target_excerpt": "y_f определён с точностью до скаляра",
                        "replacement_latex": "Представляющий вектор y_f единственен.",
                        "reason": "Так утверждает стандартная теорема Рисса.",
                        "confidence": 0.99,
                        "evidence": [
                            {"source": "transcript", "quote": "стандартная теорема Рисса"}
                        ],
                    }
                ]
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
    assert result.unresolved == []


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
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "keep",
                        "reason": "Блок соответствует текущему источнику.",
                        "confidence": 0.99,
                    }
                ]
            }
        )

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


def test_high_confidence_source_grounded_issue_marks_exact_block(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Представляющий вектор y_f определяется с точностью до скаляра.",
            )
        ],
    )
    chunk = _chunk("y_f определён однозначно.")
    evidence_json = json.dumps(
        [
            {
                "request_id": "req_riesz",
                "raw_latex": "y_f — единственный представляющий вектор",
            }
        ],
        ensure_ascii=False,
    )

    def fake_structured(self, prompt, schema, **kwargs):
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "suppress",
                        "target_excerpt": "y_f определяется с точностью до скаляра",
                        "reason": "Блок приписывает y_f свойство, которого нет в текущем источнике.",
                        "confidence": 0.96,
                        "evidence": [
                            {
                                "source": "visual",
                                "quote": "y_f — единственный представляющий вектор",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json=evidence_json,
        previous_context=None,
    )

    assert any(
        item.startswith("audit-suppress:") for item in result.blocks[0].source_evidence_ids
    )
    assert "audit-visual:req_riesz" in result.blocks[0].source_evidence_ids
    assert any("Audit block 0" in item for item in result.unresolved)


def test_strict_verdict_rejects_target_excerpt_from_another_block(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Вектор z_f определяется с точностью до скаляра.",
            ),
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Вектор y_f определяется однозначно.",
            ),
        ],
    )

    def fake_structured(self, prompt, schema, **kwargs):
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "keep",
                        "reason": "Соответствует источнику.",
                        "confidence": 0.99,
                    },
                    {
                        "block_index": 1,
                        "action": "suppress",
                        "target_excerpt": "z_f определяется с точностью до скаляра",
                        "reason": "Ошибочно выбран индекс блока.",
                        "confidence": 0.99,
                        "evidence": [
                            {
                                "source": "transcript",
                                "quote": "y_f определяется однозначно",
                            }
                        ],
                    },
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=_chunk("z_f определяется с точностью до скаляра, y_f определяется однозначно."),
        evidence_json="[]",
        previous_context=None,
    )

    assert not any(
        item.startswith("audit-suppress:") for item in result.blocks[1].source_evidence_ids
    )
    assert result.blocks[1].latex == "Вектор y_f определяется однозначно."


def test_writer_added_phase_claim_can_be_suppressed_from_source_context(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    notes = ChunkNotes(
        section_title="Норма функционала",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex=(
                    r"Выбираем $\varphi=\arg f(x)$. Тогда $xe^{-i\varphi}$ является "
                    r"вещественным вектором и $f(xe^{-i\varphi})=|f(x)|$."
                ),
            )
        ],
    )
    chunk = _chunk("Выберем фазу так, чтобы f(x e^{-i phi}) было равно |f(x)|.")

    def fake_structured(self, prompt, schema, **kwargs):
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "suppress",
                        "target_excerpt": "является вещественным вектором",
                        "reason": "Это дополнительное утверждение writer, которого источник не говорит.",
                        "confidence": 0.95,
                        "evidence": [
                            {
                                "source": "transcript",
                                "quote": "f(x e^{-i phi}) было равно |f(x)|",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert "audit-suppress:0.950" in result.blocks[0].source_evidence_ids


def test_topology_type_drift_can_be_replaced_only_from_current_source(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    notes = ChunkNotes(
        section_title="Слабая-* топология",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex=r"На $\mathbb C^X$ рассматривается слабая топология.",
            )
        ],
    )
    chunk = _chunk("На C^X рассматривается декартова топология.")

    def fake_structured(self, prompt, schema, **kwargs):
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "replace",
                        "target_excerpt": "слабая топология",
                        "replacement_latex": r"На $\mathbb C^X$ рассматривается декартова топология.",
                        "reason": "Draft подменил явно названный тип топологии.",
                        "confidence": 0.98,
                        "evidence": [
                            {
                                "source": "transcript",
                                "quote": "рассматривается декартова топология",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert "декартова топология" in result.blocks[0].latex
    assert len(result.corrections) == 1


def test_suppress_cannot_erase_literal_lecturer_statement(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True)
    statement = "Лектор утверждает, что сфера слабо компактна."
    notes = ChunkNotes(
        section_title="Слабая компактность",
        blocks=[NoteBlock(type=BlockType.PARAGRAPH, latex=statement)],
    )
    chunk = _chunk(statement)

    def fake_structured(self, prompt, schema, **kwargs):
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "suppress",
                        "target_excerpt": "сфера слабо компактна",
                        "reason": "Стандартный учебник говорит иначе.",
                        "confidence": 0.99,
                        "evidence": [
                            {
                                "source": "transcript",
                                "quote": "сфера слабо компактна",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert not any(
        item.startswith("audit-suppress:") for item in result.blocks[0].source_evidence_ids
    )



def test_malformed_audit_verdict_does_not_discard_valid_sibling(monkeypatch):
    client = object.__new__(LectureModelClient)
    client.config = LLMConfig(math_audit=True, max_tokens=4096)
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Вектор y_f определяется с точностью до скаляра.",
            ),
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Второй спорный блок остаётся без надёжного evidence.",
            ),
        ],
    )
    chunk = _chunk(
        "Вектор y_f определён однозначно. Второй фрагмент сформулирован неразборчиво."
    )
    seen = {}

    def fake_structured(self, prompt, schema, **kwargs):
        seen["max_tokens"] = kwargs["max_tokens"]
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "block_index": 0,
                        "action": "replace",
                        "target_excerpt": "y_f определяется с точностью до скаляра",
                        "replacement_latex": "Вектор y_f определён однозначно.",
                        "reason": "Источник явно утверждает однозначность.",
                        "confidence": 0.98,
                        "evidence": [
                            {
                                "source": "transcript",
                                "quote": "Вектор y_f определён однозначно",
                            }
                        ],
                    },
                    {
                        "block_index": 1,
                        "action": "suppress",
                        "target_excerpt": "Второй спорный блок",
                        "reason": "Модель забыла приложить evidence.",
                        "confidence": 0.90,
                    },
                ]
            }
        )

    monkeypatch.setattr(LectureModelClient, "_structured", fake_structured)
    result = client._audit_math(
        notes,
        chunk=chunk,
        evidence_json="[]",
        previous_context=None,
    )

    assert seen["max_tokens"] == 4096
    assert result.blocks[0].latex == "Вектор y_f определён однозначно."
    assert not any(
        item.startswith("audit-suppress:") for item in result.blocks[1].source_evidence_ids
    )
