import inspect

from automatic_lecture_tex.linear_notes import (
    GlobalBlockEdit,
    GlobalLectureEditPlan,
    GlobalSectionPlan,
    LinearPatch,
    block_id,
    block_segment_ids,
    plan_global_lecture_edit,
    provenance_claim_ids,
    scan_linear_corrections,
)
from automatic_lecture_tex.linear_pipeline import (
    _apply_global_edit_plan,
    _apply_patch,
    _sanitize_final_ir_tex,
    _stamp_chunk_provenance,
    _writer_context_notes,
    run_linear_pipeline,
)
from automatic_lecture_tex.schemas import (
    BlockType,
    ChunkNotes,
    LectureChunk,
    LectureIR,
    NoteBlock,
)


def _block(stable_id: str, latex: str, segment_id: str = "seg_old") -> NoteBlock:
    return NoteBlock(
        type=BlockType.EQUATION,
        latex=latex,
        source_claim_ids=provenance_claim_ids(stable_id, [segment_id]),
    )


def test_linear_block_provenance_survives_note_block_schema():
    block = _block("block_0000_000", r"a=2")

    assert block_id(block) == "block_0000_000"
    assert block_segment_ids(block) == ["seg_old"]
    dumped = block.model_dump(mode="json")
    restored = NoteBlock.model_validate(dumped)
    assert block_id(restored) == "block_0000_000"
    assert block_segment_ids(restored) == ["seg_old"]


def test_host_stamps_chunk_provenance_after_writer():
    notes = ChunkNotes(
        section_title="Section",
        blocks=[
            NoteBlock(type=BlockType.PARAGRAPH, latex="A"),
            NoteBlock(type=BlockType.EQUATION, latex=r"a=2"),
        ],
    )

    _stamp_chunk_provenance(notes, 3, ["seg_a", "seg_b"])

    assert block_id(notes.blocks[0]) == "block_0003_000"
    assert block_id(notes.blocks[1]) == "block_0003_001"
    assert block_segment_ids(notes.blocks[0]) == ["seg_a", "seg_b"]
    assert block_segment_ids(notes.blocks[1]) == ["seg_a", "seg_b"]


def test_writer_context_strips_host_provenance_without_mutating_ir():
    block = _block("block_0003_000", r"a=2", segment_id="seg_a")
    block.source_evidence_ids = ["visual_1"]
    notes = ChunkNotes(section_title="Section", blocks=[block], unresolved=["u"])

    projected = _writer_context_notes(notes)

    assert projected is not None
    assert projected.section_title == "Section"
    assert projected.unresolved == ["u"]
    assert projected.blocks[0].latex == r"a=2"
    assert projected.blocks[0].source_claim_ids == []
    assert projected.blocks[0].source_evidence_ids == []
    assert block_id(notes.blocks[0]) == "block_0003_000"
    assert notes.blocks[0].source_evidence_ids == ["visual_1"]


def test_linear_pipeline_uses_existing_finalize_chunk_writer_with_clean_context():
    source = inspect.getsource(run_linear_pipeline)

    assert "writer_previous_notes = _writer_context_notes(previous_notes)" in source
    assert "pipeline.llm.finalize_chunk(" in source
    assert "draft_linear_chunk" not in source
    assert "LinearChunkDraft" not in source


def test_explicit_cross_chunk_patch_replaces_existing_block():
    block = _block("block_0000_000", r"\|e_n-e_m\|=2")
    owner = ChunkNotes(section_title="Section", blocks=[block])
    patch = LinearPatch(
        target_block_id="block_0000_000",
        action="replace",
        replacement_latex=r"\|e_n-e_m\|=\sqrt{2}",
        evidence_segment_ids=["seg_correction"],
        reason="Лектор явно исправил значение.",
        confidence=0.97,
    )

    applied, issue = _apply_patch(
        patch,
        block_map={"block_0000_000": block},
        owner_map={"block_0000_000": owner},
        allowed_targets={"block_0000_000"},
        allowed_evidence_ids={"seg_correction"},
        apply_threshold=0.90,
    )

    assert applied is True
    assert issue is None
    assert block.latex == r"\|e_n-e_m\|=\sqrt{2}"
    assert len(owner.corrections) == 1
    assert owner.corrections[0].basis == "audio_context"


def test_patch_cannot_use_noncurrent_evidence():
    block = _block("block_0000_000", r"a=2")
    owner = ChunkNotes(section_title="Section", blocks=[block])
    patch = LinearPatch(
        target_block_id="block_0000_000",
        replacement_latex=r"a=3",
        evidence_segment_ids=["seg_old"],
        reason="Unsupported patch.",
        confidence=0.99,
    )

    applied, issue = _apply_patch(
        patch,
        block_map={"block_0000_000": block},
        owner_map={"block_0000_000": owner},
        allowed_targets={"block_0000_000"},
        allowed_evidence_ids={"seg_current"},
        apply_threshold=0.90,
    )

    assert applied is False
    assert "current transcript chunk" in issue
    assert block.latex == r"a=2"


class _CaptureLLM:
    def __init__(self):
        self.prompt = ""

    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        self.prompt = prompt
        assert operation == "linear_correction_scan"
        return schema.model_validate({"patches": [], "unresolved": []})


def test_distant_correction_scan_is_explicit_only_not_math_audit():
    llm = _CaptureLLM()
    chunk = LectureChunk(
        id="chunk_0002",
        start=20.0,
        end=30.0,
        segment_ids=["seg_current"],
        text="Нет, выше я оговорился: там должен быть минус.",
        timestamped_text="[00:20-00:30] Нет, выше я оговорился: там должен быть минус.",
    )
    earlier = [_block("block_0000_000", r"f(x)=u(x)+iu(ix)")]

    scan_linear_corrections(
        llm,
        chunk=chunk,
        evidence=[],
        earlier_blocks=earlier,
        output_language="ru",
        catalog_chars=320,
    )

    assert "block_0000_000" in llm.prompt
    assert "Do NOT patch because standard mathematics" in llm.prompt
    assert "explicit corrections or retractions" in llm.prompt



def test_global_editor_plan_merges_sections_drops_duplicate_and_repairs_math():
    a = _block("block_0000_000", "Определим слабую топологию.", "seg_a")
    b = _block("block_0000_001", r"\|f\| \ge \|y_f\|.", "seg_a")
    c_dup = _block("block_0001_000", "Определим слабую топологию.", "seg_b")
    d = _block("block_0001_001", "Следующий шаг доказательства.", "seg_b")
    draft = LectureIR(
        lecture_id="lecture",
        title="Lecture",
        chunks=[
            ChunkNotes(section_title="Chunk A", start=0, end=10, blocks=[a, b]),
            ChunkNotes(section_title="Chunk B", start=10, end=20, blocks=[c_dup, d]),
        ],
    )
    plan = GlobalLectureEditPlan(
        sections=[
            GlobalSectionPlan(
                title="Слабая топология",
                block_ids=["block_0000_000", "block_0000_001", "block_0001_001"],
            )
        ],
        patches=[
            GlobalBlockEdit(
                target_block_id="block_0001_000",
                action="drop",
                reason="Дословный дубль предыдущего определения.",
                confidence=0.99,
            ),
            GlobalBlockEdit(
                target_block_id="block_0000_001",
                action="replace",
                replacement_latex=r"\|f\| \le \|y_f\|.",
                reason="В неравенстве был обращён знак.",
                confidence=0.99,
            ),
        ],
    )

    result = _apply_global_edit_plan(draft, plan, apply_threshold=0.85)

    assert len(result.chunks) == 1
    assert result.chunks[0].section_title == "Слабая топология"
    assert [block_id(block) for block in result.chunks[0].blocks] == [
        "block_0000_000",
        "block_0000_001",
        "block_0001_001",
    ]
    assert result.chunks[0].blocks[1].latex == r"\|f\| \le \|y_f\|."
    assert len(result.chunks[0].corrections) == 1


def test_global_editor_rejects_silent_block_loss():
    a = _block("block_0000_000", "A")
    b = _block("block_0000_001", "B")
    draft = LectureIR(
        lecture_id="lecture",
        title="Lecture",
        chunks=[ChunkNotes(section_title="Chunk", blocks=[a, b])],
    )
    plan = GlobalLectureEditPlan(
        sections=[GlobalSectionPlan(title="Section", block_ids=["block_0000_000"])],
    )

    import pytest

    with pytest.raises(ValueError, match="coverage mismatch"):
        _apply_global_edit_plan(draft, plan, apply_threshold=0.85)


def test_final_ir_tex_sanity_repairs_repeated_unmatched_display_lines():
    block = _block(
        "block_0000_000",
        "$$\\|f\\| = \\|u\\|.\nТекст.\n$$x \\in X.",
    )
    draft = LectureIR(
        lecture_id="lecture",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title=r"Слабая сходимость в \ell\_2",
                blocks=[block],
            )
        ],
    )

    result = _sanitize_final_ir_tex(draft)

    assert "$$" not in result.chunks[0].blocks[0].latex
    assert r"\[\|f\| = \|u\|.\]" in result.chunks[0].blocks[0].latex
    assert r"\[x \in X.\]" in result.chunks[0].blocks[0].latex
    assert "$" in result.chunks[0].section_title



class _GlobalEditorLLM:
    def __init__(self):
        self.calls = []

    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        self.calls.append((operation, prompt, max_tokens))
        if operation == "global_lecture_structure":
            return schema.model_validate(
                {
                    "sections": [
                        {"title": "Основы", "first_block_id": "block_0000_000"},
                        {"title": "Продолжение", "first_block_id": "block_0001_001"},
                    ],
                    "drops": [
                        {
                            "target_block_id": "block_0001_000",
                            "action": "drop",
                            "reason": "Повтор предыдущего блока.",
                            "confidence": 0.99,
                        }
                    ],
                }
            )
        assert operation == "global_lecture_section_edit"
        return schema.model_validate({"patches": [], "unresolved": []})


def test_global_editor_uses_compact_structure_plus_bounded_full_text_batches():
    long_a = "A" * 180 + " UNIQUE_A_TAIL"
    long_b = "B" * 180 + " UNIQUE_B_TAIL"
    long_dup = "A" * 180 + " UNIQUE_A_TAIL"
    long_d = "D" * 180 + " UNIQUE_D_TAIL"
    draft = LectureIR(
        lecture_id="lecture",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="Chunk A",
                blocks=[
                    _block("block_0000_000", long_a),
                    _block("block_0000_001", long_b),
                ],
            ),
            ChunkNotes(
                section_title="Chunk B",
                blocks=[
                    _block("block_0001_000", long_dup),
                    _block("block_0001_001", long_d),
                ],
            ),
        ],
    )
    llm = _GlobalEditorLLM()

    plan = plan_global_lecture_edit(
        llm,
        draft_ir=draft,
        output_language="ru",
        apply_threshold=0.85,
        batch_chars=300,
        catalog_excerpt_chars=40,
    )

    assert [section.title for section in plan.sections] == ["Основы", "Продолжение"]
    assert plan.sections[0].block_ids == ["block_0000_000", "block_0000_001"]
    assert plan.sections[1].block_ids == ["block_0001_001"]
    assert [patch.target_block_id for patch in plan.patches] == ["block_0001_000"]

    operations = [item[0] for item in llm.calls]
    assert operations[0] == "global_lecture_structure"
    assert operations.count("global_lecture_section_edit") >= 2
    structure_prompt = llm.calls[0][1]
    assert "A" * 100 not in structure_prompt
    assert "B" * 100 not in structure_prompt
    assert "D" * 100 not in structure_prompt
    for operation, prompt, _ in llm.calls[1:]:
        assert operation == "global_lecture_section_edit"
        # A batch sees the compact whole-lecture catalog plus only a bounded subset at full length.
        assert sum(marker in prompt for marker in ["A" * 100, "B" * 100, "D" * 100]) <= 1



class _ExactDedupLLM:
    def _structured(self, prompt, schema, *, operation, max_tokens=None):
        if operation == "global_lecture_structure":
            return schema.model_validate(
                {
                    "sections": [
                        {"title": "Доказательство", "first_block_id": "block_0000_000"}
                    ],
                    "drops": [],
                }
            )
        assert operation == "global_lecture_section_edit"
        return schema.model_validate({"patches": [], "unresolved": []})


def test_exact_dedup_keeps_first_block_and_merges_later_provenance():
    repeated = (
        "Докажем утверждение. Сначала выбираем фазу, затем применяем вещественную версию "
        "теоремы и восстанавливаем комплексный функционал с сохранением нормы."
    )
    first = _block("block_0000_000", repeated, "seg_first")
    first.source_evidence_ids = ["visual_first"]
    middle = _block("block_0000_001", "Промежуточный комментарий, который должен сохраниться.")
    duplicate = _block("block_0001_000", repeated, "seg_recap")
    duplicate.source_evidence_ids = ["visual_recap"]
    draft = LectureIR(
        lecture_id="lecture",
        title="Lecture",
        chunks=[
            ChunkNotes(section_title="Chunk A", blocks=[first, middle]),
            ChunkNotes(section_title="Chunk B", blocks=[duplicate]),
        ],
    )

    plan = plan_global_lecture_edit(
        _ExactDedupLLM(),
        draft_ir=draft,
        output_language="ru",
        batch_chars=16000,
        catalog_excerpt_chars=80,
    )

    dedup = [patch for patch in plan.patches if patch.target_block_id == "block_0001_000"]
    assert len(dedup) == 1
    assert dedup[0].action == "drop"
    assert dedup[0].merge_into_block_id == "block_0000_000"
    assert plan.sections[0].block_ids == ["block_0000_000", "block_0000_001"]

    result = _apply_global_edit_plan(draft, plan, apply_threshold=0.85)
    kept = result.chunks[0].blocks[0]
    assert block_id(kept) == "block_0000_000"
    assert block_segment_ids(kept) == ["seg_first", "seg_recap"]
    assert kept.source_evidence_ids == ["visual_first", "visual_recap"]


def test_short_exact_recap_is_not_deterministically_collapsed():
    short = "Итак, получаем требуемое."
    first = _block("block_0000_000", short, "seg_first")
    duplicate = _block("block_0001_000", short, "seg_recap")
    draft = LectureIR(
        lecture_id="lecture",
        title="Lecture",
        chunks=[
            ChunkNotes(section_title="Chunk A", blocks=[first]),
            ChunkNotes(section_title="Chunk B", blocks=[duplicate]),
        ],
    )

    plan = plan_global_lecture_edit(
        _ExactDedupLLM(),
        draft_ir=draft,
        output_language="ru",
        batch_chars=16000,
        catalog_excerpt_chars=80,
    )

    assert not plan.patches
    assert plan.sections[0].block_ids == ["block_0000_000", "block_0001_000"]
