from automatic_lecture_tex.linear_notes import (
    LinearPatch,
    block_id,
    block_segment_ids,
    provenance_claim_ids,
    scan_linear_corrections,
)
from automatic_lecture_tex.linear_pipeline import _apply_patch
from automatic_lecture_tex.schemas import (
    BlockType,
    ChunkNotes,
    LectureChunk,
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
