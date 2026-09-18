from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from automatic_lecture_tex.board_crop import detect_board_roi, generate_board_crops
from automatic_lecture_tex.config import LatexConfig, NotesConfig, VisionConfig
from automatic_lecture_tex.latex import render_block, write_course_tex
from automatic_lecture_tex.linear_visual_fallback import (
    clean_stale_architecture_artifacts,
    inject_unresolved_board_snapshots,
    is_board_snapshot,
)
from automatic_lecture_tex.schemas import (
    BlockType,
    ChunkNotes,
    ExtractedFrame,
    LectureChunk,
    LectureIR,
    NoteBlock,
    Transcript,
    TranscriptSegment,
    VisualEvidence,
    VisualKind,
    VisualRequest,
)
from automatic_lecture_tex.sensory_evidence import collect_visual_evidence
from automatic_lecture_tex.tex_safety import normalize_math_spans


def _board_image(path: Path, *, width: int = 960, height: int = 540) -> None:
    image = Image.new("RGB", (width, height), (235, 225, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 105, 890, 455), fill=(62, 78, 55))
    for y in (180, 250, 320, 390):
        draw.line((90, y, 820, y - 20), fill=(215, 215, 195), width=4)
    image.save(path, quality=95)


def test_detect_board_roi_and_generate_overlapping_tiles(tmp_path: Path) -> None:
    source = tmp_path / "frame.jpg"
    _board_image(source)
    config = VisionConfig(board_crop_tiles=3)

    detected = detect_board_roi(source, config)

    assert detected is not None
    roi, score = detected
    assert score >= config.board_crop_min_score
    left, top, right, bottom = roi
    assert left < 120
    assert top < 170
    assert right > 800
    assert bottom > 400

    result = generate_board_crops(
        ExtractedFrame(timestamp=12.0, path=source), tmp_path / "crops", config
    )
    assert result is not None
    assert len(result.frames) == 4  # full board + 3 overlapping horizontal tiles
    assert result.frames[0].path.name == "board_full.jpg"
    assert all(frame.path.exists() for frame in result.frames)


def test_manual_normalized_board_roi_is_available_for_whiteboards(tmp_path: Path) -> None:
    source = tmp_path / "whiteboard.jpg"
    Image.new("RGB", (1000, 500), "white").save(source)
    config = VisionConfig(board_crop_roi=(0.1, 0.2, 0.9, 0.8))

    detected = detect_board_roi(source, config)

    assert detected == ((100, 100, 900, 400), 1.0)


def test_unresolved_math_gets_nonfloating_board_snapshot() -> None:
    notes = ChunkNotes(
        section_title="Доказательство",
        blocks=[],
        unresolved=["[omitted-math] знак в формуле не удалось подтвердить по источнику"],
    )
    evidence = [
        VisualEvidence(
            request_id="chunk_0003_req_0",
            kind=VisualKind.EQUATION,
            latex=r"f(x)=u(x)-iu(ix)",
            confidence=0.91,
            asset_path="figures/lecture_01/chunk_0003_req_0_board.jpg",
        )
    ]

    inserted = inject_unresolved_board_snapshots(
        notes, evidence, VisionConfig(), output_language="ru"
    )

    assert inserted == 1
    assert len(notes.blocks) == 1
    block = notes.blocks[0]
    assert block.type == BlockType.FIGURE
    assert is_board_snapshot(block)
    rendered = render_block(block)
    assert r"\includegraphics" in rendered
    assert r"\begin{figure}" not in rendered
    assert r"\begin{center}" in rendered


def test_generic_audit_text_without_block_marker_does_not_insert_photo() -> None:
    notes = ChunkNotes(
        section_title="Переход",
        blocks=[],
        unresolved=["Audit block 0: формула вызывает сомнение."],
    )
    evidence = [
        VisualEvidence(
            request_id="r",
            kind=VisualKind.EQUATION,
            confidence=0.95,
            asset_path="figures/r.jpg",
        )
    ]

    assert inject_unresolved_board_snapshots(notes, evidence, VisionConfig(), output_language="ru") == 0
    assert notes.blocks == []


def test_audit_issue_replaces_exact_block_at_same_position() -> None:
    bad = NoteBlock(
        type=BlockType.PARAGRAPH,
        latex="Неподтверждённое утверждение.",
        source_claim_ids=["block:block_0001_000"],
        source_evidence_ids=["audit-suppress:0.950", "audit-visual:req"],
    )
    tail = NoteBlock(type=BlockType.PARAGRAPH, latex="Следующий корректный абзац.")
    notes = ChunkNotes(section_title="Рисс", blocks=[bad, tail])
    evidence = [
        VisualEvidence(
            request_id="req",
            kind=VisualKind.EQUATION,
            confidence=0.93,
            asset_path="figures/lecture_01/req_board.jpg",
        )
    ]

    inserted = inject_unresolved_board_snapshots(
        notes, evidence, VisionConfig(), output_language="ru"
    )

    assert inserted == 1
    assert len(notes.blocks) == 2
    assert is_board_snapshot(notes.blocks[0])
    assert notes.blocks[0].asset_path == "figures/lecture_01/req_board.jpg"
    assert notes.blocks[1].latex == "Следующий корректный абзац."


def test_audit_issue_without_linked_board_is_suppressed_without_unrelated_photo() -> None:
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Неподтверждённое утверждение.",
                source_evidence_ids=["audit-suppress:0.980"],
            )
        ],
    )
    evidence = [
        VisualEvidence(
            request_id="other",
            kind=VisualKind.EQUATION,
            confidence=0.99,
            asset_path="figures/other.jpg",
        )
    ]

    inserted = inject_unresolved_board_snapshots(
        notes, evidence, VisionConfig(), output_language="ru"
    )

    assert inserted == 0
    assert notes.blocks == []


def test_existing_figure_prevents_duplicate_snapshot() -> None:
    asset = "figures/lecture_01/req_board.jpg"
    notes = ChunkNotes(
        section_title="Рисс",
        blocks=[
            NoteBlock(type=BlockType.FIGURE, latex="", asset_path=asset),
            NoteBlock(
                type=BlockType.PARAGRAPH,
                latex="Плохой блок.",
                source_evidence_ids=["audit-suppress:0.950", "audit-visual:req"],
            ),
        ],
    )
    evidence = [
        VisualEvidence(
            request_id="req",
            kind=VisualKind.EQUATION,
            confidence=0.95,
            asset_path=asset,
        )
    ]

    inserted = inject_unresolved_board_snapshots(
        notes, evidence, VisionConfig(), output_language="ru"
    )

    assert inserted == 0
    assert len(notes.blocks) == 1
    assert notes.blocks[0].type == BlockType.FIGURE


def test_nonmathematical_unresolved_does_not_insert_photo() -> None:
    notes = ChunkNotes(
        section_title="Переход",
        blocks=[],
        unresolved=["Служебный ASR-фрагмент пропущен."],
    )
    evidence = [
        VisualEvidence(
            request_id="r",
            kind=VisualKind.EQUATION,
            confidence=0.95,
            asset_path="figures/r.jpg",
        )
    ]

    assert inject_unresolved_board_snapshots(notes, evidence, VisionConfig(), output_language="ru") == 0
    assert notes.blocks == []


def test_single_unmatched_double_dollar_is_closed_deterministically() -> None:
    value = r"$$\|f\| = \sup_{\|x\| \le 1}|f(x)|."

    normalized = normalize_math_spans(value)

    assert normalized.startswith(r"\[")
    assert normalized.endswith(r"\]")
    assert "$$" not in normalized


def test_orphan_sizing_command_before_non_delimiter_is_removed() -> None:
    assert normalize_math_spans(r"$\bigl\mathbb{C}$") == r"$\mathbb{C}$"
    assert normalize_math_spans(r"$\bigl(x\bigr)$") == r"$\bigl(x\bigr)$"


def test_indexed_bare_cap_and_cup_become_big_operators() -> None:
    assert normalize_math_spans(r"$cap_{i=1}^m V_i$") == r"$\bigcap_{i=1}^m V_i$"
    assert normalize_math_spans(r"$cup_{i=1}^m V_i$") == r"$\bigcup_{i=1}^m V_i$"
    assert normalize_math_spans(r"$bigcap_{i=1}^m V_i$") == r"$\bigcap_{i=1}^m V_i$"


def test_linear_hygiene_removes_retired_architecture_artifacts(tmp_path: Path) -> None:
    for name in (
        "knowledge_windows",
        "knowledge_sections",
        "knowledge_episode_batches",
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "stale.json").write_text("{}")
    for name in (
        "episode_hierarchy.json",
        "global_validation.json",
        "lecture_kb.json",
        "lecture_outline.json",
    ):
        (tmp_path / name).write_text("{}")
    (tmp_path / "frames" / "window_0001_req").mkdir(parents=True)
    (tmp_path / "frames" / "chunk_0001_req").mkdir(parents=True)
    (tmp_path / "linear_chunks").mkdir()

    clean_stale_architecture_artifacts(tmp_path)

    assert not (tmp_path / "knowledge_windows").exists()
    assert not (tmp_path / "episode_hierarchy.json").exists()
    assert not (tmp_path / "frames" / "window_0001_req").exists()
    assert (tmp_path / "frames" / "chunk_0001_req").exists()
    assert (tmp_path / "linear_chunks").exists()


class _FakeLLM:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def analyze_chunk(self, chunk, notation):
        return SimpleNamespace(
            visual_requests=[
                VisualRequest(id="req", timestamp=20.0, reason="board", question="formula")
            ]
        )

    def resolve_visual_request(self, request, chunk, frame_paths, frame_timestamps):
        self.paths = list(frame_paths)
        return VisualEvidence(
            request_id=request.id,
            kind=VisualKind.EQUATION,
            raw_latex="x=y",
            latex="x=y",
            confidence=0.9,
            best_frame_index=2,
        )


class _FakeSource:
    def extract_frames(self, times, frame_dir):
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, timestamp in enumerate(times):
            path = frame_dir / f"frame_{index:02d}.jpg"
            _board_image(path)
            frames.append(ExtractedFrame(timestamp=timestamp, path=path))
        return frames


def test_visual_collection_sends_full_context_plus_board_crops_and_keeps_asset(tmp_path: Path) -> None:
    llm = _FakeLLM()
    output_dir = tmp_path / "tex"
    pipeline = SimpleNamespace(
        config=SimpleNamespace(
            notes=NotesConfig(visual_rule_selector=False, visual_llm_selector=True),
            vision=VisionConfig(board_crop_tiles=3, board_crop_max_vlm_images=5),
            latex=LatexConfig(output_dir=output_dir),
        ),
        llm=llm,
    )
    lecture = SimpleNamespace(id="lecture_01")
    chunk = LectureChunk(
        id="chunk_0000",
        start=0.0,
        end=180.0,
        segment_ids=["seg_0"],
        text="Смотрим на формулу на доске.",
        timestamped_text="[00:20] Смотрим на формулу на доске.",
    )
    transcript = Transcript(
        lecture_id="lecture_01",
        language="ru",
        segments=[TranscriptSegment(id="seg_0", start=0.0, end=30.0, text=chunk.text)],
    )

    requests, evidence, _ = collect_visual_evidence(
        pipeline,
        lecture,
        chunk,
        transcript,
        _FakeSource(),
        tmp_path / "work",
        output_dir / "figures" / "lecture_01",
        {},
    )

    assert len(requests) == 1
    assert len(llm.paths) == 5
    assert llm.paths[0].name.startswith("frame_")
    assert any(path.name == "board_full.jpg" for path in llm.paths)
    assert sum(path.name.startswith("board_tile_") for path in llm.paths) == 3
    assert evidence[0].asset_path is not None
    asset = output_dir / evidence[0].asset_path
    assert asset.exists()
    # The persisted user-facing asset is the wide full-board ROI even though the fake VLM selected
    # image index 2 (a narrow tile) for recognition.
    with Image.open(asset) as image:
        assert image.width > 700


def test_course_writer_prunes_unreferenced_generated_assets(tmp_path: Path) -> None:
    output_dir = tmp_path / "tex"
    figures = output_dir / "figures" / "lecture_01"
    figures.mkdir(parents=True)
    keep = figures / "keep.jpg"
    stale = figures / "stale.jpg"
    keep.write_bytes(b"keep")
    stale.write_bytes(b"stale")
    ir = LectureIR(
        lecture_id="lecture_01",
        title="Лекция 1",
        chunks=[
            ChunkNotes(
                section_title="Тема",
                blocks=[
                    NoteBlock(
                        type=BlockType.FIGURE,
                        latex="",
                        asset_path="figures/lecture_01/keep.jpg",
                    )
                ],
            )
        ],
    )

    write_course_tex("Курс", [ir], output_dir)

    assert keep.exists()
    assert not stale.exists()
