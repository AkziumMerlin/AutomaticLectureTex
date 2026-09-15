from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from automatic_lecture_tex.frame_selection import select_least_occluded_frame
from automatic_lecture_tex.gigaam_vad import merge_speech_intervals
from automatic_lecture_tex.knowledge_integrity import GeneratedLectureObservation
from automatic_lecture_tex.latex import render_lecture, write_course_tex
from automatic_lecture_tex.schemas import (
    BlockType,
    ChunkNotes,
    ExtractedFrame,
    LectureIR,
    NoteBlock,
    ObservationKind,
    SourceStatus,
    VisualEvidence,
)
from automatic_lecture_tex.tex_safety import normalize_heading_math, normalize_math_spans
from automatic_lecture_tex.visual_formula_gate import find_formula_gate_violations


def test_gigaam_vad_merges_short_neighbors_without_crossing_limit():
    intervals = [(1.0, 5.0), (5.2, 8.0), (10.0, 35.0)]
    merged = merge_speech_intervals(
        intervals,
        duration=40.0,
        max_seconds=22.0,
        merge_gap_seconds=0.35,
        pad_seconds=0.1,
    )

    assert merged[0][0] == 0.9
    assert merged[0][1] == 8.1
    assert all(0 < end - start <= 22.0 for start, end in merged)
    assert len(merged) >= 3


def _write_frame(path: Path, *, occlusion_width: int) -> None:
    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    image[38:41, 10:110] = 20
    if occlusion_width:
        image[10:70, 45 : 45 + occlusion_width] = 30
    Image.fromarray(image).save(path)


def test_least_occluded_selection_prefers_clean_raw_frame(tmp_path):
    composite = tmp_path / "composite.png"
    _write_frame(composite, occlusion_width=0)
    frames = []
    for index, width in enumerate([45, 18, 0]):
        path = tmp_path / f"frame_{index}.png"
        _write_frame(path, occlusion_width=width)
        frames.append(ExtractedFrame(timestamp=10.0 + index, path=path))

    selected = select_least_occluded_frame(frames, composite, target_timestamp=11.0)
    assert selected.path.name == "frame_2.png"


def _observation(latex: str) -> GeneratedLectureObservation:
    return GeneratedLectureObservation(
        kind=ObservationKind.CLAIM,
        text=f"Формула ${latex}$.",
        latex=latex,
        confidence=0.9,
        source_status=SourceStatus.RECONSTRUCTED,
        source_segment_ids=["seg_0"],
        visual_evidence_ids=["visual_0"],
    )


def test_visual_formula_gate_catches_symbol_and_root_mutations():
    visual = VisualEvidence(
        request_id="visual_0",
        kind="equation",
        latex=r"\|e_n-e_m\|_2=\sqrt{2}",
        raw_latex=r"\|e_n-e_m\|_2=\sqrt{2}",
        confidence=0.95,
    )

    assert not find_formula_gate_violations(
        [_observation(r"\|e_n-e_m\|_2=\sqrt{2}")],
        [visual],
    )
    violations = find_formula_gate_violations(
        [_observation(r"\|e_n-e_m\|_2=2")],
        [visual],
    )
    assert len(violations) == 1
    assert "sqrt" in violations[0].reason


def test_visual_formula_gate_catches_variable_substitution():
    visual = VisualEvidence(
        request_id="visual_0",
        kind="equation",
        latex=r"v(ix)=u(x)\\v(x)=-u(ix)",
        confidence=0.9,
    )
    violations = find_formula_gate_violations(
        [_observation(r"v(x)=-u(ix)")],
        [visual],
    )
    assert not violations
    violations = find_formula_gate_violations(
        [_observation(r"u(x)=-u(ix)")],
        [visual],
    )
    assert len(violations) == 1


def test_tex_safety_repairs_double_dollars_unicode_and_serialization_damage():
    source = r"Имеем $$φ(x) \tleq varepsilon$$, далее Φ ∈ X."
    result = normalize_math_spans(source)
    assert "$$" not in result
    assert r"\varphi" in result
    assert r"\leq" in result
    assert r"\varepsilon" in result
    assert r"\(\Phi\)" in result
    assert r"\(\in\)" in result


def test_heading_math_commands_are_wrapped_in_math_mode():
    result = normalize_heading_math(r"Определение функционала \varphi\_x при ε")
    assert r"$\varphi_x$" in result
    assert r"$\varepsilon$" in result


def test_renderer_keeps_audit_out_of_tex_and_writes_sidecar(tmp_path):
    ir = LectureIR(
        lecture_id="lecture_01",
        title="Лекция 1",
        chunks=[
            ChunkNotes(
                chunk_id="chunk_0",
                start=0.0,
                end=10.0,
                section_title="Раздел",
                blocks=[NoteBlock(type=BlockType.PARAGRAPH, latex="Текст.")],
                unresolved=["сомнительный фрагмент"],
            )
        ],
    )

    rendered = render_lecture(ir)
    assert "сомнительный фрагмент" not in rendered
    assert "% Unresolved" not in rendered

    write_course_tex("Курс", [ir], tmp_path)
    audit_path = tmp_path / "audit" / "lecture_01.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["chunks"][0]["unresolved"] == ["сомнительный фрагмент"]
