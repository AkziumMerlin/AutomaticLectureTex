import json
from pathlib import Path

import pytest

from automatic_lecture_tex import knowledge_pipeline as knowledge_pipeline_module
from automatic_lecture_tex import pipeline_robust as pipeline_robust_module
from automatic_lecture_tex.config import NotesConfig, load_config
from automatic_lecture_tex.episode_graph import apply_episode_tracking
from automatic_lecture_tex.generated_notes import (
    GeneratedChunkNotes,
    GeneratedObservationStatePatch,
)
from automatic_lecture_tex.knowledge import make_lecture_state
from automatic_lecture_tex.llm import StructuredTaskTooLargeError
from automatic_lecture_tex.knowledge_pipeline import (
    _split_state_section_evidence_by_observations,
    _state_section_batches,
    _write_state_section_batch_resilient,
)
from automatic_lecture_tex.schemas import (
    ChunkNotes,
    EpisodeStatus,
    EpisodeTrackingUpdate,
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
    SourceStatus,
    SymbolRecord,
    Transcript,
    TranscriptSegment,
)


def test_state_ir_fingerprint_depends_on_state_pipeline_version(monkeypatch):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "functional_analysis_vk_lecture01_state.yaml"
    )
    config = load_config(config_path)
    pipeline = pipeline_robust_module.Pipeline(config)
    transcript = Transcript(
        lecture_id="lecture",
        segments=[
            TranscriptSegment(
                id="seg_0",
                start=0.0,
                end=1.0,
                text="test",
            )
        ],
    )

    before = pipeline._ir_fingerprint(transcript, {})
    monkeypatch.setattr(
        pipeline_robust_module,
        "STATE_PIPELINE_VERSION",
        pipeline_robust_module.STATE_PIPELINE_VERSION + 1,
    )
    after = pipeline._ir_fingerprint(transcript, {})

    assert before != after


def test_functional_analysis_state_config_uses_qwen3_asr_and_change_sampling():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "functional_analysis_vk_lecture01_state.yaml"
    )
    config = load_config(config_path)

    assert config.asr.backend == "qwen3"
    assert config.asr.model == "Qwen/Qwen3-ASR-1.7B"
    assert config.asr.aligner_model == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert config.notes.architecture == "state"
    assert config.notes.global_validation is False
    assert config.vision.board_sampling_mode == "change"
    assert config.notes.visual_chunk_board_scan is True
    assert config.notes.state_observation_history == 4
    assert config.notes.state_observation_max_images == 2


def test_lecture_state_is_projection_of_semantic_state():
    observation = LectureObservation(
        id="obs_1",
        window_id="window",
        start=0.0,
        end=1.0,
        kind=ObservationKind.CLAIM,
        text="Claim",
        source_status=SourceStatus.OBSERVED,
        evidence_refs=["seg_1"],
    )
    episode = SemanticEpisode(
        id="episode_1",
        title="Topic",
        start=0.0,
        end=1.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=["obs_1"],
    )
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=[observation],
        episodes=[episode],
    )
    outline = LectureOutline(
        sections=[
            OutlineSection(
                id="section_1",
                title="Topic",
                start=0.0,
                end=1.0,
                episode_ids=["episode_1"],
            )
        ]
    )

    state = make_lecture_state(kb, outline=outline)

    assert state.lecture_id == "lecture"
    assert [item.id for item in state.episodes] == ["episode_1"]
    assert state.outline is not None
    assert state.outline.sections[0].episode_ids == ["episode_1"]


def test_state_section_batches_do_not_reintroduce_raw_asr():
    observation = LectureObservation(
        id="obs_1",
        window_id="window",
        start=0.0,
        end=1.0,
        kind=ObservationKind.CLAIM,
        text="Canonical mathematical statement",
        source_status=SourceStatus.RECONSTRUCTED,
        evidence_refs=["seg_1"],
    )
    episode = SemanticEpisode(
        id="episode_1",
        title="Topic",
        start=0.0,
        end=1.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=["obs_1"],
    )
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=[observation],
        episodes=[episode],
    )
    section = OutlineSection(
        id="section_1",
        title="Topic",
        start=0.0,
        end=1.0,
        episode_ids=["episode_1"],
    )
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_1",
                start=0.0,
                end=1.0,
                text="bad raw asr variant",
            )
        ],
    )

    class Config:
        state_section_max_evidence_chars = 28000
        boundary_context_seconds = 0.0

    batches = _state_section_batches(kb, section, transcript, Config())

    assert len(batches) == 1
    assert "transcript" not in batches[0]
    assert batches[0]["observations"][0]["text"] == "Canonical mathematical statement"
    assert "bad raw asr variant" not in str(batches[0])





def test_state_writer_raw_context_is_bounded_bidirectional_and_keeps_literal_ocr():
    evidence = {
        "section": {"start": 100.0, "end": 200.0},
        "episodes": [{"id": "episode_5", "window_ids": ["window_far"]}],
        "observations": [
            {
                "id": "obs_5",
                "window_id": "window_5",
                "window_ids": ["window_5"],
                "start": 100.0,
                "end": 110.0,
            }
        ],
    }
    raw_windows = [
        {
            "window_id": "window_4",
            "start": 80.0,
            "end": 90.0,
            "asr": "before",
            "visual_latex": [],
            "math_ocr_candidates": [],
        },
        {
            "window_id": "window_5",
            "start": 100.0,
            "end": 110.0,
            "asr": "current",
            "visual_latex": [],
            "math_ocr_candidates": [
                {
                    "timestamp": 105.0,
                    "text": r"y=x-\frac{f(x)}{f(z_f)}z_f\in\ker f",
                    "source_id": "ocr_1",
                }
            ],
        },
        {
            "window_id": "window_6",
            "start": 130.0,
            "end": 140.0,
            "asr": "after",
            "visual_latex": [],
            "math_ocr_candidates": [
                {
                    "timestamp": 135.0,
                    "text": r"f(z_f)\neq 0",
                    "source_id": "ocr_2",
                }
            ],
        },
        {
            "window_id": "window_far",
            "start": 500.0,
            "end": 510.0,
            "asr": "unrelated future material",
            "visual_latex": [],
            "math_ocr_candidates": [],
        },
    ]
    config = NotesConfig(
        state_section_raw_context_seconds=40.0,
        state_section_raw_evidence_chars=16000,
    )

    context = knowledge_pipeline_module._state_raw_evidence_context(
        evidence,
        raw_windows,
        config,
    )

    assert [item["window_id"] for item in context] == [
        "window_4",
        "window_5",
        "window_6",
    ]
    assert next(item for item in context if item["window_id"] == "window_5")["direct"] is True
    future = next(item for item in context if item["window_id"] == "window_6")
    assert future["math_ocr_candidates"][0]["text"] == r"f(z_f)\neq 0"
    assert "unrelated future material" not in str(context)


def test_state_writer_consumes_resolved_state_without_raw_ocr():
    class FakeOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.prompt = ""

        def _structured(self, prompt, schema, **kwargs):
            del schema, kwargs
            self.prompt = prompt
            return GeneratedChunkNotes(section_title="Topic", blocks=[])

    orchestrator = FakeOrchestrator()
    section = OutlineSection(
        id="section_0",
        title="Topic",
        start=0.0,
        end=1.0,
        episode_ids=[],
    )
    raw_context = [
        {
            "window_id": "window_0",
            "start": 0.0,
            "end": 1.0,
            "asr": "",
            "visual_latex": [],
            "math_ocr_candidates": [
                {"timestamp": 0.5, "text": r"f(z_f)\neq0", "source_id": "ocr"}
            ],
            "direct": True,
        }
    ]

    knowledge_pipeline_module._write_state_section_batch(
        orchestrator,
        section,
        {"claims": [], "observations": [], "symbols": [], "episodes": []},
        outline_context=[],
        previous_context=[],
        raw_evidence_context=raw_context,
    )

    assert "Sequentially resolved state evidence" in orchestrator.prompt
    assert "primary local hypothesis" in orchestrator.prompt
    assert "CANONICAL MATH ATOM" in orchestrator.prompt
    assert "source_evidence_ids" in orchestrator.prompt
    assert r"f(z_f)\\neq0" not in orchestrator.prompt


def test_writer_masks_and_restores_resolved_math_atoms():
    evidence = {
        "observations": [
            {
                "id": "obs_formula",
                "text": "source",
                "latex": r"x=bad",
                "resolved_text": "resolved formula",
                "resolved_latex": r"x=\sum_{n=1}^{\infty} x_n",
            }
        ]
    }

    masked, atoms = knowledge_pipeline_module._writer_evidence_with_math_atoms(evidence)
    token = masked["observations"][0]["latex"]

    assert token.startswith("MATHATOM__")
    assert r"\sum" not in str(masked)
    assert atoms[token] == r"x=\sum_{n=1}^{\infty} x_n"

    generated = GeneratedChunkNotes(
        section_title="Topic",
        blocks=[
            {
                "type": "paragraph",
                "latex": f"Получаем {token}.",
                "source_evidence_ids": ["obs_formula"],
            }
        ],
    )
    knowledge_pipeline_module._restore_generated_math_atoms(generated, atoms)

    assert generated.blocks[0].latex == r"Получаем $x=\sum_{n=1}^{\infty} x_n$."


def test_state_patch_schema_is_transactional():
    keep = GeneratedObservationStatePatch(action="keep")
    assert keep.replacement_text is None
    assert keep.evidence_refs == []

    replace = GeneratedObservationStatePatch(
        action="replace",
        replacement_text="correct",
        replacement_latex=r"f(z_f)\neq0",
        evidence_refs=["ocr:crop_1"],
        reason="visible formula",
    )
    assert replace.action == "replace"
    assert replace.replacement_latex == r"f(z_f)\neq0"

    with pytest.raises(ValueError):
        GeneratedObservationStatePatch(
            action="replace",
            replacement_text="correct",
            replacement_latex=r"f(z_f)\neq0",
            reason="missing provenance",
        )


def test_state_patch_replace_applies_without_similarity_gate():
    original = {
        "id": "obs_1",
        "text": "old prose",
        "latex": r"f(z_f)=0",
    }
    patch = GeneratedObservationStatePatch(
        action="replace",
        replacement_text="nonzero denominator",
        replacement_latex=r"f(z_f)\neq0",
        evidence_refs=["visual:crop:crop_1", "history:obs_0"],
        reason="crop visibly contains neq",
    )

    resolved, accepted, issue = knowledge_pipeline_module._apply_observation_state_patch(
        original,
        patch,
        allowed_evidence_refs={"visual:crop:crop_1", "history:obs_0"},
        direct_evidence_refs={"visual:crop:crop_1"},
    )

    assert accepted is True
    assert issue is None
    assert resolved["text"] == "old prose"
    assert resolved["latex"] == r"f(z_f)=0"
    assert resolved["resolved_text"] == "nonzero denominator"
    assert resolved["resolved_latex"] == r"f(z_f)\neq0"
    assert resolved["resolution_status"] == "replaced"


def test_state_patch_cannot_change_state_from_history_only():
    original = {"id": "obs_1", "text": "claim", "latex": r"x=y"}
    patch = GeneratedObservationStatePatch(
        action="replace",
        replacement_text="different claim",
        replacement_latex=r"x\neq y",
        evidence_refs=["history:obs_0"],
        reason="history disagrees",
    )

    resolved, accepted, issue = knowledge_pipeline_module._apply_observation_state_patch(
        original,
        patch,
        allowed_evidence_refs={"history:obs_0"},
        direct_evidence_refs=set(),
    )

    assert accepted is False
    assert "direct local evidence" in issue
    assert resolved["resolution_status"] == "unresolved"
    assert resolved["resolved_text"] is None
    assert resolved["resolved_latex"] is None


def test_state_patch_reject_suppresses_current_from_canonical_state():
    original = {"id": "obs_bad", "text": "unsupported", "latex": r"0=1"}
    patch = GeneratedObservationStatePatch(
        action="reject",
        evidence_refs=["asr:window_1"],
        reason="ASR does not support this generated equation",
    )

    resolved, accepted, issue = knowledge_pipeline_module._apply_observation_state_patch(
        original,
        patch,
        allowed_evidence_refs={"asr:window_1"},
        direct_evidence_refs={"asr:window_1"},
    )

    assert accepted is True
    assert issue is None
    assert resolved["resolution_status"] == "rejected"
    assert resolved["resolved_text"] is None
    assert resolved["resolved_latex"] is None


def test_state_resolver_uses_compact_local_state_and_exact_visual_crop(tmp_path):
    crop = tmp_path / "formula.jpg"
    board = tmp_path / "board.jpg"
    crop.write_bytes(b"crop")
    board.write_bytes(b"board")

    class FakeOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.prompt = ""
            self.images = None
            self.guided_json = None
            self.max_tokens = None

        def _structured(self, prompt, schema, **kwargs):
            self.prompt = prompt
            self.images = kwargs.get("images")
            self.guided_json = kwargs.get("guided_json")
            self.max_tokens = kwargs.get("max_tokens")
            return schema(action="keep")

    orchestrator = FakeOrchestrator()
    current = {
        "id": "obs_current",
        "episode_id": "episode_1",
        "kind": "equation",
        "text": "nonzero denominator",
        "latex": r"f(z_f)\neq0",
        "start": 10.0,
        "end": 11.0,
        "window_id": "window_1",
        "window_ids": ["window_1"],
        "unused_metadata": "must not reach resolver",
    }
    history = [
        {
            "id": f"obs_{index}",
            "kind": "claim",
            "text": f"history_{index}",
            "latex": None,
            "resolved_text": f"resolved_history_{index}",
            "huge_metadata": "x" * 2000,
        }
        for index in range(10)
    ]
    raw_windows = [
        {
            "window_id": "window_1",
            "start": 0.0,
            "end": 20.0,
            "asr": "local ASR",
            "visual_latex": [],
            "math_ocr_candidates": [
                {
                    "timestamp": 10.5,
                    "text": r"f(z_f)\neq0",
                    "source_id": "crop_exact",
                }
            ],
            "formula_crops": [
                {
                    "id": "crop_exact",
                    "timestamp": 10.5,
                    "image_path": str(crop),
                    "detector_confidence": 0.9,
                }
            ],
            "board_frames": [
                {"timestamp": 10.0, "image_path": str(board)}
            ],
        }
    ]
    evidence = {
        "claims": [{"id": "claim_that_should_not_be_sent", "content": "duplicate"}],
        "symbols": [
            {
                "symbol": "z_f",
                "meaning": "Riesz witness",
                "type_hint": "H",
                "introduced_at": 1.0,
            },
            {
                "symbol": "unrelated_symbol",
                "meaning": "irrelevant",
                "type_hint": None,
                "introduced_at": 2.0,
            },
        ],
        "episodes": [
            {"id": "episode_1", "title": "Riesz proof", "kind": "proof"}
        ],
    }
    config = NotesConfig(
        state_observation_history=4,
        state_observation_lookahead=2,
        state_observation_max_raw_windows=2,
        state_observation_max_images=2,
    )

    knowledge_pipeline_module._resolve_single_state_observation(
        orchestrator,
        section=OutlineSection(
            id="section_1",
            title="Riesz",
            start=0.0,
            end=20.0,
            episode_ids=["episode_1"],
        ),
        evidence=evidence,
        current=current,
        lookahead=[],
        resolved_history=history,
        raw_windows=raw_windows,
        config=config,
    )

    assert orchestrator.images == [crop, board]
    assert orchestrator.guided_json is False
    assert orchestrator.max_tokens == 512
    assert "claim_that_should_not_be_sent" not in orchestrator.prompt
    assert "unused_metadata" not in orchestrator.prompt
    assert "huge_metadata" not in orchestrator.prompt
    assert "resolved_history_5" not in orchestrator.prompt
    assert "resolved_history_6" in orchestrator.prompt
    assert "resolved_history_9" in orchestrator.prompt
    assert "visual:crop:crop_exact" in orchestrator.prompt
    assert len(orchestrator.prompt) < 9000


def test_resolver_raw_prompt_does_not_leak_visual_paths(tmp_path):
    raw = [
        {
            "window_id": "window_1",
            "asr": "speech",
            "visual_latex": [r"x=y"],
            "math_ocr_candidates": [
                {"timestamp": 1.0, "text": r"x=y", "source_id": "crop_1"}
            ],
            "formula_crops": [
                {"id": "crop_1", "image_path": str(tmp_path / "secret.jpg")}
            ],
            "board_frames": [
                {"image_path": str(tmp_path / "board.jpg"), "timestamp": 1.0}
            ],
        }
    ]

    prompt_raw = knowledge_pipeline_module._resolver_raw_prompt_windows(raw)

    serialized = json.dumps(prompt_raw)
    assert "secret.jpg" not in serialized
    assert "board.jpg" not in serialized
    assert prompt_raw[0]["math_ocr_candidates"][0]["source_id"] == "crop_1"


def test_observation_resolver_uses_transaction_schema():
    class FakeOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.schema = None

        def _structured(self, prompt, schema, **kwargs):
            del prompt, kwargs
            self.schema = schema
            return schema(action="keep")

    orchestrator = FakeOrchestrator()
    current = {
        "id": "obs_1",
        "text": "equation",
        "latex": r"x=y",
        "start": 0.0,
        "end": 1.0,
        "window_id": "window_1",
        "window_ids": ["window_1"],
    }

    knowledge_pipeline_module._resolve_single_state_observation(
        orchestrator,
        section=OutlineSection(id="section_1", title="Topic", start=0.0, end=1.0),
        evidence={"claims": [], "symbols": []},
        current=current,
        lookahead=[],
        resolved_history=[],
        raw_windows=[],
        config=NotesConfig(),
    )

    assert orchestrator.schema is GeneratedObservationStatePatch


def test_raw_windows_for_sequential_resolution_use_current_only():
    current = {
        "id": "obs_1",
        "window_id": "window_1",
        "window_ids": ["window_1"],
        "start": 10.0,
        "end": 20.0,
    }
    lookahead = [
        {
            "id": "obs_2",
            "window_id": "window_2",
            "window_ids": ["window_2"],
            "start": 20.0,
            "end": 30.0,
        }
    ]
    raw_windows = [
        {"window_id": "window_0", "start": 0.0, "end": 10.0},
        {"window_id": "window_1", "start": 10.0, "end": 20.0},
        {"window_id": "window_2", "start": 20.0, "end": 30.0},
        {"window_id": "window_3", "start": 30.0, "end": 40.0},
    ]

    selected = knowledge_pipeline_module._raw_windows_for_observation_sequence(
        current,
        lookahead,
        raw_windows,
        max_windows=6,
    )

    assert [item["window_id"] for item in selected] == ["window_1"]
    assert selected[0]["role"] == "current"


def test_current_formula_ocr_filter_rejects_next_proof_step():
    current = {"latex": r"\|f\| \leq \|y_f\|"}
    candidates = [
        {
            "text": (
                r"| f ( \frac { y _ { f } } { \parallel y _ { f } \parallel } ) | = "
                r"\parallel y _ { f } \parallel \leq \parallel f \parallel"
            )
        },
        {"text": r"| | f | | = | | y + 1 | |"},
    ]

    filtered = knowledge_pipeline_module._filter_current_window_ocr_candidates(
        current,
        candidates,
    )

    assert filtered == []


def test_sequential_state_patches_use_accepted_history_without_mutating_source(tmp_path):
    prompts = []

    class FakeOrchestrator:
        output_language = "ru"

        def _structured(self, prompt, schema, **kwargs):
            del schema, kwargs
            prompts.append(prompt)
            if len(prompts) == 1:
                return GeneratedObservationStatePatch(
                    action="replace",
                    replacement_text="First resolved",
                    replacement_latex=r"f(z_f)\neq0",
                    evidence_refs=["ocr:ocr_1"],
                    reason="direct OCR",
                )
            return GeneratedObservationStatePatch(
                action="replace",
                replacement_text="Second resolved",
                replacement_latex=r"y=x-\frac{f(x)}{f(z_f)}z_f",
                evidence_refs=["ocr:ocr_2"],
                reason="direct OCR",
            )

    evidence = {
        "observations": [
            {
                "id": "obs_1",
                "window_id": "window_1",
                "window_ids": ["window_1"],
                "episode_id": "episode_1",
                "start": 0.0,
                "end": 1.0,
                "kind": "claim",
                "text": "First",
                "latex": r"f(z_f)=0",
            },
            {
                "id": "obs_2",
                "window_id": "window_2",
                "window_ids": ["window_2"],
                "episode_id": "episode_1",
                "start": 1.0,
                "end": 2.0,
                "kind": "equation",
                "text": "Second",
                "latex": r"y=x-z_f",
            },
        ],
        "claims": [],
        "symbols": [],
        "episodes": [{"id": "episode_1", "observation_ids": ["obs_1", "obs_2"]}],
    }
    section = OutlineSection(
        id="section_1",
        title="Proof",
        start=0.0,
        end=2.0,
        episode_ids=["episode_1"],
    )
    config = NotesConfig(
        state_observation_lookahead=1,
        state_observation_history=4,
        state_observation_max_raw_windows=2,
        state_observation_max_images=0,
    )
    history = []

    resolved, corrections, unresolved, cache_hits = (
        knowledge_pipeline_module._resolve_state_batch_sequential(
            FakeOrchestrator(),
            section=section,
            evidence=evidence,
            section_observations=list(evidence["observations"]),
            resolved_history=history,
            raw_windows=[
                {
                    "window_id": "window_1",
                    "start": 0.0,
                    "end": 1.0,
                    "asr": "",
                    "visual_latex": [],
                    "math_ocr_candidates": [
                        {"text": r"f(z_f)\neq0", "source_id": "ocr_1"}
                    ],
                },
                {
                    "window_id": "window_2",
                    "start": 1.0,
                    "end": 2.0,
                    "asr": "",
                    "visual_latex": [],
                    "math_ocr_candidates": [
                        {
                            "text": r"y=x-\frac{f(x)}{f(z_f)}z_f",
                            "source_id": "ocr_2",
                        }
                    ],
                },
            ],
            work=tmp_path,
            config=config,
            llm_config={},
            force=True,
        )
    )

    assert cache_hits == 0
    assert corrections == []
    assert unresolved == []
    assert [item["latex"] for item in resolved["observations"]] == [
        r"f(z_f)=0",
        r"y=x-z_f",
    ]
    assert [item["resolved_latex"] for item in resolved["observations"]] == [
        r"f(z_f)\neq0",
        r"y=x-\frac{f(x)}{f(z_f)}z_f",
    ]
    assert [item["resolution_status"] for item in resolved["observations"]] == [
        "replaced",
        "replaced",
    ]
    assert len(history) == 2
    assert r"f(z_f)\\neq0" in prompts[1]
    assert "Accepted state immediately before CURRENT" in prompts[1]


def test_resolved_episode_split_preserves_resolved_observation_values():
    evidence = {
        "episodes": [
            {"id": "episode_1", "observation_ids": ["obs_1"]},
            {"id": "episode_2", "observation_ids": ["obs_2"]},
        ],
        "observations": [
            {
                "id": "obs_1",
                "episode_id": "episode_1",
                "start": 0.0,
                "end": 1.0,
                "text": "source one",
                "latex": r"f(z_f)=0",
                "resolved_text": "resolved one",
                "resolved_latex": r"f(z_f)\neq0",
                "sequentially_resolved": True,
            },
            {
                "id": "obs_2",
                "episode_id": "episode_2",
                "start": 1.0,
                "end": 2.0,
                "text": "source two",
                "latex": r"y=x-z_f",
                "resolved_text": "resolved two",
                "resolved_latex": r"y=x-\frac{f(x)}{f(z_f)}z_f",
                "sequentially_resolved": True,
            },
        ],
        "claims": [],
        "symbols": [],
        "sequential_resolution": {
            "resolved_observation_ids": ["obs_1", "obs_2"],
        },
    }

    child = knowledge_pipeline_module._subset_state_evidence_by_episode_ids(
        evidence,
        ["episode_2"],
    )

    assert [item["id"] for item in child["observations"]] == ["obs_2"]
    assert child["observations"][0]["latex"] == r"y=x-z_f"
    assert child["observations"][0]["resolved_latex"] == r"y=x-\frac{f(x)}{f(z_f)}z_f"
    assert child["observations"][0]["sequentially_resolved"] is True
    assert child["sequential_resolution"]["resolved_observation_ids"] == ["obs_2"]


def test_episode_tracking_derives_symbol_introduced_at_from_evidence():
    observation = LectureObservation(
        id="obs_1",
        window_id="window_1",
        start=12.5,
        end=13.0,
        kind=ObservationKind.NOTATION,
        text="Introduce z_f",
        source_status=SourceStatus.OBSERVED,
    )
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=[observation],
    )
    update = EpisodeTrackingUpdate(
        symbols=[
            SymbolRecord(
                symbol="z_f",
                meaning="chosen vector",
                evidence_ids=["obs_1"],
            )
        ]
    )

    apply_episode_tracking(kb, update, ["obs_1"], window_id="window_1")

    assert len(kb.symbols) == 1
    assert kb.symbols[0].introduced_at == 12.5


def test_state_section_evidence_excludes_symbols_introduced_after_batch():
    observation = LectureObservation(
        id="obs_now",
        window_id="window_now",
        start=5.0,
        end=10.0,
        kind=ObservationKind.CLAIM,
        text="Current claim",
        source_status=SourceStatus.OBSERVED,
        episode_id="episode_now",
    )
    episode = SemanticEpisode(
        id="episode_now",
        title="Now",
        start=5.0,
        end=10.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=["obs_now"],
    )
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=[observation],
        episodes=[episode],
        symbols=[
            SymbolRecord(
                id="past",
                symbol="x",
                meaning="current symbol",
                introduced_at=5.0,
            ),
            SymbolRecord(
                id="future",
                symbol="y_f",
                meaning="future symbol",
                episode_id="episode_future",
                introduced_at=50.0,
            ),
        ],
    )
    section = OutlineSection(
        id="section_0",
        title="Long section",
        start=0.0,
        end=100.0,
        episode_ids=["episode_now"],
    )
    transcript = Transcript(lecture_id="lecture", language="ru", segments=[])
    config = NotesConfig()

    payload = knowledge_pipeline_module._state_section_payload(
        kb,
        section,
        transcript,
        config,
    )

    assert [item["id"] for item in payload["symbols"]] == ["past"]


def test_state_section_batches_split_oversized_single_episode_by_observations():
    observations = [
        LectureObservation(
            id=f"obs_{index}",
            window_id="window_0",
            start=float(index),
            end=float(index + 1),
            kind=ObservationKind.CLAIM,
            text=("canonical evidence " + str(index) + " ") * 120,
            source_status=SourceStatus.OBSERVED,
            episode_id="episode_0",
        )
        for index in range(4)
    ]
    episode = SemanticEpisode(
        id="episode_0",
        title="Long proof",
        start=0.0,
        end=4.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=[item.id for item in observations],
    )
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=observations,
        episodes=[episode],
    )
    section = OutlineSection(
        id="section_0",
        title="Topic",
        start=0.0,
        end=4.0,
        episode_ids=["episode_0"],
    )
    transcript = Transcript(lecture_id="lecture", language="ru", segments=[])

    class Config:
        state_section_max_evidence_chars = 5000
        boundary_context_seconds = 0.0

    batches = _state_section_batches(kb, section, transcript, Config())

    assert len(batches) > 1
    assert [
        item["id"]
        for batch in batches
        for item in batch["observations"]
    ] == [item.id for item in observations]
    assert all(batch["episodes"][0]["id"] == "episode_0" for batch in batches)
    assert [batch["batch"]["index"] for batch in batches] == list(range(len(batches)))
    assert all(batch["batch"]["count"] == len(batches) for batch in batches)


def test_functional_analysis_20s_ablation_uses_fine_windows_and_five_image_budget():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "functional_analysis_vk_lecture01_state_20s.yaml"
    )
    config = load_config(config_path)

    assert config.notes.architecture == "state"
    assert config.notes.chunk_target_seconds == 20
    assert config.notes.chunk_overlap_seconds == 5
    assert config.notes.visual_chunk_board_scan is True
    assert config.notes.state_section_raw_context_seconds == 90
    assert config.notes.state_section_raw_evidence_chars == 16000
    assert config.notes.state_observation_lookahead == 2
    assert config.notes.state_observation_history == 4
    assert config.notes.state_observation_max_raw_windows == 2
    assert config.vision.board_sampling_mode == "change"
    assert config.vision.board_crop_max_vlm_images == 5
    assert config.vision.board_change_probe_seconds == 4.0
    assert config.vision.board_change_min_gap_seconds == 2.0
    assert config.vision.board_change_threshold == 0.10
    assert config.vision.formula_detection.enabled is True
    assert config.vision.formula_detection.backend == "yolov8"
    assert config.vision.formula_detection.model_path is not None
    assert config.vision.formula_detection.model_path.name == "yolo_v8_ft.pt"
    assert config.vision.math_ocr.backend == "unimumer"
    assert config.vision.math_ocr.unimumer_model == "phxember/Uni-MuMER-Qwen3.5-2B"
    assert config.vision.math_ocr.unimumer_load_in_4bit is False
    assert config.vision.math_ocr.unimumer_max_gpu_memory_gib == 6.0
    assert config.vision.math_ocr.unimumer_max_tokens == 384
    assert config.vision.math_ocr.unimumer_temperature == 0.0
    assert config.vision.math_ocr.unimumer_python_path is not None
    assert config.vision.math_ocr.board_scan_enabled is True
    assert config.vision.math_ocr.board_scan_max_images == 8
    assert config.vision.math_ocr.device == "cuda"
    assert config.vision.math_ocr.unimumer_python_path.name == "python"
    assert config.latex.compile is False
    assert config.latex.output_dir.name == "functional_analysis_vk_20s"




def test_single_episode_evidence_can_split_by_canonical_observations():
    evidence = {
        "section": {"id": "section_0"},
        "episodes": [
            {
                "id": "episode_0",
                "observation_ids": ["o0", "o1", "o2", "o3"],
                "claim_ids": ["c0", "c1"],
            }
        ],
        "observations": [
            {"id": "o0", "start": 0.0},
            {"id": "o1", "start": 1.0},
            {"id": "o2", "start": 2.0},
            {"id": "o3", "start": 3.0},
        ],
        "claims": [
            {"id": "c0", "evidence_ids": ["o0", "o1"]},
            {"id": "c1", "evidence_ids": ["o2", "o3"]},
        ],
        "symbols": [{"id": "s0", "symbol": "x"}],
    }

    split = _split_state_section_evidence_by_observations(evidence)

    assert split is not None
    left, right = split
    assert [item["id"] for item in left["observations"]] == ["o0", "o1"]
    assert [item["id"] for item in right["observations"]] == ["o2", "o3"]
    assert [item["id"] for item in left["claims"]] == ["c0"]
    assert [item["id"] for item in right["claims"]] == ["c1"]
    assert left["episodes"][0]["observation_ids"] == ["o0", "o1"]
    assert right["episodes"][0]["observation_ids"] == ["o2", "o3"]
    assert left["symbols"] == right["symbols"] == evidence["symbols"]


def test_state_section_writer_splits_episode_batch_after_context_limit(monkeypatch):
    observations = [
        LectureObservation(
            id=f"obs_{index}",
            window_id=f"window_{index}",
            start=float(index),
            end=float(index + 1),
            kind=ObservationKind.CLAIM,
            text=f"Claim {index}",
            source_status=SourceStatus.OBSERVED,
            evidence_refs=[f"seg_{index}"],
            episode_id=f"episode_{index}",
        )
        for index in range(2)
    ]
    episodes = [
        SemanticEpisode(
            id=f"episode_{index}",
            title=f"Episode {index}",
            start=float(index),
            end=float(index + 1),
            status=EpisodeStatus.CLOSED,
            observation_ids=[f"obs_{index}"],
        )
        for index in range(2)
    ]
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=observations,
        episodes=episodes,
    )
    section = OutlineSection(
        id="section_0",
        title="Topic",
        start=0.0,
        end=2.0,
        episode_ids=["episode_0", "episode_1"],
    )
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(id=f"seg_{index}", start=index, end=index + 1, text=f"raw {index}")
            for index in range(2)
        ],
    )
    config = NotesConfig(
        architecture="state",
        chunk_target_seconds=20,
        chunk_overlap_seconds=5,
    )
    evidence = knowledge_pipeline_module._state_section_payload(
        kb,
        section,
        transcript,
        config,
    )
    calls = []

    def fake_write(orchestrator, child_section, child_evidence, **kwargs):
        del orchestrator, kwargs
        episode_ids = [item["id"] for item in child_evidence["episodes"]]
        calls.append(episode_ids)
        if len(episode_ids) > 1:
            raise StructuredTaskTooLargeError("backend context limit")
        return ChunkNotes(
            chunk_id=child_section.id,
            start=child_section.start,
            end=child_section.end,
            section_title=child_section.title,
            blocks=[],
            unresolved=[f"wrote:{episode_ids[0]}"],
        )

    monkeypatch.setattr(
        knowledge_pipeline_module,
        "_write_state_section_batch",
        fake_write,
    )

    result = _write_state_section_batch_resilient(
        object(),
        section,
        evidence,
        outline_context=[],
        previous_context=[],
        kb=kb,
        transcript=transcript,
        config=config,
    )

    assert calls == [
        ["episode_0", "episode_1"],
        ["episode_0"],
        ["episode_1"],
    ]
    assert result.unresolved == ["wrote:episode_0", "wrote:episode_1"]


def test_state_section_writer_splits_episode_batch_after_structured_json_failure(
    monkeypatch,
):
    observations = [
        LectureObservation(
            id=f"obs_{index}",
            window_id=f"window_{index}",
            start=float(index),
            end=float(index + 1),
            kind=ObservationKind.CLAIM,
            text=f"Claim {index}",
            source_status=SourceStatus.OBSERVED,
            evidence_refs=[f"seg_{index}"],
            episode_id=f"episode_{index}",
        )
        for index in range(2)
    ]
    episodes = [
        SemanticEpisode(
            id=f"episode_{index}",
            title=f"Episode {index}",
            start=float(index),
            end=float(index + 1),
            status=EpisodeStatus.CLOSED,
            observation_ids=[f"obs_{index}"],
        )
        for index in range(2)
    ]
    kb = LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=observations,
        episodes=episodes,
    )
    section = OutlineSection(
        id="section_0",
        title="Topic",
        start=0.0,
        end=2.0,
        episode_ids=["episode_0", "episode_1"],
    )
    transcript = Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(id=f"seg_{index}", start=index, end=index + 1, text=f"raw {index}")
            for index in range(2)
        ],
    )
    config = NotesConfig(
        architecture="state",
        chunk_target_seconds=20,
        chunk_overlap_seconds=5,
    )
    evidence = knowledge_pipeline_module._state_section_payload(
        kb,
        section,
        transcript,
        config,
    )
    calls = []

    def fake_write(orchestrator, child_section, child_evidence, **kwargs):
        del orchestrator, kwargs
        episode_ids = [item["id"] for item in child_evidence["episodes"]]
        calls.append(episode_ids)
        if len(episode_ids) > 1:
            raise json.JSONDecodeError("Unterminated string", '{"x":"', 5)
        return ChunkNotes(
            chunk_id=child_section.id,
            start=child_section.start,
            end=child_section.end,
            section_title=child_section.title,
            blocks=[],
            unresolved=[f"wrote:{episode_ids[0]}"],
        )

    monkeypatch.setattr(
        knowledge_pipeline_module,
        "_write_state_section_batch",
        fake_write,
    )

    result = _write_state_section_batch_resilient(
        object(),
        section,
        evidence,
        outline_context=[],
        previous_context=[],
        kb=kb,
        transcript=transcript,
        config=config,
    )

    assert calls == [
        ["episode_0", "episode_1"],
        ["episode_0"],
        ["episode_1"],
    ]
    assert result.unresolved == ["wrote:episode_0", "wrote:episode_1"]


def test_load_config_preserves_unimernet_venv_python_symlink(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    worker_bin = tmp_path / "models" / "formula" / "unimernet-env" / "bin"
    worker_bin.mkdir(parents=True)
    base_python = tmp_path / "base-python"
    base_python.write_text("", encoding="utf-8")
    worker_python = worker_bin / "python"
    worker_python.symlink_to(base_python)

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        """
course:
  id: test
  title: Test
  lectures:
    - id: lecture
      source:
        type: youtube
        url: https://example.com/video
vision:
  math_ocr:
    backend: unimernet
    unimernet_config_path: ../models/formula/unimernet_small/model.yaml
    unimernet_python_path: ../models/formula/unimernet-env/bin/python
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vision.math_ocr.unimernet_python_path == worker_python.absolute()
    assert config.vision.math_ocr.unimernet_python_path != base_python.resolve()
