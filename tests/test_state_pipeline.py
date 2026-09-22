import json
from pathlib import Path

from automatic_lecture_tex import knowledge_pipeline as knowledge_pipeline_module
from automatic_lecture_tex.config import NotesConfig, load_config
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
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
    SourceStatus,
    Transcript,
    TranscriptSegment,
)


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
    assert config.vision.board_sampling_mode == "change"
    assert config.vision.board_crop_max_vlm_images == 5
    assert config.vision.board_change_probe_seconds == 4.0
    assert config.vision.board_change_min_gap_seconds == 2.0
    assert config.vision.board_change_threshold == 0.10
    assert config.vision.formula_detection.enabled is True
    assert config.vision.formula_detection.backend == "yolov8"
    assert config.vision.formula_detection.model_path is not None
    assert config.vision.formula_detection.model_path.name == "yolo_v8_ft.pt"
    assert config.vision.math_ocr.backend == "unimernet"
    assert config.vision.math_ocr.board_scan_enabled is True
    assert config.vision.math_ocr.board_scan_max_images == 8
    assert config.vision.math_ocr.device == "cuda"
    assert config.vision.math_ocr.unimernet_config_path is not None
    assert config.vision.math_ocr.unimernet_config_path.name == "automatic_lecture_tex.yaml"
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
