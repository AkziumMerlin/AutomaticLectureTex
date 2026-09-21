from pathlib import Path

from automatic_lecture_tex.config import load_config
from automatic_lecture_tex.knowledge import make_lecture_state
from automatic_lecture_tex.knowledge_pipeline import _state_section_batches
from automatic_lecture_tex.schemas import (
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
    assert config.vision.board_change_min_gap_seconds == 3.0
    assert config.latex.compile is False
    assert config.latex.output_dir.name == "functional_analysis_vk_20s"
