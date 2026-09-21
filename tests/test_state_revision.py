from pathlib import Path

from automatic_lecture_tex.config import NotesConfig
from automatic_lecture_tex.schemas import (
    ClaimStatus,
    EpisodeStatus,
    KnowledgeClaim,
    LectureKnowledgeBase,
    LectureObservation,
    ObservationKind,
    SemanticEpisode,
    SourceStatus,
    StateObservationRevision,
    StateReviewCandidate,
    StateReviewPlan,
    Transcript,
    TranscriptSegment,
)
from automatic_lecture_tex.state_revision import run_global_state_revision
from automatic_lecture_tex.util import atomic_json_dump


class _RevisionOrchestrator:
    def __init__(self, config, *, revision):
        self.config = config
        self.revision = revision
        self.calls = []

    def _structured(
        self,
        prompt,
        schema,
        *,
        operation,
        max_tokens=None,
        images=None,
        guided_json=True,
    ):
        self.calls.append(
            {
                "operation": operation,
                "prompt": prompt,
                "images": images,
                "guided_json": guided_json,
            }
        )
        if schema is StateReviewPlan:
            return StateReviewPlan(
                candidates=[
                    StateReviewCandidate(
                        observation_id="obs_bad",
                        reason="Contradicts the later Riesz-identification observation.",
                    )
                ]
            )
        if schema is StateObservationRevision:
            return self.revision
        raise AssertionError(schema)


def _kb():
    bad = LectureObservation(
        id="obs_bad",
        window_id="window_0001",
        window_ids=["window_0001"],
        episode_id="episode_0001",
        start=100.0,
        end=110.0,
        kind=ObservationKind.CLAIM,
        text="Пространство l_2^* несепарабельно.",
        confidence=0.92,
        source_status=SourceStatus.RECONSTRUCTED,
        evidence_refs=["seg_bad", "window_0001__board_scan"],
    )
    later = LectureObservation(
        id="obs_later",
        window_id="window_0002",
        window_ids=["window_0002"],
        episode_id="episode_0001",
        start=150.0,
        end=160.0,
        kind=ObservationKind.CLAIM,
        text="По теореме Рисса l_2^* изометрически изоморфно l_2.",
        confidence=0.96,
        source_status=SourceStatus.RECONSTRUCTED,
        evidence_refs=["seg_later"],
    )
    claim = KnowledgeClaim(
        id="claim_obs_bad",
        kind=ObservationKind.CLAIM,
        content=bad.text,
        episode_id="episode_0001",
        scope="episode_0001",
        source_status=SourceStatus.RECONSTRUCTED,
        evidence_ids=["obs_bad"],
        introduced_at=100.0,
    )
    episode = SemanticEpisode(
        id="episode_0001",
        title="Теорема Рисса",
        start=100.0,
        end=160.0,
        status=EpisodeStatus.CLOSED,
        observation_ids=["obs_bad", "obs_later"],
        claim_ids=["claim_obs_bad"],
        window_ids=["window_0001", "window_0002"],
    )
    return LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=[bad, later],
        claims=[claim],
        episodes=[episode],
    )


def _transcript():
    return Transcript(
        lecture_id="lecture",
        language="ru",
        segments=[
            TranscriptSegment(
                id="seg_bad",
                start=100.0,
                end=110.0,
                text="сопряженное пространство эл два",
            ),
            TranscriptSegment(
                id="seg_later",
                start=150.0,
                end=160.0,
                text="по теореме риса оно изоморфно эл два",
            ),
        ],
    )


def _write_window(work: Path, image: Path):
    atomic_json_dump(
        work / "knowledge_windows" / "window_0001.json",
        {
            "visual_evidence": [
                {
                    "request_id": "window_0001__board_scan",
                    "kind": "board_scan",
                    "frame_paths": [str(image)],
                    "frame_timestamps": [105.0],
                }
            ]
        },
    )


def test_global_state_revision_rechecks_candidate_with_original_board_image(tmp_path):
    image = tmp_path / "board.jpg"
    image.write_bytes(b"board")
    _write_window(tmp_path, image)
    config = NotesConfig(
        state_revision_enabled=True,
        state_revision_apply_threshold=0.88,
        state_revision_max_candidates=5,
    )
    orchestrator = _RevisionOrchestrator(
        config,
        revision=StateObservationRevision(
            action="replace",
            replacement_text="По теореме Рисса пространство l_2^* изометрически изоморфно l_2.",
            replacement_latex=r"l_2^* \cong l_2",
            reason="Later statement and board evidence resolve the earlier reconstruction.",
            confidence=0.97,
        ),
    )

    revised, stats = run_global_state_revision(
        orchestrator,
        _kb(),
        _transcript(),
        tmp_path,
        force=True,
    )

    target = next(item for item in revised.observations if item.id == "obs_bad")
    assert target.text.startswith("По теореме Рисса")
    assert target.latex == r"l_2^* \cong l_2"
    assert revised.claims[0].content == target.text
    targeted = next(item for item in orchestrator.calls if item["operation"] == "state_targeted_revision")
    assert targeted["images"] == [image]
    assert targeted["guided_json"] is False
    assert stats["replaced"] == 1


def test_unresolved_global_revision_removes_observation_from_canonical_episode(tmp_path):
    config = NotesConfig(
        state_revision_enabled=True,
        state_revision_apply_threshold=0.88,
        state_revision_max_candidates=5,
    )
    orchestrator = _RevisionOrchestrator(
        config,
        revision=StateObservationRevision(
            action="unresolved",
            reason="Original evidence does not determine the claimed separability statement.",
            confidence=0.95,
        ),
    )

    revised, stats = run_global_state_revision(
        orchestrator,
        _kb(),
        _transcript(),
        tmp_path,
        force=True,
    )

    assert "obs_bad" not in {item.id for item in revised.observations}
    assert "obs_bad" not in revised.episodes[0].observation_ids
    assert revised.claims[0].status == ClaimStatus.UNRESOLVED
    assert stats["unresolved"] == 1
