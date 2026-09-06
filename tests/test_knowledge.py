from automatic_lecture_tex.episode_graph import (
    apply_episode_tracking,
    build_outline_from_episodes,
    close_open_episodes,
)
from automatic_lecture_tex.knowledge import (
    apply_global_validation,
    apply_knowledge_update,
    merge_window_observations,
)
from automatic_lecture_tex.schemas import (
    ChunkNotes,
    ClaimStatus,
    EpisodeBoundary,
    EpisodeHierarchyPlan,
    EpisodeKind,
    EpisodeTrackingUpdate,
    GlobalBlockCorrection,
    GlobalValidation,
    HierarchyBoundary,
    HierarchyLevel,
    KnowledgeClaim,
    KnowledgeUpdate,
    LectureIR,
    LectureKnowledgeBase,
    LectureObservation,
    NoteBlock,
    ObservationKind,
    WindowObservations,
)


def test_overlapping_observation_is_deduplicated():
    kb = LectureKnowledgeBase(lecture_id="l1", title="Lecture")
    first = WindowObservations(
        window_id="window_0000",
        observations=[
            LectureObservation(
                id="o1",
                start=90,
                end=100,
                kind=ObservationKind.EQUATION,
                latex=r"u(ix)=-v(x)",
                confidence=0.8,
            )
        ],
    )
    second = WindowObservations(
        window_id="window_0001",
        observations=[
            LectureObservation(
                id="o2",
                start=95,
                end=105,
                kind=ObservationKind.EQUATION,
                latex=r"u(ix) = -v(x)",
                confidence=0.95,
            )
        ],
    )

    assert merge_window_observations(kb, first) == ["o1"]
    assert merge_window_observations(kb, second) == []
    assert len(kb.observations) == 1
    assert kb.observation_aliases["o2"] == "o1"
    assert kb.observations[0].start == 90
    assert kb.observations[0].end == 105
    assert kb.observations[0].confidence == 0.95


def test_overlap_reconciliation_accepts_small_formula_variation():
    kb = LectureKnowledgeBase(lecture_id="l1", title="Lecture")
    first = WindowObservations(
        window_id="window_0000",
        observations=[
            LectureObservation(
                id="o1",
                start=10,
                end=30,
                kind=ObservationKind.PROOF_STEP,
                text="По неравенству треугольника получаем противоречие.",
                latex=r"|\varphi(x)-\varphi(y)|<2\varepsilon",
            )
        ],
    )
    second = WindowObservations(
        window_id="window_0001",
        observations=[
            LectureObservation(
                id="o2",
                start=12,
                end=31,
                kind=ObservationKind.PROOF_STEP,
                text="Из неравенства треугольника следует противоречие.",
                latex=r"|\varphi(x) - \varphi(y)| < 2 \varepsilon",
            )
        ],
    )

    merge_window_observations(kb, first)
    assert merge_window_observations(kb, second) == []
    assert len(kb.observations) == 1
    assert kb.observation_aliases["o2"] == "o1"


def test_episode_continues_across_technical_windows_by_default():
    kb = LectureKnowledgeBase(lecture_id="l1", title="Lecture")
    first = WindowObservations(
        window_id="window_0000",
        observations=[
            LectureObservation(
                id="o1",
                start=10,
                end=20,
                kind=ObservationKind.PROOF_STEP,
                text="Начало доказательства",
            )
        ],
    )
    ids = merge_window_observations(kb, first)
    apply_episode_tracking(
        kb,
        EpisodeTrackingUpdate(
            boundaries=[
                EpisodeBoundary(
                    before_observation_id="o1",
                    kind=EpisodeKind.PROOF,
                    title="Доказательство",
                )
            ]
        ),
        ids,
        window_id="window_0000",
    )

    second = WindowObservations(
        window_id="window_0001",
        observations=[
            LectureObservation(
                id="o2",
                start=21,
                end=30,
                kind=ObservationKind.PROOF_STEP,
                text="Продолжение доказательства",
            )
        ],
    )
    ids = merge_window_observations(kb, second)
    apply_episode_tracking(kb, EpisodeTrackingUpdate(), ids, window_id="window_0001")

    assert len(kb.episodes) == 1
    assert kb.observations[0].episode_id == "episode_0000"
    assert kb.observations[1].episode_id == "episode_0000"
    assert kb.episodes[0].observation_ids == ["o1", "o2"]


def test_explicit_correction_naturally_supersedes_target_claim():
    kb = LectureKnowledgeBase(lecture_id="l1", title="Lecture")
    first = WindowObservations(
        window_id="window_0000",
        observations=[
            LectureObservation(
                id="wrong",
                start=10,
                end=12,
                kind=ObservationKind.EQUATION,
                text="Первая запись",
                latex=r"f(x)=u(x)+iu(ix)",
            )
        ],
    )
    ids = merge_window_observations(kb, first)
    apply_episode_tracking(
        kb,
        EpisodeTrackingUpdate(
            boundaries=[
                EpisodeBoundary(
                    before_observation_id="wrong",
                    kind=EpisodeKind.DERIVATION,
                    title="Восстановление функционала",
                )
            ]
        ),
        ids,
        window_id="window_0000",
    )
    old_claim = kb.claims[0]
    assert old_claim.status == ClaimStatus.ACTIVE

    second = WindowObservations(
        window_id="window_0001",
        observations=[
            LectureObservation(
                id="fix",
                start=13,
                end=15,
                kind=ObservationKind.CORRECTION,
                text="Лектор исправляет знак",
                latex=r"f(x)=u(x)-iu(ix)",
                target_observation_id="wrong",
            )
        ],
    )
    ids = merge_window_observations(kb, second)
    apply_episode_tracking(kb, EpisodeTrackingUpdate(), ids, window_id="window_0001")

    assert old_claim.status == ClaimStatus.SUPERSEDED
    assert kb.claims[-1].status == ClaimStatus.ACTIVE
    assert kb.claims[-1].supersedes == [old_claim.id]
    assert kb.claims[-1].episode_id == old_claim.episode_id


def test_hierarchy_groups_fixed_episode_leaves_without_losing_content():
    kb = LectureKnowledgeBase(lecture_id="l1", title="Lecture")
    observations = [
        LectureObservation(id="o0", start=0, end=10, kind=ObservationKind.DEFINITION, text="A"),
        LectureObservation(id="o1", start=10, end=20, kind=ObservationKind.PROOF_STEP, text="B"),
        LectureObservation(id="o2", start=20, end=30, kind=ObservationKind.DEFINITION, text="C"),
    ]
    batch = WindowObservations(window_id="window_0000", observations=observations)
    ids = merge_window_observations(kb, batch)
    apply_episode_tracking(
        kb,
        EpisodeTrackingUpdate(
            boundaries=[
                EpisodeBoundary(before_observation_id="o0", kind=EpisodeKind.DEFINITION, title="A"),
                EpisodeBoundary(before_observation_id="o1", kind=EpisodeKind.PROOF, title="B"),
                EpisodeBoundary(before_observation_id="o2", kind=EpisodeKind.DEFINITION, title="C"),
            ]
        ),
        ids,
        window_id="window_0000",
    )
    close_open_episodes(kb)

    plan = EpisodeHierarchyPlan(
        boundaries=[
            HierarchyBoundary(
                before_episode_id="episode_0000",
                level=HierarchyLevel.TOPIC,
                title="Первая тема",
            ),
            HierarchyBoundary(
                before_episode_id="episode_0002",
                level=HierarchyLevel.TOPIC,
                title="Вторая тема",
            ),
        ]
    )
    sections = build_outline_from_episodes(kb, plan, lecture_title="Lecture")

    assert [item.title for item in sections] == ["Первая тема", "Вторая тема"]
    assert sections[0].episode_ids == ["episode_0000", "episode_0001"]
    assert sections[1].episode_ids == ["episode_0002"]
    assert [episode_id for section in sections for episode_id in section.episode_ids] == [
        "episode_0000",
        "episode_0001",
        "episode_0002",
    ]
    assert sections[0].start == 0
    assert sections[1].end == 30


def test_explicit_correction_supersedes_old_claim_legacy_updater():
    kb = LectureKnowledgeBase(
        lecture_id="l1",
        title="Lecture",
        claims=[
            KnowledgeClaim(
                id="claim_old",
                content="old",
                latex=r"f(x)=u(x)+iu(ix)",
            )
        ],
    )
    update = KnowledgeUpdate(
        claims=[
            KnowledgeClaim(
                content="corrected",
                latex=r"f(x)=u(x)-iu(ix)",
                supersedes=["claim_old"],
                evidence_ids=["obs_fix"],
            )
        ]
    )

    apply_knowledge_update(kb, update, window_id="window_0001")
    assert kb.claims[0].status == ClaimStatus.SUPERSEDED
    assert kb.claims[1].status == ClaimStatus.ACTIVE
    assert kb.claims[1].supersedes == ["claim_old"]


def test_global_validation_applies_only_high_confidence():
    ir = LectureIR(
        lecture_id="l1",
        title="Lecture",
        chunks=[
            ChunkNotes(
                section_title="S",
                blocks=[NoteBlock(type="paragraph", latex="wrong")],
            )
        ],
    )
    validation = GlobalValidation(
        corrections=[
            GlobalBlockCorrection(
                section_index=0,
                block_index=0,
                corrected_latex="right",
                reason="supported by later correction",
                confidence=0.9,
            )
        ]
    )

    apply_global_validation(ir, validation, threshold=0.85)
    assert ir.chunks[0].blocks[0].latex == "right"
    assert ir.chunks[0].corrections