from automatic_lecture_tex.knowledge import (
    apply_global_validation,
    apply_knowledge_update,
    merge_window_observations,
)
from automatic_lecture_tex.schemas import (
    ChunkNotes,
    ClaimStatus,
    GlobalBlockCorrection,
    GlobalValidation,
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
    assert kb.observations[0].start == 90
    assert kb.observations[0].end == 105
    assert kb.observations[0].confidence == 0.95


def test_explicit_correction_supersedes_old_claim():
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
