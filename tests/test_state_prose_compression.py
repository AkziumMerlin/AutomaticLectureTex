from automatic_lecture_tex.knowledge_pipeline import _assemble_state_section_deterministically
from automatic_lecture_tex.schemas import (
    EpisodeKind,
    LectureKnowledgeBase,
    LectureObservation,
    LectureOutline,
    ObservationKind,
    OutlineSection,
    SemanticEpisode,
)
from automatic_lecture_tex.state_canonicalization import CanonicalRenderPolicy
from automatic_lecture_tex.state_prose_compression import (
    ProseCompressionGroupProposal,
    ProseCompressionPlan,
    ProseCompressionSentence,
    _candidate_runs,
    _validate_group,
    run_state_prose_compression,
)


def _obs(
    observation_id: str,
    *,
    start: float,
    kind: ObservationKind,
    text: str,
    latex: str | None = None,
) -> LectureObservation:
    return LectureObservation(
        id=observation_id,
        episode_id="episode_0000",
        start=start,
        end=start + 1.0,
        kind=kind,
        text=text,
        latex=latex,
    )


def _kb(*observations: LectureObservation) -> LectureKnowledgeBase:
    return LectureKnowledgeBase(
        lecture_id="lecture",
        title="Lecture",
        observations=list(observations),
        episodes=[
            SemanticEpisode(
                id="episode_0000",
                title="Intro",
                kind=EpisodeKind.TOPIC,
                start=0.0,
                end=100.0,
                observation_ids=[item.id for item in observations],
            )
        ],
    )


def _section() -> OutlineSection:
    return OutlineSection(
        id="topic_000",
        title="Введение",
        start=0.0,
        end=100.0,
        episode_ids=["episode_0000"],
    )


def test_candidate_runs_never_include_formula_or_formula_like_prose():
    observations = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Пространство X рассматривается как комплексное.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.REMARK,
            text="Это пространство снова называется комплексным.",
        ),
        _obs(
            "obs_math_text",
            start=3.0,
            kind=ObservationKind.REMARK,
            text="Получаем v(ix) = u(x).",
        ),
        _obs(
            "obs_formula",
            start=4.0,
            kind=ObservationKind.REMARK,
            text="Формула на доске.",
            latex="x=1",
        ),
        _obs(
            "obs_c",
            start=5.0,
            kind=ObservationKind.REMARK,
            text="Возвращаемся к комплексному случаю.",
        ),
        _obs(
            "obs_d",
            start=6.0,
            kind=ObservationKind.NOTATION,
            text="Обозначение X используется для пространства.",
        ),
    ]

    runs = _candidate_runs(observations)

    assert [[item.id for item in run] for run in runs] == [
        ["obs_a", "obs_b"],
        ["obs_c", "obs_d"],
    ]


def test_opening_narration_can_be_compressed_to_two_plain_sentences(tmp_path):
    observations = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.TRANSITION,
            text="Начинается вторая часть курса и объявляется её содержание.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.REMARK,
            text=(
                "Перечисляются нормированные пространства, сопряжённые пространства, "
                "спектр и спектральная теория операторов."
            ),
        ),
        _obs(
            "obs_c",
            start=3.0,
            kind=ObservationKind.TRANSITION,
            text="После перечисления тем начинается первый вопрос.",
        ),
        _obs(
            "obs_d",
            start=4.0,
            kind=ObservationKind.REMARK,
            text="Исходное пространство X рассматривается как комплексное пространство.",
        ),
        _obs(
            "obs_e",
            start=5.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
        _obs(
            "obs_definition",
            start=6.0,
            kind=ObservationKind.DEFINITION,
            text="Вводится линейный функционал.",
            latex=r"f:X\to\mathbb{C}",
        ),
    ]
    kb = _kb(*observations)
    outline = LectureOutline(sections=[_section()])

    plan = ProseCompressionPlan(
        groups=[
            ProseCompressionGroupProposal(
                source_observation_ids=["obs_a", "obs_b", "obs_c", "obs_d", "obs_e"],
                sentences=[
                    ProseCompressionSentence(
                        text=(
                            "Во второй части курса рассматриваются нормированные и сопряжённые "
                            "пространства, спектр и спектральная теория операторов."
                        ),
                        source_observation_ids=["obs_a", "obs_b", "obs_c"],
                    ),
                    ProseCompressionSentence(
                        text="Исходное пространство X рассматривается как комплексное.",
                        source_observation_ids=["obs_d", "obs_e"],
                    ),
                ],
            )
        ]
    )

    class StubOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.calls = 0

        def _structured(self, *args, **kwargs):
            self.calls += 1
            return plan

    orchestrator = StubOrchestrator()
    policy, stats, unresolved = run_state_prose_compression(
        orchestrator,
        kb=kb,
        outline=outline,
        work=tmp_path,
        llm_config={"model": "stub"},
        pipeline_version=10,
        max_sentences=2,
        max_ratio=0.55,
        max_summary_chars=700,
        force=False,
    )

    assert unresolved == []
    assert orchestrator.calls == 1
    assert stats["accepted_groups"] == 1
    assert stats["compressed_observations"] == 5
    assert [item.id for item in kb.observations] == [item.id for item in observations]

    notes = _assemble_state_section_deterministically(
        kb,
        _section(),
        render_policy=CanonicalRenderPolicy(),
        prose_policy=policy,
    )
    assert notes.blocks[0].latex == (
        "Во второй части курса рассматриваются нормированные и сопряжённые пространства, "
        "спектр и спектральная теория операторов. "
        "Исходное пространство X рассматривается как комплексное."
    )
    assert notes.blocks[0].source_evidence_ids == [
        "obs_a",
        "obs_b",
        "obs_c",
        "obs_d",
        "obs_e",
    ]
    assert notes.blocks[1].latex == "Вводится линейный функционал."
    assert notes.blocks[2].latex == r"f:X\to\mathbb{C}"


def test_noncontiguous_group_is_rejected():
    run = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Первое описание комплексного пространства X.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.REMARK,
            text="Отдельная содержательная деталь.",
        ),
        _obs(
            "obs_c",
            start=3.0,
            kind=ObservationKind.NOTATION,
            text="Повторяется обозначение комплексного пространства X.",
        ),
    ]
    proposal = ProseCompressionGroupProposal(
        source_observation_ids=["obs_a", "obs_c"],
        sentences=[
            ProseCompressionSentence(
                text="Пространство X рассматривается как комплексное.",
                source_observation_ids=["obs_a", "obs_c"],
            )
        ],
    )

    group, reason = _validate_group(
        proposal,
        section_id="topic_000",
        runs=[run],
        used_ids=set(),
        max_sentences=2,
        max_ratio=0.55,
        max_summary_chars=700,
    )

    assert group is None
    assert "contiguous" in reason


def test_generated_prose_cannot_introduce_math_or_reconstruction_narration():
    run = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Пространство X рассматривается как комплексное.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
    ]

    for text in [
        r"Получаем \(X=\mathbb{C}\).",
        "По OCR пространство X является комплексным.",
        "Пространство Y является комплексным.",
    ]:
        proposal = ProseCompressionGroupProposal(
            source_observation_ids=["obs_a", "obs_b"],
            sentences=[
                ProseCompressionSentence(
                    text=text,
                    source_observation_ids=["obs_a", "obs_b"],
                )
            ],
        )
        group, _ = _validate_group(
            proposal,
            section_id="topic_000",
            runs=[run],
            used_ids=set(),
            max_sentences=2,
            max_ratio=0.90,
            max_summary_chars=700,
        )
        assert group is None


def test_cached_plan_avoids_second_model_call(tmp_path):
    observations = [
        _obs(
            "obs_a",
            start=1.0,
            kind=ObservationKind.REMARK,
            text="Пространство X рассматривается как комплексное.",
        ),
        _obs(
            "obs_b",
            start=2.0,
            kind=ObservationKind.NOTATION,
            text="X обозначает комплексное пространство.",
        ),
    ]
    kb = _kb(*observations)
    outline = LectureOutline(sections=[_section()])
    plan = ProseCompressionPlan(
        groups=[
            ProseCompressionGroupProposal(
                source_observation_ids=["obs_a", "obs_b"],
                sentences=[
                    ProseCompressionSentence(
                        text="Пространство X рассматривается как комплексное.",
                        source_observation_ids=["obs_a", "obs_b"],
                    )
                ],
            )
        ]
    )

    class StubOrchestrator:
        output_language = "ru"

        def __init__(self):
            self.calls = 0

        def _structured(self, *args, **kwargs):
            self.calls += 1
            return plan

    orchestrator = StubOrchestrator()
    kwargs = dict(
        kb=kb,
        outline=outline,
        work=tmp_path,
        llm_config={"model": "stub"},
        pipeline_version=10,
        max_sentences=2,
        max_ratio=0.90,
        max_summary_chars=700,
        force=False,
    )
    run_state_prose_compression(orchestrator, **kwargs)
    _, stats, _ = run_state_prose_compression(orchestrator, **kwargs)

    assert orchestrator.calls == 1
    assert stats["cache_hits"] == 1
